"""Assert mmap CPU backup pause grows RSS and resume reclaims most of that growth."""

import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver

# Large enough that allocator noise (~0.25 GiB) cannot hide reclaim.
_TENSOR_BYTES = 2 * 1024**3
_TOLERANCE_BYTES = int(0.25 * 1024**3)


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    def rss_bytes() -> int:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("VmRSS not found in /proc/self/status")

    torch.cuda.synchronize()
    after_alloc_baseline = rss_bytes()

    with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
        tensor = torch.full((_TENSOR_BYTES,), 7, dtype=torch.uint8, device="cuda")

    torch.cuda.synchronize()
    after_alloc = rss_bytes()

    torch_memory_saver.pause()
    torch.cuda.synchronize()
    after_pause = rss_bytes()

    torch_memory_saver.resume()
    torch.cuda.synchronize()
    after_resume = rss_bytes()

    pause_delta = after_pause - after_alloc
    resume_delta = after_resume - after_pause

    print(
        f"rss_gib after_alloc={after_alloc / 1024**3:.3f} "
        f"after_pause={after_pause / 1024**3:.3f} after_resume={after_resume / 1024**3:.3f} "
        f"pause_delta={pause_delta / 1024**3:.3f} resume_delta={resume_delta / 1024**3:.3f} "
        f"(pre_region={after_alloc_baseline / 1024**3:.3f})"
    )

    assert tensor[0].item() == 7
    assert pause_delta >= _TENSOR_BYTES - _TOLERANCE_BYTES, (
        f"pause should grow RSS by ~tensor size; pause_delta={pause_delta}"
    )
    assert resume_delta <= -(_TENSOR_BYTES - _TOLERANCE_BYTES), (
        f"resume should reclaim ~tensor RSS; resume_delta={resume_delta}"
    )


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])

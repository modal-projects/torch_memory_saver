import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.utils import change_env


def run(hook_mode: str):
    """Pause/resume keeps content for mmap and pinned; CUDA frees host shadow on resume."""
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Env default reaches C++ TLS when Python does not pass an explicit backend.
    if not torch.version.hip:
        with change_env("TMS_INIT_CPU_BACKUP_BACKEND", "pinned"):
            torch_memory_saver._ensure_initialized()
            cdll = torch_memory_saver._impl._binary_wrapper.cdll
            with torch_memory_saver.region(enable_cpu_backup=True):
                assert cdll.tms_get_cpu_backup_backend() == b"pinned"
                env_tensor = torch.full((1_000_000,), 3, dtype=torch.uint8, device="cuda")
                # Nested backup-off must not clobber outer backend TLS.
                with torch_memory_saver.region(enable_cpu_backup=False):
                    assert not cdll.tms_get_enable_cpu_backup()
                assert cdll.tms_get_cpu_backup_backend() == b"pinned"
            del env_tensor
            torch.cuda.empty_cache()

    # One cycle per backend (default is mmap on CUDA / pinned on ROCm — covered by env/unit tests).
    backends = ["pinned"] if torch.version.hip else ["mmap", "pinned"]
    for backend in backends:
        print(f"Allocate tensor_with_backup backend={backend}")
        with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend=backend):
            tensor_with_backup = torch.full((20_000_000,), 10, dtype=torch.uint8, device="cuda")
            typed_tensor_with_backup = torch.randn((10, 20, 30), dtype=torch.float32, device="cuda")
            typed_tensor_with_backup_cpu_expected = typed_tensor_with_backup.cpu()

        print("Allocate tensor_without_backup")
        with torch_memory_saver.region(enable_cpu_backup=False):
            tensor_without_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device="cuda")

        assert tensor_with_backup[:3].tolist() == [10, 10, 10]
        assert tensor_without_backup[:3].tolist() == [20, 20, 20]

        torch_memory_saver.pause()
        typed_actual = torch_memory_saver.get_cpu_backup(typed_tensor_with_backup)
        assert typed_actual is not None
        assert torch.all(typed_tensor_with_backup_cpu_expected == typed_actual)

        tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device="cuda")
        torch_memory_saver.resume()

        if not torch.version.hip:
            assert torch_memory_saver.get_cpu_backup(typed_tensor_with_backup) is None

        assert tensor_with_backup[:3].tolist() == [10, 10, 10]
        assert tensor_without_backup[:3].tolist() != [20, 20, 20]

        del tensor_with_backup, typed_tensor_with_backup, tensor_without_backup, tensor_unrelated
        torch.cuda.empty_cache()

    # MemPool key must include backend (torch mode reuses pools by key).
    if hook_mode == "torch" and not torch.version.hip:
        impl = torch_memory_saver._impl
        backends_in_keys = {key[3] for key in impl._mem_pools}
        assert "mmap" in backends_in_keys and "pinned" in backends_in_keys, backends_in_keys


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])

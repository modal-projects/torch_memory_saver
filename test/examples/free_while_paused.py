"""Repro: freeing a pauseable tensor while it is paused.

pause() already cuMemUnmap + cuMemRelease's the VMM handle. free() still
unmaps/releases the same handle, which is undefined and may abort.
"""

import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    with torch_memory_saver.region():
        t = torch.full((8 * 1024 * 1024,), 1, dtype=torch.uint8, device="cuda")
    ptr = t.data_ptr()
    print(f"allocated ptr={ptr:#x}")
    torch_memory_saver.pause()
    print("paused; deleting tensor (triggers free while PAUSED)")
    del t
    torch.cuda.synchronize()
    print("free-while-paused completed without abort")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])

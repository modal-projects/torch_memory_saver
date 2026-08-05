"""Free an allocation while it is paused.

`pause()` already unmaps the memory and releases its VMM handle, thus `free()`
must not release that handle a second time.
See https://github.com/fzyzcjy/torch_memory_saver/issues/92

Calls the hooked CUDA API directly (like cuda_vmm_granularity.py), because
torch's caching allocator decides on its own when to release a block, and thus
cannot be used to free at a chosen point in time.
"""

import ctypes
import sys


_SIZE = 4 * 1024 * 1024
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _bind(api, name, argtypes, restype=ctypes.c_int):
    fn = getattr(api, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _assert_round_trip(cuda_memcpy, ptr, expected):
    host_write = ctypes.c_ubyte(expected)
    host_read = ctypes.c_ubyte()
    assert (
        cuda_memcpy(
            ptr,
            ctypes.byref(host_write),
            ctypes.sizeof(host_write),
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        == 0
    )
    assert (
        cuda_memcpy(
            ctypes.byref(host_read),
            ptr,
            ctypes.sizeof(host_read),
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        == 0
    )
    assert host_read.value == expected


def run(hook_mode: str):
    assert hook_mode == "preload"

    api = ctypes.CDLL(None)
    cuda_malloc = _bind(
        api, "cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    )
    cuda_memcpy = _bind(
        api,
        "cudaMemcpy",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    )
    cuda_free = _bind(api, "cudaFree", [ctypes.c_void_p])
    cuda_set_device = _bind(api, "cudaSetDevice", [ctypes.c_int])
    tms_pause = _bind(api, "tms_pause", [ctypes.c_char_p], None)
    tms_resume = _bind(api, "tms_resume", [ctypes.c_char_p], None)
    tms_set_enable_cpu_backup = _bind(api, "tms_set_enable_cpu_backup", [ctypes.c_bool], None)
    tms_set_enable_disk_backup = _bind(api, "tms_set_enable_disk_backup", [ctypes.c_bool], None)

    assert cuda_set_device(0) == 0
    tms_set_enable_disk_backup(False)

    # The backup (if any) has to be released by free() as well
    for enable_cpu_backup in [False, True]:
        print(f"Free while paused, {enable_cpu_backup=}")
        tms_set_enable_cpu_backup(enable_cpu_backup)

        ptr_to_free = ctypes.c_void_p()
        ptr_to_keep = ctypes.c_void_p()
        assert cuda_malloc(ctypes.byref(ptr_to_free), _SIZE) == 0
        assert cuda_malloc(ctypes.byref(ptr_to_keep), _SIZE) == 0
        _assert_round_trip(cuda_memcpy, ptr_to_keep, 0xA5)

        tms_pause(None)
        assert cuda_free(ptr_to_free) == 0
        tms_resume(None)

        # The allocation that was not freed is still usable
        _assert_round_trip(cuda_memcpy, ptr_to_keep, 0x5A)
        assert cuda_free(ptr_to_keep) == 0


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])

# Torch Memory Saver

A PyTorch library that allows tensor memory to be temporarily released and resumed later.

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## Examples and Features

### Basic Example

```python
# 1. For tensors that wants to be paused, create them within `region`
with torch_memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
torch_memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
torch_memory_saver.resume()
```

During the pause, physical memory is released and virtual address is preserved. When resume, virtual address is kept unchanged, while physical memory is re-allocated

### Multiple Tags

Please refer to https://github.com/sgl-project/sglang/issues/7009 for details.

```python
# 1. Create tensors with different tags
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="type2"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. Pause and resume with different tags selectively
torch_memory_saver.pause("type1")
torch_memory_saver.pause("type2")

torch_memory_saver.resume("type2")
torch_memory_saver.resume("type1")

torch_memory_saver.pause("type1")
torch_memory_saver.resume("type1")
```

### Release Memory in CUDA Graph

Not only does torch_memory_saver make tensors compatible with CUDA graph, but we can also release the memory held by CUDA graph (i.e. the intermediate tensors).

API: Change `torch.cuda.graph(...)` to `torch_memory_saver.cuda_graph(...)`

### CPU Backup

By default, in order to save time, the content is thrown away. This is useful for, for example, KV cache that are to be staled, or model weights that are to be updated.

If you want the tensor content to be kept unchanged, use `enable_cpu_backup`.

```python
with torch_memory_saver.region(enable_cpu_backup=True):
    tensor1 = torch.full((5_000_000_000,), 42, dtype=torch.uint8, device='cuda')

torch_memory_saver.pause()
torch_memory_saver.resume()

assert tensor1[0] == 42, "content is kept unchanged"
```

On CUDA, the default host shadow is anonymous `mmap` (`cpu_backup_backend="mmap"`). After resume copies data back to the device, **both** `mmap` and `pinned` host shadows are released (`munmap` / `cudaFreeHost`). Prefer `mmap` when you need process RSS to return reliably; prefer `pinned` when copy latency matters more (reclaim is not promised the same way as mmap).

```python
with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="pinned"):
    ...
```

ROCm stays pinned-only (`cpu_backup_backend="mmap"` is rejected); host shadows are retained across resume on the legacy ROCm path.

`TMS_INIT_CPU_BACKUP_BACKEND=mmap|pinned` sets the process default for preload / env-driven integrations (mmap is rejected on ROCm); an explicit `cpu_backup_backend=` argument overrides it. On CUDA, `get_cpu_backup` is only valid while allocations are paused — the host shadow is released on resume.

### Hook Modes

There are two hook modes:

* **preload**: Use `LD_PRELOAD` to hook CUDA's malloc and free API to change allocation behavior.
* **torch**: Use torch's custom allocator API to change allocation behavior.

The mode can be chosen by:

```python
torch_memory_saver.hook_mode = "torch"
```

### Example of RL with CUDA Graph

Please refer to `rl_example.py` for details.

## Development

```bash
make reinstall
```

You can use this command for local testing:

```bash
pytest /path/to/torch_memory_saver/test
```

Or this one to test a single case (e.g. the `simple` one here):

```bash
pytest /path/to/torch_memory_saver/test/test_examples.py::test_simple -s
```

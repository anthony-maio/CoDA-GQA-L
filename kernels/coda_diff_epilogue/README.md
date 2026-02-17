# Fused Differential Attention Epilogue Kernel

Replaces three separate PyTorch ops with a single CUDA kernel pass:

```
out_sig, out_noise = split(sdpa_output)     # memory read
diff = out_sig - lam * out_noise            # memory write + read
rms = rsqrt(mean(diff^2) + eps)             # memory write + read
output = diff * rms * weight                # memory write
```

Fused: one read of sig+noise, one write of output. Eliminates 2 intermediate tensor materializations.

## Build

```bash
cd kernels/coda_diff_epilogue
pip install -e .
```

## Benchmark

```bash
python benchmark_diff_epilogue.py
```

## Integration with CoDA-GQA-L

In `_sdpa_stacked_two_stream`, replace the epilogue:

```python
# Before (3 memory round-trips):
out_sig = out_cat[:, :H, :, :]
out_noise = out_cat[:, H:, :, :]
out = out_sig - lam * out_noise
out = self.head_norm(out)

# After (1 memory round-trip):
from coda_kernels import diff_epilogue
out_sig = out_cat[:, :H, :, :].contiguous()
out_noise = out_cat[:, H:, :, :].contiguous()
out = diff_epilogue(out_sig, out_noise, lam, self.head_norm.weight, eps=self.head_norm.eps)
```

Falls back to unfused PyTorch automatically if the kernel isn't compiled.

## Supported GPUs

| GPU | Compute Cap | Vectorization | Status |
|-----|-------------|---------------|--------|
| H100 | sm_90 | bf16x2 | Primary target |
| A100 | sm_80 | bf16x2 | Supported |
| T4 | sm_75 | fp16x2 | Supported (no bf16) |

## Files

```
kernel_src/diff_epilogue.cu     # CUDA kernel (bf16/fp16/fp32)
torch-ext/torch_binding.cpp     # PyTorch C++ bindings
torch-ext/coda_kernels/         # Python API
benchmark_diff_epilogue.py      # Micro-benchmark
build.toml                      # kernel-builder config
setup.py                        # pip install -e .
```

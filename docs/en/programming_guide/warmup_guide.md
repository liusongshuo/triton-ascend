# Triton-Ascend Warmup Guide

## Overview

`kernel.warmup(*args, grid=...)` compiles a kernel without executing it. The compiled artifact is written to the Triton cache, so a later real launch can load it directly.

Typical uses:

- pay compilation cost during service startup;
- pre-populate the compile cache in CI or a deployment pipeline;
- verify that a kernel compiles without allocating outputs or running it.

Warmup does not read or write input/output tensors, and it cannot detect runtime problems such as out-of-bounds access, numerical errors, or UB overflow. Do not confuse it with `do_bench(warmup=...)`: that parameter runs the kernel several times before timing, while `kernel.warmup()` only compiles.

## Plain JIT kernels

```python
compiled = add_kernel.warmup(x, y, out, n, BLOCK=256, grid=grid)
```

`grid` must be passed explicitly. The return value is a `CompiledKernel`; `metadata`, `asm`, and other compilation results are available.

To avoid allocating device memory for warmup, use `torch.dtype` instead of tensors:

```python
compiled = add_kernel.warmup(
    torch.float32,
    torch.float32,
    torch.float32,
    n,
    BLOCK=256,
    grid=grid,
)
```

A placeholder keeps only the dtype; its pointer is treated as 16-byte aligned and its shape as `[1]`. If real inputs are slices or offset views with potentially misaligned pointers, pass real tensors, otherwise the specialization may differ.

## Autotune kernels

```python
handles = kernel_add.warmup(a, b, out, N=1024, grid=(1, ))
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

- Each candidate configuration is compiled once, and the result is a list;
- The Ascend backend compiles in parallel by default, so elements may be `FutureKernel`; `.result()` returns a `CompiledKernel`, and the pattern above handles both forms;
- Warmup **does not benchmark or select a best configuration**; selection still happens on the first real launch;
- Automatic tiling with `configs=[]` is also supported;
- Compilation errors from parallel compilation surface before `warmup()` returns.

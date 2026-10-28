# Triton-Ascend Warmup Guide

## Overview

`kernel.warmup(*args, grid=...)` compiles a kernel without executing it. The compiled artifact is written to the Triton cache, so a later real launch can load it directly.

Typical uses and benefits:

- **Lower first-inference latency**: pay compilation cost at service startup or in CI so real requests no longer wait on compilation, with no first-request stall;
- **Earlier compile-error exposure**: compile failures surface at startup or CI time instead of being deferred to runtime;
- **Compile without executing**: verify a kernel compiles without allocating outputs or running it.

### Tips

- Use `torch.dtype` placeholders instead of tensors to avoid allocating device memory for warmup (see the examples below);
- Warm up every kernel you will use during service initialization, including autotune candidate configurations;
- Autotune warmup only compiles — it does not select a best configuration. Selection still happens on the first real launch, so do not treat warmup as "selection preheating".

Warmup does not read or write input/output tensors, and it cannot detect runtime problems such as out-of-bounds access, numerical errors, or UB overflow. Do not confuse it with `do_bench(warmup=...)`: that parameter runs the kernel several times before timing, while `kernel.warmup()` only compiles.

## Plain JIT kernels

This section covers how to warm up a plain `@triton.jit` kernel: compile it once ahead of execution, inspect the returned `CompiledKernel`, and avoid allocating device memory by using dtype placeholders.

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (triton.cdiv(n, 256),)

# Compile without executing. `grid` must be passed explicitly.
compiled = add_kernel.warmup(
    torch.float32, torch.float32, torch.float32, n,
    BLOCK_SIZE=256, grid=grid,
)
print(compiled.metadata)
```

`grid` must be passed explicitly. The return value is a `CompiledKernel`; `metadata`, `asm`, and other compilation results are available.

To avoid allocating device memory for warmup, pass `torch.dtype` instead of tensors (as shown above). A placeholder keeps only the dtype; its pointer is treated as 16-byte aligned and its shape as `[1]`. If real inputs are slices or offset views with potentially misaligned pointers, pass real tensors so specialization matches the real launch:

```python
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.randn(n, device="npu", dtype=torch.float32)
out = torch.empty_like(x)
compiled = add_kernel.warmup(x, y, out, n, BLOCK_SIZE=256, grid=grid)
```

## Autotune kernels

This section covers how to warm up a `@triton.autotune` kernel: every candidate configuration is compiled once and the artifacts are cached, but warmup **does not benchmark or select a best configuration** — selection still happens on the first real launch.

```python
import torch
import torch_npu
import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
]


@triton.autotune(configs=configs, key=["n_elements"])
@triton.jit
def add_kernel_autotune(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (1,)

# Compile every candidate once. No benchmark, no selection.
handles = add_kernel_autotune.warmup(
    torch.float32, torch.float32, torch.float32, n, grid=grid,
)
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

- Each candidate configuration is compiled once, and the result is a list;
- The Ascend backend compiles in parallel by default, so elements may be `FutureKernel`; `.result()` returns a `CompiledKernel`, and the pattern above handles both forms;
- Warmup **does not benchmark or select a best configuration**; selection still happens on the first real launch;
- Automatic tiling with `configs=[]` is also supported;
- Compilation errors from parallel compilation surface before `warmup()` returns.

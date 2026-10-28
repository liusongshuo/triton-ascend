# Triton-Ascend warmup 使用指南

## 概述

`kernel.warmup(*args, grid=...)` 只编译 kernel，不执行。编译产物写入 Triton 缓存，后续真实 launch 可以直接命中。

适用场景与好处：

- **降低首次推理时延**：在服务启动或 CI 阶段提前承担编译开销，真实请求不再等待编译，冷启动无卡顿；
- **提前暴露编译错误**：编译问题在启动或 CI 阶段即可发现，避免推迟到运行时；
- **只验证编译、不执行**：可不分配输出、不实际运行，仅确认 kernel 能否编译。

### 使用技巧

- 用 `torch.dtype` 占位符代替张量，避免为 warmup 分配设备内存（见下文示例）；
- 在服务初始化阶段集中 warmup 所有用到的 kernel，包括 autotune 的候选配置；
- autotune kernel 的 warmup 只编译、不选优，选优仍发生在首次真实 launch，不要把 warmup 当作"选优预热"。

warmup 不会读写输入输出，也不会发现越界、数值错误、UB 溢出等运行时问题。注意不要和 `do_bench(warmup=...)` 混淆：后者是计时前实际运行若干次，`kernel.warmup()` 只编译、不执行。

## 普通 JIT kernel

本节介绍如何对普通 `@triton.jit` kernel 进行 warmup：在真实执行之前提前编译一次，查看返回的 `CompiledKernel`，并通过 dtype 占位符避免分配设备内存。

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

# 只编译、不执行；grid 必须显式传入
compiled = add_kernel.warmup(
    torch.float32, torch.float32, torch.float32, n,
    BLOCK_SIZE=256, grid=grid,
)
print(compiled.metadata)
```

`grid` 必须显式传入；返回 `CompiledKernel`，可以读取 `metadata`、`asm` 等编译结果。

不想为 warmup 分配设备内存时，可以用 `torch.dtype` 代替张量（如上例所示）。占位符只保留 dtype，指针按 16 字节对齐、shape 按 `[1]` 处理。如果真实输入是切片、偏移视图等可能未对齐的指针，应传入真实张量，否则编译特化可能不同：

```python
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.randn(n, device="npu", dtype=torch.float32)
out = torch.empty_like(x)
compiled = add_kernel.warmup(x, y, out, n, BLOCK_SIZE=256, grid=grid)
```

## Autotune kernel

本节介绍如何对 `@triton.autotune` kernel 进行 warmup：每个候选配置都会被编译一次并缓存到 Triton 缓存，但 warmup **不做 benchmark、不选最优配置**——选优仍发生在第一次真实 launch。

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

# 只编译每个候选配置，不做 benchmark、不选优
handles = add_kernel_autotune.warmup(
    torch.float32, torch.float32, torch.float32, n, grid=grid,
)
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

- 每个候选配置编译一次，返回列表；
- Ascend 后端默认并行编译，元素可能是 `FutureKernel`；`.result()` 返回 `CompiledKernel`，可以统一按上面的写法处理；
- warmup **不做 benchmark、不选最优配置**，选优仍发生在第一次真实 launch；
- `configs=[]` 的自动 Tiling 同样支持；
- 并行编译的异常会在 `warmup()` 返回前抛出。

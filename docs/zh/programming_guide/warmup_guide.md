# Triton-Ascend warmup 使用指南

## 概述

`kernel.warmup(*args, grid=...)` 只编译 kernel，不执行。编译产物写入 Triton 缓存，后续真实 launch 可以直接命中。

适用场景：

- 服务启动阶段提前承担编译开销；
- CI 或部署流程中预填充编译缓存；
- 只验证 kernel 能否编译，不分配输出、不实际执行。

warmup 不会读写输入输出，也不会发现越界、数值错误、UB 溢出等运行时问题。注意不要和 `do_bench(warmup=...)` 混淆：后者是计时前实际运行若干次，`kernel.warmup()` 只编译、不执行。

## 普通 JIT kernel

```python
compiled = add_kernel.warmup(x, y, out, n, BLOCK=256, grid=grid)
```

`grid` 必须显式传入；返回 `CompiledKernel`，可以读取 `metadata`、`asm` 等编译结果。

不想为 warmup 分配设备内存时，可以用 `torch.dtype` 代替张量：

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

占位符只保留 dtype，指针按 16 字节对齐、shape 按 `[1]` 处理。如果真实输入是切片、偏移视图等可能未对齐的指针，应传入真实张量，否则编译特化可能不同。

## Autotune kernel

```python
handles = kernel_add.warmup(a, b, out, N=1024, grid=(1, ))
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

- 每个候选配置编译一次，返回列表；
- Ascend 后端默认并行编译，元素可能是 `FutureKernel`；`.result()` 返回 `CompiledKernel`，可以统一按上面的写法处理；
- warmup **不做 benchmark、不选最优配置**，选优仍发生在第一次真实 launch；
- `configs=[]` 的自动 Tiling 同样支持；
- 并行编译的异常会在 `warmup()` 返回前抛出。

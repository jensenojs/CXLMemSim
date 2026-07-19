# Kimi fatbin image selection

## 当前修订：cuBLAS 没有 sm_89 CUBIN 时的候选选择

这份计划保留此前“先选 ELF，避免把大 PTX 解压进 1 MiB BAR2 window”的历史原因；它的旧目标
`exact sm_89 ELF，否则第一个 PTX`已经被正式 Kimi failure 否决。fresh same-VM task
[`cnb-tv8-1jtu4cmdu`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-tv8-1jtu4cmdu) 在
`cublasCreate_v2` 前进入 `cuLibraryLoadData`。L40 是 `sm_89`，此份精确 `libcublas.so.12` fatbin
包含 `sm_80` ELF CUBIN 和 `sm_120` PTX，却没有 `sm_89` ELF。旧规则把第一个 PTX 交给真实 Driver，
Driver 返回 `CUDA_ERROR_INVALID_PTX (218)`，baseline 以 exit 134 终止，concordia case 没有启动。

当前修订只修复这个共享 image-selection 前提。它不把 lower-SM CUBIN 的接受性当成 CUDA 通用规则：
先由 `probe_cuda_fatbin_cubin_load.py` 从 SHA256 冻结的同一 library 提取 `sm_80` CUBIN，并在 L40
上以 `cuModuleLoadData` 验证；只有该 Driver probe 成功，才发布 selector 修复并重跑 Kimi paired
correctness。probe 的通过证明一份 image 被一张 L40 接受，不能代替 guest、BAR2、QEMU、HetGPU 或
baseline/concordia 输出比较。

## 背景与问题存在性

正式 Kimi same-VM development run 中，CUDA 12.9 runtime 传给 `cuLibraryLoadData` 的同一个 fatbin 包含两个 `sm_89` record：一个 PTX record 和一个 ELF record。PTX 的 zstd payload 为 460338 bytes，解压后为 5087083 bytes；ELF 的 zstd payload 为 379217 bytes，解压后为 2056864 bytes。

guest shim 当前按 record 顺序处理。它遇到 PTX 后立即在 guest 解压；解压目标超过 1 MiB BAR2 data buffer 时返回 `CUDA_ERROR_INVALID_VALUE`，后面的 ELF 从未进入候选选择。QEMU 已经实现 `CXL_GPU_CMD_MODULE_LOAD_CUBIN` 的 zstd 输入：guest 只需发送小于 1 MiB 的压缩 payload，QEMU 在 host 侧允许解压到 64 MiB 后再调用 HetGPU/NVIDIA backend。因此当前问题来自 fatbin image 选择顺序，不需要扩大 BAR2 或增加传输协议。

## 目标

在同一个 fatbin 已经过 record bounds 检查后，选择器使用以下窄排序：

```text
exact target-SM ELF
  -> target 同一 compute-capability major、SM 不高于 target 的最高 ELF
  -> SM 不高于 target 的最高 PTX
  -> 显式 CUDA_ERROR_NOT_SUPPORTED
```

ELF candidate 交给现有 CUBIN load 路径后的真实错误原样返回；它不会回退为 PTX，也不会把拒绝变成 success。
本轮只把 L40 `sm_89 -> sm_80` 的 library CUBIN 接受性作为发布 gate，不推断跨 major 或高于 target 的兼容性。

## 非目标

- 不改变 BAR2 register map、command number、data buffer size或QEMU实现。
- 不增加分块module传输。
- 不在ELF真实加载失败后静默退回PTX。
- 不实现CUDA fatbin对所有架构兼容规则的通用模拟。
- 不把本机 RTX 或 synthetic test 的结果写成 L40 Driver acceptance。

## Inventory

- `qemu_integration/guest_libcuda/libcuda.c`
  - `cudart_load_module_from_fatbin`拥有record验证与image选择。
  - `cxl_module_load_cubin`拥有压缩payload、encoding和uncompressed size的BAR2提交。
- `qemu_integration/guest_libcuda/cxl_gpu_cmd.h`
  - `CXL_GPU_DATA_SIZE`是单条BAR2数据区容量，保持1 MiB。
  - `CXL_GPU_MODULE_DATA_ZSTD`已经声明压缩CUBIN编码。
- `qemu-cxl-type2/hw/cxl/cxl_type2.c`
  - `CXL_GPU_CMD_MODULE_LOAD_CUBIN`验证压缩输入不超过data buffer，在host解压到最多64 MiB，再交给backend。

## 改后形状

`cudart_load_module_from_fatbin`先遍历并验证所有record，同时保存最高的合法候选：PTX 必须不高于
target SM；ELF 除此前提外还必须与 target 位于同一 compute-capability major。遍历完成后：

1. 有兼容ELF时，调用现有CUBIN load路径。zstd payload保持压缩状态跨BAR2传输。
2. 没有兼容ELF而存在兼容PTX时，调用现有PTX解码和load路径。
3. 两者都不存在时显式失败。

被选择的ELF如果被QEMU、HetGPU或NVIDIA Driver拒绝，现有错误路径保持可见。选择器不吞掉错误，也不执行第二次表示切换。QEMU当前会把backend CUBIN load失败映射为协议已有错误码；底层错误身份仍需从同窗QEMU与HetGPU日志区分。

## 验收

- `test_cxl_gpu_context_shim`构造 `sm_80 ELF + sm_90 ELF + sm_120 PTX`，target 为 `sm_89`；只接受
  `CXL_GPU_CMD_MODULE_LOAD_CUBIN`，并验证发送的 payload 来自 `sm_80`。旧 first-PTX 规则必须使该测试失败。
- guest shim构建、transport/case-control测试和 Python probe 编译通过。
- L40 diagnostic 以失败任务中冻结的 `libcublas.so.12` SHA256、`files_size=95856`、`cubin_sm=80`、
  `expected_device_sm=89` 执行 `--load`；immutable result 必须保存 library/image SHA256、fatbin offset、
  Driver version、GPU SM、`cuModuleLoadData` 与 unload 返回码。
- L40 probe 成功后才触发组件 build、candidate、fresh-pull、promotion、guest rebuild 和新的 fresh Kimi
  baseline -> concordia paired task；最终 comparator 的 `verdict=pass` 才关闭教程 correctness。
- L40 probe 若拒绝 CUBIN，保留失败 result 与原始错误，停止 selector promotion，并从 Driver 错误重新收窄候选规则。

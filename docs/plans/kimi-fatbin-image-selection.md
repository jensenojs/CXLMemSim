# Kimi fatbin image selection

## 背景与问题存在性

正式 Kimi same-VM development run 中，CUDA 12.9 runtime 传给 `cuLibraryLoadData` 的同一个 fatbin 包含两个 `sm_89` record：一个 PTX record 和一个 ELF record。PTX 的 zstd payload 为 460338 bytes，解压后为 5087083 bytes；ELF 的 zstd payload 为 379217 bytes，解压后为 2056864 bytes。

guest shim 当前按 record 顺序处理。它遇到 PTX 后立即在 guest 解压；解压目标超过 1 MiB BAR2 data buffer 时返回 `CUDA_ERROR_INVALID_VALUE`，后面的 ELF 从未进入候选选择。QEMU 已经实现 `CXL_GPU_CMD_MODULE_LOAD_CUBIN` 的 zstd 输入：guest 只需发送小于 1 MiB 的压缩 payload，QEMU 在 host 侧允许解压到 64 MiB 后再调用 HetGPU/NVIDIA backend。因此当前问题来自 fatbin image 选择顺序，不需要扩大 BAR2 或增加传输协议。

## 目标

逐个验证可解析 fatbin record 的边界后，优先选择与当前 Type-2 GPU compute capability 精确匹配的 ELF。没有匹配 ELF 时，才使用已有 PTX 路径。

## 非目标

- 不改变 BAR2 register map、command number、data buffer size或QEMU实现。
- 不增加分块module传输。
- 不在ELF真实加载失败后静默退回PTX。
- 不实现CUDA fatbin对所有架构兼容规则的通用模拟。

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

`cudart_load_module_from_fatbin`先遍历并验证所有record，同时保存第一个PTX和第一个与当前设备`major * 10 + minor`精确匹配的ELF。遍历完成后：

1. 有精确匹配ELF时，调用现有CUBIN load路径。zstd payload保持压缩状态跨BAR2传输。
2. 没有匹配ELF而存在PTX时，调用现有PTX解码和load路径。
3. 两者都不存在时显式失败。

被选择的ELF如果被QEMU、HetGPU或NVIDIA Driver拒绝，现有错误路径保持可见。选择器不吞掉错误，也不执行第二次表示切换。QEMU当前会把backend CUBIN load失败映射为协议已有错误码；底层错误身份仍需从同窗QEMU与HetGPU日志区分。

## 验收

- guest shim构建通过。
- 现有transport与case control测试通过。
- exact CUDA 12.9 sm_89 library probe出现`library CUBIN load selected`，其中compressed size为379217、uncompressed size为2056864。
- `cuLibraryLoadData`成功，`ggml_cuda_init`不再报告`invalid argument`，且不得出现`no usable GPU found`或`--gpu-layers option will be ignored`。
- probe在模型读取前结束；通过后才重新组装type2-guest并进入完整Kimi。

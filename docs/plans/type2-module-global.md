# Type-2 module global 查询

## 问题与证据

扰动等级为 BR2。固定 1.5B 的受控重放在首个 module 初始化时调用：

```text
cuModuleGetGlobal_v2(dptr=<valid>, bytes=NULL, mod=0x1, name=kmask_iq2xs)
```

未导出入口时，libcudart 在调用 guest `cuMemGetInfo` 前返回错误 36；诊断入口返回 `CUDA_ERROR_NOT_SUPPORTED` 后，最终错误同步变为 801。这证明 module global 查询是当前运行路径直接消费的必要接口。

首次跨BAR2接通后，Concordia现有导出从通用`unimplemented`宏返回成功，但`dptr=0,size=0`。固定1.5B因此越过801，却没有获得真实global。这个结果证明“符号已导出”不能作为backend能力证据；Concordia的NVIDIA路径必须直接调用系统Driver。

NVIDIA Driver API允许`dptr`或`bytes`其中一个为空，但不能同时为空：<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>。

## 目标、非目标与不变量

目标是让guest通过现有BAR2命令通道查询backend module中的真实device global地址和大小，并保留CUDA错误语义；Concordia的NVIDIA导出必须透传系统`cuModuleGetGlobal_v2`。非目标包括实现managed global、复制global内容、改变module加载方式、扩展Concordia backend抽象，或直接解决查询成功之后的模型加载问题。

长期不变量是：guest handle只编码QEMU持有的module-id；QEMU只把backend返回的device pointer交回guest；符号不存在必须可见；`bytes=NULL`不影响地址查询；tiny计算路径不因新增命令发生变化。

## 影响面 inventory

```text
CXLMemSim/
└── qemu_integration/guest_libcuda/
    ├── cxl_gpu_cmd.h       # guest命令枚举与CUDA错误镜像
    └── libcuda.c           # cuModuleGetGlobal入口和BAR2请求

qemu-cxl-type2/
├── include/hw/cxl/
│   ├── cxl_type2_gpu_cmd.h # host命令枚举与错误镜像
│   └── cxl_hetgpu.h        # QEMU内部backend查询接口
└── hw/cxl/
    ├── cxl_type2.c         # BAR2命令handler与module-id检查
    └── cxl_hetgpu.c        # dlsym cuModuleGetGlobal_v2并调用backend

Concordia/
├── zluda/src/lib.rs                         # 将cuModuleGetGlobal_v2移出通用未实现导出
├── zluda/src/impl/module.rs                 # 用wrapper中的真实NVIDIA module handle查询global
├── ext/nvidia_runtime-sys/src/lib.rs        # 动态Driver函数表的最薄调用包装
└── ext/ze_runtime-sys/build.rs              # 识别Fedora /usr/lib64中的现有Level Zero loader
```

定义位置有两份命令枚举镜像；主要调用点各一个：guest Driver API入口、QEMU switch handler、hetGPU backend wrapper。直接构造点是`libcuda.c`写`PARAM0=module-id`并把符号名写入data buffer。输入边界是libcudart提供的module handle和NUL结尾符号名。读取点是QEMU handler。受影响仓库为CXLMemSim和qemu-cxl-type2两个；Concordia只作为现有backend合同消费者。
定义位置有两份命令枚举镜像；主要调用点是guest Driver API入口、QEMU switch handler、hetGPU backend wrapper和Concordia NVIDIA导出。直接构造点是`libcuda.c`写`PARAM0=module-id`并把符号名写入data buffer。输入边界是libcudart提供的module handle和NUL结尾符号名。读取点是QEMU handler与系统NVIDIA Driver。受影响仓库为CXLMemSim、qemu-cxl-type2和Concordia三个；不增加第四套module身份。

## 最小接口约束

冻结命令`CXL_GPU_CMD_MODULE_GET_GLOBAL = 0x34`：

```text
请求
  PARAM0 = guest module handle对应的module-id
  DATA   = NUL结尾的global symbol name

成功响应
  RESULT0 = backend device pointer
  RESULT1 = global size in bytes

失败响应
  module-id越界       -> CUDA_ERROR_INVALID_HANDLE
  symbol不存在        -> CUDA_ERROR_NOT_FOUND
  backend无此能力     -> CUDA_ERROR_NOT_SUPPORTED
  非法输入            -> CUDA_ERROR_INVALID_VALUE
```

guest可以根据调用方是否提供`dptr`或`bytes`选择写回哪个输出，但QEMU/backend始终查询并返回两个事实。下游不得虚构device pointer、分配替代内存、返回host地址，或把NOT_FOUND改成成功。

## 改后预期调用链

```text
libcudart
  -> guest cuModuleGetGlobal[_v2]
     -> BAR2 MODULE_GET_GLOBAL(module-id, symbol-name)
        -> QEMU cxl_type2 handler
           -> hetgpu_get_global
              -> Concordia/NVIDIA cuModuleGetGlobal_v2
           <- device pointer + size
        <- RESULT0 + RESULT1
     <- 按调用方非空输出写回
```

## 删减与边界

删除guest当前只返回NOT_SUPPORTED的诊断函数体，用真实BAR2调用替换。将Concordia的`cuModuleGetGlobal_v2`移出“未实现即成功”导出，直接复用现有NVIDIA函数表和`Module.cuda_module`。保留同一个`cuModuleGetGlobal`兼容别名。不开辟新的module registry或backend插件层；沿用QEMU已有`modules[]`及guest module-id编码。

## 验收与回滚

1. guest shim和QEMU增量构建通过，两个命令头的opcode和错误码一致。
2. local-kvm tiny仍返回`1234`。
3. 固定1.5B日志显示`kmask_iq2xs`查询进入系统NVIDIA Driver，成功结果的device pointer和size均非零，并越过当前801；出现的新失败必须保留为下一边界，不能用默认值关闭。
4. 若真实backend返回错误，按CUDA错误向guest传播；若只能得到host指针或size不可确定，停止，不建立兼容映射层或零值成功。

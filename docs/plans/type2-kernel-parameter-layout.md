# Type-2 kernel 参数布局

## 问题与存在性

扰动等级为 BR2。固定1.5B在真实module global和HtoD async通过后完成模型tensor加载、约68.24 MiB GPU模型buffer、约28.09 MiB compute buffer与warmup准备，随后第一次真实kernel launch以`rc=139`退出。guest最后一条日志是：

```text
cuLaunchKernel(f=0x1472, grid=(6,1,1), block=(128,1,1), shared=0)
```

QEMU日志没有对应的backend launch入口。guest当前通过“读取`kernelParams[i]`直到遇到NULL”猜参数数量，并把每项截成固定八字节槽。CUDA Driver API没有规定`kernelParams`以NULL结尾；真实函数参数还包含大于八字节的by-value类型。源码因此在BAR2命令前越过调用方数组边界，tiny手工NULL终止的夹具不能代表真实libcudart。

CUDA 12.9本机头文件`/usr/local/cuda-12.9/targets/x86_64-linux/include/cuda.h`声明`cuFuncGetParamInfo`：按参数索引返回device-side参数布局中的offset与size，索引超出参数数量时返回`CUDA_ERROR_INVALID_VALUE`。它提供了真实Driver已有的参数布局权威，不需要解析mangled name、PTX或维护第二套kernel签名表。

## 目标、非目标与不变量

目标是让guest从backend查询每个参数的offset和size，按真实布局把参数字节写入BAR2 DATA，并让QEMU使用同一backend布局构造`kernelParams[]`后调用现有launch链。

非目标包括实现`extra` launch buffer协议、stream并发、动态并行、kernel签名缓存、PTX签名解析或新的function registry。

长期不变量：function身份继续是guest可见的QEMU function-id；参数布局只由持有真实NVIDIA function的backend回答；guest不得用NULL哨兵或固定八字节猜参数；QEMU不得直接解引用guest指针；tiny与真实模型使用同一套布局协议。

## 影响面 inventory

```text
CXLMemSim/
└── qemu_integration/guest_libcuda/
    ├── cxl_gpu_cmd.h       # FUNC_GET_PARAM_INFO opcode镜像
    └── libcuda.c           # cuFuncGetParamInfo与launch参数buffer构造

qemu-cxl-type2/
├── include/hw/cxl/
│   ├── cxl_type2_gpu_cmd.h # opcode镜像
│   └── cxl_hetgpu.h        # backend param-info接口
└── hw/cxl/
    ├── cxl_type2.c         # 查询handler与launch参数指针重建
    └── cxl_hetgpu.c        # dlsym并调用Concordia cuFuncGetParamInfo

Concordia/
├── zluda/src/lib.rs                         # NVIDIA feature下启用cuFuncGetParamInfo
├── zluda/src/impl/function.rs               # 从NvidiaKernel取真实CUfunction
└── ext/nvidia_runtime-sys/src/lib.rs        # 系统Driver函数表与最薄wrapper
```

定义位置是两份command enum镜像与CUDA公开API。主要调用点是guest `cuLaunchKernel`、QEMU launch handler和Concordia NVIDIA function wrapper。直接构造点是libcudart提供的`kernelParams`指针数组。输入边界是guest进程地址；QEMU不能读取这些地址，因此参数字节必须在guest侧复制。受影响仓库为三个，既有function-id和NvidiaKernel wrapper继续作为唯一function身份。

## 最小接口约束

冻结命令`CXL_GPU_CMD_FUNC_GET_PARAM_INFO = 0x35`：

```text
请求
  PARAM0 = function-id
  PARAM1 = parameter index

成功响应
  RESULT0 = parameter offset
  RESULT1 = parameter size

失败响应
  function-id越界  -> CUDA_ERROR_INVALID_HANDLE
  index越界        -> CUDA_ERROR_INVALID_VALUE
  backend无能力    -> CUDA_ERROR_NOT_SUPPORTED
```

launch继续使用现有`CXL_GPU_CMD_LAUNCH_KERNEL`，但DATA从“连续八字节值”收紧为“backend device-side parameter layout”：

```text
DATA[param_offset : param_offset + param_size] = guest kernelParams[i]指向的字节
PARAM4[63:32] = parameter count
PARAM4[31:0]  = shared memory bytes
PARAM5        = parameter buffer extent
```

guest从index 0顺序查询，首次`INVALID_VALUE`表示参数列表结束；其他错误必须传播。QEMU按相同function和index重新查询布局，并令`host_kernel_params[i] = DATA + offset`。任一offset、size、count或extent越过1 MiB DATA边界即返回`INVALID_VALUE`。不得把参数截成八字节、解析mangled name、复制guest指针值作为host地址，或新增签名缓存。

## 改后预期调用链

```text
libcudart cuLaunchKernel(kernelParams)
  -> guest逐项FUNC_GET_PARAM_INFO
     -> QEMU hetgpu_get_param_info
        -> Concordia NvidiaKernel
           -> NVIDIA cuFuncGetParamInfo
  -> guest按offset/size复制参数字节到BAR2 DATA
  -> BAR2 LAUNCH_KERNEL(function-id, count, extent)
     -> QEMU按同一布局构造host_kernel_params[]
     -> Concordia/NVIDIA cuLaunchKernel
```

## 删减、停止与回滚

删除guest的NULL哨兵扫描和固定八字节槽假设；保留现有function registry、launch opcode与1 MiB DATA buffer。第一刀只增加参数布局查询，不实现缓存或完整stream模型。

如果系统NVIDIA Driver对当前函数不能提供参数布局，停止并保留错误；不回退到mangled-name解析或按某个kernel特判。若参数extent超过DATA buffer，记录真实需求后重新设计bulk参数通道，不静默截断。

## 验收

1. Concordia准确构建入口通过，已包装的`NvidiaKernel`能返回真实offset/size。
2. QEMU增量构建通过，0x35与guest镜像一致；launch handler不再假设八字节槽。
3. guest shim构建通过，tiny仍返回1234。
4. 固定1.5B首次kernel进入QEMU和真实NVIDIA backend，越过当前guest `rc=139`；新失败按实际日志成为下一边界。
5. run manifest、argv、payload/initrd provenance和日志SHA256齐全；不得把warmup开始或单次kernel进入backend写成模型correctness pass。

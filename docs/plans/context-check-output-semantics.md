# CONTEXT_CHECKS 输出语义

## 问题存在性

真实 L40 Driver 的 immutable diagnostic result `cnb-4ng-1ju16mq3q` 保存了
`CONTEXT_CHECKS` UUID `263e8860-7cd2-6143-92f6-bbd5006dfa7e` slot 2 的三次自然调用。
三次调用都把 `result1` 指向的四字节写成零，保留 `result2` 指向的八字节原值，并返回
`CUDA_SUCCESS`。当前 guest shim 同样写零到 `result1`，随后无条件把 `result2` 清成
`NULL`。这项写入是成功 native cuBLAS 初始化与失败 guest `cublasCreate_v2=14` 之间最早的
具体输出差异。

完整输入、原始字节与证明边界由 cxl-lab 的
`docs/evidence/kimi-context-checks-output-cnb-4ng-native-slot2.md` 保存。

## 目标

让 guest `context_check()` 复现已经观察到的输出合同：

```text
有效 result1  -> 写 0
有效 result2  -> 保留调用方原值
return         -> CUDA_SUCCESS
```

修改后使用同一 Type-2 cheap gate 检查中间调用输入与 `cublasCreate_v2` 是否继续前进。

## 影响边界

定义与直接构造点都位于 `qemu_integration/guest_libcuda/libcuda.c`：

- `context_check()` 是 `CONTEXT_CHECKS_TABLE[2]` 的唯一实现。
- `cuGetExportTable()` 为该 UUID 返回 `CONTEXT_CHECKS_TABLE`。
- `context_check_result2_stub()` 只被两个诊断环境分支引用。

直接 fixture 位于 `qemu_integration/guest_libcuda/test_cxl_gpu_context_shim.c`，它已经通过
`cuGetExportTable()` 测试其他 private table，不需要新增测试入口。

这项修改不改变 BAR2 ABI、QEMU handler、HetGPU、CUDA public API、context-storage map 或
component payload shape。

## 改动

- 删除 `context_check_result2_stub()`。
- 删除 `context_check()` 中的 `*result2 = NULL`。
- 删除 `CXL_CUDA_CONTEXT_CHECK_RESULT2_HOSTLIKE` 与
  `CXL_CUDA_CONTEXT_CHECK_RESULT2_ARG4_STUB` 两个诊断分支。
- 保留 `result1=0`、返回 `CUDA_SUCCESS` 和现有调用/返回日志。
- 增加一个通过 export table 取得 slot 2 的 fixture，按 native 三次调用形状放入不同的非空
  `result2` sentinel，验证每次调用后 `result1=0` 且 `result2` 原值不变。

## 删除方案成立的原因

真实 Driver 在三次自然调用中都没有写 `result2`。guest 无需构造替代对象、复制 host 地址或维护
新的 private 生命周期。删除 guest 多余写入即可得到同一可观察输出。

保留诊断环境分支会让同一函数拥有两种相互冲突的输出合同；根据 `arg4` 或 `arg5` 构造 guest
函数指针会引入本轮证据没有观察到的状态。两种路径都不进入实现。

## 验收

- 新 fixture 在旧实现上因 `result2` 被清空而失败，在新实现上通过。
- `test-context-shim` 通过。
- guest shim 完整构建通过。
- 源码中不再存在两个 `RESULT2` 诊断环境变量或 `context_check_result2_stub`。
- CNB component build、fresh pull 和 type2-guest promotion 使用 exact 新提交。
- Type-2 cheap gate 保存三次 guest 调用，首先判断中间 `arg5` 是否恢复为 `0xffffffff`，以及
  `cublasCreate_v2` 是否从 `14/NULL` 继续前进。

该 gate 通过以后仍需执行正式 same-VM `baseline → concordia → comparator`；本改动与 fixture
不签署 Kimi correctness。

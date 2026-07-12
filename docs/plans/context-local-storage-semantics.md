# CUDA context-local storage 语义收拢

## 问题与影响面

固定 1.5B 的受控 Type-2 重放已经加载 133 个 CUBIN，并在 `cudaMemGetInfo` 进入 guest shim 前返回错误 36。更小的 fatbin probe 已证明：fatbin 注册前 `cudaMemGetInfo` 会进入 `cuMemGetInfo` 并成功，注册后失败发生在 libcudart 消费 context-local state 的阶段。

当前 `CONTEXT_LOCAL_STORAGE_TABLE` 偏离同仓 Concordia/ZLUDA 参考实现：它把空 context 转换为全局 context，以 `(context, state_mgr)` 为键，根据调用次数返回不同错误，并把 vtable 第二槽实现为删除操作。这些行为没有来自 runtime 合同的依据。

本切片为 BR1。修改限制在 guest `libcuda.so.1` 的进程内状态表，不改变 BAR2 命令 ABI、QEMU 设备模型、CXLMemSim server 或公开 CUDA Driver API。

## 改后预期形状

```text
guest libcudart
  -> CONTEXT_LOCAL_STORAGE_TABLE.ctor(raw CUcontext, state_mgr, ctx_state, dtor)
     -> 按 raw CUcontext 保存 ctx_state 与 state_mgr
  -> CONTEXT_LOCAL_STORAGE_TABLE.get(out, raw CUcontext, state_mgr)
     -> 命中时返回同一 ctx_state
     -> 缺失时固定返回 CUDA_ERROR_INVALID_VALUE
  -> CONTEXT_LOCAL_STORAGE_TABLE.dtor(...)
     -> 保持参考 ABI 的无状态占位，不伪造删除语义
```

`qemu_integration/guest_libcuda/libcuda.c`仍是唯一实现位置。状态表只负责保存和取回 runtime 交付的对象，不修改对象私有字节，不推断当前 context，也不为缺失状态选择不同错误。

## 删除与保留

删除空 context 到 `g_context`/`g_primary_context` 的转换、`state_mgr`键比较、调用次数驱动的错误码和错误的 delete 槽。保留固定容量、进程内锁、runtime 提供的析构回调记录，以及 context reset/destroy 时已有的清理路径。

## 验收

同一改动必须依次满足：

1. guest shim构建成功。
2. 受控local-kvm tiny重放继续得到`kernel_probe result=1234 expected=1234`，完整manifest和verifier通过。
3. fatbin前后meminfo probe能区分错误发生在runtime还是driver；若仍失败，日志必须给出新的首个状态差异。
4. 固定1.5B受控重放不回退133个CUBIN加载。只有实际越过`cudaMemGetInfo`才算本切片解决了阻断。

出现tiny回归、vtable调用ABI不匹配或新的静默fallback时立即拒绝该改动。

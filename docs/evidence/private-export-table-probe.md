# L40 CUDA private export-table probe

## 失败链与工具链修复

Kimi same-VM baseline 的 core 已定位到 `cudaFuncSetAttribute`：CUDA Runtime 从 guest shim 提供的
`TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE` 取 slot 1，随后以 selector `0x111` 间接调用；该槽位为 NULL，
进程跳转到 RIP 0。调用方是
`ggml_cuda_flash_attn_ext_mma_f16_case<576, 512, 2, 16>`。原始 core 继续保留在
`cxl-lab/.codex-runs/kimi-paired-callback-hooks/cnb-jho-1jtrgsfn7/`，不压缩、不删除。

通用动态 probe 的首个 L40 task [`cnb-ll9-1jts8onjj`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-ll9-1jts8onjj)
尚未执行 probe：CNB stage 默认使用 `/bin/sh`，而 event 中的 `set -o pipefail` 使它在 shell
入口失败。runner log SHA256 是
`f275bbf9f250e7ea2a6f268830285af93276c3648c4ae0577db83ba34f344d7b`。

提交 `51eca64be07016b92334f19dfd0fc402e8622f88` 将 event 固定为 Bash 后，第二次 L40 task
[`cnb-e8o-1jts8u69n`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-e8o-1jts8u69n)
完成了 tiny trigger 和 probe build，但在运行前明确失败：固定工具镜像没有 Python-enabled `gdb`。
runner log SHA256 是
`eb1889417117018e3e8ddace6222b7d30e894f7cfdbbf86d20d49a98521030e2`。

工具链提交 `6caeb848ec20514347d8b793bdcf3b7d436b563b` 添加 `gdb` 和构建期 Python 断言；它的首次
构建 [`cnb-buo-1jts96cbu`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-buo-1jts96cbu)
暴露旧 `FROM` digest 已从 registry 消失。该失败 runner log SHA256 是
`e00a1bc74401282a81f94906a2c3dbec52f705ee682437c4bd26fad28b33cbe9`。

随后提交 `652975c7174134dc638546650456e13f0da3a58d` 把 base 切换到已由其他 L40 event 拉取的
`sha256:8732c435c97964f2ba95b42fbbcaf79b3c23feffc3b1f7683531917122f4f59e`。
工具链任务 [`cnb-nu8-1jts9fbnb`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-nu8-1jts9fbnb)
在约两分钟内成功，确认 base pull 和 `gdb-python-ok`。它发布的 immutable manifest-list digest 是：

```text
docker.cnb.cool/gevico.online/jensen/cxl-lab@sha256:f5199f028c436e6d08b698602ce8207abf5ad650dba6c95eb6388aaa8393a77f
```

该任务的 runner log SHA256 是
`edc2a15766ef9ec48c715b6a806640c8f8fb4b7140ef6d05ba56e24f322372b0`。

## L40 观察结果

提交 `dda577d61efa7f29fe522f50b6da4b9fa95fe23a` 只把
`api_trigger_private_export_table_probe` 切换到上述 GDB toolchain，未改 component build、已冻结的
Kimi run manifest 或历史 result。

任务 [`cnb-f8g-1jts9mpgc`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-f8g-1jts9mpgc)
在 L40、16 CPU、64 GiB runner 上成功完成。runner log SHA256 是
`36aa9512780e87e94d32f1f851289dda6e00073cf5fd3958d77209716a8e5236`。它的两次 trigger identity
完全一致：

```text
GPU             NVIDIA L40 / GPU-96e4c3b4-886e-1f1f-1c87-760b2dcba46c / SM 8.9
Driver          580.95.05 / Build ID b21cc51b8b73433abb27468d0d2ed55a0135ee44
CUDA Runtime    libcudart.so.12.9.79 / Build ID dee4b68d7f29b160323eff1e31338d5ebf73d190
Debugger        /usr/bin/gdb / SHA256 3832cc070ae1716e322105d3b39fb398695e5f031c9d39224cf227a8c2b889f6
Trigger         cuda-runtime-dlopen-kernel-probe / SHA256 6b599fbe845a366f5438425fcb43ca2aff393921f5804cfbbfc93e833261c9d4
```

tiny trigger 实际完成 `cudaFuncSetAttribute`、`cudaMalloc`、HtoD、kernel launch、同步、DtoH 与
`cudaFree`，最终输出 `result=1234`。discovery 成功解析七张 Runtime 请求的 private table，并记录
22 个自然调用。目标 callback-hooks UUID
`a094798c-2e74-2e74-93f2-0800200c0a66` 的自然调用分布是：

```text
slot 1  0 次
slot 2  1 次
slot 6  1 次
```

capture 以 UUID、slot 1、selector `0x111` 和 `rsi:64` 运行；其 `capture_status` 为
`not_reached`，`captures.jsonl` 为空，且 discovery/capture identity comparison 为 pass。完整
`identity.json`、`tables.jsonl`、`calls.jsonl`、`captures.jsonl`、`summary.json` 与 GDB transcript
均以明确 `BEGIN/END` 标记保存在 runner log 中。

## 证明边界与下一步

这次任务证明 L40 上的 Python GDB probe 能够记录真实 CUDA Runtime 对 private export table 的自然
请求，并能以 GPU、Driver、Runtime、trigger 和源码身份 fail-closed 地关联 discovery 与 capture。
它证明这个 tiny CUDA 路径没有到达 Kimi core 所需的 callback-hooks slot 1 / selector `0x111`。

它没有给出 slot 1 的入口寄存器、返回值或 `rsi` 状态变化，因此不能实现 guest shim callback，不能关闭
Kimi baseline，也不能证明 Type-2、HetGPU translation 或 TPS。

下一步是让同一个 probe 包裹真实 `libggml-cuda.so.0` / llama 图执行路径，最小化到足以自然进入
`ggml_cuda_flash_attn_ext_mma_f16_case<576, 512, 2, 16>` 的调用。只有该 trigger 得到 slot 1 的成对
entry/return capture，才可以按实际 ABI 修改 `TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[1]`；随后仍需重建
component、guest并重新运行同一 Kimi paired correctness contract。

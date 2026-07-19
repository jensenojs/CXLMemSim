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

## exact DSO trigger 的自然返回与 companion 输出

cxl-lab 任务 [`cnb-8ro-1jtsh5ghm`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-8ro-1jtsh5ghm)
消费 CXLMemSim `9b9b2ca83cb61ba41c130ef299458de338a6a3e8`，在同一次 exact
`libggml-cuda.so.0` public trigger 中为十四条自然 private 调用保存了十四条 sequence 配对的 return。
runner log SHA256 是
`03d75f822fd63ec2c3c2eb77d7af67fed20d8010b55b10b233405d992f0e325b`。immutable result 身份是：

```text
result digest       sha256:3cacae52ec56e55e3cd1e351df27f15b969e45a47c524e679744958e3d36270c
archive SHA256      1e9528151077fea160bfd9b2e0ed7adb3362573edb54f6a69760564e3b3cb8dc
manifest SHA256     785b004c0eb01033682d80386ff15482afde9db7d87196f8c56cfe6513f061e4
```

该结果首次证明探针在真实 L40 Runtime 上能保持 entry/return 完整配对：`call_records=14`、
`return_records=14`、`unreturned_sequences=[]`。callback-hooks slot 2 与 slot 6 均返回非零 `RAX`，但
guest 实现已经表明它们是 `void (void **ptr, size_t *size)`；void 函数的残留 `RAX` 不能解释 ABI。
因此下一条判别量是两个 output word，而不是继续推断返回码。

CXLMemSim `cf18bc60f66c62421ab06411755cce3bf874b82e` 随后允许一个 capture 声明多个整数参数寄存器窗口，
每个窗口仍有独立字节上限，且不沿读出的指针继续解引用。cxl-lab 任务
[`cnb-dvo-1jtshu9k3`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-dvo-1jtshu9k3)
在一个 Job 内对同一 public trigger 顺序执行 target、slot 2 companion 和 slot 6 companion。三次运行的
GPU、Driver、Runtime、trigger、debugger 和 source identity 完全一致。runner log SHA256 是
`aa3c1b88e46d9c5048853dddb7b146eac4e064c9ebf77350a4cd7397875380e3`，immutable result 身份是：

```text
result digest       sha256:a1c16a34bd8a1ed6f90907eef3e7a36f5f831b898001fa3980178b64979088f9
archive SHA256      9becfdf0b479f699d59e1f3404b06d37d50e502cc0c81b76009ff1365a565e43
manifest SHA256     5acb166d5e0db3f1e4482b8b1293dc6ef89c1777af1aa16f7e75e84c6b4e604c
```

L40 companion capture 得到：

| callback-hooks slot | `*ptr` before | `*ptr` after | `*size` before | `*size` after |
| ---: | --- | --- | ---: | ---: |
| 2 | `NULL` | 非空 Driver buffer | 0 | 1024 |
| 6 | `NULL` | 非空 Driver buffer | 0 | 14 |

guest 的 `tools_get_buffer1` 与 `tools_get_buffer2` 分别返回 1024-byte 和 14-byte 静态 buffer。这组证据
排除了“这两个 getter 写出错误长度，因而让 Runtime 在简化 trigger 与 Kimi 中选择不同后续路径”这一解释。
它还没有读取返回 buffer 的内容，也没有观察 Runtime 后续如何使用这些 buffer，因此不能宣布 slot 2/6 的
完整生命周期等价。

## 返回 buffer 内容排除了最后一个 getter 初始化候选

CXLMemSim `372b32b0b0631c23e98f7ec82acafd3376c80dc1` 为自然调用的已知
`void (void **ptr, size_t *size)` companion 增加了显式的一层 output-buffer projection：GDB 只在 return
后读取 `*ptr`、`*size`，再读取至多 spec 声明的字节数。它不调用 private slot，也不继续解引用 buffer 内的
任意内容。

cxl-lab 任务 [`cnb-bh8-1jtsl45h8`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-bh8-1jtsl45h8)
消费同一 exact `libggml-cuda.so.0` public trigger。runner log SHA256 是
`ac590d53358a7461b8950c2bd634bd37a4c5f7cce83abe75baf8f49dc21e9039`；虽然 top-level target slot 1
继续 `not_reached` 而按合同返回 failure，61 个文件的 immutable diagnostic result 已成功发布并 fresh-pull：

```text
result digest       sha256:3b466ce819ff1af038adc809d79f5119dd0dd5ee39e92a8c2b5bd915b9dd73bc
archive SHA256      637982aa2ea82e707b9a0dfe4e29ba32661db71bcc5e2edcc1340430e5e13feb
manifest SHA256     c8a6f2e458bfe7b0871b51ef2e3b962419d7d703aa7c7c549d2d0d7f81e8a069
```

同一个 L40 Runtime 下，slot 2 的 1024 bytes 和 slot 6 的 14 bytes 都是全零。已有 exact Kimi core 在
崩溃点读取 guest `TOOLS_RUNTIME_BUFFER1` 与 `TOOLS_RUNTIME_BUFFER2`，也分别得到全零。因此当前可排除：

```text
真实 Driver 在 slot 2/6 写入非零初始化内容
→ guest 静态零 buffer 缺少该内容
→ CUDA Runtime 改走 slot 1
```

这只比较了 getter return 后的初始字节。buffer 后续所有权、写入时机、并发和生命周期仍未观测；不过这些
变量不能再作为“slot 1 在首次 cudaFuncSetAttribute 前被触发”的无证据解释。下一边界是 exact DSO 的
backend/Runtime 前置状态，而不是 slot 2/6。

主目标在三轮 exact DSO trigger 中仍为 `slot 1 count=0`、`capture_status=not_reached`。下一步需要补足
真实 Kimi 在 `cudaFuncSetAttribute` 前已经建立的 module/function/Runtime 状态，继续让真实 Driver 自然
调用 slot 1。当前证据不支持修改 slot 2/6，也不支持为 slot 1 填 success stub。

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

## Backend 初始化仍未到达 slot 1

上一节把剩余差异收敛到 Kimi 在 public `cudaFuncSetAttribute` 之前建立的状态。首先应区分
“缺少 CUDA backend 初始化”与“缺少真正的 GGML graph dispatcher”。为此，L40 任务
[`cnb-rk2-1jtsm25j9`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-rk2-1jtsm25j9)
消费 control commit `451d2ba687dd64e5c8d5b88fae5f6b9cb535c9de` 和 CXLMemSim
`ab4b1958022ecf17d8f3c71fdf874633b98d1b3a`。它在同一个 L40 上按下列公开调用顺序执行：

```text
dlopen exact guest libggml-cuda.so.0
  -> cudaSetDevice(0)
  -> ggml_backend_cuda_init(0)
  -> cudaFuncSetAttribute(exact flash-attention host stub, attribute=8, bytes=58528)
  -> ggml_backend_free(...)
```

exact DSO 的 SHA256 是
`7f0bc366d42882d62312509007caa3c7d3f847a5ba5e5fbcd16de5c6df36a9ab`，Build ID 是
`0da23c284f8a619dfe5e3160f2a60b3689fa3b3a`。运行使用 L40
`GPU-09d7c3a9-a10a-fd9d-cf7e-b62a9bac7da1`、Driver `580.95.05`（Build ID
`b21cc51b8b73433abb27468d0d2ed55a0135ee44`）和 libcudart `12.9.79`（Build ID
`dee4b68d7f29b160323eff1e31338d5ebf73d190`）。

目标 callback-hooks UUID `a094798c-2e74-2e74-93f2-0800200c0a66` 的 slot 1、selector
`0x111` 仍没有自然调用。slot 2 和 slot 6 分别自然到达一次，且复用先前已经验证的
`(void **ptr, size_t *size)` 输出形状与全零初始 buffer。由此排除的是：

```text
缺少 ggml_backend_cuda_init
  -> slot 2/6 getter 的输入或初始输出不同
  -> Runtime 改走 slot 1
```

该任务按 target 未命中的 fail-closed 合同返回远端 error，但 immutable result 已发布并 fresh-pull。
result OCI digest 是
`sha256:6b127c3f0b112a9094c31f03831a75a47a646702dd08873435a6cd995c771364`，
archive SHA256 是
`44cea721c228c0b47d801e9e2023b753ca3324fdfd8485758b965edd0b9c0b9b`，
manifest SHA256 是
`900023d42b13d5e0eeba78d26d9114c47698932ae58722ff3cf1a47220ae3df1`，
runner log SHA256 是
`42c680f7523667cc02880f208721db84f654b6a8c1c0070aee113127af9d896e`。

这条阴性证据不能给 slot 1 填任何实现。下一条最小扰动必须使 Runtime 经真实公开 GGML graph 完成
tensor allocation、module/function 准备和 flash-attention dispatcher：固定
`Q=[576,2,16,1]`、`K=[576,32,1,1]`、K 的 `[512,...]` V view 与 F16 mask 在 L40 上进入 core
记录的 `ggml_cuda_flash_attn_ext_mma_f16_case<576,512,2,16>`。它若仍未命中，只排除该图的前置条件；
它若命中，entry/return 和有界状态才可以约束 guest slot 1 的最小 oracle。

## Flash-attention 公开图先被 dispatcher 拒绝

cxl-lab 的 [`cnb-vfg-1jtsp6d52`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-vfg-1jtsp6d52)
首次消费正确 pin 的 flash-attention trigger：CXLMemSim `83a4b9587ef21810864d507d29c19f3b4091ca29`。
target 构建、exact DSO static collector 和 L40 Runtime 都成功；runner log SHA256 是
`c06f84d0450b54408666f591c33c13ebc17072737267662cb4cda253c83896c7`。immutable result digest 是
`sha256:4e48861eedfb93b391bbb51956df4b4542aeb409c570e8f76e78970070925190`，archive SHA256 是
`261da8d6ebea1209fa4ebc08024e8d25707e5498f8ca6709ce9cfb205888f470`。

触发器的 `Q=[576,2,16,1]` 与目标 V head 已正确，`K=[576,32,1,1]` 却在
`ggml_backend_supports_op` 返回 false。frozen `fattn-common.cuh` 定义
`FATTN_KQ_STRIDE=256`；DeepSeek/Kimi 的 GQA path 要求 `K.ne[1] % 256 == 0`，所以 32 在 kernel
wrapper 前被拒绝。slot 1 因此没有被自然调用，这不能当作阴性 ABI 结论。

触发器已改为最小满足条件的 `K=[576,256,1,1]`，并同步调整 V view 和 mask。运行 manifest 同时声明
`qemu_integration/guest_libcuda/Makefile:ggml-flash-attn-ext-probe`；source-only `sync-source` 会从该
声明检查已推送 commit 中的 target，再写入 closed pin。下一次 L40 probe 才能判断公开 GGML dispatcher 到
slot 1 的可达性。

## K=256 图关闭了 host public-graph 候选

cxl-lab 任务 [`cnb-37o-1jtsqajsu`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-37o-1jtsqajsu)
消费本仓 source `f10367140df75637c0e66387333c3188e1dd8f9c`。这个提交只把 flash-attention trigger 的
KV 长度从 32 改为 256，并同步 V view 与 mask；`256` 是 frozen dispatcher 已经由上一轮错误定位出的
必要整除条件。

实际 L40 transcript 为：

```text
ggml_flash_attn_ext_trigger_shape=q=[576,2,16,1] k=[576,256,1,1] v=view-[512,256,1,1] mask=[256,2,1,1]
ggml_cuda_init: found 1 CUDA devices
Device 0: NVIDIA L40, compute capability 8.9
ggml_flash_attn_ext_trigger_backend_support=1
ggml_flash_attn_ext_trigger_compute_status=0
ggml_flash_attn_ext_trigger_synchronize=pass
=== GGML_FLASH_ATTN_EXT_TRIGGER_PASS ===
```

运行使用 paired core 的 exact `libggml-cuda.so.0`，SHA256 为
`7f0bc366d42882d62312509007caa3c7d3f847a5ba5e5fbcd16de5c6df36a9ab`、Build ID 为
`0da23c284f8a619dfe5e3160f2a60b3689fa3b3a`。静态 collector 在该 DSO 中唯一找到
`ggml_cuda_flash_attn_ext_mma_f16_case<576,512,2,16>`，并保存其包含
`cudaFuncSetAttribute@plt` 和 `$0x8,%esi` 的局部反汇编。这里的静态 collector 约束 artifact 与调用点；
动态图的成功由 support、compute 和 synchronize marker 证明，两者不能互相替代。

Python-GDB observer 保存了 29 条自然 entry 和 29 条 return，`unreturned_sequences=[]`，自身状态为
`pass`。callback-hooks UUID `a094798c-2e74-2e74-93f2-0800200c0a66` 仍只自然调用 slot 2 和 slot 6
各一次；slot 1 / selector `0x111` 的 `capture_status=not_reached`。因此顶层 runner 按既定合同
fail closed，而没有把图成功伪装成 ABI 成功。

不可变结果身份为：

```text
control commit     bb7c327b8afb3c5eb6d2e64f60bb4b96cd82b36b
run spec SHA256    c411fb26921ef00202ecc4b5f0e809035723fa7dc94a0eb7bbed355b0de0a58f
result digest      sha256:df8e2291eb7fdcf06aba41072d200fac176e984b0408ab654d670c357dc3cdd8
archive SHA256     4020b624a254a9f93e9b0c558b491850f35b795a04b2605386aabc89c8418c0d
manifest SHA256    d0043c18b3ae0c76b7f6ede3be9b7c5e88581bc6d0199f9ddbccc7458e4d8a58
runner log SHA256  06d32566f59d30e1270d91c920c3304b4e8e7d1ad71167c9c11e50d841e3b2f1
```

本轮把 host trigger 的解释空间收紧到一个明确边界：继续增加“更像 Kimi”的 host public graph 已经没有
直接证据收益，因为 exact Kimi 只有在 guest Runtime 消费 guest shim 返回的 table 时自然调用了 slot 1。
当前工具仍有长期价值：它保存真实 Driver/Runtime 的动态可达集合、已知 companion 输出和 exact artifact
静态现场；它不负责把 host 阴性结果转换成 guest ABI。

下一次诊断应在 guest table 上观察已经自然发生的 slot 1 调用。该入口只能记录 selector、整数寄存器、
显式上限内的输入窗口及必要的返回前后状态，然后 fail closed。没有这份 guest-side capture 之前，仍禁止
为 slot 1 写 success stub、推测函数签名、主动调用真实 Driver 未知槽或再次用完整 paired Kimi 逐槽碰撞。

## Selector buffer 的元素类型解释了 guest 分叉

exact guest core 后续给出了比 slot 1 observer 更早的控制流。CUDA Runtime 12.9 的
`cudaFuncSetAttribute` 在调用任何 callback 之前执行：

```asm
mov 0xa8(%rbx), %rax
mov 0x444(%rax), %edx
test %edx, %edx
je   public_driver_call
```

同一 core 中，`rbx+0xa8` 指向 `TOOLS_RUNTIME_BUFFER1`。slot 2 getter 的第二个输出为 `1024`；本次
selector 是 `0x111`，而 `0x111 * sizeof(uint32_t) == 0x444`。因此该输出表示 1024 个 32-bit selector
元素。旧 shim 把 buffer 声明成 `unsigned char[1024]`，只能容纳 256 个 selector。Runtime 读取
selector 273 时越过数组结尾，落入后续的 `g_context_storage`；core 中该位置含有析构回调地址的非零高位，
于是 Runtime 进入 callback 分支并调用 NULL slot 1。

本机用 exact core 和 guest ELF 关闭了这条地址关系：旧 ELF 中 `TOOLS_RUNTIME_BUFFER1` 大小为
`0x400`，起址后 `0x444` 位于 `g_context_storage`；修复后的 ELF 将其声明为 `uint32_t[1024]`，大小为
`0x1000`，getter 仍返回元素数 `1024`，selector `0x111` 的初始值为零。这个修复保持真实 L40 已观察到的
getter 输出，只纠正内部存储单位。

host Kimi 诊断 `cnb-djo-1jtsvoc5f` 使用相同 `llama-completion` 和相同 libcudart Build ID，完整运行成功且
目标 slot 1 调用数为零。结合上述唯一分支，这说明真实 Driver 的 selector 273 在该调用窗口为零。该证据
不定义非零 selector 的 callback ABI，也不授权实现 slot 1；下一边界是发布修复后的 guest shim 并重跑
same-VM paired Kimi，确认 Runtime 直接进入公开 Driver 调用。

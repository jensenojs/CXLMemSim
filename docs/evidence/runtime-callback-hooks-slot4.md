# L40 runtime callback hooks slot four

## 失败现场

正式 paired Kimi 任务 [`cnb-q6o-1jtrcmq91`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-q6o-1jtrcmq91) 消费 control commit `4400ca04bf709aec2d307256008005c2f81fb47d` 与 run-spec SHA256 `690c814eec4c66c40c46915eb2767dc46b2b6038ddd2d289ce58378c1ee41fa1`。失败结果已按 digest 发布为 `docker.cnb.cool/gevico.online/jensen/cxl-lab@sha256:dd8b010f902ff4cb503202ceae640d939a9773600e660a4a19e2222432265bb0`；runner log SHA256 是 `584490124d25c4536f8e0c246e4a8c4f444b32de149c57e281a33c20dbc1dd17`。

baseline 读取模型、建立 CUDA context，并完成 17 次真实 Type-2 kernel launch。最后一次 guest stderr 已记录：

```text
[CXL-CUDA] TOOLS_TLS.get(out=0x7fff7d2a3dc0)
Segmentation fault (core dumped)
```

guest dmesg 仍为 `segfault at 0 ip 0000000000000000`，但 TLS getter 已执行。因此上轮 `TOOLS_TLS_TABLE[2] == NULL` 的根因已经关闭。

同一 immutable result 中的 `kimi-core-baseline.llama-completio.296.1784406641` 显示 CUDA Runtime 的返回地址为 `libcudart.so.12+0x5819f`。调用点依次为：

```asm
mov 0xa0(%rbx), %rax
call *0x10(%rax)      # TOOLS_TLS slot 2
mov 0x98(%rbx), %rax
mov -0x70(%rbp), %rdi
lea 0x8(%r15), %rsi
call *0x20(%rax)      # callback-hooks slot 4
```

core 内存中 `*(rbx+0x98)` 是 `TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE`：sentinel `0x38`、slot 2 非空、slot 4 为零。`rbp-0x70` 为 `NULL`，`r15+8` 指向有效的八字节输出位置。这一组寄存器和 table 事实说明零地址跳转来自 callback-hooks slot 4，而非 TLS getter 的返回值。

## L40 Driver 取证

诊断提交 `2261bc1175778071741ef7d79645c1b8d7eff21b` 的短任务 [`cnb-n2i-1jtrfe3h3`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-n2i-1jtrfe3h3) 只以 `RTLD_LOCAL` 打开真实 Driver 和候选 shim，读取 `TOOLS_RUNTIME_CALLBACK_HOOKS`。它没有启动 CUDA Runtime、QEMU、guest、模型或 Kimi。任务运行在 NVIDIA L40、Driver `580.95.05`、ELF Build ID `b21cc51b8b73433abb27468d0d2ed55a0135ee44`；runner log SHA256 为 `970aa21c2e209c72a4aeb00a69a6f11e94e1f8b3db13a98fd55be6b611673c4b`。

真实 Driver table 的 sentinel 是 `0x38`，七个 slots 全部 non-NULL。slot 4 的代码前缀为：

```text
4885f6741331c04885ff74038b473848
```

它先检查第二个参数；当 output 有效时，`input=NULL` 路径写零并返回成功。这个结论由修复后短任务 [`cnb-kof-1jtrfk1jh`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-kof-1jtrfk1jh) 直接验证。该任务消费 `81a6937a1feac3ca60b473b65e9e04d6429a3e5f`，运行在 L40、同一 Driver Build ID，runner log SHA256 为 `1d11c37e107660806c875d492d39c37f9a706984f444cd479625cda819d4875a`：

```text
runtime_callback_hooks_oracle_slot4 label=host input=NULL result=0 out=0
runtime_callback_hooks_oracle_slot4 label=shim input=NULL result=0 out=0
```

## 修复

`TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[4]` 现在指向 `tools_runtime_callback_hook_slot4`。这个函数只实现已在 Kimi core 中观察到的调用：有效 output 与 `input=NULL` 时写入 `0` 并返回 `CUDA_SUCCESS`。output 为 NULL 返回 `CUDA_ERROR_UNKNOWN`；non-NULL input 返回 `CUDA_ERROR_NOT_SUPPORTED`，保留下一条未恢复 ABI 的现场，而不假装已经知道 host 的完整语义。

本机 Driver `580.142` 的同一 oracle、候选 shim 构建和 null-input host/shim 对照均通过。`.cnb.yml` 新 event 已通过 CNB YAML 语法、语义和 schema 校验。现有 `libcuda.c` unused placeholder 和 `snprintf` 截断警告仍存在，未由本次改动引入。

## 证明边界

这组证据证明：当前 paired Kimi core 的第二次零地址跳转由 callback-hooks slot 4 缺失造成；正式 L40 Driver 在相同 table index 提供非空 callback；shim 对 core 实际的 `NULL` 输入调用已与 L40 的 result/output 对齐。

它不证明 callback-hooks slot 1、3、5 的 ABI，也不证明 slot 4 在 non-NULL input 下的语义。它不证明 baseline 已完整结束、concordia case 已开始、输出对齐、教程 correctness 或 TPS。

下一边界是将该 shim 修复重建为 CXLMemSim component、fresh-pull、重建依赖它的 Type-2 guest，并重跑同一个 baseline → concordia paired contract。若 Runtime 进入未恢复的 callback slot，必须从新的 core 和 exact L40 oracle 继续收敛，不能预填 success stub。

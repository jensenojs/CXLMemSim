# L40 TOOLS_TLS export table

## 问题

Kimi same-VM paired run `cnb-29f-1jtr53k0l` 已通过 guest shim、BAR2、QEMU Type-2 和真实 NVIDIA backend，baseline 在 17 次 kernel launch 后退出 139。guest core 的 RIP 为零。CUDA Runtime 12.9 的 `cudaFuncSetAttribute` 在 runtime 私有对象偏移 `0xa0` 取到 `TOOLS_TLS` table，并执行 table slot 2。

当时 guest shim 的 `TOOLS_TLS_TABLE` 是四个 machine words，slot 2 为 NULL。这个事实解释了间接 call 跳转到零地址，但尚不能定义 slot 2 的函数签名或 table sentinel。

## 同版本 Driver 取证

短 L40 oracle 不启动 CUDA Runtime、QEMU、guest、模型或 Kimi。它以 `RTLD_LOCAL` 分别打开真实 Driver 和候选 shim，只读取 `TOOLS_TLS` table，并对真实 Driver slot 2 传入一个输出指针。

第一次任务 [cnb-1vq-1jtrbbnim](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-1vq-1jtrbbnim) 在 `/bin/sh` 解析 `pipefail` 时退出；第二次任务 [cnb-7j9-1jtrbf260](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-7j9-1jtrbf260) 在 fresh checkout 缺少 `.work/` 时退出。两次都没有进入 oracle。对应控制提交分别为 `c2bac4f51ef275c613334e1d8efcf70d408f26aa` 和 `891a4052f909455a6dccdba4c99ab9ba2ca61b08`。

第三次任务 [cnb-mjo-1jtrbhn67](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-mjo-1jtrbhn67) 使用 `791fa09087d6871151cc232e834c91f03a4f0cb8` 成功。它观察到：

```text
GPU UUID: GPU-aa79a642-d423-5efb-fbc2-af54d11182a6
GPU: NVIDIA L40, 46068 MiB
Driver: 580.95.05
libcuda: /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05
ELF Build ID: b21cc51b8b73433abb27468d0d2ed55a0135ee44

host table sentinel: 0x18
host table words: 3
host slot 1: non-NULL
host slot 2: non-NULL

slot 2 before cuInit: result=3, output unchanged
slot 2 after cuInit, first call: result=0, output=NULL
slot 2 after cuInit, second call: result=0, output=NULL
slot 2 after cuInit, second pthread: result=0, output=NULL
```

这一运行的 runner log SHA256 是 `2123613f6ad21928109ede770d7a5d21a4bd30e5b508f4794f7735ac7314388f`。旧 shim 同时被观察到 `sentinel=0x20`、四个 words、slot 2 为 NULL。

## 修复与复验

`libcuda.c` 将 table 收紧为三个 words：sentinel、仍未定义的 slot 1 和 `tools_tls_get`。后者的可证实边界是 `CUresult tools_tls_get(void **out)`：空输出地址返回 `CUDA_ERROR_INVALID_VALUE`，普通调用写入 `NULL` 并返回 `CUDA_SUCCESS`。

复验任务 [cnb-ms8-1jtrbr1f3](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-ms8-1jtrbr1f3) 使用修复提交 `54134a51b56a972208cb5f017f83bf83ce248977`。在另一张同版本 L40 上，shim table 已报告 `sentinel=0x18`、三个 words、slot 2 non-NULL；oracle 调用 shim slot 2 得到 `result=0, output=NULL`。runner log SHA256 为 `2587058c27cc5dce7ddf44ae57f4c82c4a8a7ece93883057b57ba47ebe2bc94b`。

本机真实 Driver 580.142 上也完成同一 oracle 和既有 `test-context-shim`；本机只用于编译和相邻行为交叉验证，L40 580.95.05 的两次结果才定义本次修复。

## 证明范围

这组证据证明：当前 Kimi core 的零地址跳转来自 `TOOLS_TLS` slot 2 的空函数指针；在本次正式 L40 Driver 上，slot 2 的已观察 ABI 与 shim 修复一致；修复后的 shim table 和 slot 2 可以在 L40 上构造并调用。

它不证明 slot 1 的 ABI，也不证明 `CUDART_INTERFACE_TABLE[4]` 的 ABI。它不证明 Kimi baseline 已从 `cudaFuncSetAttribute` 返回，更不证明 concordia case、paired 输出对齐、教程 correctness 或 TPS。下一条边界是重新发布 CXLMemSim、重建 Type-2 guest，并在同一 VM 重新运行 paired Kimi；若 Runtime 到达 CUDART slot 4，再从同一 core/trace 继续取证。

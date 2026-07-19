# CUDA private export-table 可达调用探针

## 背景与问题存在性

Kimi same-VM correctness 已连续越过 `TOOLS_TLS_TABLE[2]` 和
`TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[4]`，随后在同一 CUDA Runtime
`cudaFuncSetAttribute` 路径调用 `TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[1]` 时再次跳到
零地址。三次失败都来自 guest shim 返回的 private export table 与真实 NVIDIA Driver
的运行时合同不完整。每轮完整 Kimi 才发现一个空槽，会重复支付 QEMU TCG、guest boot、模型供给和
llama 初始化成本。

现有 `cuda-integrity-export-oracle` 可以读取指定 UUID 的真实 Driver table 形状、函数地址、DSO
偏移和代码前缀，但它依赖手写 table 分支，也不能回答真实 CUDA Runtime 实际调用了哪些槽。现有 core
与 GDB 分析可以恢复一次失败的入口寄存器，却依赖已经发生的长任务崩溃。这两个能力之间缺少一个可复用的
动态可达性与定点 ABI 采集入口。

这个问题必须存在于当前层级：private table 是 CUDA Runtime 与 Driver 的真实进程内接口，不能通过删除
guest shim、绕过 CUDA Runtime 或回退 host native 消除。可删除的是每个 slot 单独编写的一次性 oracle
和一次性 GDB 命令；本切片把它们收拢为同一个观察工具。

## 目标

建立一个对任意真实 CUDA trigger 可复用的 host L40 探针：

1. 在真实 `cuGetExportTable` 返回时记录该进程请求的 UUID、table shape 和槽身份。
2. 每次被明细记录的自然调用保存同一 sequence 下的入口寄存器、caller、thread 与返回 `RAX`，不采集未知指针内存。
3. capture 模式只对调用方选择的 UUID、slot和可选selector额外记录有限内存差异。
4. 输出由 Driver、CUDA Runtime、trigger 和源码身份约束的机器可读证据。
5. 让未来显式的 fuzz/主动调用层复用 inventory 与选择器，但不在本切片主动调用未知槽。

首个真实验收目标是 L40 Driver `580.95.05` 上自然执行
`TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[1]` 的 `selector=0x111`，关闭当前 Kimi
failure 的 slot 1 ABI 未知量。

## 非目标

- 不自动生成 guest shim callback 或函数签名。
- 不主动调用、枚举参数或 fuzz 未知 private slot。
- 不宣称一次 trigger 穷尽 NVIDIA Driver 的全部 private API。
- 不修改 CUDA Runtime、Driver table、目标进程参数或内存。
- 不启动 QEMU、guest、Kimi 模型或 same-VM correctness。
- 不建立新的 CNB waiter、结果发布器或组件 promotion 机制。
- 不把探针通过传播为 Type-2、Kimi correctness、HetGPU translation 或 TPS 证据。

## 约束与不变量

探针观察真实 NVIDIA Driver 与真实 CUDA Runtime 的自然调用。它不能用模拟 callback、shim stub
或手工函数调用制造命中。discovery 与 capture 必须记录并比较 GPU UUID、Driver 路径和 Build ID、
`libcudart` 路径和 Build ID、trigger 文件 SHA256；身份不一致时拒绝把两次结果关联。

table 解析只接受两类仓内和 Concordia 已观察的 shape：首 word 为可验证字节长度的 size-prefixed
table，或由 NULL/`UINTPTR_MAX` 终止的 pointer table。读取必须有显式上限；长度不对齐、越过上限、
终止符缺失或指针不可读时 fail closed。

capture 只读取调用约定中的整数参数寄存器、栈顶返回地址和调用方声明上限内的可读内存。对于已经由自然
调用证明为 `void **ptr, size_t *size` 形状的函数，调用方还可显式声明两个 out-parameter 寄存器及最大
字节数；return 后探针读取 `*ptr`、`*size`，再对该唯一返回 buffer 读取至多声明上限的一层字节。不可读或
超界记录为 unavailable。探针不把寄存器中的任意整数默认解释成指针，也不继续沿返回 buffer 内容递归读取。

## 影响边界

本切片是 CXLMemSim guest shim 的 host 诊断能力，代码 owner 保持在
`qemu_integration/guest_libcuda/`。`.cnb.yml` 只增加显式 L40 diagnostic event，复用现有固定
toolchain image。`scripts/`、BAR2 协议、QEMU、guest artifact、Concordia runtime 和 cxl-lab
正式 run spec 均不改变。

结果先由现有 `run_cnb_job.py` 保存完整 runner log 和终态 `run.json`。本切片不增加 OCI result
类型；首个 slot 1 结果确认有不可替代的 ABI 证据后，再写入 `docs/evidence/` 并由后续 shim 修复消费。

## Inventory

### 定义位置

- `qemu_integration/guest_libcuda/integrity_export_oracle.c`
  - 已拥有 `cuGetExportTable` 动态解析、已知 UUID、table shape、`dladdr`、Driver/shim 对照和
    slot 级 oracle。
- `qemu_integration/guest_libcuda/tiny_cuda_fatbin.cu`
  - 已拥有真实 CUDA Runtime DSO 与 tiny kernel，可增加显式 attribute trigger。
- `qemu_integration/guest_libcuda/cuda_runtime_dlopen_kernel_probe.c`
  - 在 `main` 后 `dlopen` trigger DSO，避免 launcher 自身提前加载 CUDA Runtime。
- `qemu_integration/guest_libcuda/Makefile`
  - 拥有 oracle、tiny DSO 和 launcher 构建入口。
- `.cnb.yml`
  - 已拥有 `api_trigger_tools_tls_export_oracle` 与
    `api_trigger_runtime_callback_hooks_export_oracle` 两个 L40 diagnostic event。

### 主要使用点

- `libcuda.c::cuGetExportTable`
  - guest shim 当前返回七张已知 private table，共六十一个 machine word；其中仍有 NULL 或无 ABI
    证据的占位入口。
- `docs/evidence/tools-tls-export-oracle.md`
  - 记录 TLS slot 2 的真实 L40 shape 和行为。
- `docs/evidence/runtime-callback-hooks-slot4.md`
  - 记录 callback-hooks slot 4 的真实 L40 shape 和行为。
- `cxl-lab/scripts/run/run_cnb_job.py`
  - 触发或恢复 exact CNB task，低频等待终态并保存原始日志；本切片只消费该入口。

### 外部参考

- `concordia/zluda_dump/src/dark_api.rs::create_new_override`
  - 已处理 size-prefixed 与 terminator table，并按 UUID/index 建立调用观察。
- `concordia/zluda_dump/src/os_unix.rs::get_thunk`
  - 证明 x86-64 下未知签名函数可以在保留六个整数参数寄存器后透明转发；本切片使用 debugger
    观察，不复制运行时 thunk。
- `cxl-lab/scripts/run/run_kimi_cuda_gdb_probe.sh`
  - 已有 L40 debugger 身份、batch transcript 和失败现场保存模式；本切片复用其证据保存形状，不复制其 Kimi/CUBIN
    语义。

## 改后预期形状

```text
private-export-table-probe DRIVER -- MODE [FILTERS] -- TRIGGER [ARGS...]
  -> 记录 GPU / Driver / libcudart / trigger 身份
  -> debugger 在 cuGetExportTable entry/return 观察每张真实 table
  -> 按 table shape 生成 UUID + slot -> function address 映射
  -> discovery:
       只对实际调用记录 UUID、slot、caller、thread、入口寄存器、返回RAX与调用次数
  -> capture:
       只在匹配 UUID、slot、selector 时记录 entry/return 和有限内存差异
       对显式声明的 out-parameter 形状记录一层返回 buffer 的指针、长度和有界内容
  -> 输出 identity.json、tables.jsonl、calls.jsonl、returns.jsonl、captures.jsonl、summary.json
```

trigger 是 `--` 后的任意 argv，不属于探针硬编码。首个 trigger 使用现有 tiny CUDA DSO 自然调用
`cudaFuncSetAttribute`。如果它不能进入 `selector=0x111`，结果必须是 `not_reached`；随后可以替换为
更接近 exact `libggml-cuda` 的最小 launcher，而不修改探针主体。

Kimi core 已经把下一份 trigger 收敛到一个公开 CUDA Runtime 调用。`ggml-cuda-attribute-trigger` 只接受
调用方显式给出的 DSO 路径、动态导出 anchor、同一 exact DSO 的 local host-stub ELF offset 和动态 shared
memory 字节数。它以 `dlopen(..., RTLD_NOW | RTLD_GLOBAL)` 让该 DSO 完成真实 CUDA fatbin 注册，使用 anchor 的
`dladdr` base 加 offset 计算 host stub，并用 `dl_iterate_phdr` 拒绝不位于该 DSO executable `PT_LOAD` 的地址。
随后它调用公开的 `cudaSetDevice(0)` 与
`cudaFuncSetAttribute(host_stub, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`。调用方可显式开启
`--initialize-backend`：它只调用 exact DSO 导出的 `ggml_backend_cuda_init(0)`，再用公开
`ggml_backend_free` 释放一个 backend，用于重放 Kimi 已建立的 backend Runtime 前置状态。它不构造模型图、
不加载模型、不调用 private callback；private callback table 仍只由 debugger 被动观察。

这个适配器不从模型、源码或符号表猜输入。L40 composition 在运行前必须校验 artifact manifest 与
`libggml-cuda.so.0` SHA256，再用 `nm -an` 从该文件解析 local host-stub offset；anchor raw symbol、offset、
shared-memory bytes 和 artifact hash 都写入 runner log。它不加载 Kimi 模型、不构造 ggml context、不启动 QEMU，
也不通过手工调用 callback slot 制造命中。

`collect_cuda_elf_static_evidence.py` 是可复用的静态证据生产者。它总会保存 exact library 的
`nm -D`、`nm -an -C`、`readelf -n`、`readelf -W -l` 原始输出；调用方可以按本轮问题声明任意多个 named
symbol regex，要求某些匹配唯一，并为某个唯一 local symbol 保存一个局部 `objdump -d -C`。它把所有匹配、
原始命令、SHA256、Build ID、transcript文件名与失败原因写入 `static-evidence.json`。因此新的 core 只改变
选择条件，不复制静态分析命令，也不会丢掉未匹配的完整原始 symbol/ELF现场。

Kimi slot 1 的 composition 使用这个通用采集器声明 anchor、wrapper 和 host stub 三个选择；其后对 wrapper
transcript要求出现对应 host stub、`cudaFuncSetAttribute@plt` 和 public attribute `8`。运行时 executable-segment
检查仍由 C trigger 再做一次。动态 shared-memory bytes 是本次 Kimi core 的观测输入，采集器记录它但不从静态
反汇编猜值。

`run_cuda_debug_capture.sh` 把 debugger 现场也收拢到同一证据形状。它可使用 `gdb` 或 `cuda-gdb`，并支持
程序运行与 exact ELF + core 两种输入。每次输出 debugger version/configuration/hash、ELF/core/command-file
hash、实际 argv、完整 transcript 与 debugger exit code。core 只读，不会被复制、压缩、删除或修改。private
export-table probe 仍直接使用 Python-enabled `gdb`，因为它必须在运行时从真实 table 地址创建 breakpoint；它继续
保存等价的 identity、命令与 transcript。`cuda-gdb` 用于需要 device-code 或 kernel state 的未来问题，不必为它再
写一串临时命令或单独保存现场。

## 具体改动清单

| repo | file | symbol/field | action | reason | acceptance |
|---|---|---|---|---|---|
| cxlmemsim | `docs/plans/private-export-table-probe.md` | probe contract | add | 冻结可达性、身份和安全边界 | spec审计通过 |
| cxlmemsim | `qemu_integration/guest_libcuda/private_export_probe.py` | GDB Python commands and result writer | add | 观察table与真实调用，不改目标ABI | discovery/capture输出通过schema自检 |
| cxlmemsim | `qemu_integration/guest_libcuda/run_private_export_probe.sh` | host diagnostic entry | add | 收拢身份、argv、debugger与输出目录 | 非法输入和身份漂移fail closed |
| cxlmemsim | `qemu_integration/guest_libcuda/tiny_cuda_fatbin.cu` | explicit attribute trigger | modify | 提供第一个自然Runtime触发器 | 真实L40调用成功或明确not_reached |
| cxlmemsim | `qemu_integration/guest_libcuda/Makefile` | probe targets | modify | 复用现有tiny DSO与launcher构建 | make目标成功 |
| cxlmemsim | `qemu_integration/guest_libcuda/ggml_cuda_attribute_trigger.c` | exact host-stub adapter | add | 让已验证Kimi host stub进入公开Runtime调用 | 只在同一DSO可执行segment内调用；输出public CUDA返回码 |
| cxlmemsim | `qemu_integration/guest_libcuda/collect_cuda_elf_static_evidence.py` | generic ELF static collector | add | 收拢core/ELF的symbol、header与局部反汇编现场 | 原始transcript与命名匹配一起保存；要求唯一的条件不成立即失败 |
| cxlmemsim | `qemu_integration/guest_libcuda/run_cuda_debug_capture.sh` | generic debugger capture | add | 收拢gdb/cuda-gdb程序与core现场 | core只读；输入身份、命令、完整transcript与退出码一起保存 |
| cxlmemsim | `.cnb.yml` | `api_trigger_private_export_table_probe` | add | 分钟级L40 discovery/capture | exact event保存身份与结果marker |
| cxlmemsim | `AGENTS.md` | Commands/Key Files/DoD | modify | 让后续agent找到和正确解释探针 | 文档不复制current result |
| cxlmemsim | `docs/evidence/private-export-table-probe.md` | first L40 evidence | add after run | 保存slot 1结果及证明边界 | 引用SN、SHA、Build ID和原始结果 |

## 核心数据流

debugger 在 `cuGetExportTable` entry 保存 `ppExportTable` 和 UUID，在函数 return 后读取 CUDA result
与 table pointer。每个可执行 slot 建立内部 breakpoint，并保存该地址对应的一个或多个 UUID/index。
同一函数地址被多个槽复用时，记录全部候选映射，不能任意选择一个 owner。

discovery breakpoint 命中后读取 caller return address、thread 和整数参数寄存器，只把可配置小整数范围内的
`RDI` 另记为 selector candidate，不推断其语义。每条受 event limit 接纳的 entry 建立一个 frame-bound
return breakpoint，用同一 sequence 保存 `RAX`；进程正常退出时存在未配对 sequence 会 fail closed。
capture 根据 UUID、slot 和可选 selector 过滤，并复用同一个 return breakpoint，只有目标匹配时才比较调用方
显式选择的一个或多个整数参数寄存器指向的有限窗口。每个窗口独立声明 register 与 bytes，探针不沿其中内容
继续解引用。对于已知 `void **ptr, size_t *size` out-parameter 的自然 companion call，调用方可额外声明
`pointer-register:size-register:max-bytes`；probe 仅在 return 后读取这两个 out object，并对输出 pointer 做一次
有界读取。达到 event limit 后普通明细停止增长，累计调用计数继续更新，目标匹配仍会被捕获。

## 删除清单与保留清单

删除后续为每个新 slot 复制 oracle event 和一次性 GDB command 的前提。现有 TLS、integrity 和
callback slot 4 的语义 oracle 保留，因为它们验证具体函数行为，通用探针不替代这些已知签名的对照。

保留 private slot 的逐项证明责任。通用探针只缩短发现与取证，不允许把多个未知槽映射到同一个 generic
success 或 NOT_SUPPORTED stub。

## Fuzz 组合边界

未来主动调用或 fuzz 层可以消费 `tables.jsonl`、`calls.jsonl` 和已经审定的函数签名描述，生成参数组合
并在隔离进程执行。它必须另立 spec，至少声明函数签名来源、允许写入的内存对象、进程隔离、GPU reset
边界、停止条件和崩溃结果保存。当前 probe 不提供“调用任意地址”入口，避免把观察工具变成未约束执行器。

## 风险与回滚

debugger breakpoint 可能改变线程调度或显著放大高频调用。discovery 只记录首次调用与累计计数；达到
显式事件上限后继续目标进程但停止新增详细记录。capture 只允许一个明确过滤器，避免无界 transcript。

不同 CUDA/Driver 版本可能改变 table shape 或复用函数地址。身份文件与 fail-closed parser 防止旧 offset
静默套用。本机已验证 CUDA-GDB 不带 Python；本切片使用固定镜像内带 Python 的 host `gdb`，它只观察
host CPU 侧的 Driver/Runtime 调用与间接跳转，不承担 GPU kernel 调试。若目标镜像缺少这个能力，任务保留
身份与错误并停止，不新增另一套完整调试栈。

回滚只删除新增 probe、event 和文档，不影响现有 oracle、shim、artifact或run spec。

## 最小接口约束

- trigger 只通过 `--` 后 argv 输入，探针不得搜索模型、DSO或可执行文件。
- mode 只允许 `discovery` 或 `capture`。
- capture 必须同时声明 UUID 和 slot；selector与内存参数索引可选。
- 输出目录必须不存在，避免覆盖历史现场。
- 所有机器文件使用schema version并记录producer source commit。
- 目标进程退出码和debugger退出码分别保存；探针通过不能覆盖trigger失败。

## 验收方式

本地只运行 Python compile、shell syntax、CLI非法输入、exact trigger build和现有 oracle build；这些检查不代替L40。

真实验收使用一个 CNB L40 event，在同一 Job 内顺序执行：

1. discovery，确认真实 CUDA Runtime 请求 callback-hooks UUID并报告活跃slot。
2. capture，筛选 callback-hooks slot 1和selector `0x111`。
3. 验证五类机器文件可解析，Driver/libcudart/trigger身份一致，slot 1 entry 与 return 成对。
4. 若 tiny trigger 报 `not_reached`，保留结果并改用最小 exact ggml trigger；不能因此调用slot 1。
5. exact ggml trigger 的 capture 只有在 artifact hash、anchor、local host-stub offset、动态 shared-memory
   bytes 和 L40 Driver/Runtime identity 都保存时才可消费为 guest shim ABI 证据。

只有 capture 得到返回值和有限状态前后差异，才进入 guest shim slot 1 实现。探针本身完成不关闭 Kimi
correctness；slot 1 修复发布后仍需正式 same-VM baseline、concordia与输出比较。

## 未决问题

首个 tiny trigger 是否能产生与 Kimi 相同的 `selector=0x111` 尚未验证。这个变量只影响 trigger adapter，
不影响 probe 的 table discovery 和 capture 接口。真实运行前不预设 `rsi` 状态窗口的长度；capture 必须由
CLI显式声明有限读取长度，首轮以 core 已确认可读的对象头部为依据。

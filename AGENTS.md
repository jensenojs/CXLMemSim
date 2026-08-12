# Project: CXLMemSim

CXLMemSim 是 CMake 驱动的 C++20/C 混合项目，核心代码在 `src/` 和 `include/`。
主路径是 CXL 内存模拟服务器：QEMU 或测试客户端把 CXL.mem 请求送到 `cxlmemsim_server`，服务器维护容量、延迟、拓扑、DCD/GFAM 和一致性状态。
Type 2 GPU 路径在 `qemu_integration/guest_libcuda/` 提供 guest CUDA Driver API shim，通过 BAR2 MMIO 协议和修改后的 QEMU CXL Type 2 设备通信。
本文件是CXLMemSim源码仓职责、构建约束和Type-2局部边界的权威。本机活跃开发checkout位于`/home/jensen/Projects/cxl-cloud/cxlmemsim/`；项目目标、正确性层级和本地/CNB分工由`/home/jensen/Projects/cxl-memsim/AGENTS.md`定义。跨仓exact source组合从`cxl-lab/manifests/sources.lock.json`读取，不在本文件复制。

## 全链路位置

```text
cxl-models -> guest llama-cpp
                    |
                    | CUDA Driver API
                    v
  CXLMemSim guest_libcuda [this repository] -> BAR2 command/data window
                    |                                  |
                    |                                  v
  CXLMemSim server / memory model       QEMU cxl-type2 -> Concordia / NVIDIA L40
                    ^                                  |
                    +---- Type-2 memory and command state

linux-cxl-type2 discovers the device; type2-guest packages the shim and kernel;
cxl-lab fixes the exact component artifact and runs the complete contract.
```

本仓交付两类相连但不同的能力：server侧的CXL内存/拓扑状态，以及guest侧把CUDA API编码为BAR2命令的shim。任何修改先说明它改变哪一类状态、由哪一侧消费，并从对应run或probe取得证据。

## 控制仓回链

本仓产出的源码和组件payload由控制仓组合，具体身份与运行事实沿以下入口继续读取：

- [控制仓职责与组合边界](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/AGENTS.md)
- [source、artifact、run与result字段](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/AGENTS.md)
- [candidate、promotion与fresh-pull](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/artifacts/AGENTS.md)
- [正式运行、observer与result发布](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/run/AGENTS.md)
- [immutable result、core归档与materialize](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/results/AGENTS.md)

## Cloud Source Authority

CNB `gevico.online/jensen/cxlmemsim`是项目primary，GitHub `jensenojs/CXLMemSim`保存同SHA公开镜像。新提交先进入CNB primary，再把同一SHA推送GitHub；运行任务只消费`cxl-lab` source lock声明的exact source。

本次candidate迁移只证明CXLMemSim superproject声明的heads、tags及其可达对象可以在CNB和GitHub之间保持一致，不证明`.gitmodules`中的外部仓库、组件构建、OCI制品、Type-2运行或模型正确性。局部迁移结构和验收见`docs/specs/cloud-source-authority.md`。

组件构建、server/guest payload、OCI发布与fresh pull边界见`docs/specs/cloud-component-artifact.md`，首次真实构建与失败链见`docs/evidence/cloud-component-artifact.md`。该制品携带CXLMemSim拥有的server、guest shim、case控制CLI，以及toolchain中server需要但本地开发宿主不保证提供的`libfmt.so.9`。glibc、libstdc++等基础系统运行库仍由消费环境提供。实验声明必须显式绑定runtime library，不能从server相邻目录猜路径。

## Cloud Build and Artifact

云端组件输入按消费关系组织：

```text
manifests/build-profile.json
  定义本组件编译器、CMake开关、测试并行度和预期文件图

manifests/artifact-contract.json
  定义CNB源码/registry位置、固定toolchain digest、ORAS身份和OCI media type

.cnb.yml
  选择固定toolchain image并调用仓内build/publish/fresh-pull脚本

scripts/build_component.sh
  只构建和测试CXLMemSim，生成server、guest shim、case控制CLI和symlink

scripts/component_artifact.py + publish_component.sh + pull_component.sh
  生成文件事实、确定性归档、按digest发布与安全恢复
```

工具环境缺少系统header、编译器或归档CLI时，在`cxl-lab/.ide/Dockerfile`从当前固定digest派生最小增量层。新镜像通过CNB任务后，先在`cxl-lab`原子更新`manifests/sources.lock.json#toolchain_image`、active `.cnb.yml`镜像和README，再更新本仓`manifests/artifact-contract.json#toolchain_image`与组件pipeline镜像。不得在组件源码中添加fallback来掩盖工具镜像缺口，也不得用tag替代digest。

本组件的CMake开关、目标或payload shape变化只修改`manifests/build-profile.json`和本组件spec。toolchain升级与profile变化是两条不同的输入轴，任务日志必须同时记录exact source commit和toolchain digest。

CNB任务接管时至少核对repo、branch、exact SHA、event、runner CPU/内存、toolchain digest、当前stage和首个失败日志。build成功标记、CTest、payload文件事实、OCI digest与fresh pull各自只证明对应层级；缺少后续marker时不得传播为Type-2、固定1.5B或Kimi成功。

## Role in the Project

本仓在教程链路中负责CXLMemSim server、backing store、guest CUDA shim和guest侧BAR2协议定义。进入本仓工作前，先从当前checkout适用的项目权威确定证据层和工程入口，再说明改动作用于server、guest shim、共享协议还是测试。本机`/home/jensen/Projects/cxl-memsim/CXLAgent/`只提供guest侧CXL sysfs/trace/snapshot观测和driver参考，不是当前Type-2 kernel主线；CNB checkout不假设该目录存在。

Type-2 smoke 的观察点：启动参数含 `-device cxl-type2`，QEMU 日志出现 Type-2 realized，guest 能看到 `/dev/cxl/cache0`、`/dev/cxl/mem0`、`/dev/cxl_gpu0`。
本机`/home/jensen/Projects/cxl-memsim/CXLAgent/`可以在guest rootfs准备好后作为观测工具：先用topology/snapshot类命令确认cache/mem/sysfs/iomem，再考虑memory snapshot和tracepoint；当前诊断initramfs不适合直接跑完整`cxlagent`。

## 流语义与测试契约

Kimi 正确性战役中，`cnb-5r8-1jvj5f4kp` 暴露了非阻塞计算流的生产者与
legacy 默认流拷贝并发时会读取脏源。已入 `main` 的 `a1ac80a` 把这个边界
收拢为三条直接路径：`MEM_COPY_2D_DTOD` 在 `PARAM6` 传递 `stream_wire`，
`cuMemcpy2DAsync_v2` 转发调用流而同步 `cuMemcpy2D_v2` 使用 legacy sentinel；
`cuMemsetD8Async` 先 drain 调用流，再走既有立即执行的分片 HtoD 填充；
`cuMemcpyAsync` 按指针类型进入保留流身份的异步入口。

`qemu_integration/guest_libcuda/cxl_gpu_cmd.h` 中
`CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION=3` 是 shim 与 QEMU 的共享 ABI。版本2让
`MEM_COPY_2D_DTOD`在`PARAM6`携带stream；版本3让`STREAM_SYNC`在`PARAM1`
携带public API、blocking DtoH drain或memset emulation drain来源。QEMU对版本2
按public API解释以保留旧shim兼容，版本3的未知来源值直接失败。不能把新shim
与旧QEMU组合后宣称来源归因或流语义成立。

`tests/stream-contract/` 是这一边界的回归套件。它让延迟 producer 与异步 copy
在同一 stream 上执行，并用 legacy-stream 对照组区分流顺序丢失与一般拷贝故障。
本地 KVM/RTX 3050 已建立该套件；它证明特定 copy API 是否保留同流顺序，不证明
完整模型正确性或性能。修改 shim copy path、BAR2 wire protocol 或 QEMU copy
handler 后，先在 native 路径运行 `tests/stream-contract/build.sh` 与
`tests/stream-contract/run.sh --surface native`，再以 Type-2 guest 的
`--surface type2-fixed` 验收配对实现。

## 可复用调试证据

`qemu_integration/guest_libcuda/`中的 CUDA 诊断工具服务于下一次同类失败的低成本取证。原始 core、CNB
runner log、exact guest ELF、collector 原始输出、反汇编、阴性 L40 probe、触发器源码和对应 commit 都是
长期证据。`.work/`中的这些命名现场和`.codex-runs/`中的原始失败目录不能随 build、cache 或普通 generated
state 清理；它们不进入组件 payload、OCI component artifact 或 Git index。

四个入口按问题边界复用：

- `collect_cuda_elf_static_evidence.py`读取调用方给定的 exact ELF 与命名 selector，写入 SHA256、Build ID、
  完整 `nm`/`readelf`/`objdump` transcript 和 `static-evidence.json`。section facts进入结构化结果；存在
  `.nvFatBinSegment`时按64位CUDA registration descriptor的24字节形状输出count与remainder，不能整除时
  fail closed。声明target SM、BAR2 payload上限和QEMU decoded上限时，它还会逐个wrapper执行与guest
  shim同语义的ELF/PTX选择，输出codec、compressed/decoded size、transport route和预期
  `cuLibraryGetModule`结果；任何unsupported codec或容量越界fail closed。它用于确认符号、ELF 段、
  registration容量和每个wrapper的预运行传输结果；
  它不加载 CUDA，不能证明 Runtime private ABI、guest shim 或 Type-2。
- `libcublas_create_probe.so`由现有`cuda-runtime-dlopen-kernel-probe`加载。它先加载调用方固定的exact
  `libggml-cuda`，再从同一guest CUDA userland解析`libcudart`与`libcublas`，在同一个handle生命周期执行
  `cublasCreate_v2`、固定2x2 FP32 `cublasSgemm_v2`、同步、DtoH数值比较和`cublasDestroy_v2`。两次显式
  `dlopen`前后各以`dl_iterate_phdr`输出一组严格JSONL loader image；`libcuda.so.1`的每条
  `cuLibraryLoadData`记录同时保存`code_file`、`code_base`与`code_offset`。cxl-lab把它们与解包后的exact
  guest build manifest交叉，得到每份已加载ELF的SHA256、Build ID与fatbin descriptor count，并把每条
  registration绑定到一份实际image。它用于在模型加载前完成CUDA library registration、cuBLAS初始化、第一笔
  GEMM计算和runtime DSO集合取证；registration总数仍由static collector和上层run spec拥有，DSO本身不猜测数量、
  不链接宿主`libcublas`替代物。
- `ggml-cuda-attribute-trigger`加载调用方给定的 exact `libggml-cuda`，用已收集的 dynamic anchor 与同 DSO
  local host-stub 虚拟地址执行一次公开 `cudaFuncSetAttribute`。它会拒绝 DSO 可执行段外的地址；它绝不读取、
  写入或调用 private export-table slot。shared-memory 字节数必须来自同一 Runtime/core 观察，不能由反汇编猜测。
- `ggml-flash-attn-ext-trigger`用冻结 llama source 的公开头文件和 extracted guest artifact 的 exact GGML
  libraries 构造一次小的 public `ggml_flash_attn_ext` 图。固定 `Q=[576,2,16,1]`、`K=[576,256,1,1]`、K 的
  `[512,...]` V view 与 F16 mask 在 L40 上选择 core 中的 `<576,512,2,16>` CUDA wrapper。256 是 frozen CUDA
  GQA dispatcher 的 `FATTN_KQ_STRIDE`，较小的 KV 长度会在 `supports_op` 阶段被拒绝。构建必须显式传入
  `GGML_INCLUDE_DIR`、`GGML_LIBRARY_DIR` 和仅供 link-time closure 的 `GGML_LINK_LIBRARY_DIR`，缺失时 fail
  closed；最后一项不能进入 host runtime 搜索路径。它不读写或调用 private export-table slot。
- `run_private_export_probe.sh`在 L40 上观察真实 CUDA Runtime 自然到达的 export table。每条受明细上限接纳的
  调用以 sequence 保存 entry registers 与 return `RAX`；capture 只在显式 UUID、slot 和可选 selector 匹配时
  读取一个或多个声明的整数参数寄存器窗口。resolver 模式额外观察真实 Driver 对一个声明符号的
  `cuGetProcAddress` 查询，保存返回函数地址相对 exact `libcuda.so.1` 的 image offset、与公开符号地址的比较、
  caller frame 映射，以及经该地址自然发生的调用和返回。它不主动调用未知 slot 或 resolver 地址，不沿窗口内容
  继续解引用；负结果只表示给定 trigger 没有到达该边界。
- `run_cuda_debug_capture.sh`对程序或 exact ELF + core 保存 debugger、命令文件、输入 hash、完整 transcript
  与退出码。core 保持只读。host Runtime/Driver 调用链用 Python-enabled `gdb`；只有问题涉及 device code、
  kernel state、device memory 或 SASS PC 时才选择 `cuda-gdb`。
- `probe_cuda_fatbin_cubin_load.py`从 SHA256 已冻结的 CUDA shared library 中按声明的 fatbin `files_size`提取一个
  ELF CUBIN，并可用当前真实 CUDA Driver 执行一次`cuModuleLoadData`。它用于先回答“这份低架构 CUBIN 是否被
  目标 L40 接受”；它不启动 guest、BAR2、QEMU、HetGPU或Kimi。必须同时传入 library SHA256、fatbin shape、
  CUBIN SM 和`--load --expected-device-sm`，使输入或 GPU 身份漂移显式失败。
- `probe_cuda_fatbin_library_load.py`复原 Runtime 实际传给`cuLibraryLoadData`的完整 library wrapper，并在真实
  Driver 上按`LoadData → GetModule → Unload`观察三个返回值。它以 library SHA256、明确 fatbin offset、16-byte
  header SHA256、header/files region SHA256 和有序 entry 序列共同定位输入；`files_size`在一个大 DSO 中可以重复，
  不能单独充当 wrapper 身份。它不启动 guest、BAR2、QEMU、HetGPU 或 Kimi。成功的 library load 与后续 module
  materialization 分属两个 Driver 状态，结果必须分别保存。

上述直接CLI入口以及两个 GGML public trigger 都必须同时提供`--help`与`--hint`。`--help`解释参数与失败语义；`--hint`用稳定的`key=value`行给压缩恢复后的 agent 指出自身源码路径、相邻工具、主要输出、证明边界与下一条证据边界。新增可直接调用的CUDA诊断入口也遵守此合同；`libcublas_create_probe.so`这类由统一launcher加载的内部DSO、测试夹具和生成二进制不伪装成CLI。collector与cuBLAS probe的底层合同分别由`test-static-collector`和`test-cublas-create-probe`保护；cxl-lab测试只验证上层编排。

每个新 ABI 失败先复用上述最窄入口，保存 `proves`、`does_not_prove` 和下一边界；不要复制临时 `nm`、
`readelf`、`objdump` 或 GDB 命令。需要正式 L40、exact artifact 或 core 组合时，输入身份和结果发布由
`cxl-lab` diagnostic pipeline 管理，不手改本仓 `.cnb.yml` 的 image 行，也不在交互 shell 拼装冻结输入。

## Notes

本机实验笔记位于`/home/jensen/Projects/cxl-memsim/Note/`，并遵照`/home/jensen/obsidian/AGENTS.md`。CNB checkout只产出原始任务日志和机器证据，不假设Obsidian目录存在。
写笔记前先判断笔记回答的问题：理论、概念边界、机制模型写到本机`Note/计算机体系结构/`或`Note/计算机系统模拟/`；操作流程、环境复现、benchmark日志、踩坑复盘写到本机`Note/实践笔记/`。
粗糙捕获可以先保留日期或“草稿”标记，但不能把 AI 对话直接粘贴成成稿；成稿要有自己的 thesis，首次出现的 GPU/QEMU/CXL/Concordia 概念要给最小解释，关键外部链接要摘出能支撑判断的内容。
Obsidian wikilink 使用 vault 根目录绝对路径，不使用 `../` 相对路径。

## Commands

- install Fedora deps: `sudo dnf install -y cmake ninja-build meson python3-pip gcc gcc-c++ clang llvm-devel clang-devel libbpf-devel boost-devel fmt-devel spdlog-devel rdma-core-devel liburing-devel make git pkgconf-pkg-config glib2-devel pixman-devel zlib-ng-compat-devel bzip2-devel lzo-devel snappy-devel libzstd-devel libslirp-devel capstone-devel ncurses-devel openssl-devel elfutils-libelf-devel && git submodule update --init --recursive`
- configure local Clang: `CC=clang CXX=clang++ cmake -S . -B build-clang -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0091=NEW`
- configure like GitHub Actions: `CC=gcc-13 CXX=g++-13 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0091=NEW`
- build: `cmake --build build-clang --parallel 4`
- typecheck: `cmake --build build-clang --parallel 4 --target cxlmemsim_server test_dcd_gfam test_rob test_mem_stall test_bandwidth_model`
- test: `ctest --test-dir build-clang --output-on-failure`
- single test: `ctest --test-dir build-clang -R '^test_dcd_gfam$' --output-on-failure`
- lint: `clang-format --dry-run --Werror $(git ls-files '*.[ch]' '*.cc' '*.cpp' '*.hpp')`
- format: `clang-format -i $(git ls-files '*.[ch]' '*.cc' '*.cpp' '*.hpp')`
- build Type-2 QEMU: `cd /home/jensen/Projects/qemu-cxl-type2 && mkdir -p build && cd build && ../configure --target-list=x86_64-softmmu && ninja -j2 qemu-system-x86_64`
- check Type-2 device is compiled: `/home/jensen/Projects/qemu-cxl-type2/build/qemu-system-x86_64 -device help | grep -i 'cxl-type2'`
- build guest CUDA shim: `make -C qemu_integration/guest_libcuda`
- prepare private export-table probe: `make -C qemu_integration/guest_libcuda private-export-table-probe`
- build exact public `cudaFuncSetAttribute` trigger: `make -C qemu_integration/guest_libcuda ggml-cuda-attribute-trigger`
- build exact public GGML flash-attention graph trigger: `make -C qemu_integration/guest_libcuda ggml-flash-attn-ext-probe GGML_INCLUDE_DIR=.../ggml/include GGML_LIBRARY_DIR=.../opt/llama/bin GGML_LINK_LIBRARY_DIR=.../lib/x86_64-linux-gnu`; the third path only closes the exact guest NCCL dependency while linking and must not be added to runtime search paths
- prepare exact trigger tools: `make -C qemu_integration/guest_libcuda ggml-cuda-attribute-probe-tools`
- prepare reusable CUDA ELF static evidence collector: `make -C qemu_integration/guest_libcuda cuda-elf-static-evidence-tools`
- prepare generic gdb/cuda-gdb capture wrapper: `make -C qemu_integration/guest_libcuda cuda-debug-capture-tools`
- prepare exact CUDA fatbin CUBIN Driver-load probe: `make -C qemu_integration/guest_libcuda fatbin-cubin-load-probe`
- run private export-table discovery on an L40: `qemu_integration/guest_libcuda/run_private_export_probe.sh --mode discovery --output-dir DIR -- TRIGGER [ARGS...]`
- run a bounded private export-table capture: `qemu_integration/guest_libcuda/run_private_export_probe.sh --mode capture --output-dir DIR --uuid UUID --slot N [--selector VALUE] [--memory REGISTER:BYTES ...] -- TRIGGER [ARGS...]`
- Type 2 endpoint smoke: `./qemu_integration/smoke_type2_endpoint.sh`

## Code Conventions

- DO keep public simulator types in `include/` and their implementation in `src/`; follow `include/cxlcontroller.h` with `src/cxlcontroller.cpp` and `include/dcd_gfam.h` with `src/dcd_gfam.cpp`.
- DO include the matching project header before standard headers in implementation files; `src/dcd_gfam.cpp` includes `"dcd_gfam.h"` before `<algorithm>` and `<limits>`.
- DO follow `.clang-format`: LLVM base, 4 spaces, no tabs, 120-column limit; do not hand-format around the formatter.
- DO use C++20 standard library primitives already used in the project: `std::unique_ptr`, `std::shared_mutex`, `std::atomic`, `std::optional`, and `std::format`.
- DO keep command-line parsing local to the executable currently owning it; `src/main_server.cc`, `src/main.cc`, and `src/rob.cc` already carry local parse-result structs.
- DO make tests self-contained binaries that return nonzero on failure; mirror `tests/test_dcd_gfam.cpp` and register always-on tests with `add_test(...)` in `CMakeLists.txt`.
- DO keep guest ABI constants synchronized across guest and QEMU-facing code; `qemu_integration/guest_libcuda/cxl_gpu_cmd.h` is the high-fan-in command/register contract for the CUDA shim.
- DO use `set -euo pipefail` in new maintained shell scripts; `qemu_integration/launch_qemu_vcs_dcd_gfam.sh` and `smoke_type2_endpoint.sh` are the current good examples.
- `qemu_integration/AGENTS.md`拥有集成入口、workload inventory与归档路由；`qemu_integration/arch/AGENTS.md`拥有已证实失效文件的操作边界。归档文件不能作为模板、构建输入或运行入口；需要相同能力时在当前合同上新建维护入口。
- DON'T silently fall back from real GPU mode to simulation in Type 2 GPU work; `README.md` documents that real GPU initialization errors should remain visible.

## Key Files

- `CMakeLists.txt` — top-level build graph; defines `cxlmemsim`, `cxlmemsim_server_lib`, `cxlmemsim_server`, CTest targets, and optional RDMA/SSD backend wiring.
- `include/cxlcontroller.h` / `src/cxlcontroller.cpp` — central controller for topology, latency, migration, cache accounting, LogP, DCD, GFAM, MH-SLD, and distributed endpoint integration.
- `include/cxlendpoint.h` / `src/cxlendpoint.cpp` — CXL endpoint and switch model, queueing/credit accounting, bandwidth model, and remote expander abstractions.
- `src/main_server.cc` — `cxlmemsim_server` entry point, CLI parsing, TCP/SHM/PGAS request loop, atomics, DCD/GFAM checks, and back-invalidation handling.
- `src/shared_memory_manager.cc` — POSIX shared-memory/file-backed memory pool used by server modes.
- `include/distributed_server.h` / `src/distributed_server.cpp` — distributed memory server, SHM/TCP/RDMA transports, remote read/write forwarding, and calibration paths.
- `include/dcd_gfam.h` / `src/dcd_gfam.cpp` — Dynamic Capacity Device and GFAM allocation/access-control model; `tests/test_dcd_gfam.cpp` is the smallest executable test.
- `qemu_integration/launch_qemu_vcs_dcd_gfam.sh` — maintained QEMU launcher for Zettai VCS, Type 3 DCD/GFAM, and optional Type 2 endpoint.
- `/home/jensen/Projects/cxl-memsim/CXLAgent/` — 本机CXL Type-2 guest observability/reference repo；不属于本组件CNB checkout输入。
- `qemu_integration/smoke_type2_endpoint.sh` — bounded host-side Type 2 QEMU realization smoke test.
- `qemu_integration/guest_libcuda/libcuda.c` and `qemu_integration/guest_libcuda/cxl_gpu_cmd.h` — guest CUDA Driver API shim and BAR2 command/register contract.
- `qemu_integration/guest_libcuda/private_export_probe.py` and `run_private_export_probe.sh` — host L40 Python-GDB observation of naturally reached private export tables. They record table shape, real calls and bounded capture state; they do not invoke unknown slots or prove guest/Type-2 correctness.
- `qemu_integration/guest_libcuda/ggml_cuda_attribute_trigger.c` — exact-artifact adapter for one public `cudaFuncSetAttribute` call on an explicitly resolved, registered libggml-cuda host stub. It validates the DSO address boundary before the call and must stay paired with the external artifact hash and `nm -an` evidence; it does not implement or invoke a private table ABI.
- `qemu_integration/guest_libcuda/collect_cuda_elf_static_evidence.py` — generic static collector for exact CUDA-related ELF/core work. It always saves full `nm`/`readelf` transcripts and accepts named symbol/disassembly selectors for one investigation; it never loads CUDA or a target DSO.
- `qemu_integration/guest_libcuda/run_cuda_debug_capture.sh` — generic `gdb`/`cuda-gdb` wrapper for program or exact ELF/core runs. It records debugger/input identity and the full transcript without copying, compressing or changing the core; private export-table discovery remains a separate Python-GDB consumer because it needs dynamic table breakpoints.
- `script/build_qemu.sh` — builds the CXL-capable QEMU submodule using the vendored Meson wheel.
- `lib/qemu/` — Type-2 QEMU source checkout after submodule initialization; expected to line up with `qemu-cxl-type2`, not arbitrary upstream QEMU.

## Boundaries

- Treat `build/`, `cmake-build-*`, `CMakeFiles/`, QEMU build directories, logs, generated raw images, and files under `/dev/shm` as generated state.
- Treat `workloads/` datasets and benchmark checkouts as inputs; only edit them when the task explicitly targets a workload.
- Treat `lib/qemu/` and other `.gitmodules` entries as submodules. For current local Type-2 QEMU work, use `/home/jensen/Projects/qemu-cxl-type2/`.
- Avoid editing `include/vmlinux.h`; it is a large copied/generated Linux header used for BPF/perf compatibility.
- Keep secrets and host-specific paths out of committed scripts; pass disk images, kernels, QEMU paths, CUDA paths, and network choices through environment variables like `DISK_IMAGE`, `KERNEL_IMAGE`, `QEMU_BINARY`, `CXL_MEMSIM_PORT`, and `QEMU_NET_MODE`.
- Preserve the Type 2 command ABI when touching `cxl_gpu_cmd.h`; every register/command change must be reflected in the guest shim, QEMU Type 2 implementation, and tests.
- Do not swap the guest Type-2 kernel path to CXLAgent out-of-tree modules unless the task explicitly targets module packaging. Current validated guest path is `linux-cxl-type2` built into the test kernel.
- The static include graph currently has high fan-in at `qemu_integration/guest_libcuda/cxl_gpu_cmd.h`, `include/cxlendpoint.h`, `include/helper.h`, `include/cxlcontroller.h`, and `include/policy.h`; edits there require a full build and targeted tests.
- No `.beads/` task store exists in this checkout.

## Definition of Done

- 云端组件build、四个CTest、shim构建、OCI发布与fresh pull必须由本仓CNB任务在exact SHA和固定toolchain digest上执行；Fedora只运行shell/Python/JSON/YAML检查与小型归档测试。
- 若任务明确要求本地验证C++源码，先确认内存预算，再用不超过`--parallel 2`的隔离build目录；本地结果不替代CNB artifact证据。
- Use `CC=gcc-13 CXX=g++-13 ...` only when `gcc-13`/`g++-13` are installed and the goal is to reproduce GitHub Actions exactly.
- Run `ctest --test-dir build-clang --output-on-failure` for simulator/server changes; at minimum run the CTest target nearest to the touched code.
- Run `clang-format --dry-run --Werror $(git ls-files '*.[ch]' '*.cc' '*.cpp' '*.hpp')` before committing formatted languages.
- Run `make -C qemu_integration/guest_libcuda` after guest CUDA shim or `cxl_gpu_cmd.h` changes.
- Run `./script/build_qemu.sh` and `./qemu_integration/smoke_type2_endpoint.sh` after QEMU Type 2, Zettai, or BAR command protocol changes.
- Document any skipped command with the missing host dependency, for example absent `lib/qemu`, absent NVIDIA driver, missing guest disk, or missing kernel image.

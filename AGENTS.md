# Project: CXLMemSim

CXLMemSim 是 CMake 驱动的 C++20/C 混合项目，核心代码在 `src/` 和 `include/`。
主路径是 CXL 内存模拟服务器：QEMU 或测试客户端把 CXL.mem 请求送到 `cxlmemsim_server`，服务器维护容量、延迟、拓扑、DCD/GFAM 和一致性状态。
Type 2 GPU 路径在 `qemu_integration/guest_libcuda/` 提供 guest CUDA Driver API shim，通过 BAR2 MMIO 协议和修改后的 QEMU CXL Type 2 设备通信。
本文件是CXLMemSim源码仓职责、构建约束和Type-2局部边界的权威。本机活跃开发checkout位于`/home/jensen/Projects/cxl-cloud/cxlmemsim/`；项目目标、正确性层级和本地/CNB分工由`/home/jensen/Projects/cxl-memsim/AGENTS.md`定义。跨仓exact source组合从`cxl-lab/manifests/sources.lock.json`读取，不在本文件复制。

## Cloud Source Authority

CNB `gevico.online/jensen/cxlmemsim`是项目primary，GitHub `jensenojs/CXLMemSim`保存同SHA公开镜像。新提交先进入CNB primary，再把同一SHA推送GitHub；运行任务只消费`cxl-lab` source lock声明的exact source。

本次candidate迁移只证明CXLMemSim superproject声明的heads、tags及其可达对象可以在CNB和GitHub之间保持一致，不证明`.gitmodules`中的外部仓库、组件构建、OCI制品、Type-2运行或模型正确性。局部迁移结构和验收见`docs/specs/cloud-source-authority.md`。

组件构建、server/guest payload、OCI发布与fresh pull边界见`docs/specs/cloud-component-artifact.md`，首次真实构建与失败链见`docs/evidence/cloud-component-artifact.md`。该制品只携带CXLMemSim拥有的server、guest shim和case控制CLI；运行时系统动态库由消费该制品的固定镜像提供。

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
- DON'T treat `qemu_integration/launch_qemu_type2_gpu.sh` or `launch_qemu_type2_hetgpu.sh` as reliable templates; both contain malformed shell syntax in this checkout.
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

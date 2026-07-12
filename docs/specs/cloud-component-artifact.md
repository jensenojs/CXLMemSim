# 云端组件构建与制品边界

## 与项目目标的关系

本切片把已迁移到CNB的CXLMemSim源码构建为可按OCI digest恢复的组件文件。它服务于后续CNB TCG Type-2 discovery、tiny `1234`、固定1.5B输出对齐和Kimi K2.6正式实验。本切片只证明CXLMemSim文件集合可构建、通过局部测试并从新任务恢复，不证明Type-2运行或模型正确性。

## Blast Radius

BR3。该切片首次冻结后续组件共同使用的OCI envelope与文件事实表达，同时修改CXLMemSim仓的build、test、publish和fresh pull入口。共享格式必须由真实产物推出，不能让后续组件自行发明第二套manifest或归档格式。

## 问题存在性

CNB workspace会销毁。源码commit和build成功不能让后续运行任务获得同一组server与guest shim文件。重新构建可以临时恢复文件，却会让每次运行重新解释profile和输出集合，破坏组件独立回滚。需要一个由源码、profile和固定工具环境推出的不可变digest。

## 目标结构

```text
CXLMemSim/
├── AGENTS.md
├── docs/specs/
│   ├── cloud-source-authority.md
│   └── cloud-component-artifact.md       # 本切片目标与接口
├── manifests/
│   ├── build-profile.json                # 本组件构建输入与预期文件图
│   ├── artifact-contract.json            # OCI envelope与固定工具身份
│   └── artifacts/                        # fresh pull通过后才保存正式digest
├── scripts/
│   ├── build_component.sh                # 只构建、测试并生成payload
│   ├── component_artifact.py             # 文件图、manifest和安全恢复
│   ├── publish_component.sh              # 通用归档与ORAS push，不识别文件名
│   └── pull_component.sh                 # 按digest拉取并验证
├── tests/
│   └── test_component_artifact.py        # 正反归档性质
└── .cnb.yml                              # build/publish与fresh pull独立任务
```

制品payload固定为：

```text
payload/
├── bin/cxlmemsim_server
└── guest/
    ├── libcuda.so.1
    └── libcuda.so -> libcuda.so.1
```

payload不捆绑glibc、libstdc++、libgcc或其他系统动态库。组件仓只拥有自身生成的三个文件；运行与fresh pull环境负责提供系统库。manifest记录ELF的`DT_NEEDED`名称，fresh pull使用同一固定工具镜像执行`ldd`和`cxlmemsim_server --help`，让缺失依赖直接失败。

## 核心调用链

```text
CNB exact source checkout
  → verify_source_checkout.sh
  → build-profile.json
  → build_component.sh
      CMake configure/build
      四个always-on CTest
      guest shim Makefile
      server --help
      生成三文件payload
  → component_artifact.py create-manifest
  → publish_component.sh
      deterministic tar + zstd
      pinned ORAS push到本仓registry
      registry返回digest
  → 销毁build workspace
  → 新CNB任务读取候选digest
  → pull_component.sh
      pinned ORAS按digest拉取
      解包前验证完整归档图
      恢复payload并核对manifest
      readelf/ldd/server --help
  → Git提交正式artifact manifest
```

## 构建输入

`manifests/build-profile.json`是唯一组件build profile。脚本机械消费以下事实：

- `target_arch=amd64`，CNB runner必须报告`x86_64`；
- `CC=clang`、`CXX=clang++`；
- Release与`CMP0091=NEW`；
- microbenchmarks、RDMA、SlugAllocator全部关闭；
- 并行度为4，保持迁移时冻结的候选profile；该值只由CNB固定构建任务验证，本地Fedora完整构建不参与冻结云端资源边界；
- guest shim只构建`libcuda.so.1`和`libcuda.so`；
- `outputs`完整定义预期文件路径、类型、mode和symlink target。

构建任务额外构建并运行`test_dcd_gfam`、`test_rob`、`test_mem_stall`和`test_bandwidth_model`。测试二进制不进入payload。

## OCI合同

`manifests/artifact-contract.json`冻结：

- 组件registry repository；
- 固定toolchain image digest；
- 固定ORAS image digest；
- OCI artifact type；
- `component.tar.zst`和`manifest.json`的media type。

OCI内容恰好有两个blob：

```text
component.tar.zst
manifest.json
```

归档只含payload中的普通文件与symlink，不含目录entry、build日志、workspace路径、CNB任务ID或manifest自身。GNU tar使用POSIX格式、epoch mtime、uid/gid 0、numeric owner、NUL分隔排序文件表；zstd固定`-T1 -19`。相同payload必须生成相同归档SHA256。

`manifest.json`字段闭集为：

```text
schema_version
component
source.repository
source.commit
build.profile_sha256
build.toolchain_image
files[]
dynamic_dependencies[]
```

普通文件事实为`path,type,mode,size,sha256`；symlink事实为`path,type,mode,link_target`。动态依赖只记录每个ELF文件的`DT_NEEDED`名称，不记录宿主绝对路径。manifest不记录OCI digest，因为digest只有push完成后才存在。

tag只作为并发隔离的push句柄，由CNB build ID和source commit生成。所有消费只接受`repository@sha256:...`。

## 安全恢复

puller在写输出目录前读取完整tar图，并拒绝：绝对路径、空路径、`.`或`..`分量、非UTF-8路径、重复路径、hardlink、目录entry、FIFO、socket、设备节点、稀疏文件、特殊权限位、绝对symlink、逃逸payload的symlink，以及任何经过symlink祖先的entry。

验证通过后，puller在全新目录中先写普通文件、后建symlink，再从文件系统重新生成文件图与manifest比较。错误直接返回非零，不加入fallback格式、默认文件或兼容壳。

## 权威交接

build/publish任务只输出候选digest和证据，不直接修改Git。候选digest先进入明确标为candidate的Git文件，fresh pull任务只读取该完整digest。fresh pull通过后，人工审查的后续commit才把`manifests/artifacts/cxlmemsim.json`作为正式引用提交；失败不得改变现有正式引用。

组件源码、artifact与GitHub镜像的顺序为：

```text
实现commit先进入CNB primary
  → 相同SHA进入GitHub mirror
  → cxl-lab source lock更新到该SHA
  → CNB任务按branch与exact SHA构建
```

## 最小接口约束

本任务可以决定CXLMemSim的构建命令、测试目标和三文件payload。它不得捆绑系统动态库，不得初始化十五个submodule，不得恢复或构建QEMU、Concordia、kernel、llama或guest，不得修改CUDA/BAR2协议，不得用artifact证据关闭Type-2或模型任务。

下游可依赖的最小事实是：给定正式`repository@digest`，固定puller会恢复上述三文件图，并证明server在声明工具镜像内可装载、四个局部CTest在build任务通过。下游仍需自行证明该server和shim进入实际Type-2运行路径。

## 停止与回滚

出现profile无法机械消费、CTest失败、server help失败、文件图漂移、动态依赖缺失、归档非确定、OCI push失败、fresh pull失败或Git源码身份不一致时停止。保留原始日志和候选digest；不得提升正式manifest。

实现commit可整体回滚。已发布但未提升的digest保持候选状态，不修改运行输入。正式manifest提升后需要回滚时，以Git commit恢复上一个已验证digest。

## 验收

- 目标结构符合上述ASCII树，组件build和通用publish/pull职责分离；
- `bash -n`通过，Python单元测试覆盖普通文件、symlink、篡改、绝对路径、`..`、hardlink、重复路径和symlink祖先；
- fresh CNB checkout按profile构建server和shim，四个CTest与server help通过；
- payload恰好三项，类型、mode和link target与profile一致；
- 两次本地归档SHA256一致；
- artifact发布到`docker.cnb.cool/gevico.online/jensen/cxlmemsim`并取得digest；
- 新CNB任务只按digest恢复，manifest、归档、ELF依赖、symlink和server help均通过；
- 正式Git manifest只在fresh pull成功后提交；
-结论停在CXLMemSim组件文件可恢复，不外推Type-2、固定1.5B、Kimi或tps。

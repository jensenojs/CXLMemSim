# CXLMemSim 首次组件制品证据

## 输入与目标

- 组件源码基线：`af3f2b6d62cbd59294b4a96e7ca376a9a10d0362`
- 固定toolchain：`docker.cnb.cool/gevico.online/jensen/cxl-lab@sha256:8732c435c97964f2ba95b42fbbcaf79b3c23feffc3b1f7683531917122f4f59e`
- payload：`bin/cxlmemsim_server`、`guest/libcuda.so.1`、`guest/libcuda.so -> libcuda.so.1`
- 正式manifest：`manifests/artifacts/cxlmemsim.json`

## 失败链

首次CNB build [`cnb-bvo-1jtar2vrc`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-bvo-1jtar2vrc)在Clang编译入口立即失败：`include/perf.h`找不到`bpf/bpf.h`。该任务没有进入CTest、shim构建或发布，也没有候选digest。

修复没有修改CXLMemSim源码。`cxl-lab/.ide/Dockerfile`从当时的固定toolchain digest派生最小增量层，只安装`libbpf-dev`。工具链任务[`cnb-l6o-1jtardo6d`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-l6o-1jtardo6d)成功并产生当前toolchain digest。

## 构建与发布

CNB任务[`cnb-js9-1jtarqp1k`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-js9-1jtarqp1k)在4 CPU、8 GiB runner上完成：

- Clang 21 configure和`cxlmemsim_server`构建；
- `test_dcd_gfam`、`test_rob`、`test_mem_stall`、`test_bandwidth_model`全部通过；
- guest shim构建并保留符号链接；
- server help、文件事实、动态依赖和两次确定性归档检查通过；
- ORAS 1.3.3发布成功。

候选制品：

```text
repository=docker.cnb.cool/gevico.online/jensen/cxlmemsim
digest=sha256:d7e6996f155d0867916b8bab947ec4f481243717858e9d52441ccaa9d6c2255a
archive_sha256=cffa109c3956140835ab1d38e15d919c617ecc9308a52c4c88ed0b0cd220ef89
manifest_sha256=c4946b6434414f94704fdc908472f82d59ad15503e0df3281353531cadb5032e
profile_sha256=437408d971928437d3ab4f43c01829f215be48b636ee6d688ca06d2fac4483be
```

## Fresh pull

独立任务[`cnb-c5t-1jtasc95s`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/build/logs/cnb-c5t-1jtasc95s)只读取完整digest，在2 CPU、4 GiB runner中完成7项归档合同测试、双blob hash检查、解包前安全预检、全新目录恢复、manifest重算、`ldd`、server help和symlink检查，输出`component_fresh_pull=pass`。

验证后，commit `708b74e1443d0a2c598229779e4e6d5d370ed627`把候选文件提升为正式`manifests/artifacts/cxlmemsim.json`。已发布但未进入该manifest的其他digest不得作为运行输入。

## 证明范围

该证据证明CXLMemSim三文件组件可以在新CNB任务中按digest恢复，并证明四个局部CTest在构建任务通过。它不证明该server或shim进入实际QEMU/guest路径，不证明Type-2 discovery、tiny、固定1.5B、Kimi或性能。

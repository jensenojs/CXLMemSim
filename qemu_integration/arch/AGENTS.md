# 已归档的 QEMU 集成入口

这个目录保存已经确认不能作为当前执行入口的历史文件。保留文件是为了保存当时的参数、命名和失败现场；它们不参与构建、CI、CNB制品或Kimi运行，也不能作为新脚本模板。

| 文件 | 归档依据 | 当前替代入口 |
| --- | --- | --- |
| `launch_qemu_type2_gpu.sh` | `bash -n`在原第34行失败；文件含损坏的shell token与未形成的QEMU argv | 本地使用`qemu_integration/launch_qemu_vcs_dcd_gfam.sh`；CNB Kimi使用`cxl-lab/scripts/run/run_kimi_same_vm.sh` |
| `launch_qemu_type2_hetgpu.sh` | `bash -n`在原第18行失败；文件含损坏的shell token、历史硬编码路径和不成立的fallback | 同上 |

新发现的失效入口先保留原始文件，再记录失败命令、最小错误、最后已知调用点、当前替代入口和不能证明的范围。恢复历史能力时必须在当前QEMU、guest与CXLMemSim合同上建立维护入口；禁止修补归档文件后直接移回或调用。

归档文件不要求`--help`或`--hint`，因为它们已经失去CLI地位。本目录`AGENTS.md`是任何agent进入这些文件时的第一条解释边界。

# QEMU 集成 workload 清单

这份清单回答现有benchmark源码是否仍有当前调用者。它不把历史源码提升为当前入口，也不替代真实运行证据。状态只依据当前checkout中的Makefile、README、脚本、artifact与调用引用；没有找到调用者时写`unverified`，不推断源码已经无价值。

| 路径 | 当前调用者或证据 | 当前状态 | 下一步 |
| --- | --- | --- | --- |
| `qemu_integration/guest_libcuda/ggml_flash_attn_ext_trigger.cpp` | `type2-guest/manifests/tiny-1234-profile.json`从CXLMemSim exact source commit提取此文件；guest builder把它与同一profile的llama公开头文件、GGML DSO编成`/bin/ggml-flash-attn-ext-trigger` | active-diagnostic-source | 保持为公开GGML图的唯一源码；变更图shape、oracle或依赖时，同时更新type2-guest profile与cxl-lab tiny gate合同 |
| `qemu_integration/guest_libcuda/cxl_bar_benchmark.c` | `guest_libcuda/Makefile`的默认BAR target；`qemu_integration/README.md`给出static运行入口 | active-workload | 保持当前位置，后续与guest shim测试边界一起评估是否形成独立workload目录 |
| `qemu_integration/zettai_benchmark.sh` | `qemu_integration/README.md`仍给出launch、guest与Type-2 benchmark命令 | active-workload | 保持当前入口；补齐最近真实运行身份后再决定目录迁移 |
| `qemu_integration/ssd_stream_two_qemu_bench.sh` | 直接编译并消费`dax_stream_bench.c`；设计与验证命令保存在对应plan | active-workload | 作为同一SSD/DAX workload组合治理 |
| `qemu_integration/dax_stream_bench.c` | 由`ssd_stream_two_qemu_bench.sh`直接消费 | active-workload | 与调用脚本一起迁移或保留，不单独归档 |
| `qemu_integration/guest_libcuda/gpu_benchmark.c` | 仓库根README列出名称；当前Makefile、CNB和维护脚本未找到调用者 | unverified-public-workload | 核对README所指构建/运行合同和最后结果后决定active或historical |
| `qemu_integration/guest_libcuda/cxl_bias_benchmark.c` | 仓库根README列出名称；当前Makefile、CNB和维护脚本未找到调用者 | unverified-public-workload | 核对README所指构建/运行合同和最后结果后决定active或historical |
| `qemu_integration/guest_libcuda/rq1_graph_bfs.c` | `artifact/splash_sweep/`保存历史manifest、CSV和报告；当前Makefile/CI未找到调用者 | historical-workload | 保留现状，待历史结果身份闭合后整体移入`arch/benchmarks/splash-sweep/` |
| `qemu_integration/guest_libcuda/rq2_dir_sizing.c` | `artifact/splash_sweep/`保存历史manifest、CSV和报告；当前Makefile/CI未找到调用者 | historical-workload | 与splash-sweep历史证据整体归档 |
| `qemu_integration/guest_libcuda/rq3_bias_kv.c` | `artifact/splash_sweep/`保存历史结果；当前Makefile/CI未找到调用者 | historical-workload | 与splash-sweep历史证据整体归档 |
| `qemu_integration/guest_libcuda/rq4_alloc_policy.c` | `artifact/splash_sweep/`保存历史结果；当前Makefile/CI未找到调用者 | historical-workload | 与splash-sweep历史证据整体归档 |
| `qemu_integration/guest_libcuda/rq4_devfrac_sweep.c` | `artifact/splash_sweep/`保存历史结果；当前Makefile/CI未找到调用者 | historical-workload | 与splash-sweep历史证据整体归档 |
| `qemu_integration/guest_libcuda/plot_results.py` | 解析RQ日志，也内置placeholder数据；未发现当前自动调用者 | historical-helper | 与splash-sweep源码和结果一起治理，禁止把placeholder当新实验结果 |
| `qemu_integration/guest_libcuda/agentic_bias_benchmark.c` | 源码内保留独立编译命令；未发现Makefile、README或当前脚本调用者 | unverified | 查Git历史与最后结果；证据不足前不删除 |
| `qemu_integration/guest_libcuda/cpu_gpu_hitm_benchmark.c` | 只有`tools/mig_nccl_hitm/`中的后续benchmark注释引用其设计 | historical-reference | 保留作为后续实现的来源证据，确认最后结果后归档 |
| `qemu_integration/guest_libcuda/gpu_validation_bench.c` | 源码内保留native/guest编译命令；未发现当前调用者 | unverified | 查最后运行证据后分类 |

后续迁移必须同时更新Makefile、README、脚本调用点和证据路径。`arch/`不进入自动发现、构建、CNB或Kimi运行；任何入口一旦仍有current caller，就不能只凭文件名或时间归档。

## 公开 GGML 图在 cheap gate 中的位置

`ggml_flash_attn_ext_trigger.cpp`把一次已经在Kimi core中观察到的公开GGML CUDA图缩小为合法、可重复的输入。它用公开GGML API建立F16的Q、K、V、mask和输出tensor，要求CUDA backend支持该图，计算完成后同步，并逐元素检查零输入应得到有限的`0.0f`输出。这个数值oracle让“程序退出零”与“计算结果正确”成为两个独立事实。

该源文件属于CXLMemSim，因为它定义了要触及的guest CUDA shim、BAR2、QEMU和HetGPU边界；它的编译输入属于type2-guest，因为guest executable必须链接本轮profile已经冻结的GGML动态库。最终组合关系是：

```text
CXLMemSim exact source commit
  -> ggml_flash_attn_ext_trigger.cpp

llama-cpp exact source commit
  -> ggml/include

llama-cpp component artifact
  -> libggml-cuda.so.0 + libggml-base.so.0 + libggml.so.0

type2-guest builder
  -> /bin/ggml-flash-attn-ext-trigger

cxl-lab Type-2 tiny run
  -> C_init -> C_ggml -> C_tiny
```

这个关系避免从开发机checkout或不同版本的头文件借用编译输入。源文件、头文件、library和initramfs executable的SHA256由type2-guest materialization/build manifest保存；CXL Type-2真实执行、同步和数值结果由cxl-lab的run result保存。入口、输入参数和调试边界见仓库根[`AGENTS.md`](../AGENTS.md)中的可复用调试证据。

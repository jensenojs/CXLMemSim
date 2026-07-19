# QEMU Integration

本目录保存CXLMemSim与QEMU、guest、Type-2 CUDA shim之间的集成资产。当前维护入口、历史workload与归档证据必须分开：

```text
guest_libcuda/
  guest CUDA Driver shim、BAR2协议、直接ABI probe和紧邻测试

当前目录的launcher与benchmark
  仍被Makefile、README、当前脚本或已验证命令直接消费的集成入口

arch/
  已证实损坏或已经失去当前调用合同的历史入口

workload-inventory.md
  benchmark源码的调用者、最后证据、维护状态与下一步分类
```

`manifests/`只记录某次运行消费的workload身份、参数和结果，不拥有benchmark源码。仍维护的benchmark最终进入明确的`bench/<name>/`或`workloads/<name>/`；只剩历史学习价值的源码进入`arch/benchmarks/`。在inventory确认调用者、最后可验证命令与替代入口之前，不批量移动或删除现有workload。

由人或agent直接调用的维护中Shell/Python入口必须提供`--help`与`--hint`。`--help`说明参数和失败语义；`--hint`指出项目/仓库/目录权威、相邻工具、输入、输出、证明边界和下一条证据边界。内部库、测试夹具、生成二进制和`arch/`中的失效文件不伪装成CLI。

进入`guest_libcuda/`修改公开CUDA入口、private export-table探针或BAR2协议前，继续读取仓库根`AGENTS.md`中的ABI与验证边界。任何正式L40/exact artifact运行由cxl-lab diagnostic run spec拥有，不在本目录手工拼接CNB身份。

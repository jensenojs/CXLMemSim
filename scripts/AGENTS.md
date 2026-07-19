# CXLMemSim 组件脚本

本目录把本仓的 server 与 guest CUDA shim 从源码变成可恢复的组件 payload。它不决定哪一个组件进入 Kimi run，也不解释 benchmark、correctness 或 TPS；这些职责由控制仓的 run manifest、observer 与 result 合同拥有。

```text
exact CXLMemSim checkout + manifests/build-profile.json
  → build_component.sh
  → .work/component/payload/ 与 build evidence
  → publish_component.sh
  → candidate JSON + OCI component
  → cxl-lab component_candidate.py
  → official manifest
  → pull_component.sh
  → fresh payload + local runtime/ELF checks
```

## 入口与输入

| 入口 | 读取什么 | 写出什么 | 用来回答的问题 |
| --- | --- | --- | --- |
| `verify_source_checkout.sh` | expected source SHA 与当前 Git checkout | exit status | 当前 checkout 是否可作为声明的构建输入 |
| `build_component.sh` | `manifests/build-profile.json`、`manifests/artifact-contract.json`、当前 source | `.work/component/payload/` 与构建/测试日志 | 本仓声明的 server、shim、case CLI 是否可由精确源码构建 |
| `component_artifact.py` | payload、profile、contract、显式 archive/manifest | manifest、确定性 archive 校验结果 | payload 文件图和归档是否完整、无路径逃逸并可验证 |
| `publish_component.sh PAYLOAD_DIR` | 已闭合 payload、profile、contract、CNB registry身份 | OCI双 blob 与 `component-candidate` 输出 | 该 payload 是否已被不可变 digest 发布 |
| `pull_component.sh [--container-runtime docker|podman] CANDIDATE_JSON OUTPUT_DIR` | 完整 candidate digest、artifact contract | 新的输出目录与 `.work/component/fresh-pull/` 验证记录 | 已发布的 exact bytes 能否独立恢复并通过本仓 runtime/ELF smoke |

`pull_component.sh` 默认使用 CNB Docker service。只在本机已明确选择 rootless Podman 时传 `--container-runtime podman`；Docker 失败不会静默改用 Podman。参数形状错误以 exit 2 结束，已经通过解析后的 container、OCI、digest、archive、ELF 或运行时检查失败以非零运行错误结束。

## 证据与边界

`.work/`、build目录、下载的 archive、恢复 payload 和日志都是一次执行现场，不进入 Git。candidate 的传递、official artifact manifest 的更新、fresh-pull event 的触发和正式 run 的输入选择只通过 [cxl-lab artifact contract](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/artifacts/AGENTS.md) 执行。

这个目录能证明：源码、profile 和工具链构造了指定文件；OCI 归档经 digest 与逐文件 manifest 复原后仍满足本仓的 server/shim/CLI 基础检查。

它不能证明：QEMU Type-2 已 realized、guest 已加载 shim、Concordia/NVIDIA 已执行 Kimi kernel、baseline 与 concordia 输出相同，或任何 TPS 结论。那些问题从 [cxl-lab run contract](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/run/AGENTS.md)、[source/artifact/run/result identity](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/AGENTS.md) 和 [immutable result archive](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/results/AGENTS.md) 继续追踪。

遇到 guest `libcuda.so.1`、private Runtime export、BAR2 或 CUDA API 失败，先读 `../qemu_integration/AGENTS.md`。其中保存静态 ELF collector、runtime private-export probe 与 core/debugger capture 的用途和证据边界；不要把这些现有入口重新拼成临时 one-shot shell 命令。

# CXLMemSim 组件脚本

本目录把本仓的 server 与 guest CUDA shim 从源码变成可恢复的组件 payload。它不决定哪一个组件进入 Kimi run，也不解释 benchmark、correctness 或 TPS；这些职责由控制仓的 run manifest、observer 与 result 合同拥有。

```text
exact CXLMemSim checkout + manifests/build-profile.json
  → build_component.sh
  → 调用者指定的 payload 与 build evidence
  → package_component.sh
  → manifest + deterministic archive
  → publish_component.sh
  → candidate JSON + OCI component
  → cxl-lab component_candidate.py
  → official manifest
  → pull_component.sh
  → exact toolchain image中的唯一payload verifier
  → fresh payload + runtime/ELF checks
```

## 入口与输入

| 入口 | 读取什么 | 写出什么 | 用来回答的问题 |
| --- | --- | --- | --- |
| `verify_source_checkout.sh` | expected source SHA 与当前 Git checkout | exit status | 当前 checkout 是否可作为声明的构建输入 |
| `build_component.sh --work-dir ABSENT --cache-dir EXISTING --payload-dir ABSENT` | clean source、build profile与显式cache | 指定payload与work内构建证据 | 本仓声明的server、shim、loader audit、case CLI是否可由精确源码构建 |

组件构建通过必填`--cache-dir`消费独立编译缓存。work与payload在调用前必须不存在，脚本只创建自己声明的路径，不删除已有目录。tracked或untracked dirty状态在创建输出前失败。CMake目标和guest shim的C/C++编译都经过`ccache`，并在evidence中保存构建前后统计；cache不进入source、profile或artifact identity。
| `component_artifact.py` | payload、profile、contract、显式 archive/manifest | manifest、确定性 archive 校验结果 | payload 文件图和归档是否完整、无路径逃逸并可验证 |
| `package_component.sh --payload-dir EXISTING --source-commit 40_HEX --work-dir ABSENT --manifest-out ABSENT --archive-out ABSENT` | verified payload、clean exact HEAD、profile与contract | 原子manifest与确定性archive | artifact bytes是否在无网络边界内生成并fresh verify |
| `publish_component.sh --manifest EXISTING --archive EXISTING --work-dir ABSENT --candidate-out ABSENT` | 已验证manifest/archive、contract与CNB registry身份 | OCI双blob、candidate文件与CNB output | exact artifact bytes是否已被不可变digest发布 |
| `pull_component.sh --container-runtime docker|podman --work-dir ABSENT_ABSOLUTE_PATH CANDIDATE_JSON OUTPUT_DIR` | 完整 candidate digest、artifact contract | 新的输出目录与调用者拥有的pull现场 | 已发布的 exact bytes 能否独立恢复并在exact toolchain image中通过runtime/ELF smoke |
| `verify_component_payload.sh PAYLOAD EVIDENCE` | 一个已组装或恢复的payload | 唯一runtime、ELF、symbol、CLI与文件hash证据 | build与fresh-pull是否消费同一payload语义合同 |
| `run_integrity_export_oracle.sh TABLE LOG_PREFIX` | L40真实Driver、当前guest shim、指定private export table | Driver identity与oracle输出 | 真实Driver和shim对该table的公开可比形状是否一致 |
| `run_private_export_table_probe.sh [OUTPUT_DIR]` | L40真实Driver、tiny DSO、固定discovery/capture选择 | discovery/capture transcript与identity comparison | 自然到达的private table调用及slot 1输入是什么 |

`pull_component.sh`要求显式选择Docker或Podman，并要求绝对、父目录已存在、叶子不存在且祖先无symlink的work目录。入口exclusive-create并保留失败现场，不删除或复用旧目录。OCI transport使用所选runtime；静态archive/file manifest验证由`component_artifact.py`拥有，runtime ABI与可执行gate在artifact contract的exact toolchain image中以调用用户UID/GID执行。宿主缺少组件动态库时不构成artifact失败，也不触发fallback、安装或`LD_LIBRARY_PATH`补偿。

## 证据与边界

`.work/`、build目录、下载的 archive、恢复 payload 和日志都是一次执行现场，不进入 Git。candidate 的传递、official artifact manifest 的更新、fresh-pull event 的触发和正式 run 的输入选择只通过 [cxl-lab artifact contract](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/artifacts/AGENTS.md) 执行。

`local-host-diagnostic`只调用build并停在payload，不创建manifest。`local-container-artifact`与CNB formal在`artifact-contract.json`的exact toolchain image中调用相同build/package；只有publish需要registry凭据和网络。三个入口失败都保留work现场，不创建对应终态输出，不接受旧位置参数或默认work路径。

这个目录能证明：源码、profile 和工具链构造了指定文件；OCI 归档经 digest 与逐文件 manifest 复原后仍满足本仓的 server/shim/CLI 基础检查。

guest payload中的`libcxl-loader-audit.so`是只读loader observer。`pull_component.sh`对它保存独立的`ldd`和
`readelf`记录；type2-guest profile决定它是否进入某个guest，cxl-lab决定它的case目录、结果归档和结论。

它不能证明：QEMU Type-2 已 realized、guest 已加载 shim、Concordia/NVIDIA 已执行 Kimi kernel、baseline 与 concordia 输出相同，或任何 TPS 结论。那些问题从 [cxl-lab run contract](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/scripts/run/AGENTS.md)、[source/artifact/run/result identity](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/AGENTS.md) 和 [immutable result archive](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/results/AGENTS.md) 继续追踪。

遇到 guest `libcuda.so.1`、private Runtime export、BAR2 或 CUDA API 失败，先读 `../qemu_integration/AGENTS.md`。其中保存静态 ELF collector、runtime private-export probe 与 core/debugger capture 的用途和证据边界；不要把这些现有入口重新拼成临时 one-shot shell 命令。

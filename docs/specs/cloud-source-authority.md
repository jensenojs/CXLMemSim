---
status: completed
---

# 云端源码权威边界

## 与项目目标的关系

本切片把CXLMemSim的superproject源码历史迁移到CNB组件仓，为后续独立构建、制品恢复和CNB Type-2运行提供可追溯输入。它只关闭源码仓库准备与切换边界。Kimi K2.6 correctness和同一run specification下的baseline/concordia tps仍是项目终点。

## Blast Radius

BR3。迁移会改变CXLMemSim后续新commit的接收方，并最终改变`cxl-lab`中的source URL。准备阶段不得提前改变当前权威。

## 权威状态

```text
准备状态
  GitHub jensenojs/CXLMemSim primary
    → CNB gevico.online/jensen/cxlmemsim candidate
    → 同一migration commit进入CNB和GitHub
    → 两端fresh mirror clone、ref map、git fsck和source probe通过

切换状态
  cxl-lab refs/heads/fixed-1p5b-control上的单个cutover commit
    → source URL与commit指向CNB migration commit
    → 删除控制仓候选profile副本和集中restore入口
    → CNB primary，GitHub public mirror
```

cutover commit成为有效控制ref的远端tip之前，CNB仓始终是candidate。组件仓文档和仓库存在本身都不能触发切换。

## 目标结构

```text
CXLMemSim/
├── AGENTS.md                              # 仓内约束与candidate/primary条件
├── docs/specs/cloud-source-authority.md   # 本仓迁移边界
├── manifests/build-profile.json          # 尚未经过真实build验证的候选profile
├── scripts/verify_source_checkout.sh      # 只验证checkout事实
└── .cnb.yml                               # 显式使用Bash执行source probe
```

## Superproject历史边界

迁移对象是准备时GitHub项目fork通过`git ls-remote --heads --tags`公开的refs及其可达对象：

```text
refs/heads/cloud-type2-tiny-baseline
refs/heads/main
refs/heads/memu
refs/heads/micro26
refs/heads/pnnl
refs/heads/wasm
refs/heads/yarch23
refs/tags/main
refs/tags/memu
```

这份列表是迁移准备时的历史快照。`refs/tags/main` 指向
`b6ba524d69c13929ac5ec525643cc286fb8e49e6`，该 commit 已由当前 `refs/heads/main` 保留。
2026-07-20，项目的 CNB primary 与 GitHub mirror 删除了这个同名 tag，因为裸词 `main` 会在 Git 的 source-ref
解析中同时匹配 branch 与 tag，拒绝正常推送。SlugLab upstream 保留原始 tag 作为上游历史。后续默认分支只使用
完整 ref `refs/heads/main`，新 tag 不得复用已有 branch 的短名。

只有`refs/heads/cloud-type2-tiny-baseline`允许前进到migration commit，其他ref的object ID必须保持不变。两端必须从空目录执行非shallow mirror clone，排序比较heads/tags ref map，并运行`git fsck --full`。

`.gitmodules`中的十五个外部仓库不属于本切片。真实组件构建必须在后续spec中声明实际消费的gitlink commit和恢复方式。

## Source probe

```text
CNB candidate checkout
  → bash scripts/verify_source_checkout.sh 4910c7cf2c813698952857f20988ac66dae7fe9d
  → 拒绝shallow checkout
  → 验证功能基线是HEAD祖先
  → 验证tracked与untracked状态为空
  → 输出HEAD、baseline和clean证据
```

probe不得补取历史、初始化submodule、构建代码、清理现场、写源码树或修改Git状态。完整ref map由fresh mirror验证承担，probe只证明CNB任务实际checkout的提交与clean状态。

CNB的`git`配置属于Pipeline字段，必须放在`api_trigger_source_probe`对应的pipeline对象内。把它放在`$`分支层不会覆盖默认值；失败任务`cnb-kf8-1jtamf1ni`因此仍执行了递归submodule初始化并被主动停止。有效配置必须在pipeline内明确写出`submodules: false`与`lfs: false`。参考[CNB语法手册的Pipeline Git配置](https://docs.cnb.cool/zh/build/grammar.html#pipeline-git)。

## 候选build profile

`manifests/build-profile.json`必须与迁移前`cxl-lab/manifests/sources.lock.json#build_profiles.cxlmemsim-release`对象规范化后完全一致。此次只迁移声明所有权，不改变编译器、CMake开关、目标或输出图。首次真实组件build通过前，它不是已验证构建身份。

## 停止与回滚

出现以下任一事实时停止切换：声明ref缺失或object ID漂移、任一mirror为shallow、`git fsck --full`失败、migration commit不能到达功能基线、CNB source probe失败，或CNB与GitHub migration SHA不同。

cutover前可直接放弃CNB candidate，GitHub仍是primary。cutover后若CNB恢复失败，必须整体回滚`cxl-lab` cutover commit，同时恢复GitHub URL、候选profile和集中restore入口，不能留下半切换状态。

## 验收

- 本commit只修改仓库控制文件，功能源码diff为空，现有未跟踪build与诊断二进制保持未跟踪；
- migration commit以`4910c7cf2c813698952857f20988ac66dae7fe9d`为祖先；
- profile与控制仓候选对象的`jq -S`结果一致；
- source probe通过`bash -n`，拒绝shallow，不包含恢复、构建或清理逻辑；
- `.cnb.yml`可解析，并显式调用固定baseline SHA；
- 本阶段结论只到candidate控制文件形成，不证明CNB历史已经迁移或源码权威已经切换。

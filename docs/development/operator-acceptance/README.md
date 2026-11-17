# 本地算子提交与验收

本目录属于 `llama.cpp-omni-operators` / `fix/upstream-operators` 本地测试维护工作区，不是公开PR的提交组织方案。2026-11-17用户授权在此继续细粒度提交，并暂时跳过macOS、Metal、MoltenVK及其他缺少目标验收条件的子项；其他平台通过不能替代这些验收。

## 每个提交的约定

- 每个新提交附带一份独立验收文档，使用 `<日期>-<候选/主题>.md`；文档随对应代码/测试提交，不为补写自身SHA再制造递归提交。
- 必须记录：本地输入基点、原始`llama.cpp`/`ggml`仓库及完整commit ID、抽取范围与必要依赖、本地适配、验证命令/配置/设备、实际执行/拒绝数量、退出状态、红绿或其他决定性证据、未覆盖项及PR拆分建议。
- 无直接上游来源的适配明确标为本地修改，不编造来源。文档或工作流提交同样有独立验收说明，注明无新增上游实现。
- 本地commit可用简洁主题，不强制公开PR的双语正文等格式；保留准确的来源和AI辅助事实于验收文档，公开提交时由人类重新审阅、组织及披露。
- 实际提交身份/来源仍须真实；不改全局Git配置。维护小而可验证的提交，避免仅为凑粒度拆出不能独立工作的中间状态。

查询某份验收文档对应的原始提交：

```bash
git log --diff-filter=A --format='%H %s' -- docs/development/operator-acceptance/<文件>.md
```

## 边界与证据

默认只使用本机可验收的CPU/CUDA/Vulkan子项；HIP、ARM、其他GPU等也按实际设备能力单独判断，不自动视为有条件。实现、构建、数值验收、动态检测、性能和模型端到端状态分别记录。已有构建可在相同工作区和配置下定向增量复用，迁移前旧绝对路径cache不可复用。

`build-*/evidence/`保留raw logs、probe、manifest和job元数据，全部ignored；稳定结论写入验收文档，不把二进制、日志或权重提交到Git。更新PR worktree时由用户选择提交粒度和范围，不自动cherry-pick、推送、Ready或合并。LES、master、远端及CI/CD仍遵循原边界。

本地队列与总体状态见[算子执行记录](../upstream-operator-batches.zh.md)。此README与B03记录随同一提交引入；其验收包括文档链接/来源核对、既有B03源码与产物manifest重验及`git diff --check`，没有新增运行时代码。

# 上游算子修复包规划

确认时间：2026-11-19。M1 已按本机目标范围完成[组合验收](operator-acceptance/2026-11-19-m1-layout-cuda-correctness.md)。开发起点 `fix/upstream-operators@c64b5c69bf253504973e2628fb083e52c1caa0f1`；固定上游 `ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`（本地只读参考）。Draft PR #156 本地快照仍在 `b669a4a4d8dcdcef69a257e8194da0d71dc0302c`，本计划不授权更新或推送。

## 范围与交付方式

- 以依赖闭合的**修复包**为实现和回退单位，一个交付批次可以包含多个修复包。在同一棵候选树上验收，不为复用构建而混入无关变化，不等待逐算子确认；保留少量逻辑提交和一份[批次验收记录](operator-acceptance/README.md)。
- 先做本机 CPU/CUDA/Vulkan；ROCm、Metal 和其他缺目标环境的专用修复后置。HIP 构建复用 `ggml-cuda` 源码，有共享代码变更时保留条件分支与旧契约，记录可能触及的未验路径，不把 NVIDIA 结果记作 HIP 验收。
- 上游补丁、必要依赖和后续修正都在固定历史内核对。不整文件覆盖 omni、不升级主线、不混入 LES、CI、新格式、公共 API 或性能重排。若有必要改变公共构图语义，须证明现有后端的旧调用及新布局拒绝/fallback 一致；闭合不了就不开放新布局。

## M1：布局一致性 + CUDA 执行正确性

| 修复包 | 上游来源（`ggml-org/llama.cpp`） | 本地交付边界 |
|---|---|---|
| M1-A 布局与能力（B12） | `0878f7c304cc13915c831ae17694534c94940176` SUM；`08246a28f6000100433d297c4e037c02e9d2d464` CUDA unary；`1a7718b4c57227a6c6187a2359c883c4d3089936` Vulkan unary/GLU；`de5122d879d00d2895c36be7e0cee94105aeeabd` SCALE | 已有 CPU F32 SUM 复用；同步 CUDA/Vulkan SUM 选择和真实内核布局。CUDA unary 核对新旧 dtype/stride 交集；Vulkan 同步 host、push constants、shader generator 和实际执行。SCALE 必须连同构图、目标后端和不支持的后端选择一起闭合，否则保留现有限制，单列未纳入范围。 |
| M1-B CUDA 内核正确性（B10） | `b74f590eafec2fafc6e0e98ee93b2e5d3efa9042`、`1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3`、`99db719bdf07f7d537431ca556a0512bf3d995b6` FA 同步；`0eca4d490e591d4e93058d07540cf47278a72577` mask stride；`9bd4c09ea571a9020f30eeef169b552625b5b5a4` 归约；`9e58d4d692ed3d350591cc86d06c73c61c122509` PDL 配套调查 | 只修现有 FA、norm/softmax 路径；保留 B07/B11 的同步与 dispatch。只有已证实影响本地路径的 PDL 配套才纳入，不引入不相干的融合或 shared helper 改造。 |

### 执行顺序与验收门槛

1. **钉住基线。** 先独立构建打开 `GGML_CUDA_FA=ON` 的 NVIDIA 基线（现用 `build-operators-b03-cuda` 关闭 FA）。复用同工作区的 Vulkan 与 CPU sanitizer 配置；保留既有日志。记录设备、编译选项、代码树和实际 kernel/后端。
2. **核对并实现两个包。** 先列出布局类型及输入/输出约束，逐项核对 `supports_op`、调度落点、kernel 和 CPU fallback。并行分析 CUDA FA 分发与归约共享内存，按适用性取所需补丁；缺入场条件的项目先列证据，不凭静态推断宣称通过。
3. **统一组合验收。** 三后端都验证变化路径和既有连续快路，按算子挑选换轴、间隙、非零 offset、原地、GLU 两输入不同 stride、广播和 dtype。预期值或 CPU reference 必须独立于待测路径。GPU 明确区分实际执行、拒绝、fallback、过滤无命中。FA 覆盖不同实际 kernel、mask/尾块和适用的 memcheck/racecheck/synccheck；同时复验 B03/B07/B11 的交叠用例，包含 B11 Vulkan 曾有退出异常的组合测试。通过判定包括退出码，不只看用例数。
4. **收口审阅。** 对最终树记录每包的 SHA、依赖、本地适配、红绿/决定性证据、实际执行和未运行范围。一批一次交付，不自动追加到 PR。任何后续 PR 候选都从现有快照提取纯源码和测试并重新构建，不重写已审阅提交。

## 后续批次

- **M2 音频算子通路（已验收）**：C01 CUDA/Vulkan `POOL_1D`（`29ea9412a`、`876a43211`）及 C02 Vulkan `PAD_REFLECT_1D`（`3f545becc`），见[组合验收](operator-acceptance/2026-11-19-m2-audio-pool-reflect.md)。实际 GPU placement 和 CPU fallback 均已用独立预期检查；未替换 omni 自有 CUDA 反卷积快路，未带 WebGPU 或 HIP 项。
- **M3 类型与量化通路（下一批）**：核对 C04 CPU/CUDA BF16 RMS 与 CPU F16 `SET_ROWS` 的全线程 scratch 依赖，以及 C09 Vulkan 既有量化格式的 MM/MMV/ID。按真实消费者、dispatch 和可验类型分包；不纳入新量化格式或未实测的设备路径。细目来源见[原始调研](upstream-operator-survey.zh.md)。

本文件是执行计划，不是已完成清单；进度及历史验收见[执行记录](upstream-operator-batches.zh.md)。计划、上游差异核对及验收设计有 AI 辅助。

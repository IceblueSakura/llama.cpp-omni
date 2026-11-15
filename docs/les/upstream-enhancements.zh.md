# llama.cpp 可吸收的 LES 性能增强与算子修复

后续算子级调查见 [上游算子修补与增强](../development/upstream-operator-survey.zh.md)，按影响面由窄到广、结合难度安排参考顺序；没有现成上游修复的本地适配与额外疑点见 [扩展调研](../development/upstream-operator-extensions.zh.md)。本文保留 LES 整体方向和早期筛选记录；选择首批算子同步任务时，以前者的“小范围修补优先”顺序为准。

## 1. 状态、范围与版本

- 记录日期：2026-11-15。
- 状态：**只读源码调研后的候选记录，不是已批准的实施清单，也不是性能验收报告。**
- 本文所在分支：`fix/upstream-operators`，从本地 `master` 新建；文档由 `docs/upstream-les-enhancements` worktree 移入。
- 本 worktree 基点：`350a835d5f8215145aa210a7df0f67e8877b5d6e`。
- 调研时 omni 源码基线：`feat/les-backend-cache` 的 `714d80032296d92746d295c6e0fed3c798fb1df8`。
- 上游：<https://github.com/ggml-org/llama.cpp>，固定参考 `61dbf49228c8a96b72e40c89d93cb7c003411f8c`；本机源码 `/home/sakura/codespace/llama.cpp`。
- 调研没有 fetch，不声称覆盖远端最新版本；没有构建、数值或性能实测。已核对重点源码及提交 diff；收益仍需目标硬件验证。

**两个 omni 基线不能混用。** 本 worktree 没有合入 LES 工作分支的 context-owned storage、同步上传及失败恢复修改。下文的“当前实现”和源码行号默认指调研基线 `714d800`，不是本 worktree 的运行状态。另一会话正在推进该分支；实施前必须重核目标 HEAD、工作区变更及有关契约。

原 LES 工程计划和交接记录位于该工作分支的 `docs/les/backend-integration-plan.zh.md`、`docs/les/HANDOFF.md`，不在本 worktree 复制一份。需要读取固定检查点时可用：

```bash
git show 714d80032296d92746d295c6e0fed3c798fb1df8:docs/les/backend-integration-plan.zh.md
git show 714d80032296d92746d295c6e0fed3c798fb1df8:docs/les/HANDOFF.md
```

调研时工作区计划另有按 G1-C/G1-V/G1-Q/G1-M 拆分验收的未提交修订；上面命令不包含这些修订。执行顺序以届时负责 LES 的会话所维护的版本为准，不把此处记录当作修改其计划的授权。

### 目标与非目标

目标：在现有 LES 数据源和有界内存模型上，吸收能减少路由整理、无效计算、传输提交、中间张量读写及重建开销的成熟机制。

保留：packed 专家源、真实 router 语义、按需读取、lookahead、prefill/decode 管理、Apple shared 直写及有界 RAM/VRAM。

不做：整体同步 ggml/backend、独立普通 GGUF cache 产品线、恢复完整 host 专家副本、无硬件条件的支持声明、凭上游 benchmark 承诺 LES 加速比例。

## 2. 候选总览与建议顺序

优先级是调研建议，不是实施承诺；同优先级按硬件、实际热点及集成冲突选择。

| ID | 方向 | 建议 | 与现有 LES 计划的关系 |
|---|---|---|---|
| O1 | Vulkan 紧凑 `MUL_MAT_ID` tile | 优先，Linux 独立算子切片 | 新增计算侧能力 |
| O2 | CUDA 多 token MoE gate/up/GLU MMQ 融合 | CUDA 同步 LES 基线后 | 新增，omni 已有部分单 token 融合 |
| O3 | Metal 紧凑专家 tile、routing/归约融合 | 待真机 | 新增，不阻塞 Linux |
| O4 | CPU x86 tiled K/IQ-quant `MUL_MAT_ID` | CPU prefill 有需求时优先 | 新增，不等同于 T5 单 token CPU-share |
| S1 | 不复制未执行分支的输入 | 与 scheduler 负责人协调 | 新增，修改文件与现有工作重叠 |
| S2 | scheduler / graph-result 条件复用 | reserve 开销明显时 | 新增，不代替 device bank 持久化 |
| R1–R3 | NEON tile、F32 attention 累加、post-bias NaN | 小范围独立修复 | 新增，先按正确性/实际 dispatch 判断 |
| C1–C4 | few-row matmul、Vulkan FA、Hexagon、SVE | profile 和设备驱动 | 条件性后续候选 |
| P1–P4 | prefill 热集复用、split 预取、负 ID、批量 I/O | 提供给现有计划 | 实现参考，不重复建立主线 |

建议先做 O1 与适用的小修复，再按 CPU/CUDA 需求推进 O4/O2。O3 等真机；其余先测瓶颈。backend 算子级验证可以独立准备，LES 集成验收依赖当时对应平台的正确性基线，不能以单算子通过替代。

## 3. 新增高价值算子方向

### O1：Vulkan 紧凑 `MUL_MAT_ID` tile

**现状与机制**：omni 在 `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl:8–72` 的计算块中扫描 IDs；`ggml-vulkan.cpp:7787–7800` 仍按专家及 token tile 网格 dispatch。上游预处理一次生成 row IDs 和 `(expert, valid rows)` 紧凑描述，避免反复扫描及大量空工作组，同时正确展开重复 IDs。

**上游参考**：

- [90c26fcd4：row-ID / expert-count hoisting](https://github.com/ggml-org/llama.cpp/commit/90c26fcd4)。
- [966baae76：紧凑 tile 和重复 ID 预处理](https://github.com/ggml-org/llama.cpp/commit/966baae76)。
- 固定上游的 `vulkan-shaders/count_experts.comp`、`mul_mm_id_funcs.glsl`、`ggml-vulkan.cpp:6388–6407,7856–7858`。

**LES 价值**：多 token/prefill/小批验证的专家计算更直接受益；不改变文件格式，不需要全专家驻留。单 token 不一定进入该路径。LES 已缩小 bank 的专家维，必须按实际 bank 几何衡量，不能套用完整专家张量的收益。

**移植边界**：host scratch、push constants、prepass 和普通/量化/coopmat shader 必须一致；上游文件拆分不能照搬为独立 shader 更新。输入仍是物理 slot IDs。scratch 纳入峰值内存。

**最小验收**：现有 `test-backend-ops` 的 `MUL_MAT_ID`；top-k=1/8，rows=1/31/32/33/512，集中/分散路由、重复 IDs、非连续 ID view；CPU oracle 与 Vulkan validation。随后复用 LES scheduler/device/model 入口，记录 GPU 时间、scratch、读盘/上传字节与 logits。

### O2：CUDA 多 token gate/up/GLU MMQ 融合

**现状**：omni `ggml/src/ggml-cuda/ggml-cuda.cu:2389–2546` 已有 gate/up/GLU 融合识别和单 token 路径；缺的是 MMQ 专家计算扩展。

**上游参考**：

- [c49edf096：MMQ gate/up/GLU 基础融合](https://github.com/ggml-org/llama.cpp/commit/c49edf096)。
- [f5222f25c：扩展到 MoE MUL_MAT_ID](https://github.com/ggml-org/llama.cpp/commit/f5222f25c)。
- 上游 `ggml-cuda.cu:2375–2444`、`mmq.cu:169–240`。

**机制与适用性**：交错装载 gate/up，在同一 MMA 计算及写回中完成 GLU，减少 launch 与中间结果读写。面向支持的 NVIDIA GPU、量化类型、精度和 tile 几何。omni `src/llama-graph.cpp:1748–1768` 的分离 gate/up 可匹配；单张 merged `gate_up` 后拆 views 不是同一模式。

**依赖/风险**：先有 CUDA 同步 LES 基线。保留 compute-buffer view 的 padding 安全拒绝及未融合 fallback；不为命中融合删校验。若同时采用 CUDA MoE linearized layout `69ee2ecc0`，须核对空 tile 修复 `5d4c8eeb2`，不混用两代 tile ABI。

**最小验收**：融合开关对照，实际量化/stride/view 矩阵，确认执行的是 fused kernel；参考 Q4_K 17 tokens/16 experts/top-8、Q8_0 2048 tokens/40 experts/top-8，再测真实 LES。数值、kernel 次数、中间内存与端到端耗时一起记录。

### O3：Metal 紧凑专家 tile 与 MoE fusion

**上游参考**：

- [073b74034：紧凑专家 tile](https://github.com/ggml-org/llama.cpp/commit/073b74034)。
- [5b59b83f4：MoE 等融合](https://github.com/ggml-org/llama.cpp/commit/5b59b83f4)。
- [098ce526b：moe_reduce 输入/输出 alias 竞态修复](https://github.com/ggml-org/llama.cpp/commit/098ce526b)。
- `44f91a02b` 可作为融合组 command-buffer 边界的配套参考。

**现状/价值**：omni `ggml-metal-ops.cpp:2265–2280,2407` 仍采用每专家固定 token 容量及三维网格。紧凑 tile 减少稀疏专家空工作；routing fusion 合并 softmax/top-k/取权重及可选归一化，reduction fusion 合并专家加权与归约。CUDA/Vulkan 已有 top-k MoE fusion，不重复列为新增。

**边界**：这不是单 kernel 移植，需要 fusion matcher、alloc-deps/liveness、多输出及 alias 检查。不能漏 `098ce526b`，也不能跨越 LES A2 phase-B CPU 触发节点。不能以提升融合率为由改变真实 router 结果或 shared 直写。

**最小验收**：Metal 真机 top-k=8、归一化/scale、别名分配、融合开关；packed/原始同后端 oracle；A2 开/关分开验收；专家 backend 上传仍为零。紧凑 tile 另外覆盖重复 IDs 和批宽边界。当前缺真机，保持待验，不用 CPU/Vulkan 代替。

### O4：CPU x86 tiled K/IQ-quant MUL_MAT_ID

**上游参考**：[d834d44e6](https://github.com/ggml-org/llama.cpp/commit/d834d44e6)，`ggml/src/ggml-cpu/tiled/`、`ggml-cpu.c`、`tests/test-tiled-mulmat.cpp`。

**机制**：从 `src0->data + expert * nb02` 读取原量化布局，聚集当前专家的 routed rows，计算时分块解包。无需把整个 bank 永久重排，因此比直接接 CPU_REPACK 更贴合 LES 动态填槽。

**限制**：主要是 x86 AVX 系列；默认每专家至少 8 行，非整个 batch 至少 8 行；不应宣传为单 token miss 或 ARM 优化。scratch 约 `512 KiB × 线程数 + 对齐`。拒绝已有 repack `extra` 的权重。

**集成**：workspace 规划、type dispatch、CMake 和算子测试一起核对。上游 `tiled.cpp:615–639` 有类型分支提前返回、后续 src1 检查可达性问题，移植时需要确认调用方约束，不能因来自上游就省略契约审查。

**最小验收**：Q4_K/Q6_K 及实际使用 IQ 类型；每专家 rows=1/7/8/32；同槽先后装两组不同专家，与非 tiled/reference 路径对照。记录填槽、FFN、scratch 和端到端耗时。

## 4. 新增调度方向

### S1：跳过未执行分支的输入复制

**参考**：[806b3c2be](https://github.com/ggml-org/llama.cpp/commit/806b3c2be)，上游 `ggml/src/ggml-backend.cpp:1809–1817,2229–2247,2400–2411`。

omni 已有 `ggml_build_forward_select`，但 scheduler 没有 `hv_tensor_requires_copy`。上游让未选分支继续参与分配，仅对 COMPUTE 节点实际读取的输入做复制。它可能减少无意义上传及 LES copy callback，但效果取决于图中是否有这类输入。

该标志按源 tensor 保守累计，不是任意分支的精确消除；不能连分配依赖一起删除。最小测试为跨 backend 双分支、轮换选择、共享输入及 reset，核对上传/callback 次数与输出。此文件正处于 LES 同步边界工作范围，应由同一集成负责人处理。

### S2：复用 scheduler 与 graph-result

**参考**：[3e86ce2c2](https://github.com/ggml-org/llama.cpp/commit/3e86ce2c2)、[cb44fdb18](https://github.com/ggml-org/llama.cpp/commit/cb44fdb18)。

omni `src/llama-context.cpp:445–474` 在实际 reserve 时重建 graph-result 和 scheduler。上游在配置兼容、容量足够时保留 scheduler/compute buffers，并复用 graph-result arena。收益是减少 reserve/重建抖动，不是保证每 token 加速。

只提取复用条件，不整体替换不同的 graph-result 组织。保留 LES callback；park/unpark、bank 重建、配置/容量变化需失效旧绑定。验收重复同配置 reserve、扩容、并行参数变化、阶段切换与 park/unpark，记录重分配次数、峰值和数值。若 profile 表明 reserve 很少，降低优先级。

## 5. 可独立处理的小修复

| ID | 上游与当前位置 | 内容及验证 |
|---|---|---|
| R1 | [ab4a77b9a](https://github.com/ggml-org/llama.cpp/commit/ab4a77b9a)；omni `ggml/src/ggml-cpu/simd-gemm.h:12` | 将 ARM NEON 检测从 `__ARM_NEON__` 改为 `__ARM_NEON`，使适用编译器选择 4×4 tile。实际影响 CPU tiled attention，非专家量化 matmul。确认编译器宏和 kernel 路径，再测 ARM 真机 FA 正常/尾部几何。 |
| R2 | [7f24aacde](https://github.com/ggml-org/llama.cpp/commit/7f24aacde)；omni `ggml/src/ggml-cpu/ops.cpp:8344–8442` | one-chunk FA 的 F16 V 改用 F32 running accumulator，并带 SIMD helper。首先是数值稳健性，不承诺提速。构造可能使未归一化 F16 累加溢出的输入，检查 finite/F32 oracle；强制覆盖 one-chunk，不能只测本就 F32 累加的 tiled 路径。 |
| R3 | [10d4846a5](https://github.com/ggml-org/llama.cpp/commit/10d4846a5)；omni `ggml/src/ggml-cuda/topk-moe.cu:148` | 加 bias 后重新处理 selection NaN。用受控异常输入核对选择与权重；这是边界修复，没有证据表明当前 MiniCPM 已触发。 |

## 6. 条件性性能候选

### C1：few-row quantized matmul

参考 CUDA `0fb8f1577`，Metal `a3a1c4747`、`988190680`。CUDA 增加少量行的 MMQ 路径；Metal 在已有 BS 2–8 `mul_mv_ext` 之外增加 MMA 路径及类型扩展。

适合调查 DSpark 小批验证、dense projection/FFN；不是现成的专家 `MUL_MAT_ID` 加速。需核对架构、类型、K/对齐和 fallback。测 rows=1/2/4/8/16/24/32，最终看 accepted tokens/s 或多模态阶段时延，不只看 kernel。不要把上游特定 Apple GPU 的阈值直接外推其他芯片。

### C2：Vulkan Flash Attention 的 KV 一次转换/重排

参考 `dc72703fc`（Q8_0 KV）、`733f57a99`（strided F16/BF16）、`3b83d12b0`（合并 K+V scratch 上限）、`d2b3eb9b6`（CM2 与 stride 修正）。固定上游入口在 `ggml-vulkan.cpp:8517–8560,8884` 附近。

多个 query 重复消费 KV 时，一次转换可能减少反复解量化和跨 stride 读取。长上下文/多模态 prefill 是候选，不外推单 token decode；保留设备排除、复用阈值与总 scratch fallback。测试 quant/raw KV、GQA、非对齐 head、长短 query，记录总显存峰值、TTFT、pp 和 decode 无退化。

### C3：Hexagon IM2COL / patch embedding

参考 `355303eda`、DMA ring 修复 `d4d82d67f`、HVX 分块 `2e0d86beb`。omni MiniCPM-V 图由 `tools/mtmd/models/minicpmv.cpp:461` 进入 `clip.cpp:526` 的 `ggml_conv_2d()`，实际展开为 IM2COL + MUL_MAT。omni Hexagon 尚无对应 IM2COL 支持。

这是可达的视觉编码优化，不是 LES 专家缓存。上游已重构 DMA/kparams/work queue，不能单拣算子文件，尤其不可漏 DMA 队列满时回收修复。HTP 真机核对实际 MiniCPM patch geometry、`IC*KH>256` 回归、CPU 中间输出、实际 placement、视觉 TTFT 和峰值。

### C4：ARM SVE-256 repack 内核

参考 Q8 GEMV `369dcf309`、Q6_K GEMM `ac8e1694d`。需真实 SVE-256 设备，Q6_K 另受 MATMUL_INT8 条件限制。omni 已有相关量化类型及其他 ARM 路径，不是新增格式支持。

当前 CPU LES 默认 bank 不走 CPU_REPACK；必须先解决第 8 节的动态重排契约。KleidiAI 更多 dense 内核也不等于已有 MoE 支持：固定上游 `kleidiai.cpp:1793–1804` 仍只接受二维 MUL_MAT。

## 7. 提供给现有 LES 计划的实现参考

这些条目补充已有 T2/T3/T5，不重新建立一套缓存方案。

### P1：prefill 只读复用 decode 热集

参考 `d6cf9acb2`，上游 `src/llama-moe-cache.cpp:691–764`。大批次走临时计算布局，已有 device 命中可 D2D 复制，但不更新 decode 驻留热集；连续范围可合并复制。

LES 必须把上游按真实 expert 索引的大临时 tensor/完整 host miss 来源，改成自身有界 prefill slots。保护 D2D 源至复制完成。验收 decode 建热集→大批次→decode，分别核对读盘/H2D/D2D 字节、热集保留、预算与数值。

### P2：下一 CPU split 的输入预取

参考 `e95426f16`，必须同时核对 `3bce8bd50`。这是 scheduler 激活读回预取，不是 LES 专家 lookahead：提前读取更早 GPU split 已产生的输入，使后续 CPU split 不必等待当前整段 GPU 工作。

只预取合适的 COMPUTE 结果，排除之后才由 copy callback 写入的 bank。测试 GPU A→GPU B/CPU C→合并；用时间线证明 B/C 重叠，再测 C 依赖 B、callback 改写权重、view alias、copy 轮转及无事件 fallback。仅建两条分支不等于有重叠。

### P3：CPU 负 ID 语义

参考 `701fd25b3`。`MUL_MAT_ID(-1)` 清零对应输出并跳过计算；`ADD_ID(-1)` 保留输入、不加 bias。不得把任意负数都当 -1。

通用 CPU、repack、平台路径一起核对；LES CPU 分担使用 host-slot IDs，不用真实 expert ID 寻址有界 bank。覆盖全部/部分跳过、复用输出不残留、多 token、容量小于专家总数，以及 bias/scale 的真实 ID 契约。

### P4：批量 tensor I/O

参考 `418f905a4` 及 Vulkan 配套 `055c3c26e`，测试参考 `tests/test-backend-tensor.cpp`。按 buffer type 分组，优先批量 hook，退化为二维或逐项复制。

这是同步批量 API，不代替 T2.3 的事件和资源生存期。omni LES 已合并连续 slots，只有提交开销显著时才值得增加分散批量传输。先验 hook/fallback、跨 buft、非连续和零项；不得给 Metal shared 直写平添复制。

## 8. 跨候选的硬约束与移植风险

1. **真实 ID 与 slot ID 分离。** 调研基线 `src/llama-graph.cpp:1645–1648` 生成物理 IDs，部分 bias（`:1729,1753,1774,1861`）及 LoRA（`:1133–1156`）仍使用同一 ID。已知契约缺口应交给原 LES 计划收口，不在独立算子分支偷偷改图。新增融合不得扩大未经验证的支持范围；用非恒等 slot 映射的 bias/LoRA fixture 验证。
2. **持久 device bank 不等于记住 scheduler dst。** 调研基线 `src/llama-moe-storage.cpp:87–134` 的 dst 仍是 scheduler 工作副本。未来 cache 的生命周期必须独立于 copy 轮转/重新分配；scheduler 复用不能成为正确性的隐含前提。
3. **动态填槽不兼容未经设计的静态 repack。** omni `ggml/src/ggml-cpu/repack.cpp:4733–4743` 及固定上游对应接口都要求 offset=0、整 tensor 写入；LES 则通过 preadv 直接覆盖 slot。Hexagon 上游的首次 compute 重排/已 tiled 标记也没有提供自动的逐槽失效保证。必须明确选择按槽重排、独立原始/计算 bank 或禁用该优化；计入额外内存，覆盖同地址反复换专家。
4. **不照搬完整 host 专家 fallback。** 上游 CPU-share/大批次策略部分依赖完整 host weights，LES 没有这份数据。必须在有界 host slots 上实现并保护 CPU 消费期。
5. **总预算包含新 scratch。** CPU tiled workspace、GPU tile descriptors、FA 转换区、后端对齐和在飞 staging 都可能抵消 cache 容量收益，不能只报 bank 大小。
6. **新格式、新内核数量不是性能证据。** 不为追齐 FP8/NVFP4、设备 ISA 或大范围 backend 更新而扩大范围。无目标数据/设备、无实际 dispatch 或无端到端收益时保留 fallback。

## 9. 执行与验收约定

- 本文只记录方向；实现、提交及发布另行授权。不要覆盖另一个会话的工作区，也不要将其未提交变更复制到此 worktree。
- 独立 backend 修复可在隔离分支/构建树验证；scheduler/context/graph/storage 修改由同一集成负责人串行处理。集成前重新核对实际源码版本和原计划进度。
- 优先使用项目现有 `test-backend-ops`、fusion 测试及 LES tiny/model fixtures；仅补真实行为盲区，不先搭新框架。
- 测试工具参数以目标版本帮助和实际实现为准。必须确认目标 backend、case 数量和实际 kernel/fusion 路径，不能以跳过或 fallback 的成功退出冒充优化验收。
- Nix 环境使用项目 `.#default` / `.#vulkan` / `.#cuda` devShell，独立 build tree；Metal/ARM/HTP 用真实目标设备。旧构建产物不代表新源码可运行。
- 数值先行：同权重 oracle、真实 ID/slot ID、view/stride、边界与失败恢复；性能测试避免调试 callback 额外同步改变时序。
- 性能对照固定模型、量化、预算、输入、路由工作量和设备；分冷/热、prefill/decode，至少重复 5 次并报告离散程度。记录读盘/H2D/D2D、kernel/同步、峰值内存和端到端时延，不只看命中率或单算子时间。
- 每项最终状态至少区分：源码已集成、构建通过、算子正确、LES 集成正确、目标硬件收益成立。不同 backend、数据源及功能组合单独记录。

## 10. 已有证据与仍需确认的内容

已完成：固定两树源码和上游提交 diff 的比较，核实重点候选是否已有、图可达关系及明显集成前提；本文是该调研的单一记录。

未完成：任何候选的移植、构建、运行、性能测量；完整后端更新穷举；远端最新状态；另一会话后续修改后的重新审查。开始实施前还需按选中候选核对前置/后续修复的完整提交依赖，本文短 SHA 是导航入口，不是可盲目 cherry-pick 的补丁队列。

# 可参考上游的算子修补与增强调研

## 1. 阅读范围与证据口径

日期：2026-11-15。状态：**调研记录，尚未实施、构建或在 omni 上验收**。

后续执行状态与分批安排见 [上游算子修复分批记录](upstream-operator-batches.zh.md)。本文保留原始调研口径；候选是否已移植、实测及其版本差异，以执行记录为准。

- omni 比较基线：本 worktree 的 `master@350a835d5f8215145aa210a7df0f67e8877b5d6e`。
- 固定上游：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`。
- 本地上游源码：`/home/sakura/codespace/llama.cpp`。本轮未 fetch，不声称覆盖远端最新版本。
- 当前工作分支：`fix/upstream-operators`，worktree 为 `/home/sakura/workspace/llm/llama.cpp-omni-operators`，从本地 `master` 新建；文档由 `docs/upstream-les-enhancements` worktree 移入。没有合入正在另一会话推进的 LES 分支。
- CPU、CUDA/HIP、Vulkan、Metal 完成目录/历史筛查及重点源码比较；OpenCL/WebGPU/Hexagon 是针对性补查。其余后端不宣称逐算子覆盖。

**本文只收录有具体上游提交和代码依据的候选。** 本地额外发现、没有现成上游补丁的疑点，以及需要自行设计的 LES 适配，单独放在 [扩展调研](upstream-operator-extensions.zh.md)，不混入本清单。

此前 [LES 增强调研](../les/upstream-enhancements.zh.md) 仍负责缓存、传输与 LES 整体方向；本文负责算子级分类和从小到大的同步顺序。其基线为 LES 分支 `714d800`，不能和本文的 master 基线混为一谈。

### “上游已有”不等于“omni 已验证”

下面候选均有上游提交依据，并在固定上游历史中。它们不是本次原创补丁，但存在三层不同证据：

1. **源码依据**：核对上游机制与 omni 差异，证明值得参考，不证明已在当前模型触发。
2. **上游回归线索**：部分提交同时修改了 `tests/test-backend-ops.cpp`，可复用其边界案例；不等于本轮审计了其 CI 或目标硬件运行日志。
3. **omni 验收**：全部未完成。移植后仍须先复现/验证对应行为，再确认适用模型与平台。

标记 **T** 的提交已核对包含对应测试文件变更；未标 T 不代表上游没测，只表示本轮没有同等的提交内测试证据。难度为静态依赖估计，不是工期承诺。短 SHA 是源码导航入口，**不是已核清全部依赖的 cherry-pick 队列**。

查看任一参考：

```bash
git -C /home/sakura/codespace/llama.cpp show <短SHA>
# 在线： https://github.com/ggml-org/llama.cpp/commit/<短SHA>
```

## 2. 排序规则与第一批建议

排序优先考虑**改变的契约和调用范围**，其次考虑实现/验证难度；不按潜在加速比或 diff 行数排序。

| 级别 | 影响面（窄 → 广） | 常见难度 |
|---|---|---|
| A | 单后端已有算子的局部索引、边界、数值或选择条件，不新增公共合同 | 低；特殊硬件/大尺寸验证可能升为中 |
| B | 已有算子的多执行路径、共同 helper、多个后端或 scratch/dispatch 配套 | 中；复杂 GPU 同步为高 |
| C | 核心 opcode 已存在，为特定 backend 增加 kernel/类型/layout 支持 | 中；需对应硬件 |
| D | 重组计算：tile 工作表、融合、repack、额外 scratch 或设备专用快路 | 中高至高 |
| E | 公共 API、类型、op_params、状态布局或模型图跨后端变化 | 高 |

同级表内先列较容易项；ID 用于稳定引用，不代表提交顺序。级别不等于风险严重程度：一个越界修复可能影响很窄，但仍然重要。缺硬件不改变其源码影响级别，只令验收保持待办。

**建议先选带 T、影响小且有可用设备的切片：**

1. CPU `CONCAT` 量化 block 寻址：A04，`ef2d77011`（T）。
2. 有 CUDA 时做 `CPY` 非连续目标保护：A02，`134b2bb75`（T）。
3. CPU/CUDA `CLAMP` view：B01，`65840ed53`（T）；虽涉及两后端，行为边界清楚。
4. Vulkan matmul strided view：B06，`526247161`（T）；覆盖四条 MM/MMV/ID 路径，不能当一行 stride 修复。
5. Metal 真机可用时选择 A06/A07；不可用则跳过，不用其他后端替代验收。

NEON 宏等代码更小，但需要目标 ARM dispatch 证据；不要只按行数将其排在可复用上游回归的修复之前。第一批不引入融合、新格式、precision API 或 SSM/GDN 状态升级。

## 3. A 类：局部修补（最窄）

以下位置相对 omni 根目录；行号用于定位比较基线，实施时以符号和最新源码为准。

| ID / 分类 | 算子与后端、源码入口 | 上游依据与内容 | 难度 / 最小验证 |
|---|---|---|---|
| A01 性能选择 | CPU ARM attention tile；`ggml/src/ggml-cpu/simd-gemm.h:12` | `ab4a77b9a`：NEON 宏改为 `__ARM_NEON`，选择4×4 tile；不是专家量化 matmul | 低；检查编译器宏、真实 ARM `FLASH_ATTN_EXT` 路径及尾部几何 |
| A02 正确性 | CUDA `CPY`；`ggml-cuda/cpy.cu:415` | `134b2bb75`（T）：转置快路要求目标连续，避免破坏非连续 dst | 低；转置源→非连续目标，核对目标及周围哨兵 |
| A03 正确性 | Metal `CPY`；`ggml-metal/ggml-metal-ops.cpp:516` 与 shader | `36ee3caf7`：按 dst `nb[0]` 寻址 | 低；dim0 非连续目标及原连续路径 |
| A04 正确性 | CPU `CONCAT`；`ggml-cpu/ops.cpp` concat 实现 | `ef2d77011`（T）：量化数据按 block 而非普通元素寻址 | 低；量化类型、concat 各维、块边界；保留独立预期 |
| A05 正确性 | CUDA MoE selection；`ggml-cuda/topk-moe.cu:148` | `10d4846a5`：加 bias 后重新处理 selection NaN | 低；受控 NaN bias、正常路由，分别核对 IDs/weights |
| A06 正确性 | Metal `TRI`；`ggml-metal.metal` 对应 kernel | `4b46d9653`（T）：使用选择而非乘比较结果，避免0×NaN/Inf | 低；被遮蔽区域含 NaN/Inf，核对未遮蔽值 |
| A07 正确性 | Metal `SOLVE_TRI`；`ggml-metal.metal` 对应 kernel | `4180722ee`（T）：真正条件化尾部读取，避免越界 | 低；N非32倍数、Metal shader validation、解的残差 |
| A08 正确性 | Vulkan NVFP4 matmul；量化 shader 的 scale LUT | `37b0f9c36`：UE4M3 scale 查表容量修复 | 低；覆盖高编码值及实际 NVFP4 kernel，非其他量化替代 |
| A09 正确性 | CUDA IQ4_NL 转换；`ggml-cuda/convert.cu:437` | `f872b5911`：短行解量化读取边界 | 低；32/64/96元素及 memcheck |
| A10 正确性 | CPU `RMS_NORM_BACK` / `SOFT_MAX_BACK`；`ggml-cpu/ops.cpp` | `fd3271e0b`、`dd4c286f3`：防止 dst alias 输入时提前覆盖仍需读取的值 | 低；dst分别alias各输入，解析式/有限差分。主要是反向，不当作推理提速 |
| A11 正确性 | CUDA `ARGSORT`；`ggml-cuda/argsort.cu:74–133` | `b23701f77`：CUB keys 输入输出不能错误共用 | 中；CUB路径、长行、不同排序方向及源保持 |
| A12 正确性 | Metal matvec/MM；`ggml-metal-ops.cpp:2034`、shader | `9183a552e`、`9b8446b91`（后者T）：偏移乘法提前提升为64位 | 中；大偏移/输出、普通尺寸回归；大内存目标不可用时不宣称已验收 |

注：表中 backend 源码短路径均在 `ggml/src/` 下。上游 Metal 已拆分为 `kernels/*.metal`，omni 仍有单文件 shader；按对应 kernel 移植，不因路径不同覆盖整个后端。

## 4. B 类：已有算子的配套修补与兼容增强

| ID / 分类 | 算子与后端 | 上游参考与机制 | 难度 / 验证边界 |
|---|---|---|---|
| B01 正确性 | CPU/CUDA `CLAMP` | `65840ed53`（T）：按dims 1–3 stride读取view，CUDA supports_op配套 | 低；3D/4D view、in-place及行连续限制；CPU reference也需修，不能只互相比 |
| B02 正确性 | CPU/Metal `CONV_TRANSPOSE_2D` | `a43c3986b`（T）：多batch、nb3与scratch | 中；batch>1与单batch，stride/padding，独立卷积预期 |
| B03 正确性/能力 | CPU/Vulkan `NORM`/`L2_NORM`；Metal `GROUP_NORM` | `72a926917`、`80d2c0c2c`、`a194a75b7`：元素stride、batch和整SIMD线程数 | 中；非连续输入、ne03>1、尾行；host参数及shader一起改 |
| B04 正确性 | CPU `FLASH_ATTN_EXT` | `7f24aacde`：F16 V用F32累加；`6f767fe96`（T）：非向量倍数head及softcap后的padding处理 | 中；one-chunk/tiled分开测，极值、奇数head、KV尾块、softcap |
| B05 正确性 | Vulkan `FLASH_ATTN_EXT` | `ac4105d68`、`806eee984`、`6c59c4007`：rowmax偏移、scratch复用标记、量化尾行 | 中；大logit、非整tile，以及matmul→FA/softmax→再次matmul的数据链 |
| B06 正确性 | Vulkan `MUL_MAT`/`MUL_MAT_ID` | `526247161`（T）：真实batch/expert stride及精确descriptor范围 | 中；MM/MMV/ID单/多token、padded nb2、量化尾tile；不能简单将绑定范围扩大到整个buffer |
| B07 正确性 | CUDA/HIP MMF/MMID | `73a43d1f6`、`b68449857`、`9145ddb95`：shared-memory同步、真实warp size、gfx11/12 BF16 WMMA参数 | 中；F16/BF16/F32、尾tile、wave32/64及目标HIP设备 |
| B08 正确性 | CUDA量化 matmul | `8fc037672`：Q8_1 correction sum使用量化整数和×scale | 中；MMV/MMQ、非对称量化与尾块；确认配套的quantize实现 |
| B11 正确性/dispatch | CUDA/HIP大grid；Vulkan norm大dispatch | `c35b66744`、`2d8a32502`、`fda186661`、`5e4878e97`、`3837bf0bc` | 中；PAD/reflect/norm的grid拆分与循环。按实际设备上限构造用例，保留小尺寸回归 |
| B12 正确性/能力 | Vulkan unary/GLU；CPU/CUDA通用view | `1a7718b4c`、`08246a28f`、`0878f7c30`、`de5122d87`：分别处理输入stride，SUM/SCALE等view支持 | 中；分离GLU输入不同stride、broadcast、transpose、in-place；若放宽核心断言，升级到E类处理 |
| B13 公共helper | CPU dot/转换及激活 | `c16c35b81` SVE尾部累加；`284153e06` AVX512-FP16用F32累加；`119f68d7d` NEON转换；`479a52779` quick GELU不绕F16 LUT | 中；helper消费者较广，按选中ISA/激活单独切片，不能一次打包全CPU。F16C需核对后续 `685f70894`，不照搬被回退的中间方案 |
| B09 正确性 | Metal `FLASH_ATTN_EXT` | `fb34fc262`、`e01c9f187`、`1f8fa5231`、`a043d38a6`：mask/output边界、head配对和threadgroup memory限制 | 中至高；非整块mask、短query、head配对、大head量化KV；部分修复依赖新增预解量化路径 |
| B10 正确性 | CUDA `FLASH_ATTN_EXT`、norm/softmax同步 | `b74f590ea`、`1bb2b9fcb`、`99db719bd`、`0eca4d490`、`9bd4c09ea`；另核对PDL配套 `9e58d4d69` | 高；共享内存WAR、全块barrier、mask宽度；需racecheck/synccheck，不能只测普通输出 |

定位入口：CPU `ggml-cpu/ops.cpp`、`vec.cpp`、`vec.h`；CUDA `fattn-*`、`norm.cu`、`softmax.cu`、`quantize.cu`；Vulkan `ggml-vulkan.cpp`及对应shader；Metal `ggml-metal-ops.cpp`及对应kernel。每个参考提交都给出更精确的受影响文件和测试。

## 5. C 类：已有核心算子的 backend/类型能力补齐

此类不新增核心 opcode，但仍须配套 dispatch、supports_op、shader生成/注册、buffer约束。新支持的dtype不等于满足任意新precision要求。

| ID | 算子与后端 | 上游参考 | 难度 / 场景与验收 |
|---|---|---|---|
| C01 | CUDA/Vulkan `POOL_1D` | `29ea9412a`、`876a43211` | 中；omni `tools/omni/audition.cpp:698`实际使用平均池化。测AVG/MAX、边界、实际GPU placement |
| C02 | Vulkan/WebGPU `PAD_REFLECT_1D` | `3f545becc`、`af6b5243d` | 中；`token2wav-impl.cpp:5886,6339`可达；测左右边界、长短音频 |
| C03 | Metal `PAD` 左padding/circular | `4364bf723` | 中；VoxCPM2有现成left-pad workaround。先验kernel，再在独立任务中决定是否删workaround |
| C04 | CPU/Metal `SET_ROWS` F16；CPU/CUDA BF16 RMS | `68a521b59`、`7cbd61002`、`0512ef1e5`；`31953bcf7` | 中；F16 CPU写入须带 `5cdadfc32` 全线程scratch规划。Metal/Vulkan RMS不能据此宣称BF16支持 |
| C05 | CUDA/Vulkan `CONV_3D`；Metal `CONV_2D_DW` | `53ed051ce`、`c5606364b`、`92b187c97` | 中；核心已有，测F16/F32、stride/dilation、偏移view。是否模型可达另核对 |
| C06 | Metal大kernel `IM2COL`；CUDA transpose1D fallback | `2d9b7c8e9` + `daf6bc9f2`；`72874f559` | 中；1D/2D、KH×KW边界、通道多/短输入。保留omni专有SGEMM反卷积快路 |
| C07 | Vulkan GDN几何、GLA | `d5fb10429`、`5c200e0c8`、`f26efa02a` | 中；GDN增加S_v=16及合法lane/subgroup划分，GLA有head64限制。状态合同变化另归E类 |
| C08 | OpenCL浮点 `MUL_MAT_ID`；WebGPU `GROUP_NORM`等 | `53eb66943`、`cf659f240`；ARANGE `713319e28`、MEAN `553d2c2e7`、COUNT_EQUAL `a691b2345` | 中；需目标GPU；支持判定与实际dispatch一致，不能只比较switch标签 |
| C09 | Vulkan/Metal已有量化类型快路 | Vulkan `b1ff4ca23` IQ4_XS、`bbd488c42` IQ3_S、`a7cd2f0e9`/`8fe90e1fb` TQ；Metal `4a84b0ad1` TQ2_0 | 中；不是新格式。分别覆盖MM/MMV/ID、转换及fallback，按量化类型小批推进 |
| C10 | Vulkan训练/反向算子 | `92e854ab8`、`f955e394b`、`bf9421646`：GET_ROWS_BACK、OUT_PROD、交叉熵前后向 | 中；重复索引累加及CPU对照；当前推理优先级低 |

**可达性区别**：omni `tools/omni/vision.cpp:865` 的 `ggml_conv_2d()` 展开为IM2COL+MUL_MAT，并不自动进入direct CONV_2D。新增direct kernel后，不据此宣称已有视觉图加速。

## 6. D 类：性能组织变化，后于局部修补

| ID | 算子族/后端 | 上游机制与参考 | 难度 / 关键配套 |
|---|---|---|---|
| D01 | CPU x86 `MUL_MAT`/`MUL_MAT_ID` | `d834d44e6` 原布局tiled K/IQ-quant；`3dbcb7fca` AVX2归约优化 | 中高；workspace/CMake/类型dispatch与test-tiled-mulmat；每专家≥8行门槛，约512KiB/线程scratch，不等同单token加速 |
| D02 | Vulkan/Metal `MUL_MAT_ID` | `90c26fcd4`→`966baae76`；Metal `073b74034`及 `704366d76`：row IDs、紧凑tile、重复ID/9专家 | 中高；host与shader私有描述布局一致。测重复IDs、批宽边界、strided专家、scratch；LES实际bank几何单测 |
| D06 | `TOP_K`选择策略 | Metal `2a74817f9`；CUDA/Metal `57d04820d`；Vulkan `daef7b687` | 中高；radix、大k fallback、ties。CUB及各backend保证不同，不宣称跨后端bitwise一致 |
| D10 | 布局、归约、通用融合 | Vulkan `98d1e92c2` tiled CONT、`ae4893a09` CONCAT、`70c5f2632` RMS；Metal `6d194531e` threadgroup、`df1170213`归约、`b3c9d1b84` Snake融合 | 中高至高；按具体算子拆，profile证明热点；Snake需要图匹配。不把整组作为一次提交 |
| D03 | CUDA `MUL_MAT_ID`组织和MMQ融合 | `69ee2ecc0` linearized layout；`c49edf096`→`f5222f25c` gate/up/GLU | 高；带对应padding/fixup后续修复 `fc9ce6b9d`、`8a1a9b512`、`5d4c8eeb2`。不是单文件kernel更新 |
| D04 | CUDA/Metal少行matmul | `0fb8f1577`；`a3a1c4747`、`988190680` | 高；精确GPU/类型/shape门槛；Metal融合须核对 `7e8324f5f`。测试实际dispatch，不能直接宣传专家MMID收益 |
| D05 | MoE routing/reduce融合 | Metal `5b59b83f4`、`ea40a3a1e`；CUDA `41ef91f7c`、`3466812d1`；Vulkan `50182a53f` | 高；allocator/liveness、多输出、alias。Metal带 `098ce526b`，Vulkan需 `57291f264` alloc-dep合同；后者依赖扩大时按E类集成 |
| D07 | Dense `FLASH_ATTN_EXT`快路 | Vulkan `6e093b80e`、`dc72703fc`、`733f57a99`、`3b83d12b0`、`d2b3eb9b6`、`a963f6595`；Metal `70aff2525`、`926862e57`、`6e66cb5d4`；OpenCL `e1f7593c3` | 高；按backend拆包，KV转换/scratch上限/设备dispatch配套；长短query、GQA、量化KV、总峰值。稀疏/row-list另归E类 |
| D08 | ARM静态repack/KleidiAI、AMX | SVE `369dcf309`、`c02b5c641`、`ac8e1694d`；KAI `45451c366`、`dfeea5200`、`3ecfb150a`；AMX `37a77fb05` | 高；ISA/库版本/packing配套。KAI仅二维dense MUL_MAT，Q6_K路径有重新量化精度影响；LES动态bank适配不属于本条上游现成能力 |
| D09 | 移动端matmul/vision/SSM | OpenCL `4b77dee05`、`c19043f67` 小批验证；Hexagon `355303eda`、`d4d82d67f`、`2e0d86beb` IM2COL，`669204ee2`卷积，`ded522f83` SSM | 高；Hexagon DMA/kparams/work queue配套且SSM涉及新状态合同；必须HTP/Adreno真机，不孤立摘kernel |

## 7. E 类：公共合同与模型能力（最广）

| ID | 内容 | 上游参考与范围 | 难度 / 验收 |
|---|---|---|---|
| E01 | precision API | `5a6caa05f`、`494d0f47b`，区分src与acc precision，图、reference、supports_op和kernel共同遵守 | 高；各类型/精度允许与拒绝矩阵，现有模型设置兼容；不是单backend更新 |
| E02 | `MUL_MAT_ID(-1)` / `ADD_ID(-1)` | `701fd25b3`；前者清零并跳过，后者保留输入不加bias | 高；generic/repack/各执行backend一致、全部/部分跳过、旧输出不残留。LES host-slot适配另属扩展问题 |
| E03 | RoPE offset / YaRN truncate | `fe8156f78`、`1fa7f1c48`、`9f0d80cf4`，API与op_params | 高；NEOX/MRoPE/IMRoPE及未旋转区；VISION offset限制，Metal truncate=false支持限制；不得推定全backend可用 |
| E04 | SSM_SCAN rollback、GDN状态与MMA | `1692f9e50`、`e95dae18d`；性能参考 `b62b35098`、`11cd98842`、`62df1815b`、`a2878d30d` | 高；K快照、state维数/顺序、图/缓存/回滚一起变。测试K=1/>1、tokens<K、chunk尾部、多seq和恢复后续算 |
| E05 | FA row-list / sparse hint | `1b6935b22`、`8e93a9773`、Metal `7bb0fc18f`、Vulkan `fc82583e6` | 高；KV行表、hint容量、模型图与scratch。固定上游Vulkan仍不接受FA src[5] row-list，不把API存在等同backend支持 |
| E06 | `COL2IM_1D`、`SWIGLU_CLAMP` | `26021699b`、CUDA `3a3edc9ac`、Vulkan `ad39ccaa1`、Metal `f36e5c348`；GLU `0190529ec` | 高；新opcode/子类型、构图与CPU/GPU。COL2IM独立对照overlap-add，不直接替换omni音频反卷积快路 |
| E07 | Lightning Indexer、DSV4 HC、TOP_K_MASK | CPU `00f5442cc`、`0dc74e332`、`d454a204e`；GPU有对应实现 | 高；模型专用索引、连接融合和稀疏mask。没有目标模型需求时后置；对照展开图 |
| E08 | Q2_0 / F8_E4M3新格式 | `bec4772f6`、`68d20b884`、CUDA `9e9619d03` | 高；type traits、量化/加载、转换与backend配套；非单算子增强，不为追齐格式扩大第一批范围 |

以上E类是已有上游方向，但不能据“上游验证过”省略omni迁移设计。不同硬件上的具体支持和精度保证须逐项查证。

## 8. 不重复同步与不能覆盖的内容

- omni CUDA `conv-transpose-1d-gemm.*` 是自有音频快路；上游无同名方案。只更新适用fallback，不能整目录替换。
- Metal alloc-only/shared host-addressable/park-unpark 与LES有关，不属于本轮算子同步对象。更新device/dispatch时保留本地扩展。
- 上游移除了 `ADD1`、`DIAG_MASK_INF`、`DIAG_MASK_ZERO`，omni还有对应构造与执行路径；本次不跟随删除。上游能力不是omni严格超集。
- 部分基础kernel已相同或基本相同：CPU pooling/softmax前向/SSM_CONV，多个GPU的RWKV、upscale及基础逐元素等。文件相同仍可能受共享helper变化影响，但不单列无行为差异的“优化”。
- 不把已吸收的CPU simd_gemm scalar-tail修复再次列为待修；不移植已被上游删除的Metal activation amax rescale中间方案。
- 性能重排与新融合应包含对应后续正确性修复；例如Metal reduction alias修复是引入新fusion后的配套，不是omni当前已发生的同一缺陷。

## 9. 验收与交付边界

1. **先从上游现成回归开始**：移植选中提交的行为测试/独立oracle，确认旧实现失败或确实覆盖被修行为；不是先跑整套测试求覆盖率。
2. CPU reference也可能含同一错误（如CLAMP/CONCAT），用独立预期和原始数据，不能只做有共同缺陷的两条路径互比。
3. 显式指定目标backend并核对实际case数/dispatch；硬件缺失、被skip或fallback成功不算该优化通过。测试CLI以目标版本为准。
4. 复用 `test-backend-ops`、现有fusion/架构/音视频入口；复杂CUDA同步用适用的memcheck/racecheck/synccheck，Vulkan用validation，Metal用shader validation。
5. Nix使用项目devShell，独立build tree；测试绑定源码版本。单算子数值通过不等于模型图、LES动态填槽或整个平台完成。
6. 性能项固定模型、量化、输入、预算和实际工作量，至少5次，记录离散程度、scratch/峰值及端到端结果。不存在测量时不填写加速百分比。
7. 本次只写调研。具体实现、提交与发布另行授权；不触碰另一会话的未提交修改。

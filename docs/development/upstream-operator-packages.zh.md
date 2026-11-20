# 上游算子回移主控路线图与 PR 批次看板

> **唯一权威源（SSOT）**，更新日期：2026-11-20。
>
> 候选稳定 ID（53 组）索引见 [原始调研底表](upstream-operator-survey.zh.md)（只读）；额外本地疑点见 [扩展调研](upstream-operator-extensions.zh.md)；逐批实测证据与 PR 提交格式规范见 [验收目录](operator-acceptance/README.md)。

## 1. 基线快照与本机验证边界

- **开发入口**：`/home/sakura/workspace/llm/omni/llama.cpp-omni-operators`，分支 `fix/upstream-operators`（M3 已落盘提交，精确 HEAD 以 `git log` 为准）。
- **PR #156 快照**：`/home/sakura/workspace/llm/omni/llama.cpp-omni-pr-operators`，分支 `pr/ggml-operator-correctness@a93c067`（7 笔提交，对应 [Draft PR #156](https://github.com/tc-mb/llama.cpp-omni/pull/156)）。
- **omni 基点**：`350a835d5f8215145aa210a7df0f67e8877b5d6e`；**固定只读上游**：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`（本地 `/home/sakura/codespace/llama.cpp`，不 fetch）。
- **进度概览（53 组）**：**10 组完整入 PR + 2 组部分入 PR + 11 组已在本机 dev 验收待入 PR（含 M3）= 23 组已覆盖**；其中 **A/B 类本机可验证项已 100% 在 dev 闭合**，剩余 30 组中绝大多数为硬件/模型暂缓项或可选性能项。

| 维度 | 本机实测配置 | 可直接验证范围 | 不可代验（必须暂缓） |
|---|---|---|---|
| **CPU** | AMD Ryzen 9 9900X（x86_64，SSE4.2/AVX2/F16C/AVX-512F/DQ/BW/VL/BF16/VNNI） | 通用 CPU、x86 AVX2/AVX-512、ASan/UBSan | 无 `avx512_fp16`、无 Intel AMX、无 ARM NEON/SVE、无 RVV |
| **独立 GPU** | NVIDIA RTX 5060（8 GiB，sm_120a，Driver 615.78.08，CUDA 12.9.86，Sanitizer 2026.3） | CUDA (`GGML_CUDA_FA=ON`) + memcheck/racecheck/synccheck；Vulkan（CM2/CM1/generic + validation） | 无 ROCm/HIP 独显、无 Metal/MoltenVK、无 Intel GPU、无 OpenCL/WebGPU/Hexagon |
| **核显** | AMD Radeon 610M（2 CU，Nix 同 pin Mesa 26.2.3 RADV） | Vulkan RADV generic 交叉验证 | 不代表 AMD 独显性能或 ROCm/HIP |
| **本地模型** | `MiniCPM-V-4.7-35B-A3B`（MoE Q4_K_M/mix_q3gud + mmproj）、`MiniCPM5-2B`（Q8_0）、`MiniCPM-OCR-1.0`（Q8_0 + mmproj） | Dense LLM、MoE、视觉 ViT | 无 DeepSeek-V3.2/V4、Qwen 3.8 Flash Next、Mamba/GDN/RWKV6、3D 卷积、Q2_0/FP8 权重 |

### 交付与 PR 拆分原则

1. **优先本机可验证**：优先推进 CPU x86 / NVIDIA CUDA / Vulkan（含 RADV）可直接运行独立 oracle 与动态检测的批次；ROCm、Metal、无本机硬件与需特殊模型的候选统一暂缓并写明原因（见第 4 节）。
2. **粗粒度整批闭环**：以依赖闭合的**功能批次**为单位实现、验证与回退——同一批共享构建与验证脚本（**可整批验证**），组内按依赖顺序保留少量逻辑提交，出错可逆序整体撤回（**出错整批回退**），不再按单个算子反复开验收轮（**减少过细粒度**）。
3. **按主题拆 PR**：
   - **Draft PR #156（`pr/ggml-operator-correctness`）** 仅收口 A/B 类纯正确性修复（追平 P1–P3 后封板评审），不再混入 C 类新能力或 D 类性能重排。
   - **Wave 2（P4、P5）** 与 **Wave 3（P6–P9）** 按主题拆为独立 PR（不携带 dev 过程文档、根目录 `AGENTS.md` 改动、LES 生产代码或 CI 变更）。
4. **PR 复验硬门槛**：在目标 PR worktree 现路径下使用新建构建目录独立复验，详见 [验收规范与 PR 准入](operator-acceptance/README.md#pr-功能批准入与拆分规范)；推送远端、标记 Ready、合并或修改 `master` 均需显式授权。

---

## 2. 已进入 Draft PR #156 的基线（`27fd60a..a93c067`，7 笔提交）

| PR 提交 | 涵盖候选 | 内容摘要 |
|---|---|---|
| `27fd60a` | A02, A04, B01 | CPU 量化 `CONCAT` + CUDA dtype 门控、CUDA `CPY` transpose 连续性、CPU/CUDA `CLAMP` view |
| `530b464` | A05, A09, A11 | CUDA MoE selection NaN、IQ4_NL 短行边界、`ARGSORT` CUB 非 alias 缓冲 |
| `b690242` | B05, B06 | Vulkan matmul 真实 stride/descriptor 范围、FA rowmax/scratch/MMQ shared 边界 |
| `56ca84d` | A08 | Vulkan NVFP4 UE4M3 scale LUT 全编码覆盖 |
| `4889ca0` | B04（部分） | CPU FA F16 V F32 累加与 softcap padding 顺序 |
| `b5f8a64` | A10 | CPU `RMS_NORM_BACK` / `SOFT_MAX_BACK` exact-src alias 保护 |
| `a93c067` | B02（CPU） | CPU `CONV_TRANSPOSE_2D` 多 batch、`nb13` 与匹配 scratch |

---

## 3. PR 交付批次路线图（P1–P9）

文件依赖拓扑（严格无环）：`P1 (ops.cpp)` → `P2 (ops.cpp, ggml-cuda.cu, norm.cu, ggml-vulkan.cpp)` → `P3 (common.cuh, norm.cu)` → `P4 (ggml-cuda.cu, ggml-vulkan.cpp, vulkan-shaders-gen.cpp)` → `P5 (ops.cpp, ggml-cuda.cu, norm.cu, ggml-vulkan.cpp, vulkan-shaders-gen.cpp)`；Wave 3 各性能专题彼此解耦、按需进入。

| 批次 | 目标 PR | 涵盖候选与 dev 来源 | 整批验证门槛 | 整批回退边界 | 当前状态 |
|---|---|---|---|---|---|
| **P1** | **PR #156**（增量批 1） | **CPU 正确性收口**：`B04`收口（`959a23d`）+ `B13` x86 quick GELU（`d9ab965`） | 新建 CPU AVX2/AVX-512 ASan/UBSan：`FLASH_ATTN_EXT` + `test-gelu-quick` | 零 GPU 依赖，可独立逆序回退 | dev 已验收，待提取至 PR #156 |
| **P2** | **PR #156**（增量批 2） | **跨后端视图/步长与大 Dispatch**：`B03`（`bdbb3fb`）+ `B11`（`e094a79`,`c64b5c6`）+ `B12/M1-A`（`4072918`,`23b967c`） | 1 次三后端组合复验：CPU ASan + CUDA + Vulkan validation（`NORM/L2_NORM/GROUP_NORM/RMS_NORM/PAD/SUM/unary/GLU` 含交叠用例） | 依赖 P1（`ops.cpp` 相邻）；是 P3 `norm.cu` 前置，回退 P2 前须先回退 P3 | dev 已验收，待提取至 PR #156 |
| **P3** | **PR #156**（增量批 3，**封板**） | **CUDA 量化与同步正确性**：`B08`（`81502de`）+ `B07`（`51f51e2`）+ `B10/M1-B`（`bba8cdf`） | 1 次 CUDA (`GGML_CUDA_FA=ON`)：`test-cuda-quantize` + FA/MMF/MMID/GROUP_NORM/SOFT_MAX + 全量 `compute-sanitizer` (memcheck/racecheck/synccheck) | 位于 PR #156 栈顶，零耦合整批回退；**合入后 PR #156 封板评审** | dev 已验收，待提取至 PR #156 |
| **P4** | **独立 PR #2**（音频通路） | **音频 GPU 通路与 Fallback**：M2（`b6f505e`：`C01` CUDA/Vk `POOL_1D`、`C02` Vk `PAD_REFLECT_1D`）+ **`C06` CUDA `conv_transpose_1d` 14 行修复（`72874f559`）** | CPU ASan + CUDA memcheck + Vulkan validation：`POOL_1D`/`PAD_REFLECT_1D`/`CONV_TRANSPOSE_1D`（含 `p0`/`d0`/multi-plane fallback，保留 omni 自有 GEMM 快路）+ 落点探针 | 独立 PR，出错整支撤回，不阻塞 PR #156 | M2 已提交；待 dev 补 `C06` 14 行后整批开 PR |
| **P5** | **独立 PR #3**（类型与量化） | **类型与 Vulkan 量化通路**：M3（`ccfc624`：`C04` CPU/CUDA BF16 RMS 前向 + CPU F16 `SET_ROWS`；`C09` Vk IQ3_S/IQ4_XS + TQ1_0/TQ2_0） | CPU ASan + CUDA memcheck/racecheck + Vulkan 四管线模式（CM2/CM1/int-dot/fp）及实际 `ggml_vk_dispatch_pipeline` 证据 | 独立 PR（组内 C04/C09 代码正交，评审需要时可一刀拆分） | dev 已提交验收，待基于 P4 后/master 独立开 PR |
| **P6** | **独立 PR #4**（*可选性能*） | **低风险确定性与 Vulkan 布局/FA KV**：`D06-CUDA`（`57d04820d`）+ `D10-Vk`（`98d1e92c2`/`ae4893a09`/`70c5f2632`）+ `D07-Vk 可验子集`（`6e093b80e`/`dc72703fc`/`3b83d12b0`/`d2b3eb9b6`） | CUDA + Vulkan (RTX 5060/RADV) 数值与 validation + `MiniCPM5-2B`/`MiniCPM-V-4.7` 固定输入 ≥5 次延迟对照 | 独立 PR，无净收益直接放弃 | 待准入评估 |
| **P7** | **独立 PR #5**（*可选性能*） | **Vulkan MoE 紧凑 Prepass**：`D02-Vk`（`90c26fcd4` + `966baae76`） | Vulkan `MUL_MAT_ID`（重复 ID/批宽边界/validation）+ `MiniCPM-V-4.7-35B-A3B` Vulkan MoE 端到端实测 | host/shader 私有布局捆绑，必须整批回退 | 待准入评估 |
| **P8** | **独立 PR #6**（*可选高门槛*） | **CUDA MoE 融合与 MMQ 重构**：`D05-CUDA`（`41ef91f7c`/`3466812d1`）→ `D03`（`69ee2ecc0`/`c49edf096`/`f5222f25c` + 必带 3 fixups `fc9ce6b9d`/`8a1a9b512`/`5d4c8eeb2`）→ `D04-CUDA`（`0fb8f1577`） | 先做 `MiniCPM-V-4.7` profile 确认热点；全量 `compute-sanitizer` + 固定预算端到端实测 | 改动大、高风险；增益不足直接搁置 | 待准入评估 |
| **P9** | **独立 PR #7**（*可选高门槛*） | **CPU x86 Tiled K/IQ-Quant**：`D01`（`d834d44e6` + `3dbcb7fca`，~3050 行） | **前置**：先核清 `X01`（`tiled.cpp` switch 控制流）与 `X06`（~512 KiB/线程 scratch）；实测 `MiniCPM-V-4.7` CPU prefill（每专家 ≥8 tokens）净收益 | 独立 PR，不引入零影响 | 待准入评估 |

---

## 4. 暂缓任务登记表（Deferred Registry）

下列候选与子项因缺目标硬件、需要特殊模型或涉及跨后端公共合同大改，**明确移出日常队列**，直至解除条件满足：

| 暂缓类别 | 候选 ID / 子项 | 暂缓原因 | 解除条件 |
|---|---|---|---|
| **Metal / macOS 平台** | `A03`, `A06`, `A07`, `A12`, `B02-Metal`, `B03-Metal`, `B09`, `C03`, `C04-Metal`, `C05-Metal`, `C06-Metal`, `C09-Metal`, `D02/D04/D05/D06/D07/D10` Metal 子项 | 阶段策略暂缓 Metal；需 Apple Silicon 真机 + Metal Shader Validation（`A12` 还需大内存），不能以 Linux 代验（`C03` 在 VoxCPM2 已有 workaround） | 安排独立 `sakura-mac` M5 验收窗口（不与 LES 任务抢占） |
| **ROCm / HIP 平台** | `B07-HIP`（`b68449857` wave64、`9145ddb95` gfx11/12 BF16 WMMA）、`B11-HIP`、`B12-HIP/MUSA` | 本机无 ROCm/HIP 独立 GPU（仅 2CU 核显），不能用 NVIDIA CUDA 代验；共享 `ggml-cuda` 代码已保留条件分支与旧契约 | 具备 ROCm gfx11/12 独立 GPU 及工具链环境 |
| **无本机硬件的 ISA / 其他 GPU** | `A01`, `B13-SVE/NEON`, `B13-AVX512FP16`（`284153e06`）, `D08`（ARM SVE/KleidiAI/AMX）, `C02-WebGPU`, `C08`（OpenCL/WebGPU）, `D07-Intel/AMD/OCL`（`a963f6595`/`733f57a99`/`e1f7593c3`）, `D09`（OpenCL/Hexagon） | Ryzen 9 9900X **无 `avx512_fp16` / AMX / ARM ISA**；本机无 Intel GPU / OpenCL / WebGPU / Adreno / Hexagon 验收环境；`D08` 静态 repack 亦与 LES 动态填槽冲突（`X05`） | 具备对应目标真机并明确平台交付需求 |
| **不适用项（永久跳过）** | `B13-F16C`（`685f70894`） | 上游仅用于回退中间提交 `#30270` 造成的退化，omni 从未引入 `#30270`，无此缺陷 | 无需回移 |
| **无模型可达 / 需特殊模型 / 非推理** | `C05-CUDA/Vk`（`CONV_3D`）、`C07`（Vk GDN/GLA 4150 行）、`C10`（Vk 训练/反向）、`D06-Vk`（`daef7b687` `k>=1024`）、`E04`（SSM/GDN）、`E05`（稀疏 FA/row-list）、`E06`（`COL2IM_1D`/`SWIGLU_CLAMP`）、`E07`（DSV4 Lightning Indexer/HC/TOP_K_MASK）、`E08`（Q2_0/OCP FP8） | `CONV_3D` 全仓零调用点；本地无 DeepSeek-V3.2/V4、Qwen 3.8 Flash Next、Mamba/GDN/RWKV6 或 Q2_0/FP8 权重模型（omni 音频已有自有 CUDA `conv-transpose-1d-gemm`）；`C10` 属训练非当前推理主线 | 对应模型正式接入 omni 或明确开启训练需求时作为独立 RFC 评估 |
| **公共合同未闭合 / LES 耦合** | `B12-SCALE`（`de5122d87`）、`B10-PDL`（`9e58d4d69`）、`D05-Vk`（`50182a53f`+`57291f264` `add_alloc_dep`）、`E01`（precision API）、`E02`（`MUL_MAT_ID(-1)`）、`E03`（RoPE offset/YaRN truncate） | `SCALE` 放宽后 SYCL/WebGPU 越界未闭合；PDL 大范围开关缺跨 kernel 同步验证手段；`D05-Vk`/`E01`/`E03` 改变核心构图或全后端 API；`E02` 跨 45 文件且与冻结的 **LES T5.1** 强耦合 | 算子与 CI 收尾后，针对全后端合同或 LES T5.1 协同开立独立 RFC |

---

## 5. 可复用本地验证环境速查

统一使用各 worktree 锁定的 `flake.nix` / `flake.lock`（不改 `flake.lock` 与主机环境），使用 `bash -c` 保留 devShell 编译器；每次新建独立构建目录（不复用迁移前旧路径 cache），配置加 `-DGGML_CCACHE=OFF` 避开主机 `sccache` 冲突：

- **CUDA（含 FA 与 Sanitizer）**：`nix develop --no-update-lock-file .#cuda --command bash -c '...'`，配置 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_FA=ON`，运行前设 `LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`；动态检测用 `compute-sanitizer --tool memcheck|racecheck|synccheck --error-exitcode 99`。
- **CPU ASan/UBSan**：复用 `.#cuda` devShell，`-DGGML_CUDA=OFF -DLLAMA_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_ADDRESS=ON`（可配 UBSan）。
- **Vulkan（NVIDIA / RADV + Validation）**：`nix develop --no-update-lock-file .#vulkan --command bash -c '...'`，配置 `-DGGML_VULKAN=ON -DGGML_VULKAN_DEBUG=ON -DGGML_VULKAN_VALIDATE=ON`；强制关闭 CM2 时必须同时设 `GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR=1`；RADV 交叉验证须用项目同 nixpkgs pin 的进程级 Mesa `VK_DRIVER_FILES`，避免主机 Mesa glibc 冲突。

---

## 6. Next Action（全局唯一执行指针）

1. **当前第一步 — 追平并封板 Draft PR #156（P1 → P2 → P3）**：按第 3 节顺序在 `llama.cpp-omni-pr-operators` 从 `a93c067` 提取 P1、P2、P3 纯代码与测试，在现路径新构建目录完成 3 轮整批复验并提交，完成后申请将 PR #156 封板评审。
2. **随后推进独立能力 PR（P4 → P5）**：在 dev 补入 `C06` CUDA `conv_transpose_1d` 14 行修复（`72874f559`）并验收后，分别交付 **P4（音频 GPU 通路 PR）** 与 **P5（类型与 Vulkan 量化 PR）**。
3. **节点决策**：P5 完成后评估切换至 **CI/CD 修复阶段** 或按需准入 **Wave 3（P6–P9）性能专题 PR**。

*本路线图规划与上游差异核对有 AI 辅助。*

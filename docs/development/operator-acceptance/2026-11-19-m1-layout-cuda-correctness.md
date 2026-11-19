# M1 验收：布局一致性与 CUDA 执行正确性（2026-11-19）

本地输入 `c64b5c69bf253504973e2628fb083e52c1caa0f1`；固定来源 `ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`。完整上游 SHA 和包划分见 [M1 规划](../upstream-operator-packages.zh.md)。保持既有 B03/B07/B11、PR 历史及 LES 不变。依赖核对、补丁适配、测试和验收有 AI 辅助。

## M1-A：布局与能力

- **SUM (`0878f7c30`)**：沿用已有 CPU F32 stride 支持。CUDA/Vulkan 各自只接收 F32 紧密分配张量（含 packed permutation），和实际平坦归约一致；明确排除间隙及不支持的 dtype。未移植该提交的 WebGPU 部分。
- **CUDA unary (`08246a28f`)**：只为本地已有 F16/F32 路径新增 stride 入口，src/dst 独立寻址；保留密集 kernel、算式、PDL 执行顺序。未引入该上游的 BF16 前置变化或将 xIELU/其他 opcode 误纳入；HIP/MUSA 保留原 standalone unary 门控。同步约束 unary→MUL fusion 的 dim-0 与输出密集性，不让新增布局进入旧融合 kernel。
- **Vulkan unary/GLU (`1a7718b4c`)**：使用本地既有 128 B unary ABI，普通 unary 独立选 src/dst 密集路径及 offset；保留既有每个算式与 SPIR-V 名称，不带新 unary 功能。xIELU 仍用四参数的密集 ABI。GLU 使用本地 88 B 私有常量，A/B/D 独立步长和 offset、共用逻辑坐标；保留 packed/swapped/split 以及 OAI 参数，不改其他 shader 的 ABI。
- **SCALE (`de5122d87`) 未纳入**：`ggml_scale_impl` 放宽后，SYCL 旧 `supports_op` 会接收非连续行组，但 kernel 平坦寻址；WebGPU 等也未闭合。保持旧构图断言，不以 CPU/CUDA 两端通过代替跨后端合同。

| 验收 | 结果 |
|---|---|
| `SUM` 能力及调度（独立预期） | 旧 CUDA/Vulkan 分别误拒 packed transpose、误接带空洞输入；新 CUDA/Vulkan 各 **4/4**。dense/packed 真正在 GPU 执行，gapped/F16 拒绝后实际由 CPU 执行，四种结果均精确等于 21。原独立 packed 数值用例在两后端 `SUM` 组各 **9/9**。 |
| Vulkan REGLU 不同步长/offset（独立预期） | 旧等 pitch 对照 525/525；不同 pitch 16/525；非零 offset 触发描述符对齐断言（退出 134）。新版本三组均 **525/525**、退出 0。 |
| 维护测试 | 新增 F16/F32 非零 offset 的普通/原地 NEG 与不同 pitch 的 split REGLU，CUDA/Vulkan 各 **6/6**。CPU ASan/UBSan 结合 SUM、相关布局 oracle 共 **69/69**，无报告（沿用既有测试驱动 enum 例外，不排除生产代码）。 |

## M1-B：CUDA 内核正确性

应用 FA MMA barrier、tile scratch 同步、vec KQ 读后同步、mask stride 扩宽和归约 scratch 修复；vec 使用本地已有 `ggml_cuda_syncwarp()` 适配，不整体重取 B07。归约按调用者处理，没把每次 `block_reduce` 都改成无条件同步。未搬 `9e58d4d6` 的跨文件 PDL 开关，也未改 FA 主 launch 或 enrollment。已有 PDL 路径开关与跨 kernel 内存可见性不算完整验收。

| RTX 5060 CUDA 核验 | 基线 → 候选 |
|---|---|
| FA F16 MMA 分歧 barrier | synccheck **704 → 0** 报告，退出 **99 → 0**；同一用例 1/1。 |
| FA F16 MMA 和 vector 共享内存 | `192/128` MMA racecheck **3 → 0 hazards**；`64/64` 单 token vec **1 → 0**。各 1/1，退出 **99 → 0**，确认 vec 实际 `grid=(1,7,64)` 且遍历多轮 KV。 |
| Mask scan、generic tile | trace 命中 `flash_attn_mask_to_KV_max` 与 `flash_attn_tile`，各 1/1，memcheck 0 错误。仅证明实际 scan/tile 路径，不称为超 32-bit 地址的动态红绿。 |
| GROUP_NORM 与 SOFT_MAX scratch | GROUP_NORM 旧数值 0/1 且 racecheck 1 hazard，候选 **1/1、0 hazards**；普通多 warp 和宽行 cooperative softmax 旧 0/2、3 hazards，候选 **2/2、0 hazards**。 |
| PDL 限定检查 | vec+combine 用例在 `GGML_CUDA_PDL=1` 数值 **1/1**；不证明跨 kernel overlap 或无 PDL race。CUDA Graph capture/replay 未验。 |

数值红灯时进程可能优先返回测试的 1，而非 sanitizer 的 99；不把两者混记。FA memcheck 使用 `--report-api-errors explicit` 排除旧/新环境的设备代码重编译提示；记录另存的 `--dump-kernel-launches` 输出只证明分发，不把诊断 launch 数量计入检测错误。

## 组合树与复验

组合回归在同一源码状态运行：CUDA norm/softmax/PAD **460/460**（另 50 条拒绝）；Vulkan unary/GLU、SUM 和 norm（含 B11）**470/470**（另 16 条拒绝），正常退出、无 validation 错误；B07 既有 MMF/MMID **1/1 + 8/8**。拒绝或过滤无命中不计作通过。仅记录本次 B11 交叠回归正常退出，不声称定位了历史偶发堆错误。

使用 Ryzen 9 9900X 和 RTX 5060、driver 615.78.08，项目锁定 `.#cuda` / `.#vulkan`。CUDA 基线和候选同配置 `build-operators-m1/cuda-fa`（sm120a-real，`GGML_CUDA_FA=ON`、`-lineinfo`），Vulkan/CPU 使用当前路径的 `build-operators-b03-vulkan` / `build-operators-b04/asan-avx2`。证据在 `build-operators-m1/evidence/`，红绿产物、`check-sum.sh`、`check-fa-paths.sh`、回归命令及运行库哈希均可查。ROCm/Metal、其他设备、性能和模型端到端不在本次验收范围。PR #156 未更新。

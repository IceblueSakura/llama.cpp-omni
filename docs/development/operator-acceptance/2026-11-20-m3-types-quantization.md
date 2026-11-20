# M3 验收：BF16 RMS、F16 行写入与 Vulkan 量化（2026-11-20）

本地输入 `b6f505e6baa3c54a18fbce3472f6fa773f7e9635`，固定只读参考 `ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`。下列上游提交均在该参考的祖先历史中。上游差异核对、回移适配及验收有 AI 辅助。

| 修复包 | 原始上游提交 | 本地适配 |
|---|---|---|
| C04 BF16 `RMS_NORM` | [`31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`](https://github.com/ggml-org/llama.cpp/commit/31953bcf74ba489a8bbb0c936c5fcf24fcf9485c) | 只启用通用 CPU 与 NVIDIA CUDA 前向。保留本地 CUDA 256/1024 线程、Y/Z 循环、归约同步和连续输出限制；不带 CUDA 向量化、BF16 ADD/MUL/融合或反向。未适配后端的能力门控按原 kernel 对齐。 |
| C04 F16 源 `SET_ROWS` | [`68a521b591edd2f36a456809230d63aa81003dfc`](https://github.com/ggml-org/llama.cpp/commit/68a521b591edd2f36a456809230d63aa81003dfc)、[`7cbd61002d0815d440a48db011360998ca61346b`](https://github.com/ggml-org/llama.cpp/commit/7cbd61002d0815d440a48db011360998ca61346b) 的 CPU 部分，加 [`5cdadfc326163ed1fa66cf609fb1761f7928900c`](https://github.com/ggml-org/llama.cpp/commit/5cdadfc326163ed1fa66cf609fb1761f7928900c) | F16 到 F16 按行原样复制；F16 到有 `from_float` 的类型先用逐线程 F32 scratch。只改变需转换的节点的线程规划，保留 F32 源和 F16 直拷路径。CUDA/Vulkan 等未移植 F16 源的后端拒绝；可回退的图由 CPU 执行。 |
| C09 Vulkan IQ 整数通路 | [`bbd488c42abbcf725e7c713bc184eb0ca4a73986`](https://github.com/ggml-org/llama.cpp/commit/bbd488c42abbcf725e7c713bc184eb0ca4a73986)、[`b1ff4ca23630ac7c5a275405353ddb5c486332c9`](https://github.com/ggml-org/llama.cpp/commit/b1ff4ca23630ac7c5a275405353ddb5c486332c9) | IQ3_S 的 MM/MM-ID，以及 IQ4_XS 的 MM/MM-ID/MMV/MMV-ID。沿用本地 `CREATE_MMQ`、shader 生成和 push constants；IQ3_S 计入 2048 B LUT 并采用 K 类 tile，IQ4_XS 计入 int8 codebook。没有引入上游 IQ4_XS 浮点 loader 重排或 matmul specialization 重构。 |
| C09 Vulkan TQ 通路 | [`a7cd2f0e98ffa4e939276287be1fa76891908d2e`](https://github.com/ggml-org/llama.cpp/commit/a7cd2f0e98ffa4e939276287be1fa76891908d2e)、[`8fe90e1fbfc065f17a0b233c9df239423cd24a75`](https://github.com/ggml-org/llama.cpp/commit/8fe90e1fbfc065f17a0b233c9df239423cd24a75) | 复用现有 TQ1_0/TQ2_0 格式，接入浮点 MM/MMV/ID、`GET_ROWS` 与内部反量化到 F16；不启用公共 `CPY`。TQ2_0 反量化按 shader 的每组 `256 * 256` 元素分发，修正固定上游中的 `256 * 64` 注册。Metal/SYCL 的相邻能力调整仅为拒绝未实现的路径。 |

未改变模型图、权重格式或现有 KV 写入路径。BF16 权重并不意味着模型会向 `RMS_NORM` 输入 BF16；F16 缓存目的地并不意味着 `SET_ROWS` 源为 F16。不据算子验收宣称模型端到端收益。GPU 固定分配的目标视图也不一定能回退到 CPU；此处只验证可调度的图。

## 同一候选树的结果

| 实际检查 | 结果 |
|---|---|
| `RMS_NORM,RMS_NORM_BACK,SET_ROWS` | CPU AVX2 ASan/UBSan **287/287**，CUDA **175/175**，Vulkan **180/180**。新增 BF16 RMS CPU 8 条、CUDA 7 条执行；CUDA 拒绝 1 条带行间距的原地用例，Vulkan 拒绝 8 条 BF16。新增 F16 `SET_ROWS` CPU 3 条执行，CUDA/Vulkan 各拒绝这 3 条。拒绝不计通过。 |
| BF16 RMS 独立数值和调度 | 含非零偏移、带间距输入、原地及零行；四条用例各以 host double RMS oracle 验证，BF16 允差 1 ULP。CUDA 三条实际落 GPU，带间距的原地用例落 CPU；Vulkan 四条均落 CPU。F16 `SET_ROWS` 源放在两个 GPU 上时各验证一条 CPU 回退，精确匹配。 |
| CPU F16 `SET_ROWS` scratch 对照 | 4 线程、独立 `MUL` 与写入在同图，含 I32/I64 索引广播和 padded views。模拟旧的单槽 scratch 尺寸 `1280` B 时，ASan 在 F16→F32 行转换报告越界写；实际规划 `4352` B 后 F32、Q8_0 均通过独立数值与 padding 检查。F16 目的地 `0` B scratch，原样保留 NaN payload，三条均通过。这里是缩减工作区的对照，不是旧源码构建。 |
| CUDA BF16 RMS 动态检测及 F32 回归 | memcheck、racecheck 各 **7/7**，分别 **0 errors / 0 hazards**（racecheck 关闭 PDL）。含 `ne[2]`/`ne[3]` 各自跨 65535 的循环。F32 `RMS_NORM,RMS_NORM_MUL_ADD` fusion 开关两组各 **55/55**。 |
| Vulkan C09 数值、实际管线 | 默认 `NV_coopmat2v`、`KHR_coopmat`、普通 shader + 整数点积、普通 shader + 禁用整数点积，四组各 **52/52**。普通整数模式实际发射 IQ3_S/IQ4_XS 的 `matmul_*_q8_1` 和 `matmul_id_subgroup_*_q8_1`，并发射 activation 量化。默认模式发射 IQ4_XS `mul_mat_vec_*_q8_1_f32`，另强制 MMVQ 的 ID 组 **4/4**、命中 `mul_mat_vec_id_iq4_xs_q8_1_f32`。禁用整数点积的对照只走浮点。 |
| Vulkan TQ 专项 | 默认 CM2 和仅关闭 vector decode 的 CM2 分别验证 TQ2_0 向量/标量解码，标量专项 **10/10**。TQ1_0/TQ2_0 的 `GET_ROWS` 和浮点 MM/MMV/ID 实际执行。F16 RHS 探针每模式 **4/4**，逐元素与 CPU 解码加独立 dot oracle 精确匹配；非 CM2 同时命中两个 `dequant_tq*_0` 和独立 F16 matmul，其中 TQ2_0 为 100608 元素、确实发射 2 个工作组。 |
| 关联回归 | C09 CPU ASan/UBSan **52/52**，CUDA **34/34**（18 条 TQ 预期拒绝）；Vulkan 原有 Q4_0/Q2_K/Q6_K 共 **33/33**。最终 Vulkan 运行全部退出 0，未报告 validation 错误。 |

## 复验和边界

Ryzen 9 9900X，RTX 5060 / driver 615.78.08。使用当前路径下 `build-operators-b04/asan-avx2`、`build-operators-m1/cuda-fa`（sm120a，FA 保持开启，验证关闭 CUDA Graph）、`build-operators-b03-vulkan`（再配置 `GGML_VULKAN_DEBUG=ON`、`GGML_VULKAN_VALIDATE=ON`，启用 `VK_LAYER_KHRONOS_validation`）。用仓库 `.#cuda`/`.#vulkan` devShell 定向构建 `test-backend-ops`，保持 `flake.lock` 不动。

```bash
# T 为目标构建目录的 bin/test-backend-ops，在对应 devShell/设备环境中：
"$T" test -b "$backend" -o RMS_NORM,RMS_NORM_BACK,SET_ROWS -j 1
"$T" test -b "$backend" -o MUL_MAT,MUL_MAT_ID,GET_ROWS \
  -p '^(type_a|type)=(iq3_s|iq4_xs|tq1_0|tq2_0),' -j 1
```

Vulkan 普通整数模式设 `GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR=1 GGML_VK_DISABLE_COOPMAT=1`；再设 `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 GGML_VK_DISABLE_MMVQ=1` 为浮点对照。保留 KHR coopmat 时，在这张卡上仅关闭 CM2 **不**足以命中整数 MM/MM-ID。用 `ggml_vk_dispatch_pipeline` 证据判定实际选择，不用 pipeline 创建日志代替。

`build-operators-m3/evidence/` 保存日志、逐任务复验命令/退出状态、`types-probe.cpp`、`tq-f16-probe.cpp` 和 BLAKE3 manifest。未验收 HIP/MUSA、Metal、RISC-V/SpaceMIT、SYCL、OpenCL、CANN、OpenVINO、其他 Vulkan 设备或吞吐量。若提交，按 C04 类型与 C09 的 IQ / TQ 依赖边界拆分；当前仅留在开发 worktree，未提交或更新 Draft PR #156。

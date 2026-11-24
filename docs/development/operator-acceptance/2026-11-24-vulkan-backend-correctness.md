# Vulkan 后端正确性批次

**状态：`fix/operator-sync` 组合树定向验收通过，待本地提交批准。** 输入 HEAD：`88fae5773134352433bd4963c66cf303a1dd1a28`。基线保留已单独验证的 [FP16 `DIAG`/`TRI` 两行](2026-11-24-vulkan-diag-tri-fp16.md)，未包含本批 FA 和 buffer 修补。工作区原有 CUDA 反卷积源码、测试与 P4/P5 记录不纳入本批。

| 上游 `ggml-org/llama.cpp` | 提取范围 |
|---|---|
| `ecafc67145201e59336bf389d0ab1a91b89a8fac` (#32379) | Vulkan `diag.comp`/`tri.comp` 的浮点零构造，保留 omni 的 `float val` 路径。 |
| `301db633702f1d2c107d4feaa4268f07c0b508c5` (#32244) | 在 `ggml-vulkan.cpp` 的标量 FA 选参中保护单行 tile；staging 超预算时关闭；共享内存预算计入 `occupancy_limiter`。不引入 CM1/CM2 调整或 Qualcomm 策略。 |
| `b052049276778d47eb216739c9e1537d764c5585` (#32725) | 将同步非 UMA buffer 填充分为非对齐两端的既有同步写入、对齐中段的 `fillBuffer`；仅适配单文件 Vulkan 布局。 |
| `9c10c94e16ebc4f3ed7c06264a93bf7e019c3f26` (#32727) | Vulkan `clear` 广播单字节到 32 位 `fillBuffer` 图案；不提取 WebGPU 部分。 |

未改公共 API、`ggml_vk_buffer_memset_async`、KV fallback、测试框架或 `flake.lock`。建议本地提交仅包含上述三个生产文件和本批两份验收记录；其余工作区修改不纳入。

## 验收

锁定 `.#vulkan` devShell，两个新构建目录 `build-vk-correctness-base/`、`build-vk-correctness-candidate/`。Release，`GGML_VULKAN=ON`、`GGML_VULKAN_DEBUG=ON`、`GGML_VULKAN_VALIDATE=ON`、`GGML_CCACHE=OFF`，构建 `test-backend-ops`。设备固定 NVIDIA ICD、RTX 5060 / 615.78.08；日志确认 `uma: 0`、共享内存上限 49,152 B，且 instance/device 都加载 Khronos 验证层。用 `test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT -p ...` 分别筛选 `128/128`、`192/128` × KV `113/1024`（`nh=4,nr23=[1,1],nb=1,mask=1,sinks=0,prec=f32,type_K=f16,type_V=f16`），各跑一次默认和强制标量环境；后者使用 [FP16 写零记录](2026-11-24-vulkan-diag-tri-fp16.md)的四个禁用开关。

| 目标 | 基线 → 最终组合树 |
|---|---|
| **标量 FA `hsk=192,hsv=128,nb=1`，F16 K/V、关闭 FP16 算术** | KV=113：基线数值对照通过、退出 0，但报 `VUID-RuntimeSpirv-Workgroup-06530`；KV=1024：单行 tile 被减为 0，命中 `wg_denoms[0] > 0` 断言，退出 134。候选两例 `block_rows=1`、staging 关闭；主机估算 54,048 B（超限）→ 3,888 B，实际标量 Vulkan dispatch 后与 CPU 数值对照通过。 |
| **FA 邻近和默认对照** | 强制标量的 `128/128` 两例仍保留 staging（估算 37,408 B）。最终去掉诊断后，强制标量 **4/4**、默认模式 **4/4**，均有 Vulkan dispatch，均退出 0、无验证错误。默认模式这些几何也选择标量，并不是 CM1/CM2 实测。 |
| **非 UMA buffer** | 基线四例未对齐填充均错且报 `vkCmdFillBuffer` 的 offset/size VUID；非零 `clear` 读回 `5a 00 00 00…`，对齐、零长度和零清除通过。候选 64 字节逐一比较（含范围外 `0x11` 哨兵），默认映射路径 **8/8**，`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` 的不可映射显存路径 **8/8**，均退出 0、无验证错误；`offset=9,size=13` 的首尾分别有 3/2 字节 staging 拷贝。覆盖含 `view_offs` 的 F16 view。 |
| **`DIAG`/`TRI` 组合树回归** | `test-backend-ops test -b Vulkan0 -o DIAG,TRI` F32 **7/7**；FP16/F32 逐元素探针 **10/10**，真实 shader dispatch、退出 0、无验证错误。F16 `TRI` 仍仅后端探针可达，不表示公共 API 新增支持。 |

FA 的 staging 与最终预算用临时选参诊断确认，诊断已移除，之后重建并重跑全部必验项目。最终构建/默认矩阵日志先通过，附加不可映射显存的首轮核对因匹配了错误的日志格式而提前退出；修正匹配器后直接对同一最终二进制重跑，8/8。未运行其他 GPU、AMD UMA、模型端到端、性能或目标 PR 基线验收。

证据保留在两个构建目录的 `evidence/`：基线 runner 与 `buffer-probe.cpp`、基线 `runtime-summary.log`、候选 `runtime-with-probe.log`、候选 `final/runtime-and-build.log` 和 `final/device-staging-summary.log`。探针复用现有 backend API，不引入新测试框架。AI 辅助适配已发表的上游补丁、准备本地探针及整理证据；未提交或发布。

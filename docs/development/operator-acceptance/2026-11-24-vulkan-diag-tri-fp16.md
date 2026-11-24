# Vulkan `DIAG` / `TRI` storage-only FP16 写零

**状态：开发工作树定向验证通过，未提交；不等同目标 PR 基线验收。**

- 上游：[ggml-org/llama.cpp #32379](https://github.com/ggml-org/llama.cpp/pull/32379)，`ecafc67145201e59336bf389d0ab1a91b89a8fac`。
- 输入：`fix/upstream-operators@88fae5773134352433bd4963c66cf303a1dd1a28`。目标参考 `tc-mb/llama.cpp-omni@dc4a6c382357b4b90a3a0f15c3c962b4b5ffe929` 的两个 shader 同样使用整数零。工作树已有音频相关的未提交修改，不是本批的一部分。
- 只在 `ggml/src/ggml-vulkan/vulkan-shaders/{diag,tri}.comp` 的填零分支将 `D_TYPE(0)` 改为 `D_TYPE(0.0)`。保留本树的 `float val` 路径，不导入 #32202 的公共类型扩展或改 host 分发。两行可独立回退。

## 验证

在 `llama.cpp-omni-operators/` 的锁定 `.#vulkan` devShell 使用新构建目录 `build-sync-diag-tri/`。编译配置为 Release、Vulkan debug/validation 开启、ccache 关闭，构建 `test-backend-ops` 成功。未修改 `flake.lock`。

| 检查 | 结果 |
|---|---|
| 基线/候选 SPIR-V：shaderc 2026.1 / glslang 16.4.0，生成器相同的 `-fshader-stage=compute --target-env=vulkan1.2 -O`，`A_TYPE`/`D_TYPE` 为 `float16_t` 或 `float` | 两个 FP16 基线的零分支均为 `OpFConvert %half` 从 `OpConstant %float 0x1p+128`（`+Inf`）转换，候选改为从 `OpConstant %float 0` 转换。FP16 候选需要 `StorageBuffer16BitAccess`，不声明 `Float16` / `Int16` 算术能力；F32 前后各自二进制相同。8 个 SPIR-V 的 `spirv-val --target-env vulkan1.2` 均退出 0；基线错误是常量语义，不是 SPIR-V 格式无效。 |
| RTX 5060 / 615.78.08，`fp16: 0`，validation 实际加载 | 逐元素读回 10/10（两种 dtype，各一次 `DIAG` 与四种 `TRI` 模式，每例 18 个元素）；确认 dispatch 次数 `diag_f16:1, tri_f16:4, diag_f32:1, tri_f32:4`。对照固定输入位置，分别验证正零和保留输入。 |
| `build-sync-diag-tri/bin/test-backend-ops test -b Vulkan0 -o DIAG,TRI` | F32 7/7 实际执行并通过（4 个 `TRI`、3 个 `DIAG`）；不把跳过的 backend 算作实测。读回探针和回归命令退出 0，均无 Vulkan validation error；记录有目标 Vulkan 1.2 的 `WorkgroupSize` best-practices warning。 |

设备对照固定 `VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`，验证层来自 `/run/opengl-driver/share/vulkan/explicit_layer.d`。运行设置 `GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR=1`，编译环境与原始日志、SPIR-V 汇编、读回探针保留在 `build-sync-diag-tri/evidence/`。

**边界：** `ggml_diag` 能用公共 API 构造 F16 图。目标的公共 `ggml_tri` 仍断言 F32；最初的公共构图探针触发该断言。F16 `TRI` 的四次通过仅验证已注册的后端 shader，使用专用探针直接构造 Vulkan 节点，不声称公共 API 已支持 F16。上游扩展来自 [#32202](https://github.com/ggml-org/llama.cpp/pull/32202)，留作独立范围。未跑旧编译器、其他 GPU 或目标 PR worktree；基线红灯来自 SPIR-V 语义对照，未在 GPU 上重跑旧 shader。

本次为已有上游补丁的机械回移；AI 辅助整理验证、编写一次性读回探针和记录。未提交、未发布。

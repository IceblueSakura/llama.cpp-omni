# M2 验收：GPU 一维池化与反射填充（2026-11-19）

本地输入 `bba8cdf34bc068e7d6262e0c01403ad665d4528a`；固定只读上游 `ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`。按[修复包规划](../upstream-operator-packages.zh.md)统一交付。上游摘取、适配、测试及验收有 AI 辅助。

| 原始上游提交 | 本地适配 |
|---|---|
| [`29ea9412a6b5e343dc6736a5a4349158cf78b30c`](https://github.com/ggml-org/llama.cpp/commit/29ea9412a6b5e343dc6736a5a4349158cf78b30c) | 摘取 CUDA `POOL_1D` kernel；dispatch 与能力门控仅启用 NVIDIA。只接收 F32 紧密输入/输出、合法 AVG/MAX 参数及 kernel 的整数索引范围；F16/行带 padding 不由平坦 GPU kernel 处理。 |
| [`876a4321163249c43ca4e986818fab5ab081f282`](https://github.com/ggml-org/llama.cpp/commit/876a4321163249c43ca4e986818fab5ab081f282) | Vulkan `POOL_1D` 同样对齐门控，使用现有分段线性 dispatch，并处理非零 descriptor offset。AVG 先求和、只除以窗口中有效元素数，同 CPU/CUDA；空窗口保持 AVG=0、MAX=`-FLT_MAX`。 |
| [`3f545beccee69d9975f466ec7e45fd9aacd8ba90`](https://github.com/ggml-org/llama.cpp/commit/3f545beccee69d9975f466ec7e45fd9aacd8ba90) | Vulkan `PAD_REFLECT_1D` 一起接入 pipeline、shader 生成、分发与 debug CPU 对照；沿用本地 128 B unary ABI 的独立 fastdiv 字段，保留四维寻址与 descriptor offset，不搬上游 packed-fastdiv ABI。 |

原公共图契约不变。`audition.cpp` 调用 `POOL_1D` 前将 tokens 拷为紧密 F32（`k=5,s=5,p=0`）；`token2wav-impl.cpp` 的反射填充也是现有 F32 调用点。只启用对应算子后端，不改这两个调用方，也不替换 omni 自有 CUDA 反卷积快路。

## 组合验收

| 实际检查 | 结果 |
|---|---|
| CPU AVX2 ASan/UBSan | `POOL_1D,PAD_REFLECT_1D` **59/59**，无报告；原测试驱动 enum 排除仅用于旧测试表，不排除算子。 |
| RTX 5060 CUDA / Vulkan | 同组各 **58/58**，均退出 0；各拒绝 1 条 F16 `POOL_1D`，拒绝不计通过。Vulkan validation 无错误。 |
| CUDA `POOL_1D` memcheck | **54/54**，0 错误，退出 0。 |
| 相邻 `POOL_2D,PAD` | CUDA/Vulkan 各 **159/159**，退出 0。 |
| 同图实际调度（独立数值） | 基线全部落 CPU，CUDA 探针 **5/5**、Vulkan **6/6** 数值正确；候选 CUDA 三条 F32 池化实际落 CUDA，Vulkan 三条 F32 池化及 offset reflect 实际落 Vulkan。两端均确认 F16 与 padded-row 池化拒绝后由 CPU 执行；探针各 **5/5、6/6**，全元素精确匹配。 |

复用现有池化 fixture，补上空窗口、非整步长、音频 `k=5`、F16 拒绝和 266175 输出元素（跨 Vulkan 线性分发边界）。反射 fixture 修正为真正的四维输入，覆盖原 3000 宽长输入、左右不对称最大合法左 pad、无 pad。调度探针使用显式六行不同值以及已知期望，含紧密输入非零 offset、行带空洞和 F16；基线是能力/落点对照，不把旧 CPU 数值正确写成旧 GPU 失败。

## 复验

Ryzen 9 9900X，RTX 5060 / driver 615.78.08；复用当前路径 `build-operators-m1/cuda-fa`（sm120a，FA 保持开启，测试关闭 CUDA Graph）、`build-operators-b03-vulkan` 和 `build-operators-b04/asan-avx2`。CUDA 显式再 configure 以发现新增 `.cu`。用原配置从仓库根目录运行（复跑前保留原日志）：

```bash
nix develop --no-update-lock-file .#cuda --command bash build-operators-m2/evidence/verify.sh cuda
nix develop --no-update-lock-file .#vulkan --command bash build-operators-m2/evidence/verify.sh vulkan
nix develop --no-update-lock-file .#cuda --command bash build-operators-m2/evidence/verify.sh cpu
```

`build-operators-m2/evidence/` 保留运行日志、`audio-placement.cpp`、编译命令（基线 job）、运行库/源文件 manifest 及 job 退出状态。本次不包含 ROCm/Metal/WebGPU，不宣称真实模型最终落点、音频端到端质量或加速比。Draft PR #156 未更新，下一批为 M3。

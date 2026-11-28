# `experiment/les-ssd-streaming` 交接卡与现行方向

> **维护约定（2026-11-28 精简）**：`docs/les/` 仅维护两份活文档——本文件（就地覆盖）与 [RESULTS.zh.md](RESULTS.zh.md)（只增不改实测台账）；上游原版 5 个文件保持不动。提交历史以 `git log --oneline 49dc920..HEAD` 为准，不再手工抄写 commit 表。

---

## 1. 现行目标与硬约束

- **首要目标**：在消费级 Linux x86 + Vulkan 独显（第一基准：Ryzen 9 9900X / RTX 5060 `8151 MiB` / `61.94 GiB` RAM / Btrfs NVMe）上，在明确的主机内存、显存和上下文预算内按需从 NVMe 流式运行 routed experts 超过物理 RAM 的 MoE 模型，完成单序列长文本或图文请求。
- **不可妥协底线**：
  - 保留完整 top-k、原路由与归并语义；数值门槛 `NMSE < 1e-4`（同后端同配置优先逐字节一致）。
  - 单槽保持 `ubatch=1`、中断/读 EIO fail-closed（返回 `-3`/`2` 阻断后续并要求重建 context）；所有实验开关默认关闭（opt-in）。
  - 可选 Host/VRAM 热缓存不能成为运行前提；不把“能跑完”写成“已交互可用”。
  - 先保证 Linux 独显基准站稳；Windows 独显、UMA 核显与 Android 须独立验证，不用本机外推。
- **多档 RAM 预算口径**：`2 GiB`/禁 swap 保留为可重复冷盘压力回归档；实用评测另增 `8/16 GiB` 与不设额外上限的本机实用档，分别区分首跑与重复请求。

<a id="les-next"></a>

## 2. 当前状态与下一步

- **已过线基线（见 [RESULTS.zh.md](RESULTS.zh.md)）**：
  - **D3-1**：MiniCPM-V 4.7 35B LES 在 `2 GiB`/禁 swap 下完成长文本（`645+40`）与真实图文（`775+40`），`41/41` 逐字节一致（TTFT `168.46 / 193.00 s`，prefill 占请求读盘 `93% / 95%`）。
  - **D3-2**：`Qwen3.5-122B-A10B Q4_K_M`（仅 routed experts `65.25 GiB > 61.94 GiB` RAM）显式 CPU 输出头（关 `op_offload`）后，全部 48 层 top-8 专家走 Vulkan 单槽在 `2 GiB`/禁 swap 下完成 `28+40`，`41/41` 逐字节一致。
  - **D3-3（Linux 单槽 `FADV_RANDOM`）**：Linux 单槽默认关闭开关 `LLAMA_MOE_VK_SINGLE_SLOT_FADV_RANDOM=1`（仅关专家 fd 内核预读，保留同层双缓冲）将 Qwen 续步物理读从 `4.39 → 1.87 GiB/tok`（`-57%`）、TTFT 降 `20%`（`~46.3 → ~37.0 s`）、P95 降 `19%`（`1.67–1.68 → 1.36 s`）。
- **下阶段重启顺序**：
  1. **有界 prefill 同专家复用（第一优先）**：以 `LLAMA_MOE_VK_SINGLE_SLOT=1` + `_READ_AHEAD=1` + `_FADV_RANDOM=1` 为冷盘基线，验证一小块 token 完成真实路由后按专家合并装算与多行 `Q8_1` 数值一致性（暂不加载已下载的 Qwen `mmproj-F16.gguf`）。
  2. **多档 RAM 实用评测**：保留 `2 GiB` 压力档，补充 `8/16 GiB` 与不限额本机档（首跑 vs 重复请求），再按需扩 Qwen 长文本及 `mmproj-F16.gguf` 图文。
- **暂停项**：A1 两层 Vulkan L1（**Gate 1 = STOP**）不扩层、不送审；PR 拆分暂不改写本分支，待送审时另开独立 `pr/` 分支。

---

## 3. 实验开关、复验入口与本地资产

- **默认关闭实验开关**：
  - 图文 staged 提速：`LLAMA_MOE_VK_PINNED_STAGING=1`（需要充裕 Host RAM，与单槽互斥）
  - 单槽超内存流式：`LLAMA_MOE_VK_SINGLE_SLOT=1`（需配合 `LLAMA_MOE_DECODE_WINDOW=0`）
  - 同层双 Host 缓冲：额外加 `LLAMA_MOE_VK_SINGLE_SLOT_READ_AHEAD=1`
  - Linux 单槽关内核预读：额外加 `LLAMA_MOE_VK_SINGLE_SLOT_FADV_RANDOM=1`
  - A1 两层 L1 实验：`LLAMA_MOE_VK_L1_SLOTS=32`（可选 `LLAMA_MOE_VK_L1_TRACE=1`）
- **日常回归（`.#vulkan` devShell + 单一构建目录 `build-vulkan/` + [run.sh](../../tools/moe/vulkan-single-slot/run.sh)）**：

```sh
ctest --test-dir build-vulkan -R "^test-les" --output-on-failure
bash tools/moe/vulkan-single-slot/run.sh build-vulkan "$LES_GGUF" pinned-staging
bash tools/moe/vulkan-single-slot/run.sh build-vulkan "$LES_GGUF" decode-l1
bash tools/moe/vulkan-single-slot/run.sh build-vulkan "$NATIVE_GGUF" "$LES_GGUF" smoke-no-eval
bash tools/moe/vulkan-single-slot/run.sh build-vulkan "$LES_GGUF" faults
bash tools/moe/vulkan-single-slot/run.sh build-vulkan "$LES_GGUF" bounded
# Qwen 122B 冷盘短文本复验（不覆盖既有证据；输出头留 CPU）
QWEN_RUN_FADV_RANDOM=1 python3 build-vulkan/qwen122b-d3/run-short.py <new-name> 40 2 single cpu-output
```

- **本地模型与证据留存**：
  - 共享权重 `/home/sakura/workspace/llm/models/Qwen3.5-122B-A10B-GGUF/`（Unsloth `51eab4d59d53f573fb9206cb3ce613f1d0aa392b`）：三片 `Q4_K_M`（`71.28 GiB`）及已校验但尚未加载的 `mmproj-F16.gguf`（`866.6 MiB`）。
  - 忽略证据目录 `build-vulkan/`：`phase1-evidence/`（`7.0 MiB`）、`oversize-evidence/`（`157 MiB`）、`qwen122b-d3/`（`72 GiB`，含已过 783 项校验的派生 `Qwen3.5-122B-A10B-Q4_K_M-LES.gguf`）。

---

## 4. 未来独立 `pr/` 分支拆分参考（不改动本实验分支）

后续向上游 `tc-mb/master` 提 PR 时另开干净 `pr/` 分支提取（`experiment/les-ssd-streaming` 保持完整实验栈不动），已核对的 6 模块边界如下：

- **M2（格式 / 打包 / 量化加固）**：`src/llama-moe-experts.{h,cpp}`、`src/llama-quant.cpp`、`tools/moe/convert_les_gguf.py` 与对应单测——零后端依赖，可作独立前置 PR 或并入 M1。
- **M1（非 UMA LES 运行时闭环）**：`ggml-backend.{h,cpp}` `copy_callback`、`src/llama-moe-storage.{h,cpp}`、非 UMA `src/llama-moe-slot.{h,cpp}`、`#160/#161/#163` 对齐与 `tests/test-les.cpp`——送审前需补 1 项 `#165` 前置 dense 层合成单测。
- **M3（Metal 分配容错与泄漏修复）**：`ggml/src/ggml-metal/*`、`tests/test-metal-alloc.cpp` 与 `tests/test-backend-ops.cpp` 2 行——与非 UMA Vulkan 零代码依赖，开 `pr/` 分支时**务必从 M1 剥离**，留待 Apple Silicon（M5）真机窗口独立提交。
- **M4（Vulkan Pinned Host Staging）**：3 文件 `+64/-8`，堆叠在 M1 之上——送审前剥离其中 2 行尚未入上游的 `LLAMA_MOE_VK_SINGLE_SLOT` 检查。
- **M5（Vulkan 单槽 + 双缓冲 + `FADV_RANDOM`）**：单槽复用、同层双缓冲、`FADV_RANDOM` 与验证脚本——待有界 prefill 收敛、接口定型后再独立开 `pr/` 分支或先开讨论。
- **M6（不送审）**：A1 两层 L1 原型与中文文档，仅留本实验分支。

# Vulkan LES 实测结论台账（`RESULTS.zh.md`）

> **定位**：`experiment/les-ssd-streaming` 的唯一实测结论与基线台账（**只增不改**；新实验完成后在文末追加小节，不再新建零散报告或按日总览）。
>
> **关联入口**：[交接卡与现行方向](HANDOFF.md#les-next)

**固定基准机**：Linux `7.3.5` / Ryzen 9 9900X / RTX 5060 `8151 MiB` / NVIDIA `615.78.08` / `MemTotal=64946404 KiB`（`61.94 GiB`）/ Btrfs NVMe。

---

## 1. 前序验证与已止损基线（MiniCPM-V 4.7 35B `Q4_K_M`，基于 `tc-mb/master@49dc920`）

| 阶段 / 机制 | 核心实测结果 | 数值 / 故障边界 |
|---|---|---|
| **Gate C｜数值排雷** | 定位两处偏差均属 Vulkan 数值路径切换，非 LES 逻辑错误：P0 始于第 3 层 `TOPK_MOE_EARLY_SOFTMAX_NORM`；S3 两行始于第 0 层 `MUL_MAT_ID` `Q8_1` 激活量化。 | 单变量屏蔽对应融合或统一激活路径后，`6/6` logits 逐字节一致。 |
| **Gate B｜自然 D3 底线**<br/>单槽 + 同层双缓冲 | `2 GiB`/禁 swap 私有 reflink，`16+40` 第 2 请求 `40/40` 步真实读盘：同层双缓冲整请求 `11.97–12.01 s`（P95 `258–259 ms`），优于同步单槽 `13.16–13.56 s`（`-9–11%`）与 CPU `14.79–14.91 s`（`-19–20%`）。 | 三路两轮 `41/41` logits 与各自同后端 oracle 逐值一致（`NMSE=0`）；故障注入与受限冒烟通过。 |
| **A0｜三方 Vulkan 基线** | 原版 `49dc920` 加载剥离专家 LES bank 报错 `backend has no alloc-only buft`；fork 纯暖页 `P50/P95 = 80.55/83.73 ms`（仍上传 `581.31 MiB/tok`）；近等显存跨版本参照 `ggml-org/llama.cpp@61dbf49` `348.75 MiB` 缓存 `59.32` vs fork 原生 `76.32 ms`（不作 fork 开关净收益）。 | 原版与 fork 原生文本 57 份、图文 17 份逐值一致；上游 2 GiB 缓存 3 轮复测通过。 |
| **图文 Pinned Staging**<br/>独立 PR 候选（3 文件） | 默认关闭 `LLAMA_MOE_VK_PINNED_STAGING=1`。不改 `2048/512` 批形与 `32.99 GiB` 上传字节，仅将 `642 MiB` staging 改为 `Vulkan_Host`：报纸图 TTFT `6.50–6.58 → 4.44 s`，滑雪图 `6.56 → 4.35 s`，上传段 `3.60 → 1.41 s`。 | 两图各 `49/49` 逐字节一致；两类分配失败回退正常（`354 MiB pinned + 288 MiB CPU`）；同步验证通过；**与 2 GiB 单槽互斥**。 |
| **A1｜两层续步缓存**<br/>有界批量上传（**Gate 1 = STOP**） | 默认关闭 `LLAMA_MOE_VK_L1_SLOTS=32`。L0/L1 共 `124.5 MiB` VRAM + `15.56 MiB` pinned scratch；同容量零读对照写提交及等待 `1.076 → 1.331 ms/tok`（录制 `0.022 ms` + 批末等待 `1.308 ms`）；文本 P50/P95 `72.05–72.44/74.35–74.98 → 72.62–72.75/75.22–75.24 ms`，图文 `76.25/84.53 → 78.11/88.62 ms`。 | `test-les` `22/22`、`run.sh` `5/5`、全命中 `0` 读写/`0` 同步、8 槽同步验证 `0 hazard`、批中 EIO 保留旧槽映射并持续返回 `-3`；**未达延迟门槛，不扩层、不送审**。 |

---

## 2. 超内存主线 D3-1 → D3-3 实测（2026-11-28）

受限臂统一采用独立 Btrfs reflink（初始 `fincore=0 B`）、独立 cgroup v2 `MemoryMax=2 GiB` / `MemorySwapMax=0`、普通 buffered I/O，保留完整 top-8 与实际 `ubatch=1`。

| 阶段 / 模型 | 负载与关键配置 | Setup | TTFT（不含 setup） | 续步 P50 / P95 | 续步吞吐 | 物理读（prefill / decode / 总） | 整卡 VRAM 起始 → 峰值 | 数值与关键发现 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **D3-1｜MiniCPM-V 4.7 35B** | 长文本 `645 + 40`，`n_ctx=2048`，单槽双缓冲 | `0.75 s` | `168.46 s` | `297.18 / 388.36 ms` | `3.29 tok/s` | `193.81 / 14.67 / 209.78 GiB` | `2011 → 3515 MiB` | `41/41` 逐字节一致；`219,200` 安装；prefill 占请求读盘 `93%` |
| **D3-1｜MiniCPM-V 4.7 35B** | 真实图文 `43 text + 732 image + 40`（`test-1.jpeg` + `mmproj-model-f16-4x.gguf`） | `1.26 s` | `193.00 s` | `242.77 / 293.13 ms` | `4.08 tok/s` | `211.17 / 10.71 / 223.86 GiB` | `2022 → 4371 MiB` | `41/41` 逐字节一致；`260,800` 安装；视觉编码仅 `~209 ms`，prefill 占请求读盘 `95%` |
| **D3-2｜Qwen3.5-122B-A10B `Q4_K_M`** | 短文本 `28 + 40`，`n_ctx=512`、`n_batch=32`，CPU 输出头（`op_offload=false`），48 层 Vulkan 单槽双缓冲 | `2.15 s` | `45.43 s` | `1.630 / 1.656 s` | `0.616 tok/s` | `123.54 / 175.64 / 303.93 GiB` | `2215 → 7213 MiB` | 仅 routed experts `65.25 GiB > 61.94 GiB` RAM；`26,112` 安装；与 8 GiB staged 参考（`26.19 s`, `1.000/1.089 s`, `1.006 tok/s`）`41/41` 逐字节一致 |
| **D3-3｜Qwen3.5-122B-A10B**（`FADV_RANDOM=0` `A1/A2`） | 同 D3-2 条件，无插桩交错 `A1 → B1 → B2 → A2` | `~2.15 s` | `46.224 / 46.429 s` | `1.641–1.642 / 1.669–1.680 s` | `0.612 tok/s` | `123.55–123.57 GiB` / `4.387–4.388 GiB/tok` | `2449–2540 → 7574–7659 MiB` | 独立归因确认默认续步 `99.95%` 物理读（`175.43/175.52 GiB`）在专家 `preadv` 窗口内；CPU 输出头 40 步仅读 `2.46 MiB` |
| **D3-3｜Qwen3.5-122B-A10B**（`FADV_RANDOM=1` `B1/B2`） | 同上，仅对专家 fd 施加 `POSIX_FADV_RANDOM`（保留 LES 双缓冲） | `~2.14 s` | **`36.925 / 37.112 s`**（`-20%`） | **`1.319–1.322 / 1.356 s`**（`-19%`） | **`0.758–0.759 tok/s`** | **`55.17–55.21 GiB`** / **`1.869 GiB/tok`**（`-57%`） | `2539–2548 → 7584–7668 MiB` | `41/41` logits 与 D3-2 oracle 逐字节一致；staged 隔离、`test-les` `22/22` 及 MiniCPM `faults`/`bounded` 通过 |

- **Qwen3.5-122B-A10B `Q4_K_M` 结构与预算事实**：Unsloth `51eab4d59d53f573fb9206cb3ce613f1d0aa392b` 三片语言权重 `71.28 GiB`（`76,536,964,608 B`），合并 879 张量打包为 `76,537,466,880 B` LES（`735 常驻张量 + 48 专家区 = 783/783` 逐字节校验通过）。48 层×256 专家×top-8（独立 `gate/up/down`），routed experts `65.25 GiB`，非 routed `6.02 GiB`，单专家最大 `5,701,632 B`（`5.44 MiB`），全 miss 逻辑载荷 `2.039 GiB/tok`。单槽实际分配：设备槽 `5.44 MiB`、Host 双缓冲及归并区 `10.90 MiB`、子图工作区 `49,184 B`；常驻 Vulkan 另有模型缓冲 `4795.10 MiB`、循环状态 `149.06 MiB`、KV `12.00 MiB`、主图缓冲 `8.26 MiB`。

---

## 3. 核心机理、口径与 LES / Strata 边界

1. **多档 RAM 口径**：`2 GiB`/禁 swap 仅为可重复冷盘压力回归档（验证无页缓存护航时的保底能力），不代表 `61.94 GiB` 主机日常速度；实用评测另增 `8/16 GiB` 与不设额外上限的本机实用档，分别记录首跑与重复请求。
2. **Qwen 输出头放置边界**：仅把 `^output\.weight$` 移至 Host 而不关闭 `op_offload`，调度器仍会在 Vulkan 上为 `596.78 MiB` 输出头预留约 `598 MiB` 图缓冲并触发 OOM；关闭 `op_offload` 后主图计算缓冲降至 `8.26 MiB`。`result_output` 结果缓冲名显示 `Vulkan_Host` 不等于在 GPU 执行，须以 `ggml_backend_sched_get_tensor_backend` 实测为准。
3. **相比原版 `49dc920` 与 `Strata@58ee322`**：对原版增量 = 非 UMA Vulkan 兼容 + 单槽复用/同层双缓冲 + `FADV_RANDOM` + 独立 Pinned Staging；对 Strata 只借鉴执行原则（按专家读、需求读优先、有界缓冲、GPU 消费完再覆写槽），未移植自定义前向调度器、持久 VRAM 热池、Host 补集缓存、预测预取或 CPU routed experts 分流（Qwen CPU 输出头只为常驻显存准入）。
4. **当前主瓶颈**：消除内核预读放大后，冷态延迟仍由 `ubatch=1` prefill 逐 token 重复读取与上传主导（MiniCPM `168–193 s`，Qwen 28-token `~37 s`），下一结构性突破口是**有界 prefill 同专家复用**（需同时验证 Gate C 定位的多行 `Q8_1` 数值路径差异）。

---

## 4. 已知未定位异常与未覆盖范围

- **2 处历史偶发异常未定位、不得视为已修**：① A0 上游 2 GiB 缓存反序轮 1 次退出 `double free or corruption (out)`；② TTFT 诊断 `2048/512` 基线 2 次 16 步后 `SIGSEGV`（多轮复测与 gdb 正常，根因未明）。
- **尚未覆盖范围**：Qwen `8/16 GiB` 与不限额本机实用档、长文本及已下载 `mmproj-F16.gguf` 图文；Windows / Android Vulkan；Apple M5 Metal；`#165` 前置 dense 层单测；Omni 全双工 profiler。

---

## 5. 后续实验记录追加约定

后续新实验直接在下文按 `## YYYY-MM-DD｜主题` 追加条件（基线 SHA、预算、模型、批形）、关键数字、数值 oracle 结果与裁决，并同步更新 [HANDOFF.md](HANDOFF.md#les-next)。

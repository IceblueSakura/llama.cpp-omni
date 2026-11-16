# 上游算子修复分批执行记录

日期：2026-11-15。工作目录：`/home/sakura/workspace/llm/llama.cpp-omni-operators`，分支：`fix/upstream-operators`。

## 1. 依据与批次边界

- omni 基点：`350a835d5f8215145aa210a7df0f67e8877b5d6e`。
- 固定上游：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`，本地参考仓库 `/home/sakura/codespace/llama.cpp`；未 fetch。
- 候选及稳定 ID 见 [上游算子调研](upstream-operator-survey.zh.md)；没有直接上游方案的事项见 [扩展调研](upstream-operator-extensions.zh.md)，不混入本执行队列。
- 按同类契约、共享构建与设备分批移植和验收，不要求每项重建整个环境。每项保留来源、必要依赖和测试结果；批次不等于不可拆分的大提交。
- 只移植当前批次所需的修复、测试及最小前置依赖。不为套用上下文引入性能优化、公共 API 或整个后端更新。
- 算子实现不修改原 LES 源码或历史；当前用户确认的联动仅调整双方计划文档。不自动提交、推送或发布，合并到 master 前由用户审查并决定。当前分步集成与发布顺序见第 9 节。

## 2. 推荐队列

后续批次是分组建议，进入批次时仍须核对提交依赖及目标设备，不能视为已验收的 cherry-pick 清单。

| 批次 | 候选范围 | 共享验证环境 / 交付边界 |
|---|---|---|
| L1：局部索引与 view 正确性 | A04 CPU `CONCAT`、A02 CUDA `CPY`、B01 CPU/CUDA `CLAMP` | 已验证并提交为 `d5ae271`；当前 Nix CUDA devShell，CPU 与 RTX 5060 |
| L2：CUDA 局部修补 | A05 MoE selection NaN、A09 IQ4_NL 短行、A11 ARGSORT CUB alias | 已提交 `6be022a`，见第 6 节。B08 量化 correction 未纳入 |
| L3：Vulkan 正确性 | B06 matmul stride、B05 FA 边界与 scratch | 已提交 `b17420c`，见第 7 节。B03 和 D02 紧凑 tile 重组未纳入 |
| M1：Metal 局部正确性 | A03 CPY stride、A06 TRI NaN/Inf、A07 SOLVE_TRI 尾读 | LES 已有 `sakura-mac` M5 真机记录；先确认访问、工具链和资源排期，在独立源码/build 目录验收，不以旧 LES 或 Linux 结果代替 |
| M2：Metal 配套修补 | A12 大偏移、B02 Metal transpose-conv、B03 Metal group norm、B09 FA | 真机内存/shape 与 shader validation；进入批次后按实际依赖拆成可验收子组，不能为了批量而混入整套 FA 重构 |
| 后续能力与性能批次 | C 类按 backend/算子族分组，再考虑 D 类 | 先证明 omni 图可达，再做数值与性能验证；不将新 kernel 存在写成模型收益 |
| 独立迁移设计 | E 类公共 API、状态布局和格式 | 不作为上述小修复批次的隐含依赖；需要明确模型需求和跨后端合同 |

后续核对已从 LES 当前记录确认 `sakura-mac` M5 真机线索，不再笼统标为“没有 macOS 环境”。本轮没有连接或运行该机器；M1/M2 需确认访问、系统/Xcode、资源与验收排期，不与 LES 重型 oracle 并发抢占。CPU ARM ISA 专用修复仍须有目标 ISA/dispatch 证据，不能仅因 macOS 可用就视为覆盖。

## 3. L1 来源与适配说明

| 候选 | 完整上游 commit / PR | 移植内容 |
|---|---|---|
| A04 | `ef2d770117db45b05aa7ecd1b0acca36370c5470` / #25247 | CPU 量化 CONCAT 按 block 计算 dim0 偏移、循环及源边界，保留量化输入连续/对齐断言；五种类型 × 四维的回归 |
| A02 | `134b2bb756c984d4fd2cabc26335f8de56b12ee1` / #27663 | CUDA CPY transpose 快路要求 dst 连续；F32/F16 permuted-src → strided-dst 回归 |
| A02 测试前置 | `0ed235ea2c17a19fc8238668653946721ed136fd` / #25057 | 仅摘取 `test_cpy` 的 `dst_alloc` fixture 能力，不引入该提交的 cudaMemcpy2DAsync 优化或其他 backend 修改 |
| B01 | `65840ed53c8653bfbf3e9014d9cbf71ac8c08725` / #29517 | CPU/CUDA CLAMP 按 dims 1–3 stride 寻址及 CUDA 支持判定；F32/F16 四维 view 回归 |
| A04 能力门控适配 | `85f99dca8b4539163e6858c9df7c9e8b0ad3632a` / #24011 | 参考其严格 dtype 判定和拒绝量化类型的机制，仅声明 omni 现有 CUDA CONCAT kernel 实际支持的 F32；不引入上游 scalar kernel 扩展 |

### 必要的本地测试适配

- omni `ggml_clamp()` 返回输入别名 view；上游目标修复时的普通 CLAMP 已非原地。L1 保留 omni API，不引入上游 `6036c635e2d1e9aeca11d01a84188c9db9ea1e2a` 的 API/allocator 变化。
- CPU 与 CPU reference 的 CONCAT/CLAMP 共用实现。新增量化 CONCAT 与 CLAMP view 用例使用现有 `initialize_tensors`、`run_whole_graph`、`err` 扩展点：在计算前快照实际输入，用 host 拼接/标量 clamp 独立构造预期，分别核对两个 backend 输出。此 oracle 是本地测试适配，不冒称上游提交自带。
- whole-graph oracle 只比较最终输出，不代表完成 sentinel/所有 backing 孔洞不变检查。原非量化 CONCAT 和普通 CLAMP 保留逐节点检查；CPY 保留原框架的 CPU reference 与外部 sentinel 比较。
- CUDA 旧 CONCAT 支持判定会接受量化类型，但实际 kernel 仅支持 F32。上游 `85f99dca8b4539163e6858c9df7c9e8b0ad3632a` 在扩展 scalar kernel 时已拒绝量化类型。本批不扩大 CUDA CONCAT 类型能力；其支持判定应与现有 F32 kernel 保持一致，记录为测试引入后的必要能力门控适配。

## 4. 可复用测试环境

使用仓库已有 `flake.nix` / `flake.lock` 的 `.#cuda` devShell，无需 Podman 或修改主机配置。使用 `bash -c` 保留 devShell 编译器，不用 login shell 重新覆盖 PATH。

已识别设备：NVIDIA GeForce RTX 5060，8151 MiB，driver 615.78.08；backend 报告可见 VRAM 7796 MiB。已实现的 devShell 提供 CUDA 12.9.86，CUDA host compiler 为 GCC 14.4.0，普通 C/C++ 为 GCC 15.3.0；memcheck 使用本机已有 Compute Sanitizer 2026.3.0.0。未修改 flake 输入和 lock。

```bash
cd /home/sakura/workspace/llm/llama.cpp-omni-operators
nix develop --no-update-lock-file .#cuda --command bash -c '
  set -e
  cmake -S . -B build-operators-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real \
    -DGGML_CCACHE=OFF -DGGML_BLAS=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
  cmake --build build-operators-cuda --target test-backend-ops -j 6
'
```

`GGML_CCACHE=OFF` 是本环境的构建选项：自动发现的主机 `sccache` 在调用 Nix nvcc 时报告 `cannot find binary path`；不修改主机 cache 服务，也不因此修改算子源码。

```bash
# 在上述 devShell 中执行；backend 名必须与实际设备列表一致。
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
build-operators-cuda/bin/test-backend-ops test -b CPU -o CONCAT,CLAMP -j 1
build-operators-cuda/bin/test-backend-ops test -b CUDA0 -o CPY,CLAMP,CONCAT -j 1

compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-operators-cuda/bin/test-backend-ops test -b CUDA0 -o CPY \
  -p 'dst_alloc=\[32,64,16,1\]' -j 1
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-operators-cuda/bin/test-backend-ops test -b CUDA0 -o CLAMP -p 'v=1' -j 1
```

必须检查实际 case 数、unsupported 和失败详情，不能只看退出码：当前 harness 在设备名未命中或过滤后零 case 时也可能退出成功。这里只跑变更相关算子，不将全 suite/模型性能作为局部正确性修复的前置条件。

CPU ASan 复用相同 devShell 和上面的 CMake 配置，但将构建目录改为 `build-operators-asan`，将 `GGML_CUDA` 改为 `OFF`，增加 `-DLLAMA_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_ADDRESS=ON`。构建同一 target 后执行：

```bash
build-operators-asan/bin/test-backend-ops test -b CPU -o CONCAT \
  -p 'type=(q4_0|q4_1|q5_0|q5_1|q8_0),' -j 1
build-operators-asan/bin/test-backend-ops test -b CPU -o CLAMP -p 'v=1' -j 1
```

## 5. L1 验证记录

### 修复前

只加入回归与 oracle，生产源码仍为 omni 基点版本：

| 聚焦检查 | 实际结果 |
|---|---|
| CPU CLAMP `v=1` | F16/F32 共 2 例失败，exit 1 |
| CPU CONCAT `type=q4_0,.*dim=0,` | `corrupted double-linked list`，进程中止，exit 134 |
| CUDA CPY `dst_alloc=\\[32,64,16,1\\]` | F32/F16 共 2 例失败，exit 1 |
| CUDA CLAMP `v=1` | F16/F32 共 2 例失败，exit 1 |

红测脚本继续收集各子命令状态，因此脚本本身 exit 0 不是通过依据；以上是逐条记录的退出码。CONCAT 中止前 stdout 未完整刷新，不声称它完成了正常测试计数。

### 修复后

| 检查 | 实际结果 |
|---|---|
| Nix CUDA devShell 增量构建 `test-backend-ops` | exit 0 |
| CPU `CONCAT,CLAMP` | **60/60** 实际执行通过，包括 20 个量化 CONCAT、2 个 CLAMP view 回归 |
| CUDA0 `CPY,CLAMP,CONCAT` | **260/260** 实际执行通过，包括新增 CPY/CLAMP 回归。未支持的 CPY 布局、I32/量化 CONCAT 正常拒绝，不计为通过 |
| CUDA memcheck：CPY 新增回归 | **2/2**，`ERROR SUMMARY: 0 errors`，exit 0 |
| CUDA memcheck：CLAMP view | **2/2**，`ERROR SUMMARY: 0 errors`，exit 0 |
| CPU ASan：五种量化类型、四个 CONCAT 维度 | **20/20**，无 ASan 报错，exit 0 |
| CPU ASan：CLAMP view | **2/2**，无 ASan 报错，exit 0 |

本批代码与针对性验证已完成，按用户授权提交至 `fix/upstream-operators`：`d5ae271f09c3e5ed1140408b28b7564c355451ca`，含作者匹配的 DCO sign-off，不合并到 master。只检查变更相关算子，不声称模型图、性能、HIP/MUSA、Vulkan 或 Metal 已验收。两项独立只读依赖/测试审查均已完成，其适用性判断已由上述红绿检查复核。

### 本地证据

完整日志已保存至忽略的构建目录，文件首行为 job 元数据，其后为原始日志：

- `build-operators-cuda/evidence/l1-baseline-build.log`
- `build-operators-cuda/evidence/l1-red.log`
- `build-operators-cuda/evidence/l1-green.log`
- `build-operators-cuda/evidence/l1-memcheck.log`
- `build-operators-cuda/evidence/l1-asan.log`
- `build-operators-cuda/evidence/l1-sources.b3`：L1 的 5 个源码/测试文件 BLAKE3 快照；在 L1 提交对应的 checkout 根目录运行 `b3sum -c` 核对，后续批次改变同一文件后不再匹配是预期行为。

上述 job 保存时均 `captureComplete=true`。日志是本地验证产物，不纳入源码提交；清理构建目录会删除这些产物。L1 源码、测试和调研/执行文档一起纳入本批提交，master 保持不变；提交号以 Git 历史为准。

## 6. L2：本机 CUDA 局部修补

基点为 L1 `d5ae271f09c3e5ed1140408b28b7564c355451ca`，复用上述 Nix 环境和 RTX 5060。状态：**已完成针对性验证，按用户授权提交至 fork 工作分支**：`6be022af5c33acd262d754f7e76d2200d64a9411`；未推送或移动 master。不需要额外 GGUF/权重模型；不扩展到 B08、Metal 或新模型能力。

### 来源与最小适配

| 候选 | 完整上游 commit / PR | 改动与依赖 |
|---|---|---|
| A05 | `10d4846a572aebf21c0482d140b00e74c0f1ad20` / #31005 | 原样补入 post-bias selection NaN → `-FLT_MAX`。当前融合路径已存在，无新增融合前置 |
| A09 | `f872b591121761ac7b2af18283bd99bdc092a63a` / #29683 | 将实际 32 元素 block 数传入 kernel，越界 sub-block 提前返回。omni 尚未采用上游抽取的 dequantize helper，故在原 kernel 内加入同等边界保护，保留既有布局与计算 |
| A11 | `b23701f77d47dad9de834d59ebfcbe25c9e8b46f` / #28389 | 原样补入独立 `temp_keys_out`，覆盖 6 种 CUB 调用分支的 size-query/execute。额外 scratch 为 `ncols*nrows*sizeof(float)`，另有 pool 对齐；不引入后续排序策略优化 |

本机锁定的 CCCL `cub/device/device_radix_sort.cuh` 明确要求输入、输出 ranges 不重叠；不能将当前设备偶然执行成功当作允许 alias 的依据。

### 测试选择与运行证据

- A05：扩展现有 `test_topk_moe`，4 个 logits 全 0、selection bias 全 NaN，sigmoid 权重为 0.5、选择全部 4 个专家。沿用 CPU reference 检查 IDs 集合及 weights，避免依赖跨后端 NaN 排序顺序。普通 GDB 在修复前、后均记录 `ggml_cuda_try_fuse → ggml_cuda_op_topk_moe` 命中；修复前失败，修复后通过，关闭 fusion 的对照也通过。
- A09：新增窄范围 `test-cuda-convert`，直接调用实际 F32/F16 converter，以精确 `cudaMalloc` 分配验证 `k={32,64,96,256,288}`。独立 CPU 解量化 stored input 作为预期。普通 CPY 使用不同路径，backend pool padding 也可能掩盖越界，因此不以它们代替此测试。该测试只在可链接的 native CUDA backend 上构建，排除 `GGML_BACKEND_DL` 及 Windows shared；无设备返回 CTest skip 77，其他 CUDA 初始化错误仍失败。
- A11：复用 ARGSORT CPU reference，修正已有大行循环遗漏 `order` 参数的测试盲区，增加 248k 单/双行升降序。修复前本机 4/4 通过，**未复现上游的间歇性损坏**；修复后继续通过，不编造红绿结论。
- A11 另用本地小张量 probe 对同一图连续计算 3 次，覆盖普通执行、capture 和 replay，核对排序索引与源数据保持不变。4 组形状/方向共 12 次检查通过；GDB 记录各组 `capture=0/1` 且 `temp_keys != temp_keys_out`。probe、调试命令和日志只保留在本地 evidence 目录，不向生产代码添加 instrumentation。
- 辅助审查状态：MoE 子任务完成；IQ4_NL 子任务未成功返回，未将其当作证据。IQ4_NL 的调用链、精确分配复现和数值/memcheck 验证由主会话完成。

| 检查 | 实际结果 |
|---|---|
| 修复前 MoE NaN | **0/1**，exit 1；融合入口实际命中 |
| 修复前 IQ4_NL memcheck | `k=32` 实际越界读取，30 errors（仅打印前 8 项），exit 99；未继续执行其余 case |
| 修复前 ARGSORT 248k | **4/4**，exit 0，本机未复现间歇性错误 |
| 修复后 CUDA ARGSORT / TOPK_MOE / NaN 回归 | **295/295**，exit 0 |
| 修复后关闭 fusion 的 NaN 对照 | **1/1**，exit 0 |
| 修复后 IQ4_NL converter memcheck | **10/10**，0 errors，exit 0 |
| 修复后 MoE NaN / ARGSORT 聚焦 memcheck | **7/7**，0 errors，exit 0；包括 4 个 248k、2 个小尺寸排序和 1 个 MoE NaN 用例 |
| 新转换测试 CTest 注册/运行 | **1/1** test 通过，内部 10 个 conversion case 实际运行 |
| ARGSORT 同图 capture/replay probe | **12/12**，单/双行、升/降序均观察到 capture 与独立 keys 缓冲区 |
| CPU-only / CUDA dynamic-backend CMake | 配置、生成成功；dynamic-backend 的转换测试注册为 0 项，未构建或运行该模式的 CUDA backend |

首次修复后 memcheck 虽然 10/10 数值通过，但报了一条 `Selective device code recompilation in progress` API warning 并 exit 99，未当成绿色结果。同一二进制缓存预热后原命令 0 errors，最终完整检查也 0 errors。两个日志均保留。CUDA-GDB 13.4 的设备断点尝试在初始化时因 protobuf 错误中止，未取得 device breakpoint 证据；随后使用普通 GDB 获取上述 host dispatch 及 capture/缓冲区证据，无需更改主机调试器配置。

### 复验命令与本地证据

在第 4 节的 devShell 中执行，保留相同的 `LD_LIBRARY_PATH`：

```bash
cmake --build build-operators-cuda --target test-backend-ops test-cuda-convert -j 6
unset GGML_CUDA_DISABLE_FUSION
build-operators-cuda/bin/test-backend-ops test -b CUDA0 \
  -o ARGSORT,TOPK_MOE,TOPK_MOE_POST_BIAS_NAN -j 1
GGML_CUDA_DISABLE_FUSION=1 build-operators-cuda/bin/test-backend-ops test \
  -b CUDA0 -o TOPK_MOE_POST_BIAS_NAN -j 1
compute-sanitizer --tool memcheck --error-exitcode 99 build-operators-cuda/bin/test-cuda-convert
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-operators-cuda/bin/test-backend-ops test -b CUDA0 \
  -o TOPK_MOE_POST_BIAS_NAN,ARGSORT \
  -p 'ne=\[(4,1,1,1|248000,1,1,1|248000,2,1,1)\]' -j 1
ctest --test-dir build-operators-cuda -R '^test-cuda-convert$' --output-on-failure
```

`build-operators-cuda/evidence/` 中保留：

- `l2-red.log`：修复前构建/三项检查。
- `l2-green-initial.log`、`l2-green.log`：首次 instrumentation warning 与最终检查。
- `l2-dispatch-red.log`、`l2-dispatch-green.log`：融合入口及 CUB 普通/capture 分支证据。
- `l2-cmake.log`：最终 native CUDA 测试及 CPU-only / dynamic-backend 配置检查。
- `l2-argsort-graph.cpp`、`l2-argsort-dispatch.gdb`、`l2-moe-host-dispatch.gdb`：本地 probe / 调试命令。
- `l2-sources.b3`：本批 6 个源码/测试文件快照；从 worktree 根目录用 `b3sum -c` 核对。

所有性能收益均未测量；本批是正确性回移植，不据小张量验收声称模型加速或覆盖其他硬件。下一批继续优先考虑本机可验证的 Vulkan 正确性候选。

## 7. L3：本机 Vulkan 正确性

基点为 L2 `6be022af5c33acd262d754f7e76d2200d64a9411`。状态：**已移植并完成下述针对性检查，按用户授权提交至 fork 工作分支**，提交号以 Git 历史为准，未推送或移动 master。不需要 GGUF/权重模型。两项独立只读依赖审查均完成；最终结论以实际运行和已注明的检测边界为准。

### 来源与边界

| 候选 | 完整上游 commit / PR | 本批内容 |
|---|---|---|
| B06 | `52624716156a25af417b616e23bc712452d9396c` / #28956 | MM、MMV、MM-ID、MMV-ID 原位 A/B 按 `nb[2]/type_size*block_size` 取 stride；MM/ID descriptor 使用原位 strided extent 或 staged size，不能扩大到 buffer 尾部 |
| B05 rowmax | `ac4105d68b2955027115cf9bb50941ccf56974eb` / #24909 | generic/cm1 在 softmax 前应用已有 `FATTN_KQ_MAX_OFFSET`，避免未归一化 FP16 分子溢出 |
| B05 scratch | `806eee9841de5f2c20f9d43914117f157d2baacc` / #29591 | FA mask-opt、large softmax 覆盖 `prealloc_y` 时清空转换缓存标识；omni 没有上游 `use_sparse`，不引入它 |
| B05 MMQ shared | `6c59c40076c00eab49754dc955d7652d93f9e125` / #29988 | Q 量化 shared 写入增加 `r<Br`，subgroup 归约保留在分支外；保留 omni 的 `FA_TYPE_*`、`FLOAT_TYPE`，不引入上游额外 FP32 量化重构 |

B06 不引入上游后续 MM-ID K-padding staging、test `src_overlap` 或 `amax` 参数。只追加当前 fixture 所需的 `m_v/pad`，移植六个上游 A-view/padding 用例；另加四个 B-only batch/token stride 用例，验证本次同时修改的 B 寻址。没有新增测试框架。

### 关键用例和证据

- **matmul**：六个上游 A-view 用例覆盖四条执行路径及 `nb[2]` 非整行 padding；四个 B-only 用例覆盖 MM/MMV、单/多 token ID。debug dispatch 核实 MM/ID 与实际 stride，例如 F16 padded A 为 1032 元素；descriptor 不越过真实 extent。Q4_0/NVFP4 partial-tile 各一例用于防止过宽 range 让量化尾 tile 读取残留。
- **FA 溢出**：Q/K 全零、V=4096，输出独立预期为4096。初版 F16 K/V 用例在 CPU 自身也失败，故最终用 F32 K/V，使 CPU oracle 通过、GPU 仍走 FP16 中间计算。修复前 generic 出现最大浮点值、cm1 为65504，而 CPU 为4096；修复后两条路径均通过。没有在本批修改 CPU FA。
- **scratch**：同一次 graph compute 内执行 `I·X → FA mask-opt/large softmax → 2I·X`，两个 matmul 使用相同的非连续 X 指针，输出独立预期为 `1+2X`。`run_whole_graph=true`，避免逐节点 compute 清空缓存而掩盖问题；X 的转换区8192字节，大于 large-softmax 临时区4224字节，不靠中途扩容触发失效。日志确认覆盖节点、相同 RHS 的两次转换，以及 scratch buffer 复用。
- **量化 FA 尾行**：Q4_0 K、单 query、HSK64，零 Q/K、V=1，命中 generic MMQ 的小 Br 几何；数值与 validation 冒烟通过。**旧版此例也通过，普通 validation 不保证检测 Workgroup/shared 数组越界，因此不声称已取得共享内存越界的动态红绿证明。** 修复依据为上游边界条件、实际 dispatch 几何和源码核对。

### 实际结果

| 阶段 / 设备路径 | 实际结果 |
|---|---|
| 最终 fixture 的 CPU oracle | **14/14**，exit 0 |
| 修复前 NVIDIA generic | **1/14**，exit 1：10 个 matmul、FA 溢出、2 条 scratch 链失败；量化尾行冒烟通过 |
| 修复前 NVIDIA cm1 的 FA/scratch 子组 | **1/4**，exit 1：FA 溢出和2条 scratch 链失败 |
| 修复后 NVIDIA generic / cm1 | 分别 **14/14**，exit 0，且确认实际 generic / `_cm1` dispatch |
| 修复后 NVIDIA 默认 coopmat2、默认 fusion | **14/14**，exit 0 |
| NVIDIA generic Q4_0/NVFP4 partial-tile | **2/2**，exit 0，没有超时 |
| 既有混合量化/非整块 FA：NVIDIA cm2 / generic / cm1 | 分别 **8/8、4/4、4/4**；非 cm2 的4个 Q1_0 不支持项不计入通过 |
| AMD RADV generic 新用例 / 既有混合量化 FA | **14/14、4/4**；4个 Q1_0 不支持项不计入通过 |
| 最终 NVIDIA/RADV validation | 无 `Validation Error` / VUID error；仍有已有 `BestPractices-SpirvDeprecated_WorkgroupSize` warning |

初轮强制关闭 coopmat2、却未关闭 decode-vector 时，出现 `VUID-vkCreateDevice-ppEnabledExtensionNames-01387`：后者扩展依赖前者。最终非 cm2 测试同时设置两项禁用开关，消除该已有初始化告警；没有扩展本批为后端 capability 改造。

AMD 首次使用主机 Mesa 26.3 的 ICD 时，LLVM 要求 `GLIBC_2.44`，与项目环境 glibc 2.42 不匹配，Vulkan 注册失败、实际0个 Vulkan case。尽管 harness exit 0，该次**不计为通过**。随后只在本测试进程指定项目同一 nixpkgs pin 的 Mesa 26.2.3 ICD，实际识别 RADV 核显并完成上述检查；未替换主机驱动或修改 Nix inputs/lock。

### 环境与复验

```bash
cd /home/sakura/workspace/llm/llama.cpp-omni-operators
nix develop --no-update-lock-file .#vulkan --command bash -c '
  set -e
  cmake -S . -B build-operators-vulkan -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_VULKAN=ON \
    -DGGML_VULKAN_DEBUG=ON -DGGML_VULKAN_VALIDATE=ON \
    -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
  cmake --build build-operators-vulkan --target test-backend-ops -j 6
'
```

在该 devShell 内运行；用 ICD 固定设备后，测试器名称均为 `Vulkan0`，仍须核对实际设备和 case 数：

```bash
export VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
export VK_LAYER_PATH=/run/opengl-driver/share/vulkan/explicit_layer.d
export GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1
export GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR=1 GGML_VK_DISABLE_FUSION=1
T=build-operators-vulkan/bin/test-backend-ops
timeout 120s "$T" test -b Vulkan0 \
  -o MUL_MAT,MUL_MAT_ID,MUL_MAT_RHS_VIEW,FLASH_ATTN_CONSTANT,MATMUL_SCRATCH_REUSE \
  -p 'm_v=(5120|16|64)(,|$)|^(n=|hsk=|flash=)' -j 1
# cm1: unset GGML_VK_DISABLE_COOPMAT。默认 cm2/fusion: unset 全部四个 GGML_VK_DISABLE_* 开关。
timeout 120s "$T" test -b Vulkan0 -o MUL_MAT \
  -p 'type_a=(nvfp4|q4_0),type_b=f32,m=1,n=64,k=256,' -j 1
timeout 120s "$T" test -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'hsk=(64|72|128),hsv=(64|72|128),nh=4,nr23=\[1,1\],kv=(64|96|128),nb=2,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32' -j 1
```

AMD 使用兼容的 Mesa，而不是混入更新主机 glibc：

```bash
MESA=$(nix build --impure --no-link --print-out-paths --expr \
  'let f = builtins.getFlake "/home/sakura/workspace/llm/llama.cpp-omni-operators";
       p = import f.inputs.nixpkgs { system = "x86_64-linux"; };
   in p.mesa')
export VK_DRIVER_FILES="$MESA/share/vulkan/icd.d/radeon_icd.x86_64.json"
# 保持 generic 开关，复用上面的14例及混合量化 FA 命令。
```

实际解析的 Mesa 为 `/nix/store/1vkzwrp4ny20iljfnv9valq0bfn8czlv-mesa-26.2.3`；shaderc 2026.1、Vulkan loader 1.4.357，validation layer 1.4.363，NVIDIA driver 615.78.08。可复用环境未作持久声明修改。

完整本地证据保存在忽略的 `build-operators-vulkan/evidence/`：

- `l3-baseline-build.log`、`l3-red-initial.log`、`l3-red.log`
- `l3-nvidia-green.log`、`l3-nvidia-default.log`
- `l3-radv-host-unavailable.log`、`l3-mesa-env.log`、`l3-radv-green.log`
- `l3-sources.b3`：本批4个源码/测试文件快照，在对应源码状态的 worktree 根目录用 `b3sum -c` 核对。

这些检查不等于完整 Vulkan suite、模型端到端性能或 Metal 验收；本批没有测量加速比。

## 8. L3 提交时的整体进度与剩余候选

统计单位为主调研的稳定候选 ID，不是 commit 数、后端数或工时。53 组候选中，**8 组已回移植并完成已记录的针对性检查，45 组尚未开展**；不存在已修改代码、尚未收口的其他候选。这里的“已回移植”不表示所有平台、模型和检测手段均已验收，尤其保留 B05 shared-memory 动态检测缺口和 A11 本机未复现旧故障的说明。

| 批次 | 候选 ID | 提交 |
|---|---|---|
| L1 | A02、A04、B01 | `d5ae271f09c3e5ed1140408b28b7564c355451ca` |
| L2 | A05、A09、A11 | `6be022af5c33acd262d754f7e76d2200d64a9411` |
| L3 | B05、B06 | 本次 `ggml-vulkan: apply batch L3 correctness fixes` 提交 |

### 剩余清单（完整 ID）

| 分类 | 已移植 / 总数 | 尚未开展 |
|---|---|---|
| A：单后端局部修补 | 5 / 12 | A01 ARM tile；A03 Metal CPY；A06 TRI；A07 SOLVE_TRI；A08 Vulkan NVFP4 scale LUT；A10 CPU 反向算子 alias；A12 Metal 64-bit offset |
| B：配套正确性/helper | 3 / 13 | B02 transpose-conv；B03 norm；B04 CPU FA；B07 CUDA/HIP MMF/MMID；B08 CUDA 量化 correction；B09 Metal FA；B10 CUDA FA/norm/softmax 同步；B11 大 grid；B12 view/GLU；B13 CPU ISA/helper |
| C：backend/类型能力补齐 | 0 / 10 | C01–C10：池化、reflect pad、Metal PAD、SET_ROWS/BF16、卷积/IM2COL、GDN/GLA、OpenCL/WebGPU、量化快路及训练算子 |
| D：性能组织变化 | 0 / 10 | D01–D10：CPU tiled、紧凑专家 tile、MMQ/少行 matmul、routing/reduce/top-k/FA 融合、ARM repack、移动端及布局/归约优化 |
| E：公共合同/新能力 | 0 / 8 | E01–E08：precision、跳过 expert ID、RoPE、SSM/GDN 状态、稀疏 FA、新 opcode、模型索引及新格式 |

多后端候选进入实施时按实际子项记录；例如只完成 B02 的 CPU 部分，不能把其 Metal 部分也标记为完成。C–E 是候选池，不是必须全做的既定移植队列；性能组织和公共合同的成本不能按当前8组小修复线性估算。

### 后续建议：仍先做本机能验证的正确性

以下保留候选相对优先级；当前不直接启动下一批，先按第 9 节收口已有成果的集成与发布准备。

1. **下一批优先**：B04 CPU FA（L3 已观察到 F16 reference 未满足常量预期）、A08 Vulkan NVFP4 LUT、B03 的 CPU/Vulkan norm 子项。进入批次后分别核对上游依赖，B03 的 Metal 子项保留待办。
2. **后续 CUDA 正确性批次**：B08、B07 的 CUDA 子项，然后按依赖拆 B10/B11；同步问题需对应 racecheck/synccheck，不只看普通数值输出。HIP 特有部分不以 CUDA/RADV 测试代替。
3. **其余本机修补**：B02 CPU、B12 的适用 view/GLU 子项、B13 当前 x86 ISA 可达项；A10 主要服务反向/训练，推理优先级较低。
4. **能力补齐**：优先评估直接对应 omni 音频路径的 C01 CUDA/Vulkan POOL_1D、C02 Vulkan PAD_REFLECT_1D，再按模型图可达性选择其余 C 类；WebGPU 部分单独验证。
5. **待目标环境**：A03/A06/A07/A12 与 B09、其他 Metal 子项等在 macOS 真机处理；A01、ARM/SVE/NEON、AMX、HIP、Adreno/Hexagon 项需核对实际 ISA/硬件，不能因 Linux 主机或任意 macOS 可用就算覆盖。
6. **后置**：D 类须证明实际热点和端到端收益；E 类需要独立迁移设计。需要具体权重模型时再列出用途和资源需求，目前没有新增模型下载需求。

扩展文档 X01–X06 **不计入这53组**，本算子分支未推进其自研/LES 适配；涉及原 LES 工作的疑点须核对该会话最新状态，不能把旧调研位置当作仍未修复的证据。

## 9. 当前确定的集成、发布与下一批方向

当前 LES 检查点为 `81221392bc89eaf0836898d463123e123b3171aa`，算子为 `b17420c347612951dd3832794e1212dec515bbf4`，共同基点仍为 `350a835d5f8215145aa210a7df0f67e8877b5d6e`。
分步验证的**计划**在 LES 工作区 `/home/sakura/workspace/llm/llama.cpp-omni/docs/les/HANDOFF.md` 第 6 节 I0–I5；随后逐步执行的**唯一结果记录**在独立 `integration/les-operators` 分支的同名文件，工作区 `/home/sakura/workspace/llm/llama.cpp-omni-integration`。可用 `git show integration/les-operators:docs/les/HANDOFF.md` 读取，不在算子源分支复制平行执行记录。

原计划确认时仅调整文档；用户随后授权逐步执行、每步提交。组合 I0–I4 及首个 I5 本地候选的精确范围、输入和结果见上述集成记录；本算子源分支仍保留独立已测代码，没有合入 LES 生产代码。未推送/发布；已回移植仍为8组，剩余45组仍是候选池，不是发布前必须完成的清单。

### 本算子工作线的交付顺序

1. **配合集成 I0–I4**：冻结 OP-L1/L2/L3；在待建 `integration/les-operators` worktree 从 LES 检查点起逐批 merge，每步先编译并验证对应算子和 LES CPU 回归，再验最终组合的 GPU 同步链与模型数值。原两个分支不互相合并，master 不动；每个实际 merge/提交另需授权。
2. **准备当前综合发布包**：用户选择现有 OP-L1–L3 合成一个算子正确性 PR。当前分支 `pr/ggml-operator-correctness` 的三个提交已按修改范围重写为英文全文后中文全文，列完整 upstream ID，末尾各保留一份 AI assistance 和 Signed-off-by，不在公开说明中携带本地批次序号。先行 CONCAT 本地/远端分支已按用户授权删除，旧目录 detached 保留取证；精确 refs 和验证映射见 integration 的 HANDOFF。
3. **独立发布分支验收**：从已核对的 `tc-mb/llama.cpp-omni` master 提取三批纯源码/测试，保留三批提交及逐项来源；共13个文件，不带 LES、长调研文档、新能力或未测候选。该分支独立构建与复核 CPU/CUDA/Vulkan；具体结果在 integration 的 HANDOFF 第 6 节维护，不用组合通过代替。
4. **再选下一修复批次**：组合 I3/I4 收口后，优先 B04 CPU FA、A08 Vulkan NVFP4 LUT、B03 CPU/Vulkan norm 子项；每项仍核对具体前置/后续修复，不引入 core API、融合或整 backend 重排。其余第 8 节候选保持原相对优先级。
5. **Metal 独立排期**：在确认机器、构建目录和资源后做 M1；LES G1-M 原始 oracle 与 M1 是两类验收，双方原通过项不能互代。D/E 类继续后置，C类先证明真实图可达。

开发继续按同类小批共享 Nix 构建，**当前公开单位按用户选择扩大为全部 L1–L3 的一个算子正确性 PR**。以下改为包内审阅分项，不再表示必须分别打开 PR：

| 包内分项 | 不能随意丢掉的内容 / 边界 |
|---|---|
| A04 CONCAT | CPU block 寻址、必要 CUDA dtype 拒绝、fixture 和独立预期 |
| B01 CLAMP | CPU/CUDA 同一 view 契约及独立预期；保留 omni 原地 API |
| B06 Vulkan matmul | 四条 MM/MMV/ID 路径、精确 descriptor 范围及 padding/尾 tile 回归 |
| B05 FA | rowmax 溢出、scratch 失效、MMQ shared 写边界分别说明来源和测试限制，不混为同一种缺陷 |
| A02/A05/A09/A11 | 同包内各自保留来源、适配依赖与独立测试，不带入本包以外的候选 |

先人工审阅这一个综合候选，不批量创建所有剩余候选 PR。包内保留可拆提交，维护者如要求再拆，不为扩大粒度加入新修复。公开差异保留必要源码、测试和来源，长调研与本机取证路径留在 fork；新分支推送和公开发布仍另行授权。
每项状态区分“开发已实现、独立验收、组合验收、PR准备/发布、上游合入”，不因本机通过或一次 squash 自动升级为已交付。
PR 目标是 `tc-mb/llama.cpp-omni`，`ggml-org/llama.cpp` 是补丁来源。提交前重新检查目标 base、已有 PR、适用 CI 与贡献规范；人类提交者负责审阅、解释和如实披露 AI 辅助，发布仍另行授权。

### LES 与算子的所有权交接

- R3 就是已完成的 A05，不再另做；R1/R2 分别对应未开展的 A01/B04。
- D02=O1/Metal O3 tile，D03=O2，D05包含 Metal O3 fusion，D01=O4：是同一候选的不同索引，不另计一次任务。E02/P3 与 LES T5.1 统一设计。
- 真实 ID/slot、context-owned bank、scheduler、动态 repack 失效与预算归 LES 负责人；context-owned 迁移已经完成，不重做。B05 的临时 RHS scratch 不等于 LES 持久专家缓存，B06 寻址不等于 D02 紧凑 tile。
- 公共测试文件由集成负责人串行收敛；组合失败先对照独立检查点和基线定位，再将修复交回对应源分支，不在两边分别实现。既有缺陷、回归、未验收组合分开记录，不以范围扩大掩盖失败。

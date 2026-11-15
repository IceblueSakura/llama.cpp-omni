# 上游算子修复分批执行记录

日期：2026-11-15。工作目录：`/home/sakura/workspace/llm/llama.cpp-omni-operators`，分支：`fix/upstream-operators`。

## 1. 依据与批次边界

- omni 基点：`350a835d5f8215145aa210a7df0f67e8877b5d6e`。
- 固定上游：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`，本地参考仓库 `/home/sakura/codespace/llama.cpp`；未 fetch。
- 候选及稳定 ID 见 [上游算子调研](upstream-operator-survey.zh.md)；没有直接上游方案的事项见 [扩展调研](upstream-operator-extensions.zh.md)，不混入本执行队列。
- 按同类契约、共享构建与设备分批移植和验收，不要求每项重建整个环境。每项保留来源、必要依赖和测试结果；批次不等于不可拆分的大提交。
- 只移植当前批次所需的修复、测试及最小前置依赖。不为套用上下文引入性能优化、公共 API 或整个后端更新。
- 不修改原 LES 工作区，不自动提交、推送或发布。合并到 master 前由用户审查并决定。

## 2. 推荐队列

后续批次是分组建议，进入批次时仍须核对提交依赖及目标设备，不能视为已验收的 cherry-pick 清单。

| 批次 | 候选范围 | 共享验证环境 / 交付边界 |
|---|---|---|
| L1：局部索引与 view 正确性 | A04 CPU `CONCAT`、A02 CUDA `CPY`、B01 CPU/CUDA `CLAMP` | 已验证并提交为 `d5ae271`；当前 Nix CUDA devShell，CPU 与 RTX 5060 |
| L2：CUDA 局部修补 | A05 MoE selection NaN、A09 IQ4_NL 短行、A11 ARGSORT CUB alias | 已移植并完成本机针对性验证，按用户授权纳入本批提交，见第 6 节。B08 量化 correction 未纳入 |
| L3：Vulkan 正确性 | B06 matmul stride、B05 FA 边界与 scratch；B03 Vulkan 部分按依赖加入 | 独立 Vulkan build，validation；覆盖 MM/MMV/ID 及 scratch 复用链。不混入 D02 紧凑 tile 重组 |
| M1：Metal 局部正确性 | A03 CPY stride、A06 TRI NaN/Inf、A07 SOLVE_TRI 尾读 | 待用户提供 macOS 环境，在真实 Metal 设备上验收；不以 Linux 编译/静态检查代替 |
| M2：Metal 配套修补 | A12 大偏移、B02 Metal transpose-conv、B03 Metal group norm、B09 FA | 真机内存/shape 与 shader validation；进入批次后按实际依赖拆成可验收子组，不能为了批量而混入整套 FA 重构 |
| 后续能力与性能批次 | C 类按 backend/算子族分组，再考虑 D 类 | 先证明 omni 图可达，再做数值与性能验证；不将新 kernel 存在写成模型收益 |
| 独立迁移设计 | E 类公共 API、状态布局和格式 | 不作为上述小修复批次的隐含依赖；需要明确模型需求和跨后端合同 |

macOS 的具体机器、架构、系统/Xcode 版本及访问方式尚未提供，M1/M2 目前不执行。CPU ARM ISA 专用修复须有目标 ISA/dispatch 证据，不能仅因 macOS 可用就视为覆盖。

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

基点为 L1 `d5ae271f09c3e5ed1140408b28b7564c355451ca`，复用上述 Nix 环境和 RTX 5060。状态：**已完成针对性验证，按用户授权提交至 fork 工作分支**；不推送或移动 master，提交号以 Git 历史为准。不需要额外 GGUF/权重模型；不扩展到 B08、Metal 或新模型能力。

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

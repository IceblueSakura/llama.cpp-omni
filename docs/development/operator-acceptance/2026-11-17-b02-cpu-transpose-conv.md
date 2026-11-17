# B02 CPU：多batch CONV_TRANSPOSE_2D（2026-11-17）

输入：`806841b1bdd713160e46fd7931319dfd42f01c58`。本次仅CPU切片，Metal继续硬件待办，B02整体标记部分完成。上游摘取、独立测试及验收有AI辅助；本地提交查询见[目录约定](README.md)，不更新PR worktree、LES或远端。

## 原始来源与最小范围

- 固定参考：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`，未fetch。
- 原始提交：[`a43c3986b44225c8633fe938464278d20dff0ec0`](https://github.com/ggml-org/llama.cpp/commit/a43c3986b44225c8633fe938464278d20dff0ec0)，#26132。该SHA属于llama.cpp内ggml源码历史。
- `ops.cpp`输入permutation增加batch循环和`nb13`寻址，计算增加batch循环、对应scratch偏移和输出`nb3`寻址。保留清零、barrier、按输出通道分配线程及原有F16/F32 dot-product。
- `ggml-cpu.c`输入scratch规划乘以`ne13`，与kernel同时交付。所需空间为`sizeof(kernel_type)*(Kw*Kh*Cout*Cin + W*H*Cin*N)`，另有既有线程cacheline余量；不通过扩大到输入backing extent分配来掩盖stride问题。
- 两处相关原始代码块与上游父版本一致，可原样摘取，无新增前置依赖。不移植来源中的Metal参数/grid/shader，不修改API、卷积padding语义、其他backend或omni自有transpose1D音频快路。

## 测试与决定性证据

沿用`test-backend-ops`，补上游多batch实例（F16/F32各1），另加派生oracle，不改原fixture的逐节点比较：

- 5种oracle参数各跑F16/F32，共10例。含batch=1/3、stride=1/2/3、非方形kernel、多个输入/输出通道、重叠累加和输出空隙。
- 两种padded输入view使用4字节非零offset，以及真实较大row/channel/batch stride；kernel保持连续，不宣称kernel任意view能力。
- 从实际存储输入和权重读取逻辑值，独立double标量scatter-add计算预期；不使用被测CPU permutation/dot-product，也不以CPU对CPU互相比作oracle。
- 输入为整数/8、权重为整数/16，均精确可表示为F16；不把既有F32输入→F16 scratch的舍入规则变化混入本修复。两backend结果各自比较独立预期，原sentinel字节比较保留。

| 检查 | 实测 |
|---|---|
| 旧CPU实现 | **12/18**，退出1；6个多batch oracle失败，后续batch全零。例如index80得到0而非0.1875。普通多batchCPU/reference比较仍通过，证明共同实现可能同错 |
| 修复后native CPU | **18/18**，退出0，无unsupported |
| 修复后ASan/UBSan | 同18/18，halt-on-error、leak detection，退出0，无报告；测试驱动enum例外见下文 |
| 旧planner/guard探针 | **4/8**；单batch可正确计算，多batch报告scratch不足，退出1，不声称旧kernel自身已发生动态越界 |
| 修复后planner/guard探针 | native、ASan/UBSan均**8/8**，F16/F32×线程1/4×batch1/3，每配置两次compute，退出0 |
| 共享ops.cpp的A10回归 | `test-backward-alias` CTest **1/1**，退出0 |

scratch探针调用公开`ggml_graph_plan/ggml_graph_compute`，断言计划线程数等于请求的1/4，严格按返回的`work_size`提供buffer，末端紧贴PROT_NONE。使用较大输入`[10,10,9,N]`、kernel`[3,3,5,9]`、stride2，所缺空间超过线程padding，防止小case假绿；两次改变输入、预填输出，另以按输出坐标gather的标量预期核对全部batch。仅留ignored探针，不新增测试框架。

例如N=3、1线程：F16旧`work_size=2674`不足所需6210，修复后6274；F32旧5284不足所需12420，修复后12484。4线程修复后分别6466/12676。以上空间是实际scratch需求，不是性能或模型内存收益。

## 环境、复验与限制

Ryzen9900X，固定项目`.#cuda`中的GCC15.3，CPU-only AVX2/FMA/F16C、native=OFF、RelWithDebInfo。复用本轮迁移后新建的`build-operators-b04/after-avx2`及`asan-avx2`并定向重编译；证据另存本批目录，不使用迁移前cache。

```bash
nix develop --no-update-lock-file .#cuda --command bash
B=build-operators-b04/after-avx2
cmake --build "$B" --target test-backend-ops -j 8
"$B/bin/test-backend-ops" test -b CPU -o CONV_TRANSPOSE_2D,CONV_TRANSPOSE_2D_ORACLE -j 1
# ASan/UBSan目录重复相同过滤：
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1
# 完整本地探针编译、1/4线程与复验命令：
bash build-operators-b02/evidence/verify.sh
```

`-j 1`控制测试调度，不表示CPU只有一个计算线程；线程1/4由独立probe显式验证。主测试工具继承[B04](2026-11-17-b04-x86-tiled-tail.md)的本地覆盖，仅对旧`test-backend-ops.cpp`禁用enum检测，避免过滤前无关upscale枚举报告；**CPU kernel、planner和scratch probe完整启用ASan/UBSan，包括enum**。本次没有改动该旧问题或提交覆盖脚本。

独立只读合同/测试审查完成，其关于双名称过滤、F16语义、nb13和scratch余量的建议已由实测核对。未测Metal、其他GPU、ARM、任意kernel stride、非F16精确输入舍入、模型端到端或性能；不需要下载模型。由于本次仅实现CPU，不能将B02候选标为全部完成。

Ignored证据：`build-operators-b02/evidence/{red,green,final-green,scratch-before}.log`、`scratch-{after-avx2,asan-avx2}.log`、`scratch-probe.cpp`、`verify.sh`、`red-source.b3`、`red-binaries.b3`、`final-source.b3`、`final-binaries.b3`、`probe-source.b3`、`final.patch`、`sanitizer-commands.json`和`jobs/`。manifest记录各验收时刻，后续增量构建后旧库hash不再匹配是预期行为。

未来PR可独立提取两个CPU生产文件及对应测试，保留planner与计算的配套性、独立oracle及来源；不要仅带多batch比较fixture却丢掉独立预期。Metal仍是同一候选的未实施部分，待目标条件可用时另验。

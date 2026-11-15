# 算子同步的扩展调研：本地适配与额外疑点

## 1. 与主清单的边界

日期：2026-11-15。主清单见 [可参考上游的算子调研](upstream-operator-survey.zh.md)。

**本文件不属于“从上游已合入修复开始”的执行队列。** 它只记录本次源码比较中额外识别的问题和适配决策：没有确认可直接使用的上游修复，或上游并不提供omni/LES所需的完整方案。

不把这些项目包装成上游已验证的增强，不与主清单的小修复捆绑实施。只有选中的上游候选确实依赖某项时，才先完成最小的适用性检查；检查失败可暂缓该候选，而不是自动扩大到自研框架。

### 版本与证据

- omni master：`350a835d5f8215145aa210a7df0f67e8877b5d6e`。
- LES已提交检查点：`714d80032296d92746d295c6e0fed3c798fb1df8`。另一会话继续修改该分支，下面的旧位置不证明最新工作区仍有同一问题。
- 固定上游：`61dbf49228c8a96b72e40c89d93cb7c003411f8c`。
- 下列项目均未实现、未运行复现；分别标注源码事实、推断及下一步，不将风险推断写成已发生故障。

## 2. 按影响面由窄到广排序

| ID | 项目 | 证据状态 | 影响面 / 难度 | 与上游主清单的关系 |
|---|---|---|---|---|
| X01 | CPU tiled支持检查的控制流可达性 | 上游源码疑点，调用不变量未完全确认 | 单优化入口；低至中 | 仅在选择D01时需要核对，不给上游实现盲目信任 |
| X02 | 新kernel是否被omni实际模型图使用 | 已确认若干图可达差异，收益未测 | 单模型/算子组合；中 | 决定C/D类是否值得做，不是新优化方案 |
| X03 | 真实expert ID与物理slot ID的使用边界 | LES检查点存在契约疑点；未复现模型故障 | MoE图、bias/LoRA；中高 | 由原LES任务收口，不假装主清单的kernel更新会顺便修好 |
| X04 | 持久device bank与scheduler副本地址 | 已确认生命周期差异 | runtime/storage/scheduler；高 | 影响未来cache正确性，不属于算子同步 |
| X05 | 动态slot与静态repack/tiled状态失效 | 已确认接口假设不匹配，实际错误未复现 | 存储布局、后端、预算；高 | 限制D08及移动端LES，需独立适配设计 |
| X06 | 新scratch与LES总预算、资源生存期 | 已确认新增内存来源，峰值及收益未测 | context、多个后端与性能验收；高 | 对D类做实际预算检查，不引入新的通用预算框架 |

此排序仅帮助管理待调查事项。当前优先做主文档A/B类已有上游修复，不因本表扩大首批实施范围。

## 3. 具体问题与最小判别方法

### X01：上游 CPU tiled 支持检查的控制流

**源码事实**：固定上游 `ggml/src/ggml-cpu/tiled/tiled.cpp:615–639` 的类型switch分支直接return，后续部分src1类型/连续性检查无法从这些分支到达。

**尚未确认**：调用方是否已保证这些约束，使不可达检查仅是冗余代码；是否有真实图能够传入不适用输入。不能直接定性为已复现的内存错误，也没有在本次调查中找到并确认专门修复它的上游提交。

**最小下一步**：只追 `ggml_tiled_supported()` 与dense/ID两个调用入口，核对source类型、layout和fallback条件；用现有backend fixture构造临界类型/stride。若上层不变量成立，只记录约束；若能复现错误，再单独提出修复，不暗中加入D01“上游移植”。

### X02：快路存在不等于实际模型可达

**已确认例子**：

- master `tools/omni/vision.cpp:865` 使用 `ggml_conv_2d()`；`ggml/src/ggml.c:4585–4605` 将其展开为IM2COL+MUL_MAT，因此新的direct CONV_2D kernel不会自动被调用。
- `tools/omni/voxcpm2/voxcpm2_audiovae.cpp:55` 起用zeros/concat实现left-pad，原因为旧Metal不支持左padding。新增Metal PAD能力后，原图不会自动简化。
- CPU tiled MMID的默认收益门槛是每专家至少8行，不能只看总batch；few-row dense matmul也不是专家MMID的同义词。

**推断**：可能出现单算子benchmark明显改善、真实omni路径零收益的情况。当前没有profile证据确定幅度。

**最小下一步**：记录真实图的opcode、shape/layout、backend与kernel dispatch，确认它进入选中的上游路径。若不命中，保留上游能力但不声明模型收益；改变构图或删除workaround需独立授权、数值对照与回退方案。它不是主清单自动包含的“顺手优化”。

### X03：真实 expert ID 与物理 slot ID

**检查点事实**：LES `714d800` 的 `src/llama-graph.cpp:1645–1648` 生成compute slot IDs；部分bias调用（`:1729,1753,1774,1861`）及LoRA入口（`:1133–1156`）仍复用同类ID。per-expert scale的部分路径则使用真实IDs。

**风险推断**：非恒等slot映射时，普通专家bank可正确计算，但保持原专家排列的bias/adapter可能取错项。是否可由当前支持模型/参数实际触发，尚未完整验证。已有LES交接将相关组合标为未验收。

**最小下一步**：先读取另一会话最新已确认的契约和改动，避免重复修复；用2–3个专家、刻意交换的slot映射和不同bias/LoRA值作独立oracle。真实ID用于router/bias/scale/adapter，计算slot只寻址对应bank。若需要代码修改，应留在原LES图合同任务，而不是以某个上游kernel补丁的名义提交。

### X04：scheduler dst 不能直接当持久cache身份

**检查点事实**：LES `714d800` 的 `src/llama-moe-storage.cpp:87–134` 向scheduler提供的消费者dst同步上传；这个dst是工作副本，不是独立持久device bank。上游cache的稳定bank/view归属不同。

**风险推断**：若以后只记录“该地址上次是某专家”，copy轮转、图重新reserve或buffer重分配会使映射失效。单纯同步上游scheduler复用优化不能成为数据正确性的保证。

**最小下一步**：由已有T2工作明确bank所有权、图绑定与失效条件；在实际copy轮转/重新reserve后测试驻留命中。不要在本算子分支新增cache元数据或另写一套资源管理。

### X05：动态填槽与静态 repack 的合同差异

**源码事实**：omni LES用 `preadv` 直接写原布局slot；CPU_REPACK的 `set_tensor` 在master `ggml/src/ggml-cpu/repack.cpp:4733–4743`、固定上游对应位置都要求 `offset=0` 且写整个tensor。上游没有现成的逐专家slot覆盖后自动局部repack协议。

Hexagon首次compute重排/已tiled标志同样不能直接当作动态更新保证。新量化kernel或batch-set接口的存在，不足以证明同一地址反复换专家正确。

**风险推断**：直接切换buffer类型可能绕过重排，或保留与旧数据绑定的状态；若加原始/计算两份bank又会增加预算。未在本次运行中复现。

**待决选项**：保持原布局路径；按slot重排；独立原始与计算bank并显式失效。后两者需要独立设计，不属于上游已经验证好的LES方案。

**最小下一步**：针对选定backend在同图、同地址连续换两个明显不同专家，逐次对照原布局CPU oracle；核对实际repack调用、完成边界、临时副本及对齐成本。先证明一个slice，不先建跨后端重排框架。

### X06：额外 scratch 的预算与收益归因

**来源事实**：CPU tiled约512KiB/线程的workspace；GPU紧凑tile描述；FA KV转换；融合/后端的对齐和临时buffer。它们与LES bank竞争RAM/VRAM，但不一定由现有cache参数直接覆盖。

**尚未确认**：目标workload实际峰值、与I/O/上传在途区的重叠、性能提升是否被缓存容量下降抵消。没有实测就不能称其为泄漏或预算违约。

**最小下一步**：固定总资源预算，记录新旧峰值、读盘/H2D/D2D、kernel和同步耗时，分冷/热、prefill/decode。若新scratch显著改变可用cache容量，先更新已有资源报告与配置约束；不要借此新增无消费者的统一预算层。

## 4. 不纳入本轮扩展实施的事项

- 不为上述疑点自动修改正在推进的LES分支。
- 不把未找到专门修复提交误写成“上游没有解决”；这里只描述本次检索范围和固定版本。
- 不因缺硬件安排Metal/HTP仿真来冒充性能验收。
- 不把性能profiling发现的潜在热点当作已批准的新优化。
- 后续若找到明确上游修复并验证适用性，可将对应项目迁入主清单，保留来源及依赖；否则继续作为独立提案。

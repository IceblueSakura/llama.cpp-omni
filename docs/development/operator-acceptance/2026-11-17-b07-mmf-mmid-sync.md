# B07：CUDA MMF/MMID warp同步（2026-11-17）

本地输入：`81502de6fca2dff7e896f7c7d93328e204071f39`。本提交只抽取CUDA同步正确性修复；实现摘取、调查和验收整理有AI辅助。提交对应关系用[本地约定](README.md)中的`git log --diff-filter=A`查询，未更新PR worktree或远端。

## 原始来源与最小依赖

- 仓库：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，固定参考 `61dbf49228c8a96b72e40c89d93cb7c003411f8c`，未fetch。下述SHA是该仓库内ggml源码的提交，不是独立ggml仓库SHA。
- 原始提交：[`73a43d1f69345aee8bb186ef4b3172cef892f2e5`](https://github.com/ggml-org/llama.cpp/commit/73a43d1f69345aee8bb186ef4b3172cef892f2e5)，#28475。
- 仅三文件、26行新增：`common.cuh`引入原始`ggml_cuda_syncwarp`；`mmf.cuh`在slot_map初始化、共享tile读取/复用及单warp结果暂存边界同步；`mmid.cu`在共享store写入与后续读取间同步。
- 上游同提交的`fattn-vec.cuh`只是替换已有条件式syncwarp，不是本修复依赖，未抽取。保留本地MMID结构，不引入上游write_inverse、padding泛化或新helper实例。
- 不修改kernel调度、矩阵布局、数值算法或公共API。新增barrier与上游对应位置一致；每个warp执行相同循环/条件，既有多warp的`__syncthreads`保留。

## 验证与实际路径

使用已有`test-backend-ops`，不新增重复fixture：

1. F16/F32普通MMF：`m=64,n=2,k=64,bs=[8,1],nr=[4,1]`，1例，命中单warp `mul_mat_f<half2,...,nwarps=1,has_ids=false>`。
2. ID：F32/F16权重、F32 RHS、4专家、used=1/2、`m=512,k=256,n=1/17`，8例。小token命中带slot_map的MMF，n=17实际调用通用/专用`mm_ids_helper<0>/<2>`及float/half2的`mul_mat_f_ids`。

使用NVIDIA RTX 5060、CUDA12.9.86、compute-sanitizer2026.3。固定项目`.#cuda`；复用本轮迁移后新建的`build-operators-b08`，保持sm120/120a配置，增加`-lineinfo`后重编译受影响target。没有使用迁移前cache。

红灯使用本轮已验证的同路径`build-operators-b03-cuda`产物，对应B03 `bdbb3fb79404c5752b30e2cd9b4ef36589d8d964`生产代码/测试。B03到本提交输入之间，仅B08修改量化器、注释及独立测试；本修复三文件和这9例所在`test-backend-ops.cpp`均无差异，因此可以归因MMF/MMID旧实现，不能将红灯产物称为包含B08的版本。

| 检查 | 实测 |
|---|---|
| 红灯数值 | 普通1/1、ID8/8均通过；数值通过未检出race |
| 红灯racecheck，无dump | 普通15项warning、ID59项warning，各退出99；明确报告目标MMF、MMID helper共享内存读写race |
| 绿灯数值 | 普通1/1、ID8/8，退出0，无unsupported |
| 绿灯racecheck，无dump | 同9例，均`0 hazards (0 errors, 0 warnings)`，退出0 |
| 绿灯synccheck，无dump | 同9例，均`ERROR SUMMARY: 0 errors`，退出0 |
| 共用header回归 | `test-cuda-quantize` CTest 1/1，内含B08的28个格式用例，退出0 |

初次red带`--dump-kernel-launches`显示19/71项，其中含launch调试报告；最终红绿比较均去掉dump，以15/59个真实race warning对比0。launch日志只证明实际kernel/几何，不能用其调试报告计数冒充内存检测。该区别与B08消费者trace的诊断一致。

## 复验

先按[B08文档](2026-11-17-b08-q8-correction.md)配置相同构建，随后：

```bash
nix develop --no-update-lock-file .#cuda --command bash
cmake -S . -B build-operators-b08 -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build-operators-b08 --target test-backend-ops test-cuda-quantize -j 6
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 OMP_NUM_THREADS=4
B=build-operators-b08/bin/test-backend-ops
P='^type_a=f16,type_b=f32,m=64,n=2,k=64,bs=\[8,1\],nr=\[4,1\],per=\[0,1,2,3\],k_v=0,o=1,m_v=0,pad=0$'
I='^type_a=(f32|f16),type_b=f32,n_mats=4,n_used=(1|2),b=0,m=512,n=(1|17),k=256,m_v=0$'
"$B" test -b CUDA0 -o MUL_MAT -p "$P" -j 1
"$B" test -b CUDA0 -o MUL_MAT_ID -p "$I" -j 1
compute-sanitizer --tool racecheck --error-exitcode 99 \
  --racecheck-detect-level info --racecheck-report analysis \
  "$B" test -b CUDA0 -o MUL_MAT_ID -p "$I" -j 1
# 对MUL_MAT/P重复racecheck；两个过滤分别运行 --tool synccheck --error-exitcode 99。
ctest --test-dir build-operators-b08 -R '^test-cuda-quantize$' --output-on-failure
```

Ignored证据：`build-operators-b08/evidence/b07/`中的`red-nodump-*.log`、`green-{plain,racecheck,synccheck}-*.log`、`green-summary.log`、`b07-red-identity.b3`、`red-input-equivalence.txt`及最终`final-{source,binaries}.b3`、`final.patch`、`jobs/`。父目录`b07-green.sh`保存完整本地执行命令。B08与B07共用构建树，历史manifest记录各验收时刻身份，不要求后续增量构建后的二进制仍匹配旧manifest。

## 限制与PR拆分建议

- BF16数值、compact单warp epilogue、其他helper专用实例、CUDA Graph capture/replay和性能未验收；本次关闭graphs以定位目标kernel。普通小矩阵已有红绿，不以全套case数量替代路径证明。
- `b684498579899ec376ea13ab4bb3d0a9ef2c1a5b`的真实wave64资格判断及`9145ddb9556c1cfa568f3189eb7be44317826e31`的HIP BF16 WMMA类型修复暂跳过，缺少目标HIP验收；NVIDIA warp32不冒充wave64验证。因此B07只标记部分完成。
- 未来PR可独立提取这三个生产文件的同步修复，保留完整来源和racecheck/synccheck证据；无需把B08量化修改合并为同一个提交，也无需新增相同矩阵fixture。公开提交组织由用户决定。

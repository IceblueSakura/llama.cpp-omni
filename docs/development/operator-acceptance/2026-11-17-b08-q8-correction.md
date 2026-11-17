# B08：CUDA Q8_1 correction sum（2026-11-17）

本地输入：`bdbb3fb79404c5752b30e2cd9b4ef36589d8d964`。本提交的实现摘取、测试和验收文档有AI辅助；仅本地维护，不更新PR worktree或远端。提交对应关系用本目录[约定](README.md)中的`git log --diff-filter=A`查询。

## 原始来源与范围

- 仓库：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，参考固定在 `61dbf49228c8a96b72e40c89d93cb7c003411f8c`；本轮没有fetch。
- 原始提交：[`8fc0376721e9662d498a5706d7ee8e7574ce99f4`](https://github.com/ggml-org/llama.cpp/commit/8fc0376721e9662d498a5706d7ee8e7574ce99f4)，#31247。SHA属于llama.cpp仓库内ggml源码，不是独立ggml仓库SHA。
- 普通`quantize_q8_1`和MMQ的DS4/D2S6：先归约量化整数，再乘尚未转F16的FP32 scale，最后按格式存储；D4没有correction字段。
- 保留本地gather、128值转置布局和API。不带入来源上下文已有的scatter、后续few-row MMQ或其他量化重排；现有整数warp归约helper已满足依赖。
- 本地适配仅同步更正`block_q8_1_mmq`旧注释。新增`test-cuda-quantize`使用既有native CUDA测试门控，不增加测试框架或公共导出；测试/CMake不用于Windows shared或dynamic backend。

## 验证契约

新增直接CUDA测试28例：宽32/132/512、输出padding到512、两个channel、三行、真实row stride、零行、正负输入、gather重排、DS4/D2S6/D4三布局。输出预填0xa5，避免漏写零值假绿。

Host从原输入独立计算scale和payload，按显式预期layout解释输出；correction由host预期整数求和，不依赖GPU返回的qs。补一组不同32/64值scale、不同16值signed sum和非F16精确scale的数据，分别检验`half(d*sum(q))`、不能提前舍入d，以及D2S6的六个partial sums。F16字段按存储精度严格比较，D4的FP32 scale允许2e-7相对误差。无模型/权重依赖。

## 实测结果

设备：NVIDIA RTX 5060，SM120；项目固定`.#cuda`、`CMAKE_CUDA_ARCHITECTURES=120`（后端选择120a）、RelWithDebInfo、`GGML_CUDA_FA=OFF`。新建迁移后构建树`build-operators-b08`。

| 检查 | 结果 |
|---|---|
| 修复前最终直接测试 | 8/28，20例correction不一致，退出1；D4八例仍通过。例：普通/DS4得到55.9062而非63.5；D2S6得到59.8125而非63.5 |
| 修复后直接测试 | 28/28，退出0；无skip |
| 原生CUDA CTest | `test-cuda-quantize`与既有`test-cuda-convert`，2/2，无skip |
| 直接量化memcheck | 28/28，`ERROR SUMMARY: 0 errors`，退出0 |
| 最终MUL_MAT消费者 | 11/11次执行，对应6种唯一参数（3类型×n=1/9），无unsupported；既有注册有重复实例，不将11解释为11种shape |
| 最终MUL_MAT_ID消费者 | 10/10次执行，对应6种唯一参数（3类型×n=1/32），无unsupported |
| Q4_1消费者memcheck | 普通/ID各4/4次执行，均`ERROR SUMMARY: 0 errors`，退出0 |

独立launch日志确认`quantize_q8_1 → mul_mat_vec_q`及`quantize_mmq_q8_1 → mul_mat_q/stream_k_fixup`，ID多token还实际执行`mm_ids_helper<2>`。`--dump-kernel-launches`调试模式会将每个launch报告计入ERROR SUMMARY（普通9、ID12）；首轮设置`--error-exitcode 99`因此退出99，不能当作干净memcheck。去掉dump后相同消费者各4/4且0 errors，实际检测结果仅采用无dump日志；原始诊断保留，不抹去失败状态。

两个独立只读调查完成，生产最小依赖和测试盲点已核对，测试按上述精度/分组/独立预期建议补强。实际模型、性能、其他GPU架构及HIP未验收；本地gate未新增Windows/DL能力。不要据此宣称模型精度或吞吐收益。

## 复验命令

```bash
nix develop --no-update-lock-file .#cuda --command bash
cmake -S . -B build-operators-b08 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA=OFF \
  -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build build-operators-b08 --target test-cuda-quantize test-cuda-convert test-backend-ops -j 6
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ctest --test-dir build-operators-b08 -R '^test-cuda-(quantize|convert)$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build-operators-b08/bin/test-cuda-quantize
B=build-operators-b08/bin/test-backend-ops
"$B" test -b CUDA0 -o MUL_MAT -j 1 \
  -p '^type_a=(q4_1|q2_K|q8_0),type_b=f32,m=16,n=(1|9),k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,m_v=0,pad=0$'
"$B" test -b CUDA0 -o MUL_MAT_ID -j 1 \
  -p '^type_a=(q4_1|q2_K|q8_0),type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=(1|32),k=256,m_v=0$'
```

Ignored证据：`build-operators-b08/evidence/red.log`为初版21例，`red-final.log`为最终28例红灯，`green.log`为构建、CTest、memcheck和最终消费者结果；`consumer-launches.log`、`consumer-id-launches.log`仅用于launch身份，`consumer-memcheck.log`才是无dump的消费者memcheck；`red-source.b3`、`final-source.b3`、`final-binaries.b3`及`final.patch`绑定输入/产物，`jobs/`保存原生job元数据。

## 未来PR粒度

这是一个Q8_1中间格式一致性修复：普通量化器与MMQ两个带sum布局应共同交付，保留直接格式测试；D4作为无correction回归。可以从本提交提取`quantize.cu`、注释和native测试/CMake，排除本地队列/验收文档。不要拆掉MMQ一半或仅用普通数值通过替代格式红绿。

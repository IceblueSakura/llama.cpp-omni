# A10：CPU反向算子exact-src alias正确性（2026-11-17）

输入：`959a23dfe41db4523b51fffa7bcdba16fc34b728`。B04收口并提交后才进入本模块；只处理两个相同alias合同的局部上游修复，不扩展到其他反向算子、核心API或训练框架。上游摘取、测试及验收有AI辅助，仅本地提交，提交查询见[约定](README.md)。

## 原始来源与最小修改

固定参考：`ggml-org/llama.cpp@61dbf49228c8a96b72e40c89d93cb7c003411f8c`，未fetch。原始SHA均属于llama.cpp内ggml源码历史：

| 上游提交 | 内容 |
|---|---|
| [`fd3271e0b4d8422ca1d16a850edb4222f7392013`](https://github.com/ggml-org/llama.cpp/commit/fd3271e0b4d8422ca1d16a850edb4222f7392013)，#24305 | RMS_NORM_BACK完成两个归约后逐元素同时读取dz/x再写dx，避免先copy覆盖alias的梯度 |
| [`dd4c286f3809e73be087f374d9d865ca97c205e4`](https://github.com/ggml-org/llama.cpp/commit/dd4c286f3809e73be087f374d9d865ca97c205e4)，#27096 | SOFT_MAX_BACK先完成dot，再逐元素读取dy/y写dx，避免src1被copy覆盖 |

生产代码仅`ggml/src/ggml-cpu/ops.cpp`两处原样摘取，保持归约、eps/scale、stride限制及原有支持判定。既有gallocr明确允许这两个op复用输入；不通过关闭复用掩盖错误，不增加scratch。固定上游相关历史未发现另需带入的alias后续修复。

## 测试责任与红绿

现有`test-backend-ops`按context分配tensor，不会可靠触发自动输入复用；仅拿CPU作reference也可能同错。因此新增单一`test-backward-alias`，直接使用公开CPU backend、现有gallocr及RAII，无新框架：

- OUTPUT标志保护另一输入，强制且断言输出**恰好复用src0/src1**；两个都保护则断言非alias控制。
- 两op × 宽7/33 × 三布局（非alias/src0/src1）× 线程1/4 = **24配置**。每配置含六行（3×2 batch）且改变并重新上传输入计算两次，共48次compute，不把次数写成48种配置。
- 独立host double梯度预期：RMS对dz/x计算解析式，softmax对按行归一化的y计算Jacobian-vector乘积并含scale。预期不读取被覆盖的device输入，不依赖CPU reference。
- eps及scale随宽度覆盖不同数值；检查有限值和绝对/相对误差。只有exact同布局alias在范围内，不声称partial-overlap view支持。
- CMake仅在native CPU可链接target存在、非backend-DL时注册。测试调用公开CPU API，不因其他GPU存在而改用它。

| 检查 | 结果 |
|---|---|
| 旧实现AVX2 | **16/24配置**；RMS的src0 alias四配置、softmax的src1 alias四配置失败，退出1。其余alias方向和非alias控制均通过，地址断言通过 |
| 修复后AVX2 | **24/24配置、48次compute**，退出0 |
| 修复后AVX512 | 同24/24，退出0 |
| 修复后ASan/UBSan AVX2 | 同24/24，halt-on-error及leak detection，退出0，无报告 |
| 正式CTest | 上述三个构建各1/1，无skip；各自额外重复同24配置 |
| 原RMS_NORM_BACK回归 | **10/10**，含多eps、宽64/1025、4D输入 |
| 原SOFT_MAX_BACK小尺寸回归 | **6/6**，两scale、奇偶宽和batch，max_bias=0 |
| 同文件FA前一模块回归 | 常数oracle **7/7** |

红灯例：RMS width7/src0首元素得到0.999997而非-0.833333；softmax width7/src1得到0.125而非-0.00892857。两次重新上传后的compute均复现，不依赖allocator偶然选择。

设备与环境沿用[B04](2026-11-17-b04-x86-tiled-tail.md)：Ryzen9900X、固定Nix`.#cuda`中的GCC15.3，CPU-only AVX2/AVX512、native=OFF、RelWithDebInfo。本轮同路径配置可复用，定向增量编译受影响target，不用迁移前cache。**新alias测试完整启用ASan/UBSan，包括enum**；B04遗留的本地enum排除仅作用于旧`test-backend-ops.cpp`，不作用于本测试或CPU kernel。

## 复验与证据

先按B04的CPU-only ISA配置建立构建目录；以下在本worktree执行，不需要模型：

```bash
nix develop --no-update-lock-file .#cuda --command bash
B=build-operators-b04/after-avx2
cmake -S . -B "$B"
cmake --build "$B" --target test-backward-alias test-backend-ops -j 8
"$B/bin/test-backward-alias"
ctest --test-dir "$B" -R '^test-backward-alias$' --output-on-failure
"$B/bin/test-backend-ops" test -b CPU -o RMS_NORM_BACK -j 1
"$B/bin/test-backend-ops" test -b CPU -o SOFT_MAX_BACK \
  -p '^type=f32,ne=\[(16,16,1,1|15,15,1,1|16,16,2,3)\],scale=(1.000000|0.100000),max_bias=0.000000$' -j 1
# ASan/UBSan目录重复alias target，设置：
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1
```

Ignored证据在`build-operators-a10/evidence/`：`red.log`、`green.log`、`verify.sh`、`red-source.b3`、`red-binaries.b3`、`final-source.b3`、`final-binaries.b3`、`final.patch`及`jobs/`。历史manifest对应各采集时刻，后续增量构建会改变同路径库，不要求旧manifest匹配新库。

独立只读测试审查完成，无阻塞；其静态判断由上述实际alias和数值红绿复核。未测Windows、ARM、backend-DL执行、其他GPU或训练模型；未测性能、不宣称推理加速。未来PR可将两处生产修复按上游提交拆分，保留能强制真实alias的共同回归测试；本地以一个同类CPU模块交付。

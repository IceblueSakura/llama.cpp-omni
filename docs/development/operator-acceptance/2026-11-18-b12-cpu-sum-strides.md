# B12 CPU 切片：F32 SUM 的非连续输入（2026-11-18）

- 本地输入：`d9ab965a0b21322d8cf1f3af37591e967ce01087`。
- 原始来源：`ggml-org/llama.cpp` [0878f7c304cc13915c831ae17694534c94940176](https://github.com/ggml-org/llama.cpp/commit/0878f7c304cc13915c831ae17694534c94940176)（#28851），在固定参考 `61dbf49228c8a96b72e40c89d93cb7c003411f8c` 内；无额外依赖。
- 原样摘取 `ggml-cpu/ops.cpp` 的 F32 `SUM` hunk：保留行连续向量归约，在 `nb[0] != sizeof(float)` 时按实际步长读。`ggml_sum()`、F16/BF16、线程调度、其他 CPU 算子及所有 GPU 门控不变。
- 本地测试复用 `test_sum` 的图构建：四维 packed permutation 的 backing 存 `1..990`，实际结果和 reference 都必须等于独立算术级数 `490545`。带空洞的输入另用 CPU 探针验证，不把该用例无条件注册给尚未修正门控的 GPU。上游摘取、回归与验收有 AI 辅助。

## 红绿与复验

Ryzen 9 9900X，项目 `.#cuda` devShell（GCC 15.3），复用当前工作区 `build-operators-b04/asan-avx2` 的 CPU-only AVX2 RelWithDebInfo ASan/UBSan 配置。沿用测试驱动的 enum 排除，不排除生产内核或探针。实际编译含 `NDEBUG`，旧实现静默返回错值。

| 检查 | 旧实现 | 最终实现 |
|---|---|---|
| CPU `test-backend-ops -o SUM` | 8/9，退出 1；四维 oracle 实际 `470745`（两个 CPU 实现同错） | 9/9，退出 0，无 unsupported |
| packed transpose，独立精确值 | `24`，期望 `36`，退出 1 | `36`，退出 0 |
| 非零 offset、行间空洞再 transpose | `31`，期望 `21`，退出 1 | `21`，退出 0 |
| 同一有空洞 view 不转置（行连续控制） | `21`，退出 0 | `21`，退出 0 |

以上运行日志无 ASan/UBSan 报告；不以同 CPU reference 的一致性替代独立预期。

```bash
nix develop --no-update-lock-file .#cuda --command bash
export UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
B=$PWD/build-operators-b04/asan-avx2
E=$PWD/build-operators-b12/evidence
cmake --build "$B" --target test-backend-ops -j 10
"$B/bin/test-backend-ops" test -b CPU -o SUM -j 1
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iggml/include \
  "$E/sum-probe.cpp" -L"$B/bin" -Wl,-rpath,"$B/bin" -lggml -lggml-cpu -lggml-base -o "$E/sum-probe"
for shape in gapped-rows packed gapped-transpose; do "$E/sum-probe" "$shape"; done
```

本地日志与 manifest 在 `build-operators-b12/evidence/`。B12 仅部分完成：CUDA/Vulkan/WebGPU `SUM` 门控、其他类型、`SCALE`、unary/GLU、额外 backward 布局和性能不在本次验收范围。未来 PR 独立提取 CPU 内核及对应维护测试；不随带其他后端能力。

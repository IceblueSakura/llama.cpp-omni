# B11 CUDA 切片：PAD 的 y/z grid 上限（2026-11-18）

- 本地输入：`4072918e07e73874a3a071795e9febf213733d7e`。
- 原始来源：`ggml-org/llama.cpp` [c35b66744f13cb0dcc476af063e112122eee9355](https://github.com/ggml-org/llama.cpp/commit/c35b66744f13cb0dcc476af063e112122eee9355)（#30147），在固定参考 `61dbf49228c8a96b72e40c89d93cb7c003411f8c` 内；无额外依赖。
- 原样摘取 `ggml-cuda/pad.cu`：`grid.y/z` 各截到 65535，block 按实际 grid 步长遍历输出行和压平的切片索引。索引还原、零填充和循环填充语义不变；不改 shape 合同、能力门控或其他 kernel。
- 上游三例复用本地既有 `test_pad` / `test_pad_ext`；只按本地 `test_pad_ext` 构造器补 `tfrm=0`（不转换输入）。上游摘取、测试适配与验收有 AI 辅助。

## 红绿与复验

RTX 5060（compute capability 12.0，driver 615.78.08），项目 `.#cuda` devShell（CUDA 12.9.86），复用当前工作区 `build-operators-b03-cuda`（RelWithDebInfo，sm120，CUDA FA 关闭，静态注册后端）。红绿用同一配置，运行时关闭 CUDA Graphs。测试驱动固定 `-b CUDA0`，直接比较实际 GPU 输出与 CPU reference。

| 检查 | 旧实现 | 最终实现 |
|---|---|---|
| 窄张量 `grid.y`：70001 输出行，普通及 circular 两例 | 各报 `invalid configuration argument`，退出 134 | 2/2 |
| 窄张量 `grid.z`：`300*300=90000` 输出切片 | 同样报错，退出 134 | 1/1 |
| CPU 对照（上述三例） | 3/3，退出 0 | 用作绿测的数值 reference |
| CUDA `PAD` 定向组 | — | 31/31，退出 0；含三个边界例和普通尺寸/视图回归，无 unsupported |
| Compute Sanitizer 2026.3 `memcheck`（三条边界用例） | — | 3/3，退出 0，`ERROR SUMMARY: 0 errors` |

```bash
nix develop --no-update-lock-file .#cuda --command bash
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 OMP_NUM_THREADS=4
B=build-operators-b03-cuda
cmake --build "$B" --target test-backend-ops -j 8
"$B/bin/test-backend-ops" test -b CUDA0 -o PAD -j 1
# 三条边界的精确选择式和 memcheck 命令见 check-pad.sh：
PAD_EVIDENCE_DIR="$PWD/build-operators-b11/evidence/recheck" \
  bash build-operators-b11/evidence/check-pad.sh green
```

Ignored 证据在 `build-operators-b11/evidence/`：`red-summary.log`、三份 CUDA 错误日志、`red-cpu-oracle.log`、`final/green-pad.log`、`final/green-memcheck.log`、红绿 manifest 和 job 记录。保留原始带颜色日志；最终结果检查先规范化 ANSI 色码。

B11 仅部分完成：未验 CUDA Graph capture/replay、HIP、其他设备或性能；不带 HIP reflect、CUDA norm/RoPE、Vulkan norm 等切片，也不把本改动视为任意大张量的索引溢出修复。未来 PR 可独立提取本内核及三例。

#!/usr/bin/env bash
# Opt-in Linux/Vulkan LES correctness smoke for MiniCPM-V 4.7 35B Q4_K_M.
set -euo pipefail
ulimit -c 0
usage() {
    printf '%s\n' \
        'usage: bash run.sh BUILD_DIR NATIVE_GGUF LES_GGUF [smoke|smoke-no-eval|faults|bounded]' \
        '       bash run.sh BUILD_DIR LES_GGUF faults|bounded|pinned-staging|decode-l1' >&2
    exit 2
}
if (( $# < 3 || $# > 4 )); then usage; fi
native=
case "$3" in
    faults|bounded|pinned-staging|decode-l1)
        if (( $# != 3 )); then usage; fi
        les=$(realpath -e -- "$2")
        mode=$3
        ;;
    *)
        mode=${4:-smoke}
        case "$mode" in
            smoke|smoke-no-eval|faults|bounded) ;;
            *) echo "unknown mode: $mode" >&2; usage ;;
        esac
        native=$(realpath -e -- "$2")
        les=$(realpath -e -- "$3")
        ;;
esac
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
bin=$(realpath -e -- "$1/bin")
if [[ ! -r $bin/libllama.so || ! -f $les || ( -n $native && ! -f $native ) ]]; then
    echo 'a shared llama build and the requested GGUF files are required' >&2
    exit 2
fi
mkdir -p "$root/build-les-d3-data"
run_dir=$(mktemp -d "$root/build-les-d3-data/verify.XXXXXX")
printf 'artifacts=%s\n' "$run_dir"

# Keep the executable and both runners on the requested build. The caller
# supplies the ICD and driver library path appropriate for the machine.
export LD_LIBRARY_PATH=$bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_OP_OFFLOAD_MIN_BATCH=1
export LLAMA_MOE_DECODE_WINDOW=0 LLAMA_MOE_LOOKAHEAD_PREFETCH=0 LLAMA_MOE_MATMUL_SPLIT=0 LLAMA_MOE_IO_THREADS=2
export LLAMA_MOE_DUMMY_READ=0 LLAMA_MOE_SKIP_READ=0 LLAMA_MOE_PREWARM= LLAMA_MOE_LAYER_WINDOWS=
unset GGML_VK_DISABLE_FUSION GGML_VK_DISABLE_COOPMAT2 GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR
unset LLAMA_MOE_VK_PINNED_STAGING LLAMA_MOE_VK_L1_SLOTS LLAMA_MOE_VK_L1_TRACE
c++ -std=c++17 -O2 -Wall -Wextra -I"$root/include" -I"$root/ggml/include" "$here/probe.cpp" \
    -L"$bin" -Wl,-rpath,"$bin" -lllama -lggml -lggml-base -lggml-cpu -ldl -o "$run_dir/probe"

run_probe() {
    local name=$1; shift
    local log=$run_dir/$name.log
    echo "=== $name ==="
    timeout 120 "$@" >"$log" 2>&1 || { tail -60 "$log"; return 1; }
    grep -E 'storage: mode=|backend_uploaded=|Vulkan (single slot|pinned staging|L1 summary)|^read_bytes_inference=|^step=|^fault=|^retry_rc=' "$log"
}
verify_stream() {
    local log=$1
    grep -Fq 'storage: mode=vulkan-single-slot bank=0.00 MiB' "$log"
    grep -Fq 'host/shared=0.00 MiB backend_uploaded=0.00 MiB' "$log"
    grep -Fq 'Vulkan single slot: installs=2560 ' "$log"
}

case "$mode" in
    pinned-staging|decode-l1)
        run_probe staged "$run_dir/probe" "$les" 16 "$run_dir/staged.logits" off plain
        case "$mode" in
            pinned-staging)
                opt=LLAMA_MOE_VK_PINNED_STAGING=1
                active='Vulkan pinned staging: [1-9][0-9]*\.[0-9]+ MiB pinned'
                ;;
            decode-l1)
                opt=LLAMA_MOE_VK_L1_SLOTS=32
                active='Vulkan L1 summary .*payload=[1-9][0-9]*'
                ;;
        esac
        run_probe "$mode" env "$opt" "$run_dir/probe" "$les" 16 "$run_dir/$mode.logits" off plain
        if grep -Eq "$active" "$run_dir/staged.log" || ! grep -Eq "$active" "$run_dir/$mode.log"; then
            echo "$mode did not activate only in the opt-in run" >&2
            exit 1
        fi
        cmp "$run_dir/staged.logits" "$run_dir/$mode.logits"
        echo "$mode/staged logits: byte identical"
        ;;
    faults)
        cc -std=c11 -O2 -Wall -Wextra -shared -fPIC "$here/fault.c" -ldl -o "$run_dir/fault.so"
        for fault in fail abort prefill-fail; do
            run_probe "$fault" env LD_PRELOAD="$run_dir/fault.so" "$run_dir/probe" "$les" 16 "$run_dir/$fault.logits" "$fault" eval
        done
        ;;
    bounded)
        # No reference from a different run or a previous build is accepted.
        run_probe warm "$run_dir/probe" "$les" 16 "$run_dir/warm.logits" on plain
        verify_stream "$run_dir/warm.log"
        systemd-run --user --scope --quiet -p MemoryMax=2G -p MemorySwapMax=0 -- \
            bash "$here/scope.sh" "$run_dir/probe" "$les" "$run_dir"
        verify_stream "$run_dir/bounded.log"
        cmp "$run_dir/warm.logits" "$run_dir/bounded.logits"
        echo 'bounded/warm logits: byte identical'
        ;;
    smoke|smoke-no-eval)
        observation=eval
        if [[ $mode == smoke-no-eval ]]; then observation=plain; fi
        run_probe native "$run_dir/probe" "$native" 0 "$run_dir/native.logits" off "$observation"
        run_probe staged "$run_dir/probe" "$les" 16 "$run_dir/staged.logits" off "$observation"
        run_probe single "$run_dir/probe" "$les" 16 "$run_dir/single.logits" on "$observation"
        verify_stream "$run_dir/single.log"
        python3 - "$run_dir" <<'PY'
from array import array
from math import isfinite
from pathlib import Path
import sys

out = Path(sys.argv[1])
names = ('native', 'staged', 'single')
manifests = {next(line for line in (out / f'{name}.log').read_text().splitlines()
                  if line.startswith('tokens=')) for name in names}
assert len(manifests) == 1, manifests
n_vocab = int(manifests.pop().split(' n_vocab=')[1])
for ref, test in (('native', 'staged'), ('native', 'single'), ('staged', 'single')):
    with (out / f'{ref}.logits').open('rb') as a, (out / f'{test}.logits').open('rb') as b:
        for step in range(6):
            x, y = array('f'), array('f')
            x.fromfile(a, n_vocab)
            y.fromfile(b, n_vocab)
            assert all(map(isfinite, x)) and all(map(isfinite, y)), (test, step)
            error = sum((p-q)**2 for p, q in zip(x, y)) / max(sum(p*p for p in x), 1e-30)
            top_equal = max(range(n_vocab), key=x.__getitem__) == max(range(n_vocab), key=y.__getitem__)
            print(f'{ref}/{test} step={step} NMSE={error:.8g} top_equal={top_equal} bitwise={x.tobytes() == y.tobytes()}', flush=True)
            assert error < 1e-4 and top_equal, (ref, test, step)
        assert not a.read(1) and not b.read(1), 'extra logits'
PY
        ;;
esac

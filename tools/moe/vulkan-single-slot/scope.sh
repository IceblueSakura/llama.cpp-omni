#!/usr/bin/env bash
# Called inside the private user scope; the warm reference runs outside it.
set -euo pipefail
probe=$1; model=$2; out=$3
c=/sys/fs/cgroup$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup)
[[ $(<"$c/memory.max") == 2147483648 && $(<"$c/memory.swap.max") == 0 ]]
snapshot() {
    for f in memory.current memory.peak memory.swap.current memory.events memory.stat io.stat; do
        printf '\n=== %s ===\n' "$f"
        head -c 12000 "$c/$f"
    done >"$out/$1.txt"
}
snapshot before
rc=0
timeout 120 "$probe" "$model" 16 "$out/bounded.logits" on plain >"$out/bounded.log" 2>&1 || rc=$?
snapshot after
if (( rc )); then tail -60 "$out/bounded.log"; exit "$rc"; fi
awk '$1 == "oom" || $1 == "oom_kill" || $1 == "oom_group_kill" { if ($2 > 0) exit 1 }' "$c/memory.events"
[[ $(<"$c/memory.swap.current") == 0 ]]
grep -E '^read_bytes_|^peak_rss' "$out/bounded.log"
printf 'memory_max=%s memory_peak=%s swap_current=%s\n' "$(<"$c/memory.max")" "$(<"$c/memory.peak")" "$(<"$c/memory.swap.current")"
grep -E '^(max|oom|oom_kill) ' "$c/memory.events"

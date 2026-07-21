#!/usr/bin/env bash
set -euo pipefail

readonly output_root=${1:-$PWD/.work/private-export-table-probe}
[[ $# -le 1 ]] || { printf 'usage: %s [OUTPUT_DIR]\n' "$0" >&2; exit 2; }

make -C qemu_integration/guest_libcuda private-export-table-probe
qemu_integration/guest_libcuda/run_private_export_probe.sh \
  --mode discovery \
  --output-dir "$output_root/discovery" \
  -- qemu_integration/guest_libcuda/cuda-runtime-dlopen-kernel-probe \
  "$PWD/qemu_integration/guest_libcuda/libtiny_cuda.so"
qemu_integration/guest_libcuda/run_private_export_probe.sh \
  --mode capture \
  --output-dir "$output_root/callback-hooks-slot1" \
  --uuid a094798c-2e74-2e74-93f2-0800200c0a66 \
  --slot 1 \
  --selector 0x111 \
  --memory rsi:64 \
  -- qemu_integration/guest_libcuda/cuda-runtime-dlopen-kernel-probe \
  "$PWD/qemu_integration/guest_libcuda/libtiny_cuda.so"
python3 qemu_integration/guest_libcuda/private_export_probe.py compare-identity \
  --left "$output_root/discovery" \
  --right "$output_root/callback-hooks-slot1" \
  --output "$output_root/pair-identity.json"

for path in \
  "$output_root/pair-identity.json" \
  "$output_root/discovery/summary.json" \
  "$output_root/callback-hooks-slot1/summary.json"; do
  test -f "$path"
done
printf '=== PRIVATE_EXPORT_PROBE_FILE_BEGIN pair-identity.json ===\n'
cat "$output_root/pair-identity.json"
printf '\n=== PRIVATE_EXPORT_PROBE_FILE_END pair-identity.json ===\n'
for case_dir in "$output_root/discovery" "$output_root/callback-hooks-slot1"; do
  for result in identity.json probe-config.json tables.jsonl calls.jsonl returns.jsonl captures.jsonl \
      gdb-status.json summary.json debugger-command.json gdb.transcript; do
    printf '=== PRIVATE_EXPORT_PROBE_FILE_BEGIN %s/%s ===\n' "$(basename "$case_dir")" "$result"
    cat "$case_dir/$result"
    printf '\n=== PRIVATE_EXPORT_PROBE_FILE_END %s/%s ===\n' "$(basename "$case_dir")" "$result"
  done
done
printf 'private_export_table_probe=complete\n'

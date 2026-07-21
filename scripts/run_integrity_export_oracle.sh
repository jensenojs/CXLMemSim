#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <tools-tls|runtime-callback-hooks> <log-prefix>\n' "$0" >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }
readonly table=$1
readonly log_prefix=$2
case "$table" in
  tools-tls|runtime-callback-hooks) ;;
  *) usage; exit 2 ;;
esac

mkdir -p .work
nvidia-smi --query-gpu=uuid,driver_version,name,memory.total --format=csv,noheader
driver_path="$(ldconfig -p | awk '$1 == "libcuda.so.1" {print $NF; exit}')"
test -n "$driver_path"
driver_real="$(readlink -f "$driver_path")"
printf '%s_driver_path=%s\n' "$log_prefix" "$driver_real"
readelf -n "$driver_real" > ".work/${log_prefix}-driver-notes.txt"
awk -v prefix="${log_prefix}_driver_" '/Build ID/ {print prefix $0; exit}' \
  ".work/${log_prefix}-driver-notes.txt"
make -C qemu_integration/guest_libcuda libcuda.so.1 integrity-export-oracle
qemu_integration/guest_libcuda/cuda-integrity-export-oracle \
  --table "$table" \
  --driver "$driver_real" \
  --shim "$PWD/qemu_integration/guest_libcuda/libcuda.so.1"

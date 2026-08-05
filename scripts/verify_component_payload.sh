#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s PAYLOAD_DIR EVIDENCE_DIR\n' "$0" >&2
    exit 2
fi

readonly PAYLOAD=$(realpath "$1")
readonly EVIDENCE=$2
mkdir -p "$EVIDENCE"

readelf -d "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-readelf-dynamic.txt"
readelf -d "$PAYLOAD/bin/cuda-integrity-export-oracle" >"$EVIDENCE/integrity-oracle-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcxl-loader-audit.so" >"$EVIDENCE/loader-audit-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cxl-gpu-case" >"$EVIDENCE/case-control-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cxl-mem-active-gate" >"$EVIDENCE/cxl-mem-active-gate-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" >"$EVIDENCE/tiny-probe-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libtiny_cuda.so" >"$EVIDENCE/tiny-library-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcublas_create_probe.so" >"$EVIDENCE/cublas-create-probe-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libbatch_copy_probe.so" >"$EVIDENCE/batch-copy-probe-readelf-dynamic.txt"
ldd "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-ldd.txt"
ldd "$PAYLOAD/bin/cuda-integrity-export-oracle" >"$EVIDENCE/integrity-oracle-ldd.txt"
ldd "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-ldd.txt"
ldd "$PAYLOAD/guest/libcxl-loader-audit.so" >"$EVIDENCE/loader-audit-ldd.txt"
ldd "$PAYLOAD/guest/cxl-gpu-case" >"$EVIDENCE/case-control-ldd.txt"
ldd "$PAYLOAD/guest/cxl-mem-active-gate" >"$EVIDENCE/cxl-mem-active-gate-ldd.txt"
ldd "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" >"$EVIDENCE/tiny-probe-ldd.txt"
ldd "$PAYLOAD/guest/libtiny_cuda.so" >"$EVIDENCE/tiny-library-ldd.txt"
ldd "$PAYLOAD/guest/libcublas_create_probe.so" >"$EVIDENCE/cublas-create-probe-ldd.txt"
ldd "$PAYLOAD/guest/libbatch_copy_probe.so" >"$EVIDENCE/batch-copy-probe-ldd.txt"
if grep -Fh 'not found' "$EVIDENCE"/*-ldd.txt | grep -q .; then
    printf 'error: unresolved CXLMemSim payload dependency\n' >&2
    exit 1
fi
for library in liblz4.so.1 libzstd.so.1; do
    grep -F "Shared library: [$library]" "$EVIDENCE/shim-readelf-dynamic.txt" >/dev/null
    grep -F "$library =>" "$EVIDENCE/shim-ldd.txt" >/dev/null
done
! readelf -SW "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" | grep -F '.nv_fatbin' >/dev/null
! grep -F 'Shared library: [libcudart.so.12]' "$EVIDENCE/tiny-probe-readelf-dynamic.txt" >/dev/null
grep -F 'Shared library: [libcudart.so.12]' "$EVIDENCE/tiny-library-readelf-dynamic.txt" >/dev/null
for symbol in tiny_cuda_launch tiny_cuda_probe_run; do
    nm -D "$PAYLOAD/guest/libtiny_cuda.so" | grep -F " $symbol" >>"$EVIDENCE/tiny-library-symbol.txt"
done
nm -D "$PAYLOAD/guest/libcublas_create_probe.so" | grep -F ' tiny_cuda_probe_run' >"$EVIDENCE/cublas-create-probe-symbol.txt"
nm -D "$PAYLOAD/guest/libbatch_copy_probe.so" | grep -F ' tiny_cuda_probe_run' >"$EVIDENCE/batch-copy-probe-symbol.txt"
! grep -F 'Shared library: [libcublas.so.12]' "$EVIDENCE/cublas-create-probe-readelf-dynamic.txt" >/dev/null
"$PAYLOAD/guest/cxl-gpu-case" --help >"$EVIDENCE/case-control-help.txt"
"$PAYLOAD/bin/cuda-integrity-export-oracle" --help >"$EVIDENCE/integrity-oracle-help.txt"
[[ -L $PAYLOAD/guest/libcuda.so ]]
[[ $(readlink "$PAYLOAD/guest/libcuda.so") == libcuda.so.1 ]]
sha256sum \
    "$PAYLOAD/bin/cxlmemsim_server" \
    "$PAYLOAD/bin/cuda-integrity-export-oracle" \
    "$PAYLOAD/guest/libcuda.so.1" \
    "$PAYLOAD/guest/libcxl-loader-audit.so" \
    "$PAYLOAD/guest/cxl-gpu-case" \
    "$PAYLOAD/guest/cxl-mem-active-gate" \
    "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" \
    "$PAYLOAD/guest/libtiny_cuda.so" \
    "$PAYLOAD/guest/libcublas_create_probe.so" \
    >"$EVIDENCE/payload-sha256.txt"
printf 'cxlmemsim_payload_verification=pass\n'

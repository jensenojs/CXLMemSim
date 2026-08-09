#!/usr/bin/env bash
# Build the stream-contract cases. Inputs: optional env NVCC, SM (default
# sm_86 for the local RTX 3050), CCBIN (default g++-14 because the host gcc-15
# is newer than CUDA 12.9 supports). Outputs: build/<case> binaries.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$here/build"
mkdir -p "$build"

NVCC="${NVCC:-nvcc}"
SM="${SM:-sm_86}"
CCBIN="${CCBIN:-g++-14}"

ccbin_args=()
if [ -n "$CCBIN" ] && command -v "$CCBIN" >/dev/null 2>&1; then
    ccbin_args=(-ccbin "$CCBIN")
fi

for c in memcpy2d_async memcpy_dtod_async memcpy_htod_async memset_d8_async; do
    "$NVCC" "${ccbin_args[@]}" -arch="$SM" -O2 \
        -o "$build/$c" "$here/$c.cu" -lcuda
    echo "built=$build/$c"
done

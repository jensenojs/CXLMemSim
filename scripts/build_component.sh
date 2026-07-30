#!/usr/bin/env bash
set -euo pipefail

# agent-usage: bash scripts/build_component.sh --work-dir ABSENT_PATH --cache-dir EXISTING_PATH --payload-dir ABSENT_PATH
# agent-context: problem=An implicit repository work tree and CCACHE_DIR make component bytes depend on hidden mutable paths and let retries delete unrelated state.
# agent-context: mental_model=The caller owns execution and cache roots; this script owns one absent build directory and one absent payload output, after a clean source admission check.
# agent-context: role=Build and gate the cxlmemsim component payload without creating an artifact manifest or publishing it.
# agent-context: use_when=Use for a local host diagnostic, an exact-image local artifact build, or the CNB formal component build.
# agent-context: inputs=Clean cxlmemsim checkout, absent absolute work and payload paths, existing absolute cache directory, build profile, and required toolchain commands.
# agent-context: outputs=One profile-verified payload and retained build evidence under the caller-selected work directory.
# agent-context: interpret=component_build=pass means the declared component targets and payload gate passed on this execution surface.
# agent-context: proves=The payload matches the component build profile and passed its component-local build and test gates.
# agent-context: does_not_prove=It does not create or sign an OCI artifact and does not prove Type-2 or Kimi correctness.
# agent-context: next=For an exact-image artifact, pass the payload and exact HEAD to package_component.sh in the same toolchain image.

usage() {
    printf 'usage: %s --work-dir ABSENT_PATH --cache-dir EXISTING_PATH --payload-dir ABSENT_PATH\n' "$0"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf 'agent_hint=self=scripts/build_component.sh\nagent_hint=usage=bash scripts/build_component.sh --work-dir ABSENT_PATH --cache-dir EXISTING_PATH --payload-dir ABSENT_PATH\nagent_hint=boundary=builds and gates payload only; does not create an artifact manifest\n'
    exit 0
fi

work_arg= cache_arg= payload_arg=
while (( $# )); do
    case "$1" in
        --work-dir|--cache-dir|--payload-dir)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case "$1" in
                --work-dir) work_arg=$2 ;;
                --cache-dir) cache_arg=$2 ;;
                --payload-dir) payload_arg=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $work_arg && -n $cache_arg && -n $payload_arg ]] || { usage >&2; exit 2; }

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=${ROOT}/manifests/build-profile.json
[[ $work_arg == /* && $cache_arg == /* && $payload_arg == /* ]] || { printf 'paths must be absolute\n' >&2; exit 2; }
[[ ! -e $work_arg && ! -L $work_arg ]] || { printf 'work directory must not exist: %s\n' "$work_arg" >&2; exit 2; }
[[ ! -e $payload_arg && ! -L $payload_arg ]] || { printf 'payload directory must not exist: %s\n' "$payload_arg" >&2; exit 2; }
[[ -d $cache_arg && ! -L $cache_arg ]] || { printf 'cache directory must be an existing non-symlink directory: %s\n' "$cache_arg" >&2; exit 2; }
[[ -d $(dirname "$work_arg") && ! -L $(dirname "$work_arg") ]] || { printf 'work parent must be an existing non-symlink directory\n' >&2; exit 2; }
[[ -d $(dirname "$payload_arg") && ! -L $(dirname "$payload_arg") ]] || { printf 'payload parent must be an existing non-symlink directory\n' >&2; exit 2; }
readonly WORK=$(realpath -m "$work_arg")
readonly PAYLOAD=$(realpath -m "$payload_arg")
readonly CCACHE_DIR=$(realpath "$cache_arg")
case "$ROOT/" in "$WORK/"*|"$PAYLOAD/"*) printf 'output path cannot be an ancestor of the repository\n' >&2; exit 2;; esac
[[ $WORK != "$ROOT" && $PAYLOAD != "$ROOT" ]] || { printf 'output path cannot equal repository root\n' >&2; exit 2; }
[[ -z $(git -C "$ROOT" status --porcelain=v1 --untracked-files=all) ]] || { printf 'source checkout must be clean\n' >&2; exit 1; }
readonly BUILD=${WORK}/build
readonly GUEST_SOURCE=${WORK}/guest-source
readonly EVIDENCE=${WORK}/evidence
export CCACHE_DIR
export CCACHE_BASEDIR=$ROOT

cd "$ROOT"

readarray -t profile < <(python3 - "$PROFILE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(p["target_arch"])
print(p["compiler"]["cc"])
print(p["compiler"]["cxx"])
print(p["cmake"]["build_type"])
print(p["cmake"]["policy_default_cmp0091"])
print(p["cmake"]["parallel"])
for key, value in sorted(p["cmake"]["definitions"].items()):
    print(f"-D{key}={value}")
PY
)

[[ ${profile[0]} == amd64 ]]
[[ $(uname -m) == x86_64 ]]
readonly PROFILE_CC=${profile[1]}
readonly PROFILE_CXX=${profile[2]}
readonly BUILD_TYPE=${profile[3]}
readonly CMP0091=${profile[4]}
readonly PARALLEL=${profile[5]}
readonly CMAKE_DEFINITIONS=("${profile[@]:6}")

mkdir -p "$BUILD" "$GUEST_SOURCE" "$PAYLOAD/bin" "$PAYLOAD/guest" "$PAYLOAD/evidence/cuda-api" "$EVIDENCE"

ccache --show-stats | tee "$EVIDENCE/ccache-before.txt"
env CC="$PROFILE_CC" CXX="$PROFILE_CXX" cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_POLICY_DEFAULT_CMP0091="$CMP0091" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    "${CMAKE_DEFINITIONS[@]}" 2>&1 | tee "$EVIDENCE/cmake-configure.log"

cmake --build "$BUILD" --parallel "$PARALLEL" --target \
    cxlmemsim_server test_dcd_gfam test_rob test_mem_stall test_bandwidth_model \
    2>&1 | tee "$EVIDENCE/cmake-build.log"

ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tee "$EVIDENCE/ctest.log"
"$BUILD/cxlmemsim_server" --help >"$EVIDENCE/server-help.stdout" \
    2>"$EVIDENCE/server-help.stderr"

git archive HEAD qemu_integration/guest_libcuda | tar -x -C "$GUEST_SOURCE"
readarray -t guest < <(python3 - "$PROFILE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))["guest_shim"]
print(p["cc"])
print(" ".join(p["cflags"]))
for target in p["make_targets"]:
    print(target)
tiny=p["tiny_cuda"]
print(tiny["target"])
print(tiny["nvcc"])
print(tiny["host_cxx"])
print(tiny["arch"])
PY
)
readonly GUEST_CC=${guest[0]}
readonly GUEST_CFLAGS=${guest[1]}
readonly TINY_FIELD_COUNT=4
readonly GUEST_TARGET_COUNT=$((${#guest[@]} - 2 - TINY_FIELD_COUNT))
(( GUEST_TARGET_COUNT > 0 ))
readonly GUEST_TARGETS=("${guest[@]:2:GUEST_TARGET_COUNT}")
readonly TINY_TARGET=${guest[-4]}
readonly TINY_NVCC=${guest[-3]}
readonly TINY_HOST_CXX=${guest[-2]}
readonly TINY_ARCH=${guest[-1]}
readonly GUEST_DIR=${GUEST_SOURCE}/qemu_integration/guest_libcuda

make -C "$GUEST_DIR" CC="ccache $GUEST_CC" CFLAGS="$GUEST_CFLAGS" "${GUEST_TARGETS[@]}" \
    2>&1 | tee "$EVIDENCE/guest-shim-build.log"
make -C "$GUEST_DIR" CC="ccache $GUEST_CC" CFLAGS="$GUEST_CFLAGS" \
    NVCC="ccache $TINY_NVCC" NVCC_HOST_CXX="$TINY_HOST_CXX" CUDA_ARCH="$TINY_ARCH" "$TINY_TARGET" \
    2>&1 | tee "$EVIDENCE/tiny-cuda-build.log"
ccache --show-stats | tee "$EVIDENCE/ccache-after.txt"

install -m 0755 "$BUILD/cxlmemsim_server" "$PAYLOAD/bin/cxlmemsim_server"
install -m 0755 "$GUEST_DIR/cuda-integrity-export-oracle" "$PAYLOAD/bin/cuda-integrity-export-oracle"
install -m 0755 "$GUEST_DIR/libcuda.so.1" "$PAYLOAD/guest/libcuda.so.1"
install -m 0755 "$GUEST_DIR/libcxl-loader-audit.so" "$PAYLOAD/guest/libcxl-loader-audit.so"
install -m 0755 "$GUEST_DIR/cxl-gpu-case" "$PAYLOAD/guest/cxl-gpu-case"
install -m 0755 "$GUEST_DIR/cxl-mem-active-gate" "$PAYLOAD/guest/cxl-mem-active-gate"
install -m 0755 "$GUEST_DIR/cuda-runtime-dlopen-kernel-probe" \
    "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe"
install -m 0755 "$GUEST_DIR/libtiny_cuda.so" "$PAYLOAD/guest/libtiny_cuda.so"
install -m 0755 "$GUEST_DIR/libcublas_create_probe.so" "$PAYLOAD/guest/libcublas_create_probe.so"
# These are source facts consumed by cxl-lab's Q/R/N checker.  They come from
# the same git archive as the shim binary above, so a tiny result never needs
# to clone a floating component repository to reconstruct its API route.
install -m 0644 "$GUEST_DIR/libcuda.c" "$PAYLOAD/evidence/cuda-api/libcuda.c"
install -m 0644 "$GUEST_DIR/cxl_gpu_cmd.h" "$PAYLOAD/evidence/cuda-api/cxl_gpu_cmd.h"
install -m 0755 "$GUEST_DIR/collect_cuda_elf_static_evidence.py" \
    "$PAYLOAD/evidence/cuda-api/collect_cuda_elf_static_evidence.py"
ln -s libcuda.so.1 "$PAYLOAD/guest/libcuda.so"

python3 scripts/component_artifact.py verify-profile --payload "$PAYLOAD" --profile "$PROFILE"
readelf -d "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-readelf-dynamic.txt"
readelf -d "$PAYLOAD/bin/cuda-integrity-export-oracle" >"$EVIDENCE/integrity-oracle-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcxl-loader-audit.so" >"$EVIDENCE/loader-audit-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cxl-gpu-case" >"$EVIDENCE/case-control-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cxl-mem-active-gate" >"$EVIDENCE/cxl-mem-active-gate-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" \
    >"$EVIDENCE/tiny-probe-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libtiny_cuda.so" >"$EVIDENCE/tiny-library-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcublas_create_probe.so" >"$EVIDENCE/cublas-create-probe-readelf-dynamic.txt"
ldd "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-ldd.txt"
ldd "$PAYLOAD/bin/cuda-integrity-export-oracle" >"$EVIDENCE/integrity-oracle-ldd.txt"
ldd "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-ldd.txt"
ldd "$PAYLOAD/guest/libcxl-loader-audit.so" >"$EVIDENCE/loader-audit-ldd.txt"
ldd "$PAYLOAD/guest/cxl-gpu-case" >"$EVIDENCE/case-control-ldd.txt"
ldd "$PAYLOAD/guest/cxl-mem-active-gate" >"$EVIDENCE/cxl-mem-active-gate-ldd.txt"
ldd "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" >"$EVIDENCE/tiny-probe-ldd.txt"
ldd "$PAYLOAD/guest/libtiny_cuda.so" >"$EVIDENCE/tiny-library-ldd.txt"
ldd "$PAYLOAD/guest/libcublas_create_probe.so" >"$EVIDENCE/cublas-create-probe-ldd.txt"
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
nm -D "$PAYLOAD/guest/libcublas_create_probe.so" | grep -F ' tiny_cuda_probe_run' \
    >"$EVIDENCE/cublas-create-probe-symbol.txt"
! grep -F 'Shared library: [libcublas.so.12]' "$EVIDENCE/cublas-create-probe-readelf-dynamic.txt" >/dev/null
"$PAYLOAD/guest/cxl-gpu-case" --help >"$EVIDENCE/case-control-help.txt"
"$PAYLOAD/bin/cuda-integrity-export-oracle" --help >"$EVIDENCE/integrity-oracle-help.txt"
sha256sum "$PAYLOAD/bin/cxlmemsim_server" "$PAYLOAD/bin/cuda-integrity-export-oracle" "$PAYLOAD/guest/libcuda.so.1" "$PAYLOAD/guest/libcxl-loader-audit.so" \
    "$PAYLOAD/guest/cxl-gpu-case" "$PAYLOAD/guest/cxl-mem-active-gate" \
    "$PAYLOAD/guest/cuda-runtime-dlopen-kernel-probe" \
    "$PAYLOAD/guest/libtiny_cuda.so" "$PAYLOAD/guest/libcublas_create_probe.so" \
    >"$EVIDENCE/payload-sha256.txt"

printf 'component_build=pass\n'
printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
printf 'payload_dir=%s\n' "$PAYLOAD"

#!/usr/bin/env bash
# Run the stream-contract matrix on whatever CUDA stack this process sees:
# natively on the host, or inside the Type-2 guest against the shim (same
# script, same binaries).
#
# usage: run.sh [--surface native|type2-prefix|type2-fixed]
#   native        expect every case PASS (CUDA stream semantics intact)
#   type2-prefix  expect cuMemcpy2DAsync+nonblocking FAIL, everything else PASS
#                 (convicted gpu-w68.5.19 defect present)
#   type2-fixed   expect every case PASS (stream-aware fix landed on both
#                 shim and QEMU)
#
# Env: DELAY=<cycles> (default 3000000000, ~1.7 s race window; DELAY=0 for a
# quick smoke), BYTES=<n> (default 1048576).
#
# Output: each case prints its own key=value lines; the script adds
# case_result=... lines and a final matrix=PASS|FAIL. Exit 0 iff every case
# matches the surface expectation.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$here/build"

surface="native"
if [ "${1:-}" = "--surface" ]; then
    surface="${2:?missing surface name}"
    case "$surface" in native|type2-prefix|type2-fixed) ;; *)
        echo "error=bad_surface surface=$surface" >&2; exit 2 ;;
    esac
fi

delay="${DELAY:-3000000000}"
bytes="${BYTES:-1048576}"

if [ ! -x "$build/memcpy2d_async" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        "$here/build.sh"
    else
        echo "error=binaries_missing hint=run build.sh on a host with nvcc and copy build/ here" >&2
        exit 2
    fi
fi

expected_for() {
    if [ "$surface" = "type2-prefix" ] && [ "$1" = "memcpy2d_async" ] && \
       [ "$2" = "nonblocking" ]; then
        echo FAIL
    else
        echo PASS
    fi
}

deviations=0
for case_name in memcpy2d_async memcpy_dtod_async memcpy_htod_async; do
    for stream in legacy nonblocking; do
        "$build/$case_name" --stream="$stream" --delay-cycles="$delay" \
            --bytes="$bytes"
        rc=$?
        if [ $rc -eq 0 ]; then observed=PASS;
        elif [ $rc -eq 1 ]; then observed=FAIL;
        else observed=ERROR; fi
        expected="$(expected_for "$case_name" "$stream")"
        if [ "$observed" = "$expected" ]; then match=yes; else match=no; deviations=$((deviations+1)); fi
        echo "case_result=$case_name stream=$stream observed=$observed expected=$expected match=$match"
    done
done

echo "surface=$surface deviations=$deviations matrix=$([ $deviations -eq 0 ] && echo PASS || echo FAIL)"
[ $deviations -eq 0 ]

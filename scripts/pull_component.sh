#!/usr/bin/env bash
set -euo pipefail

container_runtime=docker
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-runtime)
            [[ $# -ge 2 ]] || { printf 'error: --container-runtime requires a value\n' >&2; exit 2; }
            container_runtime=$2; shift 2 ;;
        --) shift; break ;;
        -*) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [[ $# -ne 2 ]]; then
    printf 'usage: %s [--container-runtime docker|podman] CANDIDATE_JSON OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

case "$container_runtime" in
    docker|podman) ;;
    *) printf 'error: --container-runtime must be docker or podman\n' >&2; exit 2 ;;
esac
command -v "$container_runtime" >/dev/null 2>&1 || {
    printf 'error: missing container runtime: %s\n' "$container_runtime" >&2
    exit 1
}

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly CANDIDATE=$(realpath "$1")
readonly OUTPUT=$2
readonly CONTRACT=${ROOT}/manifests/artifact-contract.json
readonly WORK=${ROOT}/.work/component/fresh-pull

readarray -t candidate < <(python3 - "$CANDIDATE" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
required={"repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256"}
if set(c) != required or not c["digest"].startswith("sha256:"):
    raise SystemExit("invalid candidate manifest")
for key in ("repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256"):
    print(c[key])
PY
)
readonly REPOSITORY=${candidate[0]}
readonly DIGEST=${candidate[1]}
readonly EXPECTED_ARCHIVE_SHA256=${candidate[4]}
readonly EXPECTED_MANIFEST_SHA256=${candidate[5]}

readarray -t contract < <(python3 - "$CONTRACT" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
print(c["artifact_repository"])
print(c["oras_image"])
PY
)
[[ $REPOSITORY == "${contract[0]}" ]]
readonly ORAS_IMAGE=${contract[1]}

rm -rf "$WORK"
mkdir -p "$WORK/pulled"
readonly ORAS_BIN=${WORK}/oras
oras_container=$("$container_runtime" create "$ORAS_IMAGE")
trap '"$container_runtime" rm -f "${oras_container:-}" >/dev/null 2>&1 || true' EXIT
"$container_runtime" cp "${oras_container}:/bin/oras" "$ORAS_BIN"
"$container_runtime" rm "$oras_container" >/dev/null
oras_container=
chmod +x "$ORAS_BIN"

"$ORAS_BIN" pull "${REPOSITORY}@${DIGEST}" --output "$WORK/pulled" \
    --format json >"$WORK/pull.json"

readarray -t outer_files < <(find "$WORK/pulled" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)
[[ ${#outer_files[@]} -eq 2 ]]
[[ ${outer_files[0]} == component.tar.zst ]]
[[ ${outer_files[1]} == manifest.json ]]
printf '%s  %s\n' "$EXPECTED_ARCHIVE_SHA256" "$WORK/pulled/component.tar.zst" | sha256sum --check
printf '%s  %s\n' "$EXPECTED_MANIFEST_SHA256" "$WORK/pulled/manifest.json" | sha256sum --check

zstd -q -d "$WORK/pulled/component.tar.zst" -o "$WORK/component.tar"
python3 "$ROOT/scripts/component_artifact.py" verify-archive \
    --archive "$WORK/component.tar" \
    --manifest "$WORK/pulled/manifest.json" \
    --extract-dir "$OUTPUT"

"$OUTPUT/bin/cxlmemsim_server" --help >"$WORK/server-help.stdout" \
    2>"$WORK/server-help.stderr"
ldd "$OUTPUT/bin/cxlmemsim_server" >"$WORK/server-ldd.txt"
ldd "$OUTPUT/bin/cuda-integrity-export-oracle" >"$WORK/integrity-oracle-ldd.txt"
ldd "$OUTPUT/guest/libcuda.so.1" >"$WORK/shim-ldd.txt"
ldd "$OUTPUT/guest/cxl-gpu-case" >"$WORK/case-control-ldd.txt"
ldd "$OUTPUT/guest/cuda-runtime-dlopen-kernel-probe" >"$WORK/tiny-probe-ldd.txt"
ldd "$OUTPUT/guest/libtiny_cuda.so" >"$WORK/tiny-library-ldd.txt"
readelf -d "$OUTPUT/guest/libcuda.so.1" >"$WORK/shim-readelf-dynamic.txt"
for library in liblz4.so.1 libzstd.so.1; do
    grep -F "Shared library: [$library]" "$WORK/shim-readelf-dynamic.txt" >/dev/null
    grep -F "$library =>" "$WORK/shim-ldd.txt" >/dev/null
done
readelf -d "$OUTPUT/guest/cuda-runtime-dlopen-kernel-probe" >"$WORK/tiny-probe-readelf-dynamic.txt"
! readelf -SW "$OUTPUT/guest/cuda-runtime-dlopen-kernel-probe" | grep -F '.nv_fatbin' >/dev/null
! grep -F 'Shared library: [libcudart.so.12]' "$WORK/tiny-probe-readelf-dynamic.txt" >/dev/null
readelf -d "$OUTPUT/guest/libtiny_cuda.so" >"$WORK/tiny-library-readelf-dynamic.txt"
grep -F 'Shared library: [libcudart.so.12]' "$WORK/tiny-library-readelf-dynamic.txt" >/dev/null
for symbol in tiny_cuda_launch tiny_cuda_probe_run; do
    nm -D "$OUTPUT/guest/libtiny_cuda.so" | grep -F " $symbol" >>"$WORK/tiny-library-symbol.txt"
done
"$OUTPUT/guest/cxl-gpu-case" --help >"$WORK/case-control-help.txt"
"$OUTPUT/bin/cuda-integrity-export-oracle" --help >"$WORK/integrity-oracle-help.txt"
[[ -L $OUTPUT/guest/libcuda.so ]]
[[ $(readlink "$OUTPUT/guest/libcuda.so") == libcuda.so.1 ]]

printf 'component_fresh_pull=pass\n'
printf 'artifact_reference=%s@%s\n' "$REPOSITORY" "$DIGEST"
printf 'restored_payload=%s\n' "$OUTPUT"

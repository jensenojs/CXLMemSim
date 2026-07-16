#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s CANDIDATE_JSON OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

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
oras_container=$(docker create "$ORAS_IMAGE")
trap 'docker rm -f "$oras_container" >/dev/null 2>&1 || true' EXIT
docker cp "${oras_container}:/bin/oras" "$ORAS_BIN"
docker rm "$oras_container" >/dev/null
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
ldd "$OUTPUT/guest/libcuda.so.1" >"$WORK/shim-ldd.txt"
ldd "$OUTPUT/guest/cxl-gpu-case" >"$WORK/case-control-ldd.txt"
ldd "$OUTPUT/guest/cuda-runtime-dlopen-kernel-probe" >"$WORK/tiny-probe-ldd.txt"
ldd "$OUTPUT/guest/libtiny_cuda.so" >"$WORK/tiny-library-ldd.txt"
readelf -d "$OUTPUT/guest/libcuda.so.1" >"$WORK/shim-readelf-dynamic.txt"
for library in liblz4.so.1 libzstd.so.1; do
    grep -F "Shared library: [$library]" "$WORK/shim-readelf-dynamic.txt" >/dev/null
    grep -F "$library =>" "$WORK/shim-ldd.txt" >/dev/null
done
nm -D "$OUTPUT/guest/libtiny_cuda.so" | grep -F ' tiny_cuda_launch' >"$WORK/tiny-library-symbol.txt"
"$OUTPUT/guest/cxl-gpu-case" --help >"$WORK/case-control-help.txt"
[[ -L $OUTPUT/guest/libcuda.so ]]
[[ $(readlink "$OUTPUT/guest/libcuda.so") == libcuda.so.1 ]]

printf 'component_fresh_pull=pass\n'
printf 'artifact_reference=%s@%s\n' "$REPOSITORY" "$DIGEST"
printf 'restored_payload=%s\n' "$OUTPUT"

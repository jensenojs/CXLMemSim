#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_INITRD=${BASE_INITRD:-}
OUTPUT_INITRD=${OUTPUT_INITRD:-}
INIT_SOURCE=${INIT_SOURCE:-"$SCRIPT_DIR/type2_model_1p5b_init"}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ -n "$BASE_INITRD" ]] || die "BASE_INITRD is required"
[[ -n "$OUTPUT_INITRD" ]] || die "OUTPUT_INITRD is required"
[[ -f "$BASE_INITRD" ]] || die "missing base initrd: $BASE_INITRD"
[[ -f "$INIT_SOURCE" ]] || die "missing init source: $INIT_SOURCE"
[[ ! -e "$OUTPUT_INITRD" ]] || die "output already exists: $OUTPUT_INITRD"

WORK_DIR="$OUTPUT_INITRD.tree"
[[ ! -e "$WORK_DIR" ]] || die "work directory already exists: $WORK_DIR"
mkdir -p "$WORK_DIR"

(
    cd "$WORK_DIR"
    gzip -dc "$BASE_INITRD" | cpio -idmu --quiet
)
install -m 0755 "$INIT_SOURCE" "$WORK_DIR/init"
(
    cd "$WORK_DIR"
    find . -print0 | sort -z | cpio --null -o -H newc --quiet | gzip -n -9 > "$OUTPUT_INITRD"
)

cat > "$OUTPUT_INITRD.assembly-manifest.txt" <<EOF
base_initrd=$(readlink -f -- "$BASE_INITRD")
base_initrd_sha256=$(sha256sum "$BASE_INITRD" | awk '{print $1}')
init_source=$(readlink -f -- "$INIT_SOURCE")
init_source_sha256=$(sha256sum "$INIT_SOURCE" | awk '{print $1}')
operation=extract base initrd; replace /init from tracked source; deterministic cpio+gzip repack
output_initrd=$(readlink -f -- "$OUTPUT_INITRD")
output_initrd_sha256=$(sha256sum "$OUTPUT_INITRD" | awk '{print $1}')
EOF

echo "OUTPUT_INITRD=$OUTPUT_INITRD"

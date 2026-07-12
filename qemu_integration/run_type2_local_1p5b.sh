#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CXL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(cd -- "$CXL_ROOT/.." && pwd)
PROJECTS=$(cd -- "$WORKSPACE/.." && pwd)

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-local-1p5b}
PORT=${PORT:-10350}
QEMU_ROOT=${QEMU_ROOT:-"$PROJECTS/qemu-cxl-type2"}
LINUX_ROOT=${LINUX_ROOT:-"$PROJECTS/linux-cxl-type2"}
CONCORDIA_ROOT=${CONCORDIA_ROOT:-"$WORKSPACE/Concordia"}
QEMU_BINARY=${QEMU_BINARY:-"$QEMU_ROOT/build/qemu-system-x86_64"}
KERNEL_IMAGE=${KERNEL_IMAGE:-"$LINUX_ROOT/build/arch/x86/boot/bzImage"}
SERVER_BUILD_DIR=${SERVER_BUILD_DIR:-"$CXL_ROOT/build-replay-20260712"}
CXLMEMSIM_SERVER=${CXLMEMSIM_SERVER:-"$SERVER_BUILD_DIR/cxlmemsim_server"}
BASE_INITRD=${BASE_INITRD:-"$CXL_ROOT/build/type2-guest-initramfs/type2-model-only-initramfs-20260710-063000.cpio.gz"}
BASE_PAYLOAD=${BASE_PAYLOAD:-"$CXL_ROOT/build/type2-model-payload-20260710-062000-1p5b-compressed-cubin/payload.ext4"}
INITRD=${INITRD:-"$CXL_ROOT/build/type2-guest-initramfs/type2-model-1p5b-$RUN_ID.cpio.gz"}
PAYLOAD_DIR=${PAYLOAD_DIR:-"$CXL_ROOT/build/type2-model-payload-$RUN_ID"}
PAYLOAD_IMG="$PAYLOAD_DIR/payload.ext4"
RUN_DIR="$CXL_ROOT/build/type2-model-smoke-$RUN_ID"
HOST_LLAMA_RUNNER=${HOST_LLAMA_RUNNER:-"$HOME/llama.cpp/build/bin/llama-completion"}
HOST_MODEL=${HOST_MODEL:-"$HOME/models/sweep-next-edit-1.5b.q8_0.v2.gguf"}

die() {
    echo "error: $*" >&2
    exit 1
}

for path in "$BASE_INITRD" "$BASE_PAYLOAD" "$KERNEL_IMAGE" "$HOST_LLAMA_RUNNER" "$HOST_MODEL"; do
    [[ -f "$path" ]] || die "missing fixed input: $path"
done
[[ ! -e "$INITRD" ]] || die "initrd output already exists: $INITRD"
[[ ! -e "$PAYLOAD_DIR" ]] || die "payload output already exists: $PAYLOAD_DIR"
[[ ! -e "$RUN_DIR" ]] || die "run output already exists: $RUN_DIR"

ninja -C "$QEMU_ROOT/build" -j4 qemu-system-x86_64
cmake --build "$SERVER_BUILD_DIR" --parallel 4 --target cxlmemsim_server
(
    cd "$CONCORDIA_ROOT"
    LLVM_SYS_211_PREFIX=/usr LLVM_ZLUDA_PREBUILT=/usr CARGO_BUILD_JOBS=1 \
        MAKEFLAGS=-j1 CARGO_INCREMENTAL=0 \
        cargo build -p zluda --features nvidia --no-default-features -j1
)
make -C "$SCRIPT_DIR/guest_libcuda" -j4 libcuda.so.1

BASE_INITRD="$BASE_INITRD" OUTPUT_INITRD="$INITRD" \
    "$SCRIPT_DIR/build_type2_model_initrd.sh"

mkdir -p "$PAYLOAD_DIR"
cp --reflink=auto -- "$BASE_PAYLOAD" "$PAYLOAD_IMG"
debugfs -w -R 'rm /lib/libcuda.so.1' "$PAYLOAD_IMG" >/dev/null 2>&1 || true
debugfs -w -R "write $SCRIPT_DIR/guest_libcuda/libcuda.so.1 /lib/libcuda.so.1" "$PAYLOAD_IMG" >/dev/null
cat > "$PAYLOAD_DIR/assembly-manifest.txt" <<EOF
base_payload=$(readlink -f -- "$BASE_PAYLOAD")
base_payload_sha256=$(sha256sum "$BASE_PAYLOAD" | awk '{print $1}')
replacement_guest_libcuda=$(readlink -f -- "$SCRIPT_DIR/guest_libcuda/libcuda.so.1")
replacement_guest_libcuda_sha256=$(sha256sum "$SCRIPT_DIR/guest_libcuda/libcuda.so.1" | awk '{print $1}')
operation=copy base payload; replace /lib/libcuda.so.1 with debugfs
output_payload=$(readlink -f -- "$PAYLOAD_IMG")
output_payload_sha256=$(sha256sum "$PAYLOAD_IMG" | awk '{print $1}')
EOF

set +e
QEMU_BINARY="$QEMU_BINARY" \
KERNEL_IMAGE="$KERNEL_IMAGE" \
INITRD="$INITRD" \
PAYLOAD_IMG="$PAYLOAD_IMG" \
CXLMEMSIM_SERVER="$CXLMEMSIM_SERVER" \
HETGPU_LIB="$CONCORDIA_ROOT/target/debug/libnvcuda.so" \
ACCELERATOR=local-kvm \
RUN_ID="$RUN_ID" \
PORT="$PORT" \
TIMEOUT_SECONDS=600 \
SUCCESS_MARKER=TYPE2_MODEL_1P5B_SMOKE_PASS \
    "$SCRIPT_DIR/run_type2_local_stack.sh"
RUN_RC=$?
set -e

if [[ -f "$RUN_DIR/qemu-guest.log" ]]; then
    python3 "$SCRIPT_DIR/verify_type2_model_log.py" "$RUN_DIR/qemu-guest.log" \
        --expect-cubin-loads=133 \
        --require-module-global-success \
        --require-param-layout-launch \
        > "$RUN_DIR/boundary-verification.json" || true
    python3 "$SCRIPT_DIR/verify_type2_model_log.py" "$RUN_DIR/qemu-guest.log" \
        --expect-cubin-loads=133 \
        --require-module-global-success \
        --require-param-layout-launch \
        --require-model-success \
        > "$RUN_DIR/model-success-verification.json" || true
    RUN_DIR="$RUN_DIR" RUN_RC="$RUN_RC" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
text = (run_dir / "qemu-guest.log").read_text(errors="replace")
begin = "=== TYPE2_MODEL_STDOUT_BEGIN ===\n"
end = "=== TYPE2_MODEL_STDOUT_END ==="
model_stdout = ""
if begin in text and end in text:
    model_stdout = text.split(begin, 1)[1].split(end, 1)[0]
    (run_dir / "model.stdout").write_text(model_stdout)

summary = {
    "runner_rc": int(os.environ["RUN_RC"]),
    "model_success_marker": "TYPE2_MODEL_1P5B_SMOKE_PASS" in text,
    "model_failure_marker": "TYPE2_MODEL_1P5B_SMOKE_FAIL" in text,
    "model_stdout_sha256": hashlib.sha256(model_stdout.encode()).hexdigest() if model_stdout else None,
    "qemu_log_sha256": hashlib.sha256((run_dir / "qemu-guest.log").read_bytes()).hexdigest(),
}
for name in (
    "run-manifest.json",
    "qemu.argv",
    "server.argv",
    "input-inspection/payload-assembly-manifest.txt",
    "input-inspection/initrd-assembly-manifest.txt",
):
    path = run_dir / name
    if path.is_file():
        summary[f"{name.replace('/', '_')}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
(run_dir / "handoff-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
    if [[ -s "$RUN_DIR/model.stdout" ]]; then
        env -u LD_PRELOAD "$HOST_LLAMA_RUNNER" \
            -m "$HOST_MODEL" \
            -p 'Say that you have started in one short sentence.' \
            -n 4 -c 128 -t 2 -ngl 1 --temp 0.0 \
            > "$RUN_DIR/host-baseline.stdout" \
            2> "$RUN_DIR/host-baseline.stderr"
        set +e
        diff -u "$RUN_DIR/host-baseline.stdout" "$RUN_DIR/model.stdout" \
            > "$RUN_DIR/host-baseline.diff"
        DIFF_RC=$?
        set -e
        RUN_DIR="$RUN_DIR" DIFF_RC="$DIFF_RC" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
summary_path = run_dir / "handoff-summary.json"
summary = json.loads(summary_path.read_text())
baseline = (run_dir / "host-baseline.stdout").read_bytes()
guest = (run_dir / "model.stdout").read_bytes()
summary.update({
    "host_baseline_stdout_sha256": hashlib.sha256(baseline).hexdigest(),
    "guest_model_stdout_sha256": hashlib.sha256(guest).hexdigest(),
    "host_baseline_matches_guest": int(os.environ["DIFF_RC"]) == 0,
})
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
        if (( DIFF_RC != 0 )); then
            echo "error: Type-2 guest stdout differs from the fixed host baseline" >&2
            RUN_RC=1
        fi
    fi
fi

echo "RUN_DIR=$RUN_DIR"
exit "$RUN_RC"

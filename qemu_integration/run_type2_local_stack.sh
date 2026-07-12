#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
RUNNER_SCRIPT=$(readlink -f -- "$0")

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-local-stack}
RUN_ROOT=${RUN_ROOT:-"$ROOT/CXLMemSim/build"}
RUN_DIR="$RUN_ROOT/type2-model-smoke-$RUN_ID"
PORT=${PORT:-10255}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
ACCELERATOR=${ACCELERATOR:-local-kvm}
CXLMEMSIM_CAPACITY_MB=${CXLMEMSIM_CAPACITY_MB:-1024}
TYPE2_MEM_SIZE=${TYPE2_MEM_SIZE:-512M}
CXL_FMW_SIZE=${CXL_FMW_SIZE:-1G}

SERVER=${CXLMEMSIM_SERVER:-"$ROOT/CXLMemSim/build-clang/cxlmemsim_server"}
QEMU=${QEMU_BINARY:-/home/jensen/Projects/qemu-cxl-type2/build/qemu-system-x86_64}
KERNEL=${KERNEL_IMAGE:-/home/jensen/Projects/linux-cxl-type2/build/arch/x86/boot/bzImage}
INITRD=${INITRD:-}
PAYLOAD_IMG=${PAYLOAD_IMG:-}
HETGPU_LIB=${HETGPU_LIB:-"$ROOT/Concordia/target/debug/libnvcuda.so"}
SUCCESS_MARKER=${SUCCESS_MARKER:-}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ "$ACCELERATOR" == local-kvm ]] || die "this runner only accepts ACCELERATOR=local-kvm"
[[ -n "$INITRD" ]] || die "INITRD is required"
[[ -n "$PAYLOAD_IMG" ]] || die "PAYLOAD_IMG is required"
[[ ! -e "$RUN_DIR" ]] || die "run directory already exists: $RUN_DIR"
for path in "$SERVER" "$QEMU" "$KERNEL" "$INITRD" "$PAYLOAD_IMG" "$HETGPU_LIB"; do
    [[ -f "$path" ]] || die "missing input: $path"
done
for path in "$SERVER" "$QEMU"; do
    [[ -x "$path" ]] || die "input is not executable: $path"
done
for command in python3 sha256sum stat debugfs timeout stdbuf; do
    command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done

mkdir -p "$RUN_DIR/input-inspection"
SERVER_LOG="$RUN_DIR/cxlmemsim-server.log"
QEMU_LOG="$RUN_DIR/qemu-guest.log"

SERVER_ARGS=(
    "$SERVER"
    --comm-mode=tcp
    --port="$PORT"
    --capacity="$CXLMEMSIM_CAPACITY_MB"
    --default_latency=100
)

QEMU_ARGS=(
    "$QEMU"
    -enable-kvm
    -cpu host
    -M "q35,cxl=on,cxl-fmw.0.targets.0=cxl.0,cxl-fmw.0.size=$CXL_FMW_SIZE"
    -m 2G,maxmem=3G,slots=4
    -smp 2
    -kernel "$KERNEL"
    -initrd "$INITRD"
    -append "console=ttyS0 rdinit=/init loglevel=8 nokaslr panic=-1"
    -drive "file=$PAYLOAD_IMG,if=none,id=payload,format=raw,readonly=on"
    -device virtio-blk-pci,drive=payload,id=payloadblk
    -device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.0
    -device cxl-rp,port=0,bus=cxl.0,id=type2_rp,chassis=0,slot=2
    -device "cxl-type2,bus=type2_rp,id=cxl_type2_model_smoke,sn=205,gpu-mode=2,hetgpu-lib=$HETGPU_LIB,hetgpu-backend=3,hetgpu-device=0,cache-size=16M,mem-size=$TYPE2_MEM_SIZE,cxlmemsim-addr=127.0.0.1,cxlmemsim-port=$PORT,coherency-enabled=true,dcd=on,dcd-granularity=1M,dcd-initial-size=$TYPE2_MEM_SIZE,gfam=on,gfam-hosts=4,gfam-host-id=0,mhsld=on,mhsld-heads=4,mhsld-head-id=0"
    -nographic
)

printf '%q ' "${SERVER_ARGS[@]}" > "$RUN_DIR/server.argv"
printf '\n' >> "$RUN_DIR/server.argv"
printf '%q ' "${QEMU_ARGS[@]}" > "$RUN_DIR/qemu.argv"
printf '\n' >> "$RUN_DIR/qemu.argv"
printf '%s\0' "${SERVER_ARGS[@]}" > "$RUN_DIR/server.argv.nul"
printf '%s\0' "${QEMU_ARGS[@]}" > "$RUN_DIR/qemu.argv.nul"

printf 'CXL_TRANSPORT_MODE=tcp\nCXL_MEMSIM_HOST=127.0.0.1\nCXL_MEMSIM_PORT=%s\nHETGPU_LIB_PATH=%s\nACCELERATOR=%s\nCXLMEMSIM_CAPACITY_MB=%s\nTYPE2_MEM_SIZE=%s\nCXL_FMW_SIZE=%s\n' \
    "$PORT" "$HETGPU_LIB" "$ACCELERATOR" "$CXLMEMSIM_CAPACITY_MB" "$TYPE2_MEM_SIZE" "$CXL_FMW_SIZE" \
    > "$RUN_DIR/environment.txt"

debugfs -R "dump -p /lib/libcuda.so.1 $RUN_DIR/input-inspection/guest-libcuda.so.1" "$PAYLOAD_IMG" \
    > "$RUN_DIR/input-inspection/guest-libcuda.debugfs.log" 2>&1 \
    || die "payload does not contain /lib/libcuda.so.1"
debugfs -R "dump -p /bin/cuda_runtime_probe_dlopen_ggml $RUN_DIR/input-inspection/tiny-probe" "$PAYLOAD_IMG" \
    > "$RUN_DIR/input-inspection/tiny-probe.debugfs.log" 2>&1 \
    || true
debugfs -R "dump -p /lib/libggml-cuda.so.0.10.0 $RUN_DIR/input-inspection/tiny-dso.so" "$PAYLOAD_IMG" \
    > "$RUN_DIR/input-inspection/tiny-dso.debugfs.log" 2>&1 \
    || true
if debugfs -R "dump -p /lib/liblz4.so.1 $RUN_DIR/input-inspection/guest-liblz4.so.1" "$PAYLOAD_IMG" \
    > "$RUN_DIR/input-inspection/guest-liblz4.debugfs.log" 2>&1; then
    :
else
      rm -f -- "$RUN_DIR/input-inspection/guest-liblz4.so.1"
fi
PAYLOAD_ASSEMBLY_MANIFEST="$(dirname -- "$PAYLOAD_IMG")/assembly-manifest.txt"
if [[ -f "$PAYLOAD_ASSEMBLY_MANIFEST" ]]; then
    cp -- "$PAYLOAD_ASSEMBLY_MANIFEST" "$RUN_DIR/input-inspection/payload-assembly-manifest.txt"
fi
INITRD_ASSEMBLY_MANIFEST="$INITRD.assembly-manifest.txt"
if [[ -f "$INITRD_ASSEMBLY_MANIFEST" ]]; then
    cp -- "$INITRD_ASSEMBLY_MANIFEST" "$RUN_DIR/input-inspection/initrd-assembly-manifest.txt"
fi

RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" ROOT="$ROOT" RUNNER_SCRIPT="$RUNNER_SCRIPT" SERVER="$SERVER" QEMU="$QEMU" KERNEL="$KERNEL" \
INITRD="$INITRD" PAYLOAD_IMG="$PAYLOAD_IMG" HETGPU_LIB="$HETGPU_LIB" ACCELERATOR="$ACCELERATOR" \
      CXL_MEMSIM_PORT="$PORT" CXLMEMSIM_CAPACITY_MB="$CXLMEMSIM_CAPACITY_MB" TYPE2_MEM_SIZE="$TYPE2_MEM_SIZE" \
      CXL_FMW_SIZE="$CXL_FMW_SIZE" \
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import subprocess

run_dir = pathlib.Path(os.environ["RUN_DIR"])

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def file_fact(path: str) -> dict:
    value = pathlib.Path(path)
    info = value.stat()
    return {
        "path": str(value.resolve()),
        "size": info.st_size,
        "sha256": sha256(value),
        "mode": stat.filemode(info.st_mode),
    }

def git_fact(path: str) -> dict:
    repo = pathlib.Path(path)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).splitlines()
    return {"path": str(repo.resolve()), "head": head, "status": status}

def nul_argv(path: pathlib.Path) -> list[str]:
    return [part.decode() for part in path.read_bytes().split(b"\0") if part]

inputs = {
    "runner": file_fact(os.environ["RUNNER_SCRIPT"]),
    "cxlmemsim_server": file_fact(os.environ["SERVER"]),
    "qemu": file_fact(os.environ["QEMU"]),
    "kernel": file_fact(os.environ["KERNEL"]),
    "initrd": file_fact(os.environ["INITRD"]),
    "payload": file_fact(os.environ["PAYLOAD_IMG"]),
    "hetgpu_library": file_fact(os.environ["HETGPU_LIB"]),
    "guest_libcuda": file_fact(str(run_dir / "input-inspection/guest-libcuda.so.1")),
}
guest_lz4 = run_dir / "input-inspection/guest-liblz4.so.1"
if guest_lz4.exists():
    inputs["guest_liblz4"] = file_fact(str(guest_lz4))
tiny_probe = run_dir / "input-inspection/tiny-probe"
if tiny_probe.exists():
    inputs["tiny_probe"] = file_fact(str(tiny_probe))
tiny_dso = run_dir / "input-inspection/tiny-dso.so"
if tiny_dso.exists():
    inputs["tiny_dso"] = file_fact(str(tiny_dso))
payload_assembly_manifest = run_dir / "input-inspection/payload-assembly-manifest.txt"
if payload_assembly_manifest.exists():
    inputs["payload_assembly_manifest"] = file_fact(str(payload_assembly_manifest))
initrd_assembly_manifest = run_dir / "input-inspection/initrd-assembly-manifest.txt"
if initrd_assembly_manifest.exists():
    inputs["initrd_assembly_manifest"] = file_fact(str(initrd_assembly_manifest))

manifest = {
    "schema_version": 1,
    "run_id": os.environ["RUN_ID"],
    "accelerator": os.environ["ACCELERATOR"],
    "inputs": inputs,
    "sources": {
        "cxlmemsim": git_fact(os.environ["ROOT"] + "/CXLMemSim"),
        "qemu": git_fact("/home/jensen/Projects/qemu-cxl-type2"),
        "concordia": git_fact(os.environ["ROOT"] + "/Concordia"),
        "linux_cxl_type2": git_fact("/home/jensen/Projects/linux-cxl-type2"),
    },
    "server_argv": nul_argv(run_dir / "server.argv.nul"),
    "qemu_argv": nul_argv(run_dir / "qemu.argv.nul"),
    "environment": {
        "CXL_TRANSPORT_MODE": "tcp",
        "CXL_MEMSIM_HOST": "127.0.0.1",
          "CXL_MEMSIM_PORT": os.environ["CXL_MEMSIM_PORT"] if "CXL_MEMSIM_PORT" in os.environ else None,
          "HETGPU_LIB_PATH": os.environ["HETGPU_LIB"],
          "CXLMEMSIM_CAPACITY_MB": os.environ["CXLMEMSIM_CAPACITY_MB"],
          "TYPE2_MEM_SIZE": os.environ["TYPE2_MEM_SIZE"],
          "CXL_FMW_SIZE": os.environ["CXL_FMW_SIZE"],
      },
}
(run_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

"${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 80); do
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/$PORT" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        die "cxlmemsim server exited before readiness"
    fi
    sleep 0.1
done
[[ "$ready" -eq 1 ]] || die "cxlmemsim server did not become ready"

set +e
env CXL_TRANSPORT_MODE=tcp CXL_MEMSIM_HOST=127.0.0.1 CXL_MEMSIM_PORT="$PORT" \
    HETGPU_LIB_PATH="$HETGPU_LIB" timeout "$TIMEOUT_SECONDS" stdbuf -oL -eL \
    "${QEMU_ARGS[@]}" </dev/null > "$QEMU_LOG" 2>&1
QEMU_RC=$?
set -e
cleanup
SERVER_PID=""

QEMU_RC="$QEMU_RC" SUCCESS_MARKER="$SUCCESS_MARKER" RUN_DIR="$RUN_DIR" python3 - <<'PY'
import hashlib
import json
import os
import pathlib

run_dir = pathlib.Path(os.environ["RUN_DIR"])

def sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()

marker = os.environ["SUCCESS_MARKER"]
qemu_log = run_dir / "qemu-guest.log"
text = qemu_log.read_text(errors="replace") if qemu_log.exists() else ""
result = {
    "qemu_rc": int(os.environ["QEMU_RC"]),
    "success_marker": marker or None,
    "success_marker_found": bool(marker and marker in text),
    "qemu_log_sha256": sha256(qemu_log),
    "server_log_sha256": sha256(run_dir / "cxlmemsim-server.log"),
}
(run_dir / "run-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

echo "QEMU_RC=$QEMU_RC"
echo "RUN_DIR=$RUN_DIR"
if [[ "$QEMU_RC" -ne 0 ]]; then
    exit "$QEMU_RC"
fi
if [[ -n "$SUCCESS_MARKER" ]] && ! grep -Fq -- "$SUCCESS_MARKER" "$QEMU_LOG"; then
    echo "missing success marker: $SUCCESS_MARKER" >&2
    exit 2
fi

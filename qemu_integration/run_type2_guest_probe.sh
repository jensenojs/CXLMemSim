#!/usr/bin/env bash
set -euo pipefail

# 这个脚本维护 guest Type-2 CUDA probe 的重复实验流程：
# 构建 shim、复制并替换 payload、从 initrd tree 生成 initrd、注入环境变量，
# 最后调用现有 QEMU runner。它不修改 libcuda.c，也不删除已有实验产物。

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

SHIM_DIR=${SHIM_DIR:-"$ROOT/CXLMemSim/qemu_integration/guest_libcuda"}
SHIM_LIB=${SHIM_LIB:-"$SHIM_DIR/libcuda.so.1"}
SHIM_CFLAGS=${SHIM_CFLAGS:-'-O2 -Wall -Wextra'}
RUNNER=${RUNNER:-"$ROOT/CXLMemSim/qemu_integration/run_type2_local_stack.sh"}

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-guest-probe}
PORT=${PORT:-10255}
BASE_PAYLOAD=${BASE_PAYLOAD:-}
BASE_INITRD_TREE=${BASE_INITRD_TREE:-}
EXTRA_ENV_FILE=${EXTRA_ENV_FILE:-}
PROBE_BINARY=${PROBE_BINARY:-}
PROBE_GUEST_PATH=${PROBE_GUEST_PATH:-/bin/cuda_runtime_probe_dlopen_ggml}
EXTRA_PAYLOAD_FILE=${EXTRA_PAYLOAD_FILE:-}
EXTRA_PAYLOAD_GUEST_PATH=${EXTRA_PAYLOAD_GUEST_PATH:-}

ARTIFACT_DIR="$ROOT/CXLMemSim/build/type2-guest-probe-$RUN_ID"
PAYLOAD_DIR="$ROOT/CXLMemSim/build/type2-model-payload-$RUN_ID"
PAYLOAD_IMG="$PAYLOAD_DIR/payload.ext4"
INITRD_TREE="$ROOT/CXLMemSim/build/type2-guest-initramfs/type2-ggml-dlopen-probe-initramfs-$RUN_ID.tree"
INITRD_IMG="$ROOT/CXLMemSim/build/type2-guest-initramfs/type2-ggml-dlopen-probe-initramfs-$RUN_ID.cpio.gz"
PIPELINE_LOG="$ARTIFACT_DIR/pipeline.log"

TMP_DIR=''

die() {
  echo "错误：$*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf -- "$TMP_DIR"
  fi
}
trap cleanup EXIT

usage() {
  cat >&2 <<'EOF'
用法：
  BASE_PAYLOAD=/path/base/payload.ext4 \
  BASE_INITRD_TREE=/path/base/initrd.tree \
  EXTRA_ENV_FILE=/path/experiment.env \
  RUN_ID=experiment-name PORT=10255 \
  CXLMemSim/qemu_integration/run_type2_guest_probe.sh

环境变量：
  BASE_PAYLOAD       只读复制的 ext4 payload，必须存在
  BASE_INITRD_TREE   initrd 的目录树，必须存在
  EXTRA_ENV_FILE     可选，每行 KEY=VALUE；空行和 # 注释会被忽略
  PROBE_BINARY       可选，替换 payload 中的 probe 程序
  PROBE_GUEST_PATH   probe 在 payload 内的路径
  EXTRA_PAYLOAD_FILE 可选，放入 payload 的单个附加诊断文件
  EXTRA_PAYLOAD_GUEST_PATH  附加文件在 payload 内的绝对路径
  RUN_ID             实验标识，同时用于所有产物目录
  PORT               CXLMemSim TCP 端口
  SHIM_CFLAGS        guest shim 编译参数
EOF
}

[[ -n "$BASE_PAYLOAD" ]] || { usage; die '必须提供 BASE_PAYLOAD'; }
[[ -n "$BASE_INITRD_TREE" ]] || { usage; die '必须提供 BASE_INITRD_TREE'; }
[[ -f "$BASE_PAYLOAD" ]] || die "BASE_PAYLOAD 不存在：$BASE_PAYLOAD"
[[ -d "$BASE_INITRD_TREE" ]] || die "BASE_INITRD_TREE 不存在：$BASE_INITRD_TREE"
[[ -x "$RUNNER" ]] || die "QEMU runner 不可执行：$RUNNER"
command -v debugfs >/dev/null 2>&1 || die '找不到 debugfs'
command -v cpio >/dev/null 2>&1 || die '找不到 cpio'

if [[ -n "$EXTRA_ENV_FILE" ]]; then
  [[ -f "$EXTRA_ENV_FILE" ]] || die "EXTRA_ENV_FILE 不存在：$EXTRA_ENV_FILE"
fi
if [[ -n "$PROBE_BINARY" ]]; then
  [[ -f "$PROBE_BINARY" ]] || die "PROBE_BINARY 不存在：$PROBE_BINARY"
fi
if [[ -n "$EXTRA_PAYLOAD_FILE" ]]; then
  [[ -f "$EXTRA_PAYLOAD_FILE" ]] || die "EXTRA_PAYLOAD_FILE 不存在：$EXTRA_PAYLOAD_FILE"
  [[ "$EXTRA_PAYLOAD_GUEST_PATH" == /* ]] || die '使用 EXTRA_PAYLOAD_FILE 时必须提供绝对的 EXTRA_PAYLOAD_GUEST_PATH'
fi

# 不覆盖同名现场。重复 RUN_ID 应显式换名，而不是让脚本破坏旧证据。
for path in "$ARTIFACT_DIR" "$PAYLOAD_DIR" "$INITRD_TREE" "$INITRD_IMG"; do
  [[ ! -e "$path" ]] || die "产物已经存在，为保护现场而停止：$path"
done

mkdir -p "$ARTIFACT_DIR" "$PAYLOAD_DIR" "$(dirname -- "$INITRD_TREE")"
TMP_DIR=$(mktemp -d "$ROOT/CXLMemSim/build/type2-guest-probe-tmp.XXXXXX")

echo '=== BUILD_GUEST_SHIM ==='
make -C "$SHIM_DIR" libcuda.so.1 CFLAGS="$SHIM_CFLAGS" | tee "$ARTIFACT_DIR/shim-build.log"
[[ -f "$SHIM_LIB" ]] || die "shim 构建后找不到：$SHIM_LIB"
cp -- "$SHIM_LIB" "$ARTIFACT_DIR/libcuda.so.1"
sha256sum "$SHIM_LIB" | tee "$ARTIFACT_DIR/shim-sha256.txt"

echo '=== PACKAGE_PAYLOAD ==='
cp -- "$BASE_PAYLOAD" "$PAYLOAD_IMG"

# payload 是 ext4 镜像。只替换目标文件，不修改其它 guest 文件。
debugfs -w -R 'rm /lib/libcuda.so.1' "$PAYLOAD_IMG" >/dev/null 2>&1 || true
debugfs -w -R "write $SHIM_LIB /lib/libcuda.so.1" "$PAYLOAD_IMG" >/dev/null
if [[ -n "$PROBE_BINARY" ]]; then
  # 可选地替换 probe，让取证逻辑跟实验产物一起冻结。
  debugfs -w -R "rm $PROBE_GUEST_PATH" "$PAYLOAD_IMG" >/dev/null 2>&1 || true
  debugfs -w -R "write $PROBE_BINARY $PROBE_GUEST_PATH" "$PAYLOAD_IMG" >/dev/null
  cp -- "$PROBE_BINARY" "$ARTIFACT_DIR/$(basename -- "$PROBE_BINARY")"
fi
if [[ -n "$EXTRA_PAYLOAD_FILE" ]]; then
  # 单个附加文件用于 LD_PRELOAD tracer 等诊断产物；目标目录须已存在于 base payload。
  debugfs -w -R "rm $EXTRA_PAYLOAD_GUEST_PATH" "$PAYLOAD_IMG" >/dev/null 2>&1 || true
  debugfs -w -R "write $EXTRA_PAYLOAD_FILE $EXTRA_PAYLOAD_GUEST_PATH" "$PAYLOAD_IMG" >/dev/null
  cp -- "$EXTRA_PAYLOAD_FILE" "$ARTIFACT_DIR/$(basename -- "$EXTRA_PAYLOAD_FILE")"
fi
sha256sum "$PAYLOAD_IMG" | tee "$PAYLOAD_DIR/payload-sha256.txt"
printf 'base_payload=%s\nshim=%s\nprobe_binary=%s\nprobe_guest_path=%s\nextra_payload_file=%s\nextra_payload_guest_path=%s\n' \
  "$BASE_PAYLOAD" "$SHIM_LIB" "$PROBE_BINARY" "$PROBE_GUEST_PATH" \
  "$EXTRA_PAYLOAD_FILE" "$EXTRA_PAYLOAD_GUEST_PATH" > "$PAYLOAD_DIR/manifest.txt"

if [[ -n "$EXTRA_ENV_FILE" ]]; then
  cp -- "$EXTRA_ENV_FILE" "$ARTIFACT_DIR/extra-env.txt"
else
  : > "$ARTIFACT_DIR/extra-env.txt"
fi

echo '=== PACKAGE_INITRD ==='
cp -a -- "$BASE_INITRD_TREE" "$INITRD_TREE"

# 先解析为 export 语句，再插入 guest /init 的 probe 执行前。
# 不 source 实验文件，避免把任意 shell 代码带入 initrd。
EXPORTS_FILE="$TMP_DIR/extra-exports.sh"
: > "$EXPORTS_FILE"
if [[ -n "$EXTRA_ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line#\#}" != "$line" ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key=${BASH_REMATCH[1]}
      value=${BASH_REMATCH[2]}
      printf 'export %s=%q\n' "$key" "$value" >> "$EXPORTS_FILE"
    else
      die "EXTRA_ENV_FILE 中不是 KEY=VALUE：$line"
    fi
  done < "$EXTRA_ENV_FILE"
fi

INIT_FILE="$INITRD_TREE/init"
[[ -f "$INIT_FILE" ]] || die "initrd tree 缺少 /init：$INIT_FILE"
MARKER="echo '=== GGML_DLOPEN_PROBE_LDD ==='"
grep -Fqx "$MARKER" "$INIT_FILE" || die "无法定位 initrd probe 注入点：$MARKER"

INIT_TMP="$TMP_DIR/init"
injected=0
while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" == "$MARKER" ]]; then
    if [[ -s "$EXPORTS_FILE" ]]; then
      cat "$EXPORTS_FILE"
    fi
    injected=1
  fi
  printf '%s\n' "$line"
done < "$INIT_FILE" > "$INIT_TMP"
[[ "$injected" -eq 1 ]] || die 'initrd 环境变量未注入'
cp -- "$INIT_TMP" "$INIT_FILE"

(cd "$INITRD_TREE" && find . -print0 | cpio --null -o --format=newc | gzip -9 > "$INITRD_IMG") \
  2>&1 | tee "$ARTIFACT_DIR/initrd-build.log"
sha256sum "$INITRD_IMG" | tee "$ARTIFACT_DIR/initrd-sha256.txt"
printf 'base_initrd_tree=%s\ninitrd=%s\n' "$BASE_INITRD_TREE" "$INITRD_IMG" > "$ARTIFACT_DIR/initrd-manifest.txt"

echo '=== RUN_QEMU ==='
set +e
RUN_ID="$RUN_ID" PORT="$PORT" INITRD="$INITRD_IMG" PAYLOAD_IMG="$PAYLOAD_IMG" \
  "$RUNNER" > "$PIPELINE_LOG" 2>&1
RUNNER_RC=$?
set -e

RUN_DIR="$ROOT/CXLMemSim/build/type2-model-smoke-$RUN_ID"
printf 'run_id=%s\nport=%s\nrunner_rc=%s\nrun_dir=%s\npayload=%s\ninitrd=%s\n' \
  "$RUN_ID" "$PORT" "$RUNNER_RC" "$RUN_DIR" "$PAYLOAD_IMG" "$INITRD_IMG" \
  | tee "$ARTIFACT_DIR/result.txt"

if [[ -f "$RUN_DIR/qemu-guest.log" ]]; then
  echo '=== RELEVANT_LOG_TAIL ==='
  grep -E 'CONTEXT_CHECKS|CONTEXT_LOCAL_STORAGE|cuLibraryLoadData|after cuda(SetDevice|MemGetInfo|Free|DeviceSynchronize)|GGML_DLOPEN_PROBE_(PASS|FAIL)|QEMU_RC' \
    "$RUN_DIR/qemu-guest.log" | tail -80 || true
fi

# 保留现有 runner 的层级判定：tiny probe PASS 但未命中 1.5B marker 时仍返回 2。
exit "$RUNNER_RC"

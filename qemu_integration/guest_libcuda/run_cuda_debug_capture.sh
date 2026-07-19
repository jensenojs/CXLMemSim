#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'cuda_debug_capture_error=%s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
用法：
  run_cuda_debug_capture.sh --output-dir DIR [--debugger PATH] --command-file FILE ... \
      --program ELF -- [PROGRAM_ARGS...]

  run_cuda_debug_capture.sh --output-dir DIR [--debugger PATH] --command-file FILE ... \
      --executable ELF --core CORE

用途与边界：
  这层只收拢已有 debugger 的程序或 exact ELF + core 现场。每次运行创建一个新 output directory，写入
  debugger-identity.txt、input-identity.txt、debugger-command.txt、debugger.transcript 和 debugger-exit.txt。
  所有 ELF、core 与 command file 都记录 SHA256；给定 core 仅只读，不复制、压缩、删除或修改。

调试器选择：
  gdb 用于 host CUDA Runtime/Driver、private export table 及普通 ELF/core 调用链。
  cuda-gdb 只用于假设涉及 device code、kernel state、device memory、warp/register 或 SASS PC 的场景。
  当前 host ABI/private table 观察不因统一流程强行改用 cuda-gdb。

重放约束：
  command file 是显式版本化输入，wrapper 会保存其 hash 与实际 argv。debugger 的非零退出保留在
  debugger-exit.txt 并作为本命令退出码；后续采集不能把它掩盖成成功。完整 transcript 只证明该组
  ELF/core/debugger 输入下的观察，不能单独证明 private ABI、guest shim、Type-2 或 Kimi correctness。
EOF
}

output_dir=
debugger=gdb
program=
executable=
core=
declare -a command_files=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) output_dir=${2:-}; shift 2 ;;
        --debugger) debugger=${2:-}; shift 2 ;;
        --command-file) command_files+=("${2:-}"); shift 2 ;;
        --program) program=${2:-}; shift 2 ;;
        --executable) executable=${2:-}; shift 2 ;;
        --core) core=${2:-}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n $output_dir ]] || die "--output-dir is required"
[[ ! -e $output_dir ]] || die "output directory already exists: $output_dir"
[[ ${#command_files[@]} -gt 0 ]] || die "at least one --command-file is required"
for command_file in "${command_files[@]}"; do
    [[ -f $command_file ]] || die "command file is not a regular file: $command_file"
done

if [[ -n $program || $# -gt 0 ]]; then
    [[ -n $program && -z $executable && -z $core ]] || die "program mode requires --program and no --executable/--core"
    [[ -x $program ]] || die "program is not executable: $program"
else
    [[ -n $executable && -n $core ]] || die "core mode requires --executable and --core"
    [[ -f $executable ]] || die "executable is not a regular file: $executable"
    [[ -f $core ]] || die "core is not a regular file: $core"
fi

mkdir -p "$output_dir"
debugger_path=$(command -v "$debugger") || die "debugger is not on PATH: $debugger"
debugger_path=$(readlink -f "$debugger_path")

{
    printf 'debugger_path=%s\n' "$debugger_path"
    sha256sum "$debugger_path"
    "$debugger_path" --version
    "$debugger_path" --configuration || true
} >"$output_dir/debugger-identity.txt" 2>&1

{
    if [[ -n $program ]]; then
        printf 'mode=program\n'
        printf 'program_path=%s\n' "$(readlink -f "$program")"
        sha256sum "$program"
        printf 'program_args='
        printf '%q ' "$@"
        printf '\n'
    else
        printf 'mode=core\n'
        printf 'executable_path=%s\n' "$(readlink -f "$executable")"
        sha256sum "$executable"
        printf 'core_path=%s\n' "$(readlink -f "$core")"
        sha256sum "$core"
    fi
    for command_file in "${command_files[@]}"; do
        printf 'command_file_path=%s\n' "$(readlink -f "$command_file")"
        sha256sum "$command_file"
    done
} >"$output_dir/input-identity.txt"

declare -a command=("$debugger_path" --batch -ex 'set pagination off')
for command_file in "${command_files[@]}"; do
    command+=(-x "$command_file")
done
if [[ -n $program ]]; then
    command+=(--args "$program" "$@")
else
    command+=("$executable" "$core")
fi
printf '%q ' "${command[@]}" >"$output_dir/debugger-command.txt"
printf '\n' >>"$output_dir/debugger-command.txt"

set +e
"${command[@]}" >"$output_dir/debugger.transcript" 2>&1
debugger_exit=$?
set -e
printf 'debugger_exit_code=%d\n' "$debugger_exit" >"$output_dir/debugger-exit.txt"
printf 'cuda_debug_capture_result=%s\n' "$output_dir"
exit "$debugger_exit"

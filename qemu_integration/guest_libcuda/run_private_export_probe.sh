#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'private_export_probe_error=%s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
usage:
  run_private_export_probe.sh --mode discovery --output-dir DIR -- TRIGGER [ARGS...]
  run_private_export_probe.sh --mode capture --output-dir DIR --uuid UUID --slot N \
      [--selector VALUE] [--memory rsi:BYTES] -- TRIGGER [ARGS...]

The trigger runs once under a Python-enabled host GDB. The probe observes naturally reached
cuGetExportTable tables and table entry calls; it does not invoke private slots.
EOF
}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
readonly ROOT
PROBE=$ROOT/qemu_integration/guest_libcuda/private_export_probe.py
readonly PROBE
[[ -f $PROBE ]] || die "missing probe implementation: $PROBE"

mode=
output_dir=
uuid=
slot=
selector=
memory=
selector_max=
event_limit=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode=${2:-}; shift 2 ;;
        --output-dir) output_dir=${2:-}; shift 2 ;;
        --uuid) uuid=${2:-}; shift 2 ;;
        --slot) slot=${2:-}; shift 2 ;;
        --selector) selector=${2:-}; shift 2 ;;
        --memory) memory=${2:-}; shift 2 ;;
        --selector-max) selector_max=${2:-}; shift 2 ;;
        --event-limit) event_limit=${2:-}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ $mode == discovery || $mode == capture ]] || die "--mode must be discovery or capture"
[[ -n $output_dir ]] || die "--output-dir is required"
[[ $# -gt 0 ]] || die "trigger argv after -- is required"
[[ ! -e $output_dir ]] || die "output directory already exists: $output_dir"
if [[ $mode == capture ]]; then
    [[ -n $uuid && -n $slot ]] || die "capture requires --uuid and --slot"
else
    [[ -z $uuid && -z $slot && -z $selector && -z $memory ]] || die "discovery does not accept capture filters"
fi

prepare=(python3 "$PROBE" prepare --output-dir "$output_dir" --source-root "$ROOT" --mode "$mode")
[[ -n $uuid ]] && prepare+=(--uuid "$uuid")
[[ -n $slot ]] && prepare+=(--slot "$slot")
[[ -n $selector ]] && prepare+=(--selector "$selector")
[[ -n $memory ]] && prepare+=(--memory "$memory")
[[ -n $selector_max ]] && prepare+=(--selector-max "$selector_max")
[[ -n $event_limit ]] && prepare+=(--event-limit "$event_limit")
prepare+=(-- "$@")
"${prepare[@]}"

config=$output_dir/probe-config.json
transcript=$output_dir/gdb.transcript
debugger=${PRIVATE_EXPORT_PROBE_GDB:-gdb}
command -v "$debugger" >/dev/null 2>&1 || die "missing debugger: $debugger"
set +e
"$debugger" --batch \
    -ex 'set pagination off' \
    -ex 'set debuginfod enabled off' \
    -ex "python exec(compile(open(r'$PROBE').read(), r'$PROBE', 'exec'))" \
    -ex "private-export-probe $config" \
    -ex run \
    --args "$@" >"$transcript" 2>&1
gdb_rc=$?
set -e
printf '{"exit_code":%d,"path":"%s"}\n' "$gdb_rc" "$(command -v "$debugger")" >"$output_dir/debugger-command.json"
if [[ ! -f $output_dir/gdb-status.json ]]; then
    python3 "$PROBE" debugger-failure \
        --config "$config" \
        --exit-code "$gdb_rc" \
        --message "debugger ended without gdb-status.json; see $transcript"
fi
python3 "$PROBE" verify --config "$config"
printf 'private_export_probe_result=%s\n' "$output_dir/summary.json"
exit "$gdb_rc"

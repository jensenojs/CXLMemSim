#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'private_export_probe_error=%s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
usage:
  run_private_export_probe.sh --mode discovery --output-dir DIR \
      [--driver-symbol SYMBOL --driver-code-register REGISTER \
       --driver-output-register REGISTER --driver-event-limit N] \
      [--separate-inferior-io] -- TRIGGER [ARGS...]
  run_private_export_probe.sh --mode capture --output-dir DIR --uuid UUID --slot N \
      [--selector VALUE] [--memory REGISTER:BYTES ...] \
      [--output-buffer POINTER_OUT_REGISTER:SIZE_OUT_REGISTER:MAX_BYTES] \
      [--driver-symbol SYMBOL --driver-code-register REGISTER \
       --driver-output-register REGISTER --driver-event-limit N] \
      [--separate-inferior-io] -- TRIGGER [ARGS...]

The trigger runs once under a Python-enabled host GDB. The probe observes naturally reached
cuGetExportTable tables and table entry calls; it does not invoke private slots.
The optional Driver group observes one declared public cu* symbol and binds its code argument
to an exact mapped ELF. All four --driver-* arguments must be supplied together.
With --separate-inferior-io, the exact trigger argv is preserved in inferior-run.gdb while
inferior.stdout and inferior.stderr are kept separate from gdb.transcript.
EOF
}

hint() {
    cat <<'EOF'
private_export_probe_hint=self=qemu_integration/guest_libcuda/run_private_export_probe.sh
private_export_probe_hint=problem=running isolated GDB commands by hand loses Driver, Runtime, trigger and probe identities and makes negative private-table results impossible to compare across public triggers
private_export_probe_hint=mental_model=validate a new output directory and debugger prerequisites; generate one GDB command file loading private_export_probe.py; execute the caller-supplied trigger verbatim; preserve full transcript and identity; optionally observe one public Driver entry/return stream inside the same lifecycle; require the Python observer summary to close
private_export_probe_hint=role=provide the command-line boundary for reusable discovery or bounded capture while keeping trigger selection outside the probe implementation
private_export_probe_hint=use_when=run a low-cost public CUDA trigger on a real compatible NVIDIA environment to inventory natural private calls or capture one UUID/slot/selector already justified by discovery or a Kimi core
private_export_probe_hint=inputs=mode, new output directory, optional UUID/slot/selector/memory/private-event limits, one explicit returned-buffer projection, an optional complete driver-symbol/code-register/output-register/event-limit group, optional separated inferior I/O, Python-enabled gdb and an explicit trigger command after --
private_export_probe_hint=outputs=debugger/input identity, generated command, full gdb.transcript, identity.json,probe-config.json,tables.jsonl,calls.jsonl,returns.jsonl,captures.jsonl,optional driver-calls.jsonl and driver-returns.jsonl,gdb-status.json,summary.json; separated runs also preserve inferior-run.gdb,inferior.stdout,inferior.stderr
private_export_probe_hint=interpret=matching call/return sequence values prove one natural private call returned; public Driver records additionally require the breakpoint PC to map to the declared host Driver and the code pointer to map to one live exact ELF; capture_status separately reports whether the declared private target was reached
private_export_probe_hint=proves=the exact trigger/debugger/probe composition and naturally observed private-table events; a declared Driver stream proves its observed call count, mapped code origins, return values and output-handle transitions in this exact native window
private_export_probe_hint=does_not_prove=that an unreached slot is unused by Kimi, that a non-NULL entry has a known signature, that active calling is safe, or that guest Type-2/Kimi is correct
private_export_probe_hint=next=compare discovery sets across triggers or feed a reached bounded capture into a real-Driver oracle and minimal guest-shim repair; never replace not_reached with a guessed success stub
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
memory=()
output_buffer=
selector_max=
event_limit=
driver_symbol=
driver_code_register=
driver_output_register=
driver_event_limit=
separate_inferior_io=no
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode=${2:-}; shift 2 ;;
        --output-dir) output_dir=${2:-}; shift 2 ;;
        --uuid) uuid=${2:-}; shift 2 ;;
        --slot) slot=${2:-}; shift 2 ;;
        --selector) selector=${2:-}; shift 2 ;;
        --memory) memory+=("${2:-}"); shift 2 ;;
        --output-buffer) output_buffer=${2:-}; shift 2 ;;
        --selector-max) selector_max=${2:-}; shift 2 ;;
        --event-limit) event_limit=${2:-}; shift 2 ;;
        --driver-symbol) driver_symbol=${2:-}; shift 2 ;;
        --driver-code-register) driver_code_register=${2:-}; shift 2 ;;
        --driver-output-register) driver_output_register=${2:-}; shift 2 ;;
        --driver-event-limit) driver_event_limit=${2:-}; shift 2 ;;
        --separate-inferior-io) separate_inferior_io=yes; shift ;;
        --help|-h) usage; exit 0 ;;
        --hint) hint; exit 0 ;;
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
    [[ -z $uuid && -z $slot && -z $selector && ${#memory[@]} -eq 0 && -z $output_buffer ]] || die "discovery does not accept capture filters"
fi
driver_argument_count=0
for value in "$driver_symbol" "$driver_code_register" "$driver_output_register" "$driver_event_limit"; do
    [[ -z $value ]] || ((driver_argument_count += 1))
done
[[ $driver_argument_count -eq 0 || $driver_argument_count -eq 4 ]] || die "all four --driver-* arguments must be supplied together"

prepare=(python3 "$PROBE" prepare --output-dir "$output_dir" --source-root "$ROOT" --mode "$mode")
[[ -n $uuid ]] && prepare+=(--uuid "$uuid")
[[ -n $slot ]] && prepare+=(--slot "$slot")
[[ -n $selector ]] && prepare+=(--selector "$selector")
for window in "${memory[@]}"; do
    prepare+=(--memory "$window")
done
[[ -n $output_buffer ]] && prepare+=(--output-buffer "$output_buffer")
[[ -n $selector_max ]] && prepare+=(--selector-max "$selector_max")
[[ -n $event_limit ]] && prepare+=(--event-limit "$event_limit")
if [[ $driver_argument_count -eq 4 ]]; then
    prepare+=(
        --driver-symbol "$driver_symbol"
        --driver-code-register "$driver_code_register"
        --driver-output-register "$driver_output_register"
        --driver-event-limit "$driver_event_limit"
    )
fi
[[ $separate_inferior_io == no ]] || prepare+=(--separate-inferior-io)
prepare+=(-- "$@")
"${prepare[@]}"

config=$output_dir/probe-config.json
transcript=$output_dir/gdb.transcript
debugger=${PRIVATE_EXPORT_PROBE_GDB:-gdb}
command -v "$debugger" >/dev/null 2>&1 || die "missing debugger: $debugger"
run_command=(-ex run)
if [[ $separate_inferior_io == yes ]]; then
    run_command=(-x "$output_dir/inferior-run.gdb")
fi
set +e
"$debugger" --batch \
    -ex 'set pagination off' \
    -ex 'set debuginfod enabled off' \
    -ex "python exec(compile(open(r'$PROBE').read(), r'$PROBE', 'exec'))" \
    -ex "private-export-probe $config" \
    "${run_command[@]}" \
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

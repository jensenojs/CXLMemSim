#!/usr/bin/env python3
"""Observe naturally reached CUDA private export-table calls under GDB.

The host-side runner writes the identity/configuration.  When this file is sourced
by GDB it installs breakpoints only; it never calls a private table entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = 1
POINTER_BYTES = 8
MAX_TABLE_BYTES = 4096
MAX_TERMINATED_WORDS = 256
DEFAULT_EVENT_LIMIT = 128
INTEGER_REGISTERS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp")
DRIVER_ARGUMENT_REGISTERS = frozenset(INTEGER_REGISTERS[:-1])
HINT = """private_export_probe_hint=self=qemu_integration/guest_libcuda/private_export_probe.py
private_export_probe_hint=problem=CUDA Runtime private export tables are undocumented; a NULL guest entry and a host table address reveal neither the slot signature nor the selector-specific state transition needed for a safe shim implementation
private_export_probe_hint=mental_model=Python-enabled GDB observes real cuGetExportTable returns and naturally executed table-entry calls; an optional declared public Driver symbol records its own complete entry/return stream and binds a code pointer to the inferior's exact mapped ELF; private captures remain a separate UUID/slot/selector stream
private_export_probe_hint=role=implement the debugger-side observer and machine-readable table/call/capture event model used by multiple public CUDA triggers
private_export_probe_hint=use_when=a public CUDA API or exact application trigger naturally reaches a private table path on the matching real NVIDIA Driver/Runtime and the unknown ABI must be constrained before changing the guest shim
private_export_probe_hint=inputs=live GDB inferior running an explicit trigger; mode discovery or capture; optional UUID, slot, selector, bounded register-memory windows, one POINTER_OUT:SIZE_OUT:MAX_BYTES output-buffer projection, private event limit, one declared public Driver symbol with code/output registers and its own event limit, and explicit inferior I/O separation
private_export_probe_hint=outputs=identity.json,probe-config.json,tables.jsonl,calls.jsonl,returns.jsonl,captures.jsonl,driver-calls.jsonl,driver-returns.jsonl,gdb-status.json,summary.json and GDB-visible diagnostic messages; separated runs additionally preserve inferior-run.gdb,inferior.stdout,inferior.stderr
private_export_probe_hint=interpret=calls and returns with the same sequence are one naturally executed private call; capture reached additionally requires the declared UUID/slot/selector and bounded memory pair; not_reached supplies no target ABI authority
private_export_probe_hint=proves=which private export tables and slots the real Runtime naturally reached, their observed entry registers and return RAX, bounded argument-memory facts and, when declared, one public Driver call stream with exact mapped code origin and output-handle before/after
private_export_probe_hint=does_not_prove=the complete function signature, semantics outside observed arguments/state, safety of active fuzzing, guest shim correctness, Type-2/Kimi correctness or TPS
private_export_probe_hint=next=union results from faithful public triggers; for a Kimi blocker use the smallest reached entry/return oracle, and keep unknown or unreached slots NULL and explicit
"""


def canonical_uuid(raw: bytes) -> str:
    if len(raw) != 16:
        raise ValueError(f"UUID must contain 16 bytes, got {len(raw)}")
    hexed = raw.hex()
    return f"{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}"


def parse_uuid(text: str) -> str:
    compact = text.replace("-", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", compact):
        raise argparse.ArgumentTypeError("UUID must be 32 hexadecimal digits, with optional hyphens")
    return canonical_uuid(bytes.fromhex(compact))


def parse_nonnegative(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {text}") from error
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_memory(text: str) -> dict[str, int | str]:
    match = re.fullmatch(r"(r(?:di|si|dx|cx|8|9|sp)):(0x[0-9a-fA-F]+|[0-9]+)", text)
    if not match:
        raise argparse.ArgumentTypeError("memory window must be REGISTER:BYTES for a supported entry register")
    length = parse_nonnegative(match.group(2))
    if length == 0 or length > 4096:
        raise argparse.ArgumentTypeError("memory window must be between 1 and 4096 bytes")
    return {"register": match.group(1), "bytes": length}


def parse_output_buffer(text: str) -> dict[str, int | str]:
    match = re.fullmatch(
        r"(r(?:di|si|dx|cx|8|9)):(r(?:di|si|dx|cx|8|9)):(0x[0-9a-fA-F]+|[0-9]+)", text
    )
    if not match:
        raise argparse.ArgumentTypeError(
            "output buffer must be POINTER_OUT_REGISTER:SIZE_OUT_REGISTER:MAX_BYTES"
        )
    maximum = parse_nonnegative(match.group(3))
    if maximum == 0 or maximum > 4096:
        raise argparse.ArgumentTypeError("output buffer maximum must be between 1 and 4096 bytes")
    return {"pointer_register": match.group(1), "size_register": match.group(2), "max_bytes": maximum}


def parse_driver_register(text: str) -> str:
    if text not in DRIVER_ARGUMENT_REGISTERS:
        raise argparse.ArgumentTypeError(
            "Driver code/output register must be one of rdi,rsi,rdx,rcx,r8,r9"
        )
    return text


def parse_driver_symbol(text: str) -> str:
    if re.fullmatch(r"cu[A-Za-z0-9_]+", text) is None:
        raise argparse.ArgumentTypeError("Driver symbol must be a public cu* identifier")
    return text


def parse_proc_maps(text: str) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise RuntimeError(f"malformed /proc maps line {line_number}: {line}")
        bounds = fields[0].split("-", 1)
        if len(bounds) != 2:
            raise RuntimeError(f"malformed /proc maps bounds at line {line_number}: {fields[0]}")
        try:
            start, end = (int(value, 16) for value in bounds)
            offset = int(fields[2], 16)
            inode = int(fields[4], 10)
        except ValueError as error:
            raise RuntimeError(f"malformed /proc maps number at line {line_number}: {line}") from error
        if start >= end:
            raise RuntimeError(f"empty /proc maps range at line {line_number}: {line}")
        mappings.append(
            {
                "start": start,
                "end": end,
                "permissions": fields[1],
                "offset": offset,
                "device": fields[3],
                "inode": inode,
                "path": fields[5] if len(fields) == 6 else None,
            }
        )
    return mappings


def locate_proc_mapping(address: int, mappings: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [mapping for mapping in mappings if mapping["start"] <= address < mapping["end"]]
    if len(matches) != 1:
        raise RuntimeError(f"address 0x{address:x} maps to {len(matches)} /proc ranges")
    mapping = matches[0]
    path = mapping.get("path")
    if not isinstance(path, str) or not path.startswith("/") or path.endswith(" (deleted)"):
        raise RuntimeError(f"address 0x{address:x} is not backed by a live absolute ELF path: {path}")
    related = [
        item
        for item in mappings
        if item.get("path") == path and item.get("device") == mapping["device"] and item.get("inode") == mapping["inode"]
    ]
    if not related:
        raise RuntimeError(f"address 0x{address:x} has no related file mappings")
    load_bases = {item["start"] - item["offset"] for item in related}
    if len(load_bases) != 1:
        raise RuntimeError(f"ELF mappings for {path} disagree on load base: {sorted(load_bases)}")
    load_base = next(iter(load_bases))
    return {
        "start": mapping["start"],
        "end": mapping["end"],
        "permissions": mapping["permissions"],
        "mapping_offset": mapping["offset"],
        "device": mapping["device"],
        "inode": mapping["inode"],
        "path": path,
        "load_base": load_base,
        "file_offset": mapping["offset"] + address - mapping["start"],
        "image_offset": address - load_base,
    }


def json_dump(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def run_checked(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def resolve_library(name: str) -> pathlib.Path:
    listing = run_checked(["ldconfig", "-p"])
    for line in listing.splitlines():
        candidate = line.lstrip().split(" ", 1)[0]
        if (candidate == name or candidate.startswith(name + ".")) and " => " in line:
            return pathlib.Path(line.rsplit(" => ", 1)[1]).resolve()
    raise RuntimeError(f"could not resolve {name} through ldconfig")


def elf_build_id(path: pathlib.Path) -> str:
    output = run_checked(["readelf", "-n", str(path)])
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    if not match:
        raise RuntimeError(f"missing GNU Build ID: {path}")
    return match.group(1).lower()


def resolve_trigger(executable: str) -> pathlib.Path:
    candidate = pathlib.Path(executable)
    if candidate.parent == pathlib.Path("."):
        resolved = shutil.which(executable)
        if not resolved:
            raise RuntimeError(f"trigger is not executable or on PATH: {executable}")
        candidate = pathlib.Path(resolved)
    return candidate.resolve(strict=True)


def resolve_executable(executable: str) -> pathlib.Path:
    candidate = pathlib.Path(executable)
    if candidate.parent == pathlib.Path("."):
        resolved = shutil.which(executable)
        if not resolved:
            raise RuntimeError(f"executable is not on PATH: {executable}")
        candidate = pathlib.Path(resolved)
    return candidate.resolve(strict=True)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> int:
    output = pathlib.Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    trigger_argv = args.trigger[1:] if args.trigger[:1] == ["--"] else args.trigger
    if not trigger_argv:
        raise RuntimeError("trigger argv after -- is required")
    if args.mode == "capture" and (args.uuid is None or args.slot is None):
        raise RuntimeError("capture mode requires --uuid and --slot")
    if args.mode == "discovery" and (
        any(value is not None for value in (args.uuid, args.slot, args.selector)) or args.memory or args.output_buffer
    ):
        raise RuntimeError("discovery mode does not accept capture filters")
    driver_values = (args.driver_symbol, args.driver_code_register, args.driver_output_register)
    if any(value is not None for value in driver_values) and not all(value is not None for value in driver_values):
        raise RuntimeError(
            "public Driver observation requires --driver-symbol, --driver-code-register and --driver-output-register"
        )
    if args.driver_symbol is None and args.driver_event_limit is not None:
        raise RuntimeError("--driver-event-limit requires a public Driver observation")
    if args.driver_symbol is not None and (args.driver_event_limit is None or args.driver_event_limit == 0):
        raise RuntimeError("public Driver observation requires a positive --driver-event-limit")

    source_root = pathlib.Path(args.source_root).resolve(strict=True)
    trigger = resolve_trigger(trigger_argv[0])
    driver = resolve_library("libcuda.so.1")
    libcudart = resolve_library("libcudart.so")
    debugger = resolve_executable(os.environ.get("PRIVATE_EXPORT_PROBE_GDB", "gdb"))
    if not debugger.is_file():
        raise RuntimeError("a Python-enabled gdb is required")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required")

    output.mkdir(parents=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "producer_source_commit": run_checked(["git", "-C", str(source_root), "rev-parse", "HEAD"]),
        "gpu": run_checked(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"]
        ).splitlines(),
        "driver": {"path": str(driver), "sha256": sha256(driver), "build_id": elf_build_id(driver)},
        "libcudart": {"path": str(libcudart), "build_id": elf_build_id(libcudart)},
        "debugger": {"path": str(debugger.resolve()), "sha256": sha256(debugger.resolve())},
        "trigger": {"argv": [str(trigger), *trigger_argv[1:]], "path": str(trigger), "sha256": sha256(trigger)},
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "output_dir": str(output),
        "uuid": args.uuid,
        "slot": args.slot,
        "selector": args.selector,
        "memory": args.memory,
        "output_buffer": args.output_buffer,
        "selector_max": args.selector_max,
        "event_limit": args.event_limit,
        "driver_observation": None,
        "inferior_io": None,
    }
    if args.driver_symbol is not None:
        config["driver_observation"] = {
            "symbol": args.driver_symbol,
            "code_register": args.driver_code_register,
            "output_register": args.driver_output_register,
            "event_limit": args.driver_event_limit,
        }
    if args.separate_inferior_io:
        stdout = output / "inferior.stdout"
        stderr = output / "inferior.stderr"
        command_file = output / "inferior-run.gdb"
        run_argv = " ".join(shlex.quote(value) for value in trigger_argv[1:])
        redirections = f">{shlex.quote(str(stdout))} 2>{shlex.quote(str(stderr))}"
        command = "run"
        if run_argv:
            command += " " + run_argv
        command_file.write_text(f"{command} {redirections}\n", encoding="utf-8")
        config["inferior_io"] = {
            "mode": "separate",
            "command_file": str(command_file),
            "stdout": str(stdout),
            "stderr": str(stderr),
        }
    json_dump(output / "identity.json", identity)
    json_dump(output / "probe-config.json", config)
    print(f"private_export_probe_config={output / 'probe-config.json'}")
    return 0


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing required result file: {path}")
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(item, dict):
            raise RuntimeError(f"non-object JSONL record at {path}:{line_number}")
        items.append(item)
    return items


def verify(args: argparse.Namespace) -> int:
    config_path = pathlib.Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = pathlib.Path(config["output_dir"])
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    tables = read_jsonl(output / "tables.jsonl")
    calls = read_jsonl(output / "calls.jsonl")
    returns = read_jsonl(output / "returns.jsonl")
    captures = read_jsonl(output / "captures.jsonl")
    driver_observation = config.get("driver_observation")
    driver_call_stream = read_jsonl(output / "driver-calls.jsonl") if driver_observation is not None else []
    driver_calls = [record for record in driver_call_stream if record.get("kind") == "driver_call"]
    driver_call_errors = [record for record in driver_call_stream if record.get("kind") != "driver_call"]
    driver_returns = read_jsonl(output / "driver-returns.jsonl") if driver_observation is not None else []
    gdb_status = json.loads((output / "gdb-status.json").read_text(encoding="utf-8"))
    table_errors = [record for record in tables if record.get("kind") == "table_error"]
    identity_errors = [record for record in tables if record.get("kind") == "identity_error"]
    debugger_errors = gdb_status.get("fatal_errors", [])
    call_sequences = {record.get("sequence") for record in calls}
    return_sequences = {record.get("sequence") for record in returns}
    pair_errors: list[str] = []
    driver_errors: list[str] = []
    io_errors: list[str] = []
    inferior_io: dict[str, Any] | None = None
    if None in call_sequences or None in return_sequences:
        pair_errors.append("natural call or return record lacks a sequence")
    if call_sequences != return_sequences:
        pair_errors.append("natural call and return sequences differ")
    if gdb_status.get("unreturned_sequences"):
        pair_errors.append("natural private calls remained unreturned at process exit")
    if driver_observation is not None:
        driver_call_sequences = {record.get("sequence") for record in driver_calls}
        driver_return_sequences = {record.get("sequence") for record in driver_returns}
        expected_driver = identity["driver"]
        expected_driver_path = str(pathlib.Path(expected_driver["path"]).resolve(strict=True))
        if driver_call_errors:
            driver_errors.append(f"public Driver observation emitted {len(driver_call_errors)} error records")
        if not driver_calls:
            driver_errors.append(f"declared public Driver symbol was not reached: {driver_observation['symbol']}")
        if len(driver_call_sequences) != len(driver_calls):
            driver_errors.append("public Driver call sequence values are not unique")
        if len(driver_return_sequences) != len(driver_returns):
            driver_errors.append("public Driver return sequence values are not unique")
        if None in driver_call_sequences or None in driver_return_sequences:
            driver_errors.append("public Driver call or return record lacks a sequence")
        if driver_call_sequences != driver_return_sequences:
            driver_errors.append("public Driver call and return sequences differ")
        for record in driver_calls:
            if record.get("symbol") != driver_observation["symbol"]:
                driver_errors.append(f"unexpected public Driver call symbol: {record.get('symbol')}")
            function_mapping = record.get("function_mapping")
            code_mapping = record.get("code_mapping")
            if not isinstance(function_mapping, dict):
                driver_errors.append(f"Driver call {record.get('sequence')} lacks function mapping")
                continue
            for field, expected_value in (
                ("realpath", expected_driver_path),
                ("sha256", expected_driver["sha256"]),
                ("build_id", expected_driver["build_id"]),
            ):
                if function_mapping.get(field) != expected_value:
                    driver_errors.append(
                        f"Driver call {record.get('sequence')} function {field} mismatch: "
                        f"{function_mapping.get(field)} != {expected_value}"
                    )
            if not isinstance(code_mapping, dict):
                driver_errors.append(f"Driver call {record.get('sequence')} lacks code mapping")
            else:
                for field in ("realpath", "sha256", "build_id", "file_offset", "image_offset"):
                    if not code_mapping.get(field):
                        driver_errors.append(f"Driver call {record.get('sequence')} code mapping lacks {field}")
            for field in ("code_pointer", "output_pointer_address", "output_before"):
                if not isinstance(record.get(field), str) or re.fullmatch(r"0x[0-9a-f]+", record[field]) is None:
                    driver_errors.append(f"Driver call {record.get('sequence')} has invalid {field}")
        for record in driver_returns:
            if record.get("kind") != "driver_return":
                driver_errors.append(f"unexpected public Driver return record kind: {record.get('kind')}")
            if record.get("symbol") != driver_observation["symbol"]:
                driver_errors.append(f"unexpected public Driver return symbol: {record.get('symbol')}")
            for field in ("return_rax", "output_after"):
                if not isinstance(record.get(field), str) or re.fullmatch(r"0x[0-9a-f]+", record[field]) is None:
                    driver_errors.append(f"Driver return {record.get('sequence')} has invalid {field}")
        if gdb_status.get("driver_detailed_events") != len(driver_call_stream):
            driver_errors.append("public Driver detailed event count differs from call stream")
        if gdb_status.get("driver_call_sequence") != sum(
            record.get("count", 0) for record in gdb_status.get("driver_call_counts", [])
        ):
            driver_errors.append("public Driver total call count differs from symbol counts")
        if sum(record.get("count", 0) for record in gdb_status.get("driver_contributor_counts", [])) != len(
            driver_calls
        ):
            driver_errors.append("public Driver contributor counts differ from mapped call records")
        if gdb_status.get("driver_unreturned_sequences"):
            driver_errors.append("public Driver calls remained unreturned at process exit")
        if gdb_status.get("driver_event_limit_exhausted"):
            driver_errors.append("public Driver observation exceeded its declared event limit")
    if config.get("inferior_io") is not None:
        inferior_io = {"mode": config["inferior_io"]["mode"]}
        for stream_name in ("stdout", "stderr"):
            stream_path = pathlib.Path(config["inferior_io"][stream_name])
            if not stream_path.is_file():
                io_errors.append(f"missing separated inferior {stream_name}: {stream_path}")
                continue
            inferior_io[stream_name] = {
                "path": str(stream_path),
                "size": stream_path.stat().st_size,
                "sha256": sha256(stream_path),
            }
    expected = {
        "mode": config["mode"],
        "uuid": config.get("uuid"),
        "slot": config.get("slot"),
        "selector": config.get("selector"),
    }
    if config["mode"] == "discovery":
        capture_status = "observed" if calls else "not_reached"
    else:
        capture_status = "reached" if captures else "not_reached"
    if table_errors or identity_errors or debugger_errors or pair_errors or driver_errors or io_errors:
        status = "fail_closed"
    elif gdb_status.get("exit_code") != 0:
        status = "trigger_failed"
    else:
        status = "pass"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "expected": expected,
        "table_records": len(tables),
        "call_records": len(calls),
        "return_records": len(returns),
        "capture_records": len(captures),
        "driver_call_records": len(driver_calls),
        "driver_call_error_records": driver_call_errors,
        "driver_return_records": len(driver_returns),
        "driver_observation": driver_observation,
        "gdb": gdb_status,
        "status": status,
        "capture_status": capture_status,
        "table_errors": table_errors,
        "identity_errors": identity_errors,
        "debugger_errors": debugger_errors,
        "pair_errors": pair_errors,
        "driver_errors": driver_errors,
        "io_errors": io_errors,
        "inferior_io": inferior_io,
    }
    json_dump(output / "summary.json", summary)
    print(f"private_export_probe_status={summary['status']}")
    print(f"private_export_probe_capture_status={summary['capture_status']}")
    return 0 if summary["status"] == "pass" else 2


def debugger_failure(args: argparse.Namespace) -> int:
    config_path = pathlib.Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = pathlib.Path(config["output_dir"])
    for name in ("tables.jsonl", "calls.jsonl", "returns.jsonl", "captures.jsonl"):
        (output / name).touch(exist_ok=True)
    if config.get("driver_observation") is not None:
        for name in ("driver-calls.jsonl", "driver-returns.jsonl"):
            (output / name).touch(exist_ok=True)
    json_dump(
        output / "gdb-status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "exit_code": args.exit_code,
            "fatal_errors": [args.message],
            "call_counts": [],
            "detailed_event_limit": config["event_limit"],
            "unreturned_sequences": [],
            "driver_call_sequence": 0,
            "driver_detailed_events": 0,
            "driver_call_counts": [],
            "driver_contributor_counts": [],
            "driver_event_limit": None if config.get("driver_observation") is None else config["driver_observation"]["event_limit"],
            "driver_event_limit_exhausted": False,
            "driver_unreturned_sequences": [],
        },
    )
    return 0


def compare_identity(args: argparse.Namespace) -> int:
    left_path = pathlib.Path(args.left).resolve(strict=True) / "identity.json"
    right_path = pathlib.Path(args.right).resolve(strict=True) / "identity.json"
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    fields = ("gpu", "driver", "libcudart", "trigger", "producer_source_commit")
    differences = {
        field: {"left": left.get(field), "right": right.get(field)}
        for field in fields
        if left.get(field) != right.get(field)
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "left": str(left_path.parent),
        "right": str(right_path.parent),
        "fields": list(fields),
        "status": "pass" if not differences else "identity_drift",
        "differences": differences,
    }
    json_dump(pathlib.Path(args.output).resolve(), result)
    print(f"private_export_probe_identity_status={result['status']}")
    return 0 if result["status"] == "pass" else 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare", help="write exact probe identity and debugger configuration")
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.add_argument("--mode", required=True, choices=("discovery", "capture"))
    prepare_parser.add_argument("--uuid", type=parse_uuid)
    prepare_parser.add_argument("--slot", type=parse_nonnegative)
    prepare_parser.add_argument("--selector", type=parse_nonnegative)
    prepare_parser.add_argument("--memory", type=parse_memory, action="append", default=[])
    prepare_parser.add_argument("--output-buffer", type=parse_output_buffer)
    prepare_parser.add_argument("--selector-max", type=parse_nonnegative, default=0x3FF)
    prepare_parser.add_argument("--event-limit", type=parse_nonnegative, default=DEFAULT_EVENT_LIMIT)
    prepare_parser.add_argument("--driver-symbol", type=parse_driver_symbol)
    prepare_parser.add_argument("--driver-code-register", type=parse_driver_register)
    prepare_parser.add_argument("--driver-output-register", type=parse_driver_register)
    prepare_parser.add_argument("--driver-event-limit", type=parse_nonnegative)
    prepare_parser.add_argument("--separate-inferior-io", action="store_true")
    prepare_parser.add_argument("trigger", nargs=argparse.REMAINDER)
    prepare_parser.set_defaults(handler=prepare)
    verify_parser = subcommands.add_parser("verify", help="validate debugger output and write summary.json")
    verify_parser.add_argument("--config", required=True)
    verify_parser.set_defaults(handler=verify)
    failure_parser = subcommands.add_parser("debugger-failure", help="write fail-closed output after debugger startup failure")
    failure_parser.add_argument("--config", required=True)
    failure_parser.add_argument("--exit-code", type=int, required=True)
    failure_parser.add_argument("--message", required=True)
    failure_parser.set_defaults(handler=debugger_failure)
    compare_parser = subcommands.add_parser("compare-identity", help="fail closed on discovery/capture identity drift")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(handler=compare_identity)
    return parser.parse_args(argv)


try:
    import gdb  # type: ignore
except ImportError:
    gdb = None


if gdb is not None:

    def hex_address(value: int | None) -> str | None:
        return None if value is None else f"0x{value:x}"


    @dataclass(frozen=True)
    class TableSlot:
        uuid: str
        slot: int


    @dataclass
    class ProbeObserver:
        config: dict[str, Any]
        identity: dict[str, Any]
        output: pathlib.Path
        slot_by_address: dict[int, set[TableSlot]] = field(default_factory=dict)
        slot_breakpoints: dict[int, Any] = field(default_factory=dict)
        call_counts: dict[TableSlot, int] = field(default_factory=dict)
        detailed_events: int = 0
        call_sequence: int = 0
        capture_sequence: int = 0
        unreturned_sequences: set[int] = field(default_factory=set)
        driver_call_sequence: int = 0
        driver_detailed_events: int = 0
        driver_call_counts: dict[str, int] = field(default_factory=dict)
        driver_contributor_counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
        driver_unreturned_sequences: set[int] = field(default_factory=set)
        driver_event_limit_exhausted: bool = False
        driver_identity_cache: dict[str, dict[str, str]] = field(default_factory=dict)
        fatal_errors: list[str] = field(default_factory=list)

        def append_table(self, record: dict[str, Any]) -> None:
            append_jsonl(self.output / "tables.jsonl", record)

        def append_call(self, record: dict[str, Any]) -> bool:
            if self.detailed_events < self.config["event_limit"]:
                append_jsonl(self.output / "calls.jsonl", record)
                self.detailed_events += 1
                return True
            return False

        def append_return(self, record: dict[str, Any]) -> None:
            append_jsonl(self.output / "returns.jsonl", record)

        def append_capture(self, record: dict[str, Any]) -> None:
            append_jsonl(self.output / "captures.jsonl", record)

        def append_driver_call(self, record: dict[str, Any]) -> None:
            append_jsonl(self.output / "driver-calls.jsonl", record)

        def append_driver_return(self, record: dict[str, Any]) -> None:
            append_jsonl(self.output / "driver-returns.jsonl", record)

        def register_table(self, uuid: str, table: int) -> None:
            try:
                shape, words, callable_start = self.parse_table(table)
            except Exception as error:  # The debugger must preserve the target process and record the ambiguity.
                message = f"uuid={uuid} table={hex_address(table)} error={error}"
                self.fatal_errors.append(message)
                self.append_table({"kind": "table_error", "uuid": uuid, "table": hex_address(table), "error": str(error)})
                return
            slots: list[dict[str, Any]] = []
            for index, address in enumerate(words):
                callable_slot = index >= callable_start and address not in (0, (1 << 64) - 1)
                slots.append(
                    {
                        "slot": index,
                        "address": hex_address(address),
                        "present": address not in (0, (1 << 64) - 1),
                        "callable": callable_slot,
                    }
                )
                if callable_slot:
                    mapping = TableSlot(uuid=uuid, slot=index)
                    self.slot_by_address.setdefault(address, set()).add(mapping)
                    self.call_counts.setdefault(mapping, 0)
                    if address not in self.slot_breakpoints:
                        self.slot_breakpoints[address] = TableFunctionBreakpoint(self, address)
            self.append_table(
                {
                    "kind": "table",
                    "uuid": uuid,
                    "table": hex_address(table),
                    "shape": shape,
                    "word_count": len(words),
                    "slots": slots,
                }
            )

        def parse_table(self, table: int) -> tuple[str, list[int], int]:
            first = self.read_word(table)
            if POINTER_BYTES <= first <= MAX_TABLE_BYTES and first % POINTER_BYTES == 0:
                count = first // POINTER_BYTES
                if count < 2:
                    raise RuntimeError(f"size-prefixed table has too few words: {count}")
                return "size_prefixed", [self.read_word(table + index * POINTER_BYTES) for index in range(count)], 1
            words: list[int] = []
            for index in range(MAX_TERMINATED_WORDS):
                word = self.read_word(table + index * POINTER_BYTES)
                words.append(word)
                if word in (0, (1 << 64) - 1):
                    if index == 0:
                        raise RuntimeError("terminator table has no callable slot")
                    return "terminated", words, 0
            raise RuntimeError(f"terminator missing within {MAX_TERMINATED_WORDS} words")

        @staticmethod
        def inferior() -> Any:
            inferior = gdb.selected_inferior()
            if inferior is None:
                raise RuntimeError("no selected inferior")
            return inferior

        def read_word(self, address: int) -> int:
            return int.from_bytes(self.inferior().read_memory(address, POINTER_BYTES).tobytes(), "little")

        def read_bytes(self, address: int, count: int) -> dict[str, Any]:
            if address == 0:
                return {"status": "unavailable", "reason": "null"}
            try:
                raw = self.inferior().read_memory(address, count).tobytes()
            except gdb.MemoryError as error:
                return {"status": "unavailable", "reason": str(error)}
            return {"status": "ok", "address": hex_address(address), "bytes_hex": raw.hex()}

        def observe_output_buffer(self, entry_registers: dict[str, str]) -> dict[str, Any] | None:
            declaration = self.config.get("output_buffer")
            if declaration is None:
                return None
            pointer_out_address = int(entry_registers[declaration["pointer_register"]], 16)
            size_out_address = int(entry_registers[declaration["size_register"]], 16)
            try:
                pointer = self.read_word(pointer_out_address)
                size = self.read_word(size_out_address)
            except Exception as error:
                return {
                    "status": "unavailable",
                    "reason": str(error),
                    "pointer_out_register": declaration["pointer_register"],
                    "size_out_register": declaration["size_register"],
                    "max_bytes": declaration["max_bytes"],
                }
            result: dict[str, Any] = {
                "status": "ok",
                "pointer_out_register": declaration["pointer_register"],
                "size_out_register": declaration["size_register"],
                "pointer": hex_address(pointer),
                "size": size,
                "max_bytes": declaration["max_bytes"],
            }
            if size > declaration["max_bytes"]:
                result["content"] = {"status": "unavailable", "reason": "declared_max_exceeded"}
                return result
            result["content"] = self.read_bytes(pointer, size)
            return result

        @staticmethod
        def register(name: str) -> int:
            return int(gdb.parse_and_eval("$" + name))

        def registers(self) -> dict[str, str]:
            return {name: hex_address(self.register(name)) or "0x0" for name in INTEGER_REGISTERS}

        def caller(self) -> str | None:
            try:
                return hex_address(self.read_word(self.register("rsp")))
            except Exception:
                return None

        def read_proc_maps(self) -> list[dict[str, Any]]:
            pid = self.inferior().pid
            if not isinstance(pid, int) or pid <= 0:
                raise RuntimeError(f"inferior has no live pid: {pid}")
            return parse_proc_maps(pathlib.Path(f"/proc/{pid}/maps").read_text(encoding="utf-8"))

        def mapped_elf_identity(self, address: int) -> dict[str, Any]:
            mapping = locate_proc_mapping(address, self.read_proc_maps())
            realpath = pathlib.Path(mapping["path"]).resolve(strict=True)
            cache_key = str(realpath)
            file_identity = self.driver_identity_cache.get(cache_key)
            if file_identity is None:
                file_identity = {
                    "realpath": cache_key,
                    "sha256": sha256(realpath),
                    "build_id": elf_build_id(realpath),
                }
                self.driver_identity_cache[cache_key] = file_identity
            return {
                "path": mapping["path"],
                **file_identity,
                "start": hex_address(mapping["start"]),
                "end": hex_address(mapping["end"]),
                "permissions": mapping["permissions"],
                "mapping_offset": hex_address(mapping["mapping_offset"]),
                "load_base": hex_address(mapping["load_base"]),
                "file_offset": hex_address(mapping["file_offset"]),
                "image_offset": hex_address(mapping["image_offset"]),
                "device": mapping["device"],
                "inode": mapping["inode"],
            }

        def validate_driver_function_mapping(self, mapping: dict[str, Any]) -> None:
            expected = self.identity["driver"]
            expected_path = str(pathlib.Path(expected["path"]).resolve(strict=True))
            differences = {
                field: {"expected": expected_value, "observed": mapping.get(field)}
                for field, expected_value in (
                    ("realpath", expected_path),
                    ("sha256", expected["sha256"]),
                    ("build_id", expected["build_id"]),
                )
                if mapping.get(field) != expected_value
            }
            if differences:
                raise RuntimeError(f"public Driver breakpoint did not enter declared host Driver: {differences}")

        def observe_driver_entry(self, symbol: str) -> None:
            declaration = self.config["driver_observation"]
            self.driver_call_sequence += 1
            sequence = self.driver_call_sequence
            self.driver_call_counts[symbol] = self.driver_call_counts.get(symbol, 0) + 1
            if sequence > declaration["event_limit"]:
                self.driver_event_limit_exhausted = True
                return

            self.driver_detailed_events += 1
            registers = self.registers()
            code_pointer = int(registers[declaration["code_register"]], 16)
            output_pointer_address = int(registers[declaration["output_register"]], 16)
            try:
                function_mapping = self.mapped_elf_identity(self.register("pc"))
                self.validate_driver_function_mapping(function_mapping)
                code_mapping = self.mapped_elf_identity(code_pointer)
                output_before = self.read_word(output_pointer_address)
                contributor_key = (
                    code_mapping["realpath"],
                    code_mapping["sha256"],
                    code_mapping["build_id"],
                )
                self.driver_contributor_counts[contributor_key] = (
                    self.driver_contributor_counts.get(contributor_key, 0) + 1
                )
                thread = gdb.selected_thread()
                self.append_driver_call(
                    {
                        "kind": "driver_call",
                        "sequence": sequence,
                        "symbol": symbol,
                        "thread": None if thread is None else thread.global_num,
                        "caller": self.caller(),
                        "entry_registers": registers,
                        "code_pointer": hex_address(code_pointer),
                        "output_pointer_address": hex_address(output_pointer_address),
                        "output_before": hex_address(output_before),
                        "function_mapping": function_mapping,
                        "code_mapping": code_mapping,
                    }
                )
                self.driver_unreturned_sequences.add(sequence)
                PublicDriverReturnBreakpoint(
                    self,
                    gdb.newest_frame(),
                    sequence,
                    symbol,
                    output_pointer_address,
                )
            except Exception as error:
                message = f"{symbol} entry observation failed at sequence {sequence}: {error}"
                self.fatal_errors.append(message)
                self.append_driver_call(
                    {
                        "kind": "driver_call_error",
                        "sequence": sequence,
                        "symbol": symbol,
                        "caller": self.caller(),
                        "entry_registers": registers,
                        "code_pointer": hex_address(code_pointer),
                        "output_pointer_address": hex_address(output_pointer_address),
                        "error": str(error),
                    }
                )

        def observe_driver_return(self, sequence: int, symbol: str, output_pointer_address: int) -> None:
            try:
                output_after = self.read_word(output_pointer_address)
                return_rax = self.register("rax")
            except Exception as error:
                self.fatal_errors.append(f"{symbol} return observation failed at sequence {sequence}: {error}")
                return
            self.append_driver_return(
                {
                    "kind": "driver_return",
                    "sequence": sequence,
                    "symbol": symbol,
                    "return_rax": hex_address(return_rax),
                    "output_after": hex_address(output_after),
                }
            )
            self.driver_unreturned_sequences.discard(sequence)

        def observe_export_entry(self) -> None:
            try:
                pp_table = self.register("rdi")
                uuid_pointer = self.register("rsi")
                uuid = canonical_uuid(self.inferior().read_memory(uuid_pointer, 16).tobytes())
                ExportTableReturnBreakpoint(self, gdb.newest_frame(), uuid, pp_table)
            except Exception as error:
                message = f"cuGetExportTable entry observation failed: {error}"
                self.fatal_errors.append(message)
                self.append_table({"kind": "identity_error", "stage": "cuGetExportTable_entry", "error": str(error)})

        def observe_export_return(self, uuid: str, pp_table: int) -> None:
            result = self.register("rax")
            if result != 0:
                self.append_table({"kind": "export_result", "uuid": uuid, "result": result, "table": None})
                return
            try:
                table = self.read_word(pp_table)
            except Exception as error:
                self.fatal_errors.append(f"cuGetExportTable return read failed: {error}")
                self.append_table({"kind": "table_error", "uuid": uuid, "table": None, "error": str(error)})
                return
            if table == 0:
                self.fatal_errors.append(f"cuGetExportTable returned NULL table for {uuid}")
                self.append_table({"kind": "table_error", "uuid": uuid, "table": None, "error": "successful return with NULL table"})
                return
            self.register_table(uuid, table)

        def observe_function_entry(self, address: int) -> None:
            mappings = sorted(self.slot_by_address[address], key=lambda item: (item.uuid, item.slot))
            registers = self.registers()
            rdi = int(registers["rdi"], 16)
            selector_candidate = rdi if rdi <= self.config["selector_max"] else None
            capture_matches = [mapping for mapping in mappings if self.capture_matches(mapping, selector_candidate)]
            for mapping in mappings:
                self.call_counts[mapping] = self.call_counts.get(mapping, 0) + 1
            self.call_sequence += 1
            sequence = self.call_sequence
            thread = gdb.selected_thread()
            recorded = self.append_call(
                {
                    "kind": "call",
                    "sequence": sequence,
                    "address": hex_address(address),
                    "mappings": [{"uuid": item.uuid, "slot": item.slot} for item in mappings],
                    "caller": self.caller(),
                    "thread": None if thread is None else thread.global_num,
                    "entry_registers": registers,
                    "selector_candidate": selector_candidate,
                    "capture_match": bool(capture_matches),
                }
            )
            if recorded or capture_matches:
                self.unreturned_sequences.add(sequence)
                capture_sequence = None
                if capture_matches:
                    self.capture_sequence += 1
                    capture_sequence = self.capture_sequence
                NaturalCallReturnBreakpoint(
                    self,
                    gdb.newest_frame(),
                    sequence,
                    mappings,
                    registers,
                    recorded,
                    capture_sequence,
                    capture_matches,
                )

        def capture_matches(self, mapping: TableSlot, selector_candidate: int | None) -> bool:
            if self.config["mode"] != "capture":
                return False
            if mapping.uuid != self.config["uuid"] or mapping.slot != self.config["slot"]:
                return False
            selector = self.config.get("selector")
            return selector is None or selector == selector_candidate

        def on_exit(self, event: Any) -> None:
            calls = [
                {"uuid": item.uuid, "slot": item.slot, "count": count}
                for item, count in sorted(self.call_counts.items(), key=lambda pair: (pair[0].uuid, pair[0].slot))
            ]
            driver_calls = [
                {"symbol": symbol, "count": count}
                for symbol, count in sorted(self.driver_call_counts.items())
            ]
            driver_contributors = [
                {
                    "realpath": identity[0],
                    "sha256": identity[1],
                    "build_id": identity[2],
                    "count": count,
                }
                for identity, count in sorted(self.driver_contributor_counts.items())
            ]
            driver_observation = self.config.get("driver_observation")
            json_dump(
                self.output / "gdb-status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "exit_code": getattr(event, "exit_code", None),
                    "fatal_errors": self.fatal_errors,
                    "call_counts": calls,
                    "detailed_event_limit": self.config["event_limit"],
                    "unreturned_sequences": sorted(self.unreturned_sequences),
                    "driver_call_sequence": self.driver_call_sequence,
                    "driver_detailed_events": self.driver_detailed_events,
                    "driver_call_counts": driver_calls,
                    "driver_contributor_counts": driver_contributors,
                    "driver_event_limit": None if driver_observation is None else driver_observation["event_limit"],
                    "driver_event_limit_exhausted": self.driver_event_limit_exhausted,
                    "driver_unreturned_sequences": sorted(self.driver_unreturned_sequences),
                },
            )


    class ExportTableEntryBreakpoint(gdb.Breakpoint):
        def __init__(self, observer: ProbeObserver) -> None:
            super().__init__("cuGetExportTable", internal=True)
            self.observer = observer

        def stop(self) -> bool:
            self.observer.observe_export_entry()
            return False


    class ExportTableReturnBreakpoint(gdb.FinishBreakpoint):
        def __init__(self, observer: ProbeObserver, frame: Any, uuid: str, pp_table: int) -> None:
            super().__init__(frame, internal=True)
            self.observer = observer
            self.uuid = uuid
            self.pp_table = pp_table

        def stop(self) -> bool:
            self.observer.observe_export_return(self.uuid, self.pp_table)
            return False


    class TableFunctionBreakpoint(gdb.Breakpoint):
        def __init__(self, observer: ProbeObserver, address: int) -> None:
            super().__init__(f"*{address:#x}", internal=True)
            self.observer = observer
            self.address = address

        def stop(self) -> bool:
            self.observer.observe_function_entry(self.address)
            return False


    class NaturalCallReturnBreakpoint(gdb.FinishBreakpoint):
        def __init__(
            self,
            observer: ProbeObserver,
            frame: Any,
            call_sequence: int,
            mappings: list[TableSlot],
            entry_registers: dict[str, str],
            record_return: bool,
            capture_sequence: int | None,
            capture_mappings: list[TableSlot],
        ) -> None:
            super().__init__(frame, internal=True)
            self.observer = observer
            self.call_sequence = call_sequence
            self.mappings = mappings
            self.entry_registers = entry_registers
            self.record_return = record_return
            self.capture_sequence = capture_sequence
            self.capture_mappings = capture_mappings
            self.memory_before: list[dict[str, Any]] = []
            if capture_mappings:
                for memory in observer.config.get("memory", []):
                    pointer = int(entry_registers[memory["register"]], 16)
                    self.memory_before.append(
                        {
                            "register": memory["register"],
                            "bytes": memory["bytes"],
                            "value": observer.read_bytes(pointer, memory["bytes"]),
                        }
                    )

        def stop(self) -> bool:
            return_rax = hex_address(self.observer.register("rax"))
            self.observer.unreturned_sequences.discard(self.call_sequence)
            if self.record_return:
                self.observer.append_return(
                    {
                        "kind": "return",
                        "sequence": self.call_sequence,
                        "mappings": [{"uuid": item.uuid, "slot": item.slot} for item in self.mappings],
                        "return_rax": return_rax,
                    }
                )
            if not self.capture_mappings:
                return False
            after: list[dict[str, Any]] = []
            for memory in self.observer.config.get("memory", []):
                pointer = int(self.entry_registers[memory["register"]], 16)
                after.append(
                    {
                        "register": memory["register"],
                        "bytes": memory["bytes"],
                        "value": self.observer.read_bytes(pointer, memory["bytes"]),
                    }
                )
            self.observer.append_capture(
                {
                    "kind": "capture",
                    "sequence": self.capture_sequence,
                    "call_sequence": self.call_sequence,
                    "mappings": [{"uuid": item.uuid, "slot": item.slot} for item in self.capture_mappings],
                    "entry_registers": self.entry_registers,
                    "return_rax": return_rax,
                    "memory_windows_before": self.memory_before,
                    "memory_windows_after": after,
                    "output_buffer_after": self.observer.observe_output_buffer(self.entry_registers),
                }
            )
            return False


    class PublicDriverEntryBreakpoint(gdb.Breakpoint):
        def __init__(self, observer: ProbeObserver, symbol: str) -> None:
            super().__init__(symbol, internal=True)
            self.observer = observer
            self.symbol = symbol

        def stop(self) -> bool:
            self.observer.observe_driver_entry(self.symbol)
            return False


    class PublicDriverReturnBreakpoint(gdb.FinishBreakpoint):
        def __init__(
            self,
            observer: ProbeObserver,
            frame: Any,
            sequence: int,
            symbol: str,
            output_pointer_address: int,
        ) -> None:
            super().__init__(frame, internal=True)
            self.observer = observer
            self.sequence = sequence
            self.symbol = symbol
            self.output_pointer_address = output_pointer_address

        def stop(self) -> bool:
            self.observer.observe_driver_return(self.sequence, self.symbol, self.output_pointer_address)
            return False


    class PrivateExportProbeCommand(gdb.Command):
        def __init__(self) -> None:
            super().__init__("private-export-probe", gdb.COMMAND_OBSCURE)

        def invoke(self, argument: str, from_tty: bool) -> None:
            config_path = pathlib.Path(argument.strip()).resolve(strict=True)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            output = pathlib.Path(config["output_dir"])
            identity_path = output / "identity.json"
            if not identity_path.is_file():
                raise gdb.GdbError("missing identity.json written by prepare")
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            for name in ("tables.jsonl", "calls.jsonl", "returns.jsonl", "captures.jsonl"):
                (output / name).touch(exist_ok=False)
            driver_observation = config.get("driver_observation")
            if driver_observation is not None:
                for name in ("driver-calls.jsonl", "driver-returns.jsonl"):
                    (output / name).touch(exist_ok=False)
            gdb.execute("set breakpoint pending on")
            observer = ProbeObserver(config=config, identity=identity, output=output)
            ExportTableEntryBreakpoint(observer)
            if driver_observation is not None:
                PublicDriverEntryBreakpoint(observer, driver_observation["symbol"])
            gdb.events.exited.connect(observer.on_exit)
            gdb.write(
                f"private_export_probe_ready mode={config['mode']} "
                f"driver_symbol={None if driver_observation is None else driver_observation['symbol']}\n"
            )


    PrivateExportProbeCommand()


def main(argv: list[str]) -> int:
    if argv == ["--hint"]:
        print(HINT, end="")
        return 0
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__" and gdb is None:
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"private_export_probe_error={error}", file=sys.stderr)
        raise SystemExit(2)

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
HINT = """private_export_probe_hint=self=qemu_integration/guest_libcuda/private_export_probe.py
private_export_probe_hint=problem=CUDA Runtime private export tables are undocumented; a NULL guest entry and a host table address reveal neither the slot signature nor the selector-specific state transition needed for a safe shim implementation
private_export_probe_hint=mental_model=Python-enabled GDB observes real cuGetExportTable returns and naturally executed table-entry calls; every detailed natural call receives a sequence-bound entry/return record, while capture filters one declared UUID/slot/selector for bounded argument memory deltas and one declared returned buffer
private_export_probe_hint=role=implement the debugger-side observer and machine-readable table/call/capture event model used by multiple public CUDA triggers
private_export_probe_hint=use_when=a public CUDA API or exact application trigger naturally reaches a private table path on the matching real NVIDIA Driver/Runtime and the unknown ABI must be constrained before changing the guest shim
private_export_probe_hint=inputs=live GDB inferior running an explicit trigger; mode discovery or capture; optional UUID, slot, selector, bounded register-memory windows, one POINTER_OUT:SIZE_OUT:MAX_BYTES output-buffer projection, event limit and explicit inferior I/O separation
private_export_probe_hint=outputs=identity.json,probe-config.json,tables.jsonl,calls.jsonl,returns.jsonl,captures.jsonl,gdb-status.json,summary.json and GDB-visible diagnostic messages; separated runs additionally preserve inferior-run.gdb,inferior.stdout,inferior.stderr
private_export_probe_hint=interpret=calls and returns with the same sequence are one naturally executed private call; capture reached additionally requires the declared UUID/slot/selector and bounded memory pair; not_reached supplies no target ABI authority
private_export_probe_hint=proves=which private export tables and slots the real Runtime naturally reached, their observed entry registers and return RAX, bounded argument-memory facts and, only when declared, one first-level returned buffer pointer, length and bounded bytes
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
        "driver": {"path": str(driver), "build_id": elf_build_id(driver)},
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
        "inferior_io": None,
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
    gdb_status = json.loads((output / "gdb-status.json").read_text(encoding="utf-8"))
    table_errors = [record for record in tables if record.get("kind") == "table_error"]
    identity_errors = [record for record in tables if record.get("kind") == "identity_error"]
    debugger_errors = gdb_status.get("fatal_errors", [])
    call_sequences = {record.get("sequence") for record in calls}
    return_sequences = {record.get("sequence") for record in returns}
    pair_errors: list[str] = []
    io_errors: list[str] = []
    inferior_io: dict[str, Any] | None = None
    if None in call_sequences or None in return_sequences:
        pair_errors.append("natural call or return record lacks a sequence")
    if call_sequences != return_sequences:
        pair_errors.append("natural call and return sequences differ")
    if gdb_status.get("unreturned_sequences"):
        pair_errors.append("natural private calls remained unreturned at process exit")
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
    if table_errors or identity_errors or debugger_errors or pair_errors or io_errors:
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
        "gdb": gdb_status,
        "status": status,
        "capture_status": capture_status,
        "table_errors": table_errors,
        "identity_errors": identity_errors,
        "debugger_errors": debugger_errors,
        "pair_errors": pair_errors,
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
    json_dump(
        output / "gdb-status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "exit_code": args.exit_code,
            "fatal_errors": [args.message],
            "call_counts": [],
            "detailed_event_limit": config["event_limit"],
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
        output: pathlib.Path
        slot_by_address: dict[int, set[TableSlot]] = field(default_factory=dict)
        slot_breakpoints: dict[int, Any] = field(default_factory=dict)
        call_counts: dict[TableSlot, int] = field(default_factory=dict)
        detailed_events: int = 0
        call_sequence: int = 0
        capture_sequence: int = 0
        unreturned_sequences: set[int] = field(default_factory=set)
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
            names = ("rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp")
            return {name: hex_address(self.register(name)) or "0x0" for name in names}

        def caller(self) -> str | None:
            try:
                return hex_address(self.read_word(self.register("rsp")))
            except Exception:
                return None

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
            json_dump(
                self.output / "gdb-status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "exit_code": getattr(event, "exit_code", None),
                    "fatal_errors": self.fatal_errors,
                    "call_counts": calls,
                    "detailed_event_limit": self.config["event_limit"],
                    "unreturned_sequences": sorted(self.unreturned_sequences),
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


    class PrivateExportProbeCommand(gdb.Command):
        def __init__(self) -> None:
            super().__init__("private-export-probe", gdb.COMMAND_OBSCURE)

        def invoke(self, argument: str, from_tty: bool) -> None:
            config_path = pathlib.Path(argument.strip()).resolve(strict=True)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            output = pathlib.Path(config["output_dir"])
            if not (output / "identity.json").is_file():
                raise gdb.GdbError("missing identity.json written by prepare")
            for name in ("tables.jsonl", "calls.jsonl", "returns.jsonl", "captures.jsonl"):
                (output / name).touch(exist_ok=False)
            gdb.execute("set breakpoint pending on")
            observer = ProbeObserver(config=config, output=output)
            ExportTableEntryBreakpoint(observer)
            gdb.events.exited.connect(observer.on_exit)
            gdb.write(f"private_export_probe_ready mode={config['mode']}\n")


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

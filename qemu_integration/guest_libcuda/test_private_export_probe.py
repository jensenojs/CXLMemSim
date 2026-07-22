#!/usr/bin/env python3

import argparse
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from private_export_probe import (
    elf_load_segments,
    locate_proc_mapping,
    observe_table_slot,
    parse_driver_register,
    parse_driver_symbol,
    parse_proc_maps,
    verify,
)


class TableSlotSelectionTests(unittest.TestCase):
    def test_discovery_observes_every_callable_slot(self) -> None:
        config = {"mode": "discovery", "uuid": None, "slots": []}

        self.assertTrue(observe_table_slot(config, "first", 1))
        self.assertTrue(observe_table_slot(config, "second", 99))

    def test_capture_observes_only_the_declared_uuid_and_slot(self) -> None:
        config = {"mode": "capture", "uuid": "target", "slots": [4, 39, 51]}

        self.assertTrue(observe_table_slot(config, "target", 39))
        self.assertTrue(observe_table_slot(config, "target", 4))
        self.assertFalse(observe_table_slot(config, "target", 40))
        self.assertFalse(observe_table_slot(config, "other", 39))


class ProcMappingTests(unittest.TestCase):
    def test_locates_file_and_image_offsets_across_pt_load_mappings(self) -> None:
        mappings = parse_proc_maps(
            "\n".join(
                (
                    "0000000070000000-0000000070001000 r--p 00000000 08:01 42 /opt/libexample.so",
                    "0000000070001000-0000000070004000 r-xp 00001000 08:01 42 /opt/libexample.so",
                    "0000000070004000-0000000070005000 rw-p 00004000 08:01 42 /opt/libexample.so",
                )
            )
        )

        result = locate_proc_mapping(
            0x70001234,
            mappings,
            [
                {
                    "offset": 0,
                    "virtual_address": 0,
                    "file_size": 0x5000,
                    "memory_size": 0x5000,
                    "flags": 5,
                    "alignment": 0x1000,
                }
            ],
        )

        self.assertEqual(result["path"], "/opt/libexample.so")
        self.assertEqual(result["load_base"], 0x70000000)
        self.assertEqual(result["mapping_offset"], 0x1000)
        self.assertEqual(result["file_offset"], 0x1234)
        self.assertEqual(result["image_offset"], 0x1234)

    def test_rejects_anonymous_mapping(self) -> None:
        mappings = parse_proc_maps("0000000071000000-0000000071001000 rw-p 00000000 00:00 0 [heap]")

        with self.assertRaisesRegex(RuntimeError, "live absolute ELF path"):
            locate_proc_mapping(0x71000010, mappings, [])

    def test_rejects_deleted_mapping(self) -> None:
        mappings = parse_proc_maps(
            "0000000072000000-0000000072001000 r-xp 00000000 08:01 43 /opt/libgone.so (deleted)"
        )

        with self.assertRaisesRegex(RuntimeError, "live absolute ELF path"):
            locate_proc_mapping(0x72000010, mappings, [])

    def test_uses_pt_load_offsets_when_virtual_address_differs_by_one_page(self) -> None:
        mappings = parse_proc_maps(
            "\n".join(
                (
                    "0000000073000000-0000000073001000 r--p 00000000 08:01 44 /opt/libtwice.so",
                    "0000000073004000-0000000073005000 rw-p 00003000 08:01 44 /opt/libtwice.so",
                )
            )
        )

        result = locate_proc_mapping(
            0x73004010,
            mappings,
            [
                {
                    "offset": 0,
                    "virtual_address": 0,
                    "file_size": 0x1000,
                    "memory_size": 0x1000,
                    "flags": 4,
                    "alignment": 0x1000,
                },
                {
                    "offset": 0x3010,
                    "virtual_address": 0x4010,
                    "file_size": 0x800,
                    "memory_size": 0x1000,
                    "flags": 6,
                    "alignment": 0x1000,
                },
            ],
        )

        self.assertEqual(result["load_base"], 0x73000000)
        self.assertEqual(result["file_offset"], 0x3010)
        self.assertEqual(result["image_offset"], 0x4010)
        self.assertEqual(result["segment_offset"], 0x3010)
        self.assertEqual(result["segment_virtual_address"], 0x4010)

    def test_rejects_file_offset_outside_pt_load_file_bytes(self) -> None:
        mappings = parse_proc_maps(
            "0000000074000000-0000000074001000 r-xp 00002000 08:01 45 /opt/libgap.so"
        )

        with self.assertRaisesRegex(RuntimeError, "maps to 0 PT_LOAD origins"):
            locate_proc_mapping(
                0x74000010,
                mappings,
                [
                    {
                        "offset": 0,
                        "virtual_address": 0,
                        "file_size": 0x1000,
                        "memory_size": 0x1000,
                        "flags": 5,
                        "alignment": 0x1000,
                    }
                ],
            )

    def test_reads_pt_load_segments_from_real_elf(self) -> None:
        segments = elf_load_segments(pathlib.Path("/bin/true").resolve(strict=True))

        self.assertTrue(segments)
        self.assertTrue(all(segment["file_size"] <= segment["memory_size"] for segment in segments))


class DriverArgumentTests(unittest.TestCase):
    def test_accepts_public_symbol_and_argument_registers(self) -> None:
        self.assertEqual(parse_driver_symbol("cuLibraryLoadData"), "cuLibraryLoadData")
        self.assertEqual(parse_driver_register("rsi"), "rsi")
        self.assertEqual(parse_driver_register("rdi"), "rdi")

    def test_rejects_gdb_expressions(self) -> None:
        for value in ("$rsi", "rsi+8", "*(void**)rsi", "rsp"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_driver_register(value)
        for value in ("LibraryLoadData", "cuLibraryLoadData@plt", "cuLibraryLoadData;quit"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_driver_symbol(value)


class PrivateCallVerificationTests(unittest.TestCase):
    def test_accepts_private_call_with_exact_caller_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            config_path = output / "probe-config.json"
            caller_mapping = {
                "realpath": "/opt/libcublasLt.so.12",
                "sha256": "caller-sha",
                "build_id": "caller-build",
                "image_offset": "0x1234",
            }
            config = {
                "output_dir": str(output),
                "mode": "discovery",
                "uuid": None,
                "slot": None,
                "selector": None,
                "driver_observation": None,
                "resolver_observation": None,
                "inferior_io": None,
            }
            call = {
                "kind": "call",
                "sequence": 1,
                "caller_mapping": caller_mapping,
            }
            call_return = {"kind": "return", "sequence": 1, "return_rax": "0x0"}
            status = {"exit_code": 0, "fatal_errors": [], "unreturned_sequences": []}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (output / "identity.json").write_text(json.dumps({}), encoding="utf-8")
            (output / "tables.jsonl").write_text("", encoding="utf-8")
            (output / "calls.jsonl").write_text(json.dumps(call) + "\n", encoding="utf-8")
            (output / "returns.jsonl").write_text(json.dumps(call_return) + "\n", encoding="utf-8")
            (output / "captures.jsonl").write_text("", encoding="utf-8")
            (output / "gdb-status.json").write_text(json.dumps(status), encoding="utf-8")

            result = verify(SimpleNamespace(config=str(config_path)))

            self.assertEqual(result, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["call_records"], 1)

    def test_rejects_capture_without_exact_caller_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            config_path = output / "probe-config.json"
            config = {
                "output_dir": str(output),
                "mode": "capture",
                "uuid": "target",
                "slots": [4],
                "selector": None,
                "driver_observation": None,
                "resolver_observation": None,
                "inferior_io": None,
            }
            capture = {
                "kind": "capture",
                "sequence": 1,
                "call_sequence": 1,
                "mappings": [{"uuid": "target", "slot": 4}],
                "return_rax": "0x0",
            }
            status = {"exit_code": 0, "fatal_errors": [], "unreturned_sequences": []}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (output / "identity.json").write_text(json.dumps({}), encoding="utf-8")
            (output / "tables.jsonl").write_text("", encoding="utf-8")
            (output / "calls.jsonl").write_text("", encoding="utf-8")
            (output / "returns.jsonl").write_text("", encoding="utf-8")
            (output / "captures.jsonl").write_text(json.dumps(capture) + "\n", encoding="utf-8")
            (output / "gdb-status.json").write_text(json.dumps(status), encoding="utf-8")

            result = verify(SimpleNamespace(config=str(config_path)))

            self.assertEqual(result, 2)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("capture 1 lacks caller_mapping", summary["pair_errors"])


class DriverVerificationTests(unittest.TestCase):
    def test_accepts_one_complete_exact_driver_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            config_path = output / "probe-config.json"
            driver_path = str(pathlib.Path("/bin/true").resolve(strict=True))
            config = {
                "output_dir": str(output),
                "mode": "discovery",
                "uuid": None,
                "slot": None,
                "selector": None,
                "driver_observation": {
                    "symbol": "cuLibraryLoadData",
                    "code_register": "rsi",
                    "output_register": "rdi",
                    "event_limit": 2,
                },
                "inferior_io": None,
            }
            identity = {
                "driver": {"path": driver_path, "sha256": "driver-sha", "build_id": "driver-build"}
            }
            function_mapping = {
                "realpath": driver_path,
                "sha256": "driver-sha",
                "build_id": "driver-build",
            }
            code_mapping = {
                "realpath": "/opt/libcublas.so.12",
                "sha256": "cublas-sha",
                "build_id": "cublas-build",
                "file_offset": "0x1234",
                "image_offset": "0x1234",
            }
            driver_call = {
                "kind": "driver_call",
                "sequence": 1,
                "symbol": "cuLibraryLoadData",
                "code_pointer": "0x70001234",
                "output_pointer_address": "0x7fff0000",
                "output_before": "0x0",
                "function_mapping": function_mapping,
                "code_mapping": code_mapping,
            }
            driver_return = {
                "kind": "driver_return",
                "sequence": 1,
                "symbol": "cuLibraryLoadData",
                "return_rax": "0x0",
                "output_after": "0x90000000",
            }
            status = {
                "exit_code": 0,
                "fatal_errors": [],
                "unreturned_sequences": [],
                "driver_call_sequence": 1,
                "driver_detailed_events": 1,
                "driver_call_counts": [{"symbol": "cuLibraryLoadData", "count": 1}],
                "driver_contributor_counts": [
                    {
                        "realpath": code_mapping["realpath"],
                        "sha256": code_mapping["sha256"],
                        "build_id": code_mapping["build_id"],
                        "count": 1,
                    }
                ],
                "driver_event_limit_exhausted": False,
                "driver_unreturned_sequences": [],
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (output / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            for name in ("tables.jsonl", "calls.jsonl", "returns.jsonl", "captures.jsonl"):
                (output / name).write_text("", encoding="utf-8")
            (output / "driver-calls.jsonl").write_text(json.dumps(driver_call) + "\n", encoding="utf-8")
            (output / "driver-returns.jsonl").write_text(json.dumps(driver_return) + "\n", encoding="utf-8")
            (output / "gdb-status.json").write_text(json.dumps(status), encoding="utf-8")

            result = verify(SimpleNamespace(config=str(config_path)))

            self.assertEqual(result, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["driver_call_records"], 1)


class ResolverVerificationTests(unittest.TestCase):
    def test_accepts_resolved_pointer_and_one_natural_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            config_path = output / "probe-config.json"
            driver_path = str(pathlib.Path("/bin/true").resolve(strict=True))
            driver_mapping = {
                "realpath": driver_path,
                "sha256": "driver-sha",
                "build_id": "driver-build",
                "image_offset": "0x1234",
            }
            caller_mapping = {
                "realpath": "/opt/libcudart.so.12",
                "sha256": "caller-sha",
                "build_id": "caller-build",
                "image_offset": "0x5678",
            }
            code_mapping = {
                "realpath": "/opt/libcublasLt.so.12",
                "sha256": "code-sha",
                "build_id": "code-build",
                "image_offset": "0x9abc",
            }
            config = {
                "output_dir": str(output),
                "mode": "discovery",
                "uuid": None,
                "slot": None,
                "selector": None,
                "driver_observation": None,
                "resolver_observation": {"symbol": "cuLibraryLoadData", "event_limit": 4},
                "inferior_io": None,
            }
            identity = {
                "driver": {"path": driver_path, "sha256": "driver-sha", "build_id": "driver-build"}
            }
            query = {
                "kind": "resolver_query",
                "sequence": 1,
                "entry_point": "cuGetProcAddress",
                "symbol": "cuLibraryLoadData",
                "symbol_pointer": "0x70000000",
                "pfn_output_address": "0x70001000",
                "pfn_before": "0x0",
                "cuda_version": 12090,
                "flags": "0x0",
                "function_mapping": driver_mapping,
                "caller_mapping": caller_mapping,
            }
            query_return = {
                "kind": "resolver_return",
                "sequence": 1,
                "entry_point": "cuGetProcAddress",
                "symbol": "cuLibraryLoadData",
                "return_rax": "0x0",
                "pfn_after": "0x71001234",
                "resolved_mapping": driver_mapping,
                "public_symbol_address": "0x71005678",
                "public_symbol_mapping": driver_mapping,
                "pointer_equals_public_symbol": False,
            }
            call = {
                "kind": "resolver_call",
                "sequence": 1,
                "symbol": "cuLibraryLoadData",
                "function_pointer": "0x71001234",
                "function_mapping": driver_mapping,
                "caller_mapping": caller_mapping,
                "code_pointer": "0x72009abc",
                "code_mapping": code_mapping,
                "output_pointer_address": "0x70002000",
                "output_before": "0x0",
            }
            call_return = {
                "kind": "resolver_call_return",
                "sequence": 1,
                "symbol": "cuLibraryLoadData",
                "return_rax": "0x0",
                "output_after": "0x73000000",
            }
            status = {
                "exit_code": 0,
                "fatal_errors": [],
                "unreturned_sequences": [],
                "driver_call_sequence": 0,
                "driver_detailed_events": 0,
                "driver_call_counts": [],
                "driver_contributor_counts": [],
                "driver_event_limit_exhausted": False,
                "driver_unreturned_sequences": [],
                "resolver_query_records": 1,
                "resolver_call_records": 1,
                "resolver_event_limit": 4,
                "resolver_event_limit_exhausted": False,
                "resolver_unreturned_queries": [],
                "resolver_unreturned_calls": [],
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (output / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            for name in ("tables.jsonl", "calls.jsonl", "returns.jsonl", "captures.jsonl"):
                (output / name).write_text("", encoding="utf-8")
            (output / "resolver-queries.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
            (output / "resolver-returns.jsonl").write_text(json.dumps(query_return) + "\n", encoding="utf-8")
            (output / "resolver-calls.jsonl").write_text(json.dumps(call) + "\n", encoding="utf-8")
            (output / "resolver-call-returns.jsonl").write_text(json.dumps(call_return) + "\n", encoding="utf-8")
            (output / "gdb-status.json").write_text(json.dumps(status), encoding="utf-8")

            result = verify(SimpleNamespace(config=str(config_path)))

            self.assertEqual(result, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["resolver_query_records"], 1)
            self.assertEqual(summary["resolver_call_records"], 1)


if __name__ == "__main__":
    unittest.main()

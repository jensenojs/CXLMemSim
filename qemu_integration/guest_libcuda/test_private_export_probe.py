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
    parse_driver_register,
    parse_driver_symbol,
    parse_proc_maps,
    verify,
)


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


if __name__ == "__main__":
    unittest.main()

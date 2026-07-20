#!/usr/bin/env python3
"""Contract tests for the reusable CUDA ELF static collector."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


TOOL = pathlib.Path(__file__).with_name("collect_cuda_elf_static_evidence.py")
SPEC = importlib.util.spec_from_file_location("collect_cuda_elf_static_evidence", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def section_line(size_hex: str) -> str:
    return (
        "  [28] .nvFatBinSegment  PROGBITS        0000000003594530 3593530 "
        f"{size_hex} 00  WA  0   0  8"
    )


class StaticCollectorSectionTests(unittest.TestCase):
    def test_stripped_dso_without_local_symbols_is_valid_static_input(self) -> None:
        self.assertEqual(MODULE.parse_nm("nm: library.so: no symbols\n"), [])

    def test_exact_kimi_cuda_dso_registration_counts(self) -> None:
        for size_hex, expected in (("000c78", 133), ("001290", 198), ("011aa8", 3015)):
            with self.subTest(size_hex=size_hex):
                sections = MODULE.parse_readelf_sections(section_line(size_hex))
                self.assertEqual(
                    MODULE.fatbin_registration_facts(sections),
                    {
                        "name": ".nvFatBinSegment",
                        "section_index": 28,
                        "size_bytes": int(size_hex, 16),
                        "descriptor_size_bytes": 24,
                        "descriptor_count": expected,
                        "remainder_bytes": 0,
                    },
                )

    def test_non_divisible_registration_section_retains_remainder(self) -> None:
        sections = MODULE.parse_readelf_sections(section_line("000019"))
        self.assertEqual(
            MODULE.fatbin_registration_facts(sections),
            {
                "name": ".nvFatBinSegment",
                "section_index": 28,
                "size_bytes": 25,
                "descriptor_size_bytes": 24,
                "descriptor_count": 1,
                "remainder_bytes": 1,
            },
        )

    def test_absent_registration_section_is_explicit(self) -> None:
        sections = MODULE.parse_readelf_sections(
            "  [ 1] .text PROGBITS 0000000000001000 001000 000020 00 AX 0 0 16"
        )
        self.assertIsNone(MODULE.fatbin_registration_facts(sections))

    def test_empty_or_changed_readelf_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no parseable section headers"):
            MODULE.parse_readelf_sections("Section Headers:\n[Nr] changed columns\n")


if __name__ == "__main__":
    unittest.main()

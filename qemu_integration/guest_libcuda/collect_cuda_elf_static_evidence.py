#!/usr/bin/env python3
"""Collect reusable static evidence from one exact CUDA-related ELF.

The collector always saves raw dynamic/local symbol tables, ELF notes and
program headers.  Optional named queries and disassemblies only select views
of those full transcripts; they never replace the raw evidence.  A caller may
require a named query or disassembly text predicate to be unique/present for a
specific experiment, while other cores can reuse the same collector with
different selectors.

The .nvFatBinSegment calculation is deliberately per ELF.  Summing several
results predicts only that declared corpus; it does not prove that a dynamic
CUDA initialization loaded no additional registration-contributing DSO.  A
runtime probe must enumerate the actual cuLibraryLoadData code origins and
compare that observed DSO set with these exact ELF identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = 1
CUDA_FATBIN_DESCRIPTOR_SIZE_BYTES = 24

HELP_EPILOG = """
证据模型：
  这个工具用于把一次手工的 nm/readelf/objdump 调查变成可重放的证据目录。
  它始终保存完整 nm-dynamic.txt、nm-local-demangled.txt、readelf-notes.txt、
  readelf-program-headers.txt 和 readelf-sections.txt；named query 只是从完整现场
  选出本轮关心的符号。ELF section 会进入 static-evidence.json；若存在
  .nvFatBinSegment，还会按 64 位 CUDA fatbin registration descriptor 的 24 字节
  形状输出 descriptor_count，不能整除时 fail closed。
  descriptor_count只覆盖当前一个ELF。多个结果求和是声明集合的预运行预测；运行时是否
  出现额外DSO，必须由guest shim记录每次registration的code归属，再与静态集合核对。
  所有匹配数、匹配内容、执行 argv、退出码、ELF SHA256、Build ID 与失败原因写入
  static-evidence.json。output-dir 必须是新目录，旧 core/ELF 与既有证据不会被覆盖。

选择器：
  --dynamic-symbol-query NAME=REGEX 搜索 nm -D 的原始符号名。
  --local-symbol-query NAME=REGEX 搜索 nm -a -n -C 的 demangled 本地符号。
  --require-unique NAME 将本轮需要唯一的选择写成硬约束；0 或多于 1 个匹配时仍保存
  全部原始 transcript 和 static-evidence.json，但最终 status=fail_closed。
 --disassemble NAME=BYTES 只接受一个唯一的 local query，并保存 NAME-disassembly.txt。
  --require-text TRANSCRIPT=REGEX 可把反汇编中的公开调用形状作为本轮硬约束。
  --cuda-sass 按需保存 cuobjdump --dump-sass；--dwarf-decoded-line 按需保存 DWARF 行表。

Kimi callback-hooks slot 1 的当前静态选择示例：
  --dynamic-symbol-query 'anchor=^_Z14ggml_cuda_infov$' --require-unique anchor
  --local-symbol-query 'wrapper=^void ggml_cuda_flash_attn_ext_mma_f16_case<576, 512, 2, 16>\\(' \
  --local-symbol-query 'stub=^void flash_attn_ext_f16<576, 512, 2, 16, false, true>\\(' \
  --require-unique wrapper --require-unique stub --disassemble wrapper=0x500 \
  --require-text 'wrapper-disassembly=cudaFuncSetAttribute@plt' \
  --require-text 'wrapper-disassembly=\\$0x8,%esi'

这个示例只建立 exact libggml-cuda 的静态事实。它不加载 CUDA、不启动 QEMU、不调用
private export-table slot，也不证明 guest/Type-2/Kimi correctness。动态 ABI 仍由 L40 上
Python-enabled gdb probe 的自然调用 capture 取得。
"""

HINT = """cuda_elf_static_evidence_hint=self=qemu_integration/guest_libcuda/collect_cuda_elf_static_evidence.py
cuda_elf_static_evidence_hint=problem=manual nm/readelf/objdump commands used during a Kimi core investigation are easy to lose, rerun against the wrong ELF or quote only the matching line while omitting the full symbol and relocation context
cuda_elf_static_evidence_hint=mental_model=hash one exact ELF first; save complete tool transcripts and command exit codes; parse section identity from the saved readelf transcript; derive .nvFatBinSegment registration descriptor count only when its byte size is divisible by the 24-byte 64-bit wrapper shape; apply named selectors only as views over raw files
cuda_elf_static_evidence_hint=role=turn one exact CUDA-related ELF or core companion into reusable static identity, symbol, segment, relocation, version and bounded disassembly evidence without executing it
cuda_elf_static_evidence_hint=use_when=a crash or runtime observation names an exact guest or host DSO and the next dynamic probe needs verified symbol addresses, registered-stub virtual addresses, call sites or Build ID
cuda_elf_static_evidence_hint=inputs=exact regular ELF path; expected SHA256 when frozen by a run spec; optional dynamic/local symbol regexes, uniqueness constraints, disassembly windows, required text, DWARF or SASS requests
cuda_elf_static_evidence_hint=outputs=static-evidence.json with complete ELF section facts and optional .nvFatBinSegment size,24-byte descriptor count,remainder plus complete file,nm,readelf,objdump and optional cuobjdump/DWARF transcripts and named disassembly views
cuda_elf_static_evidence_hint=interpret=status pass means the exact ELF and every requested selector/predicate were satisfied; fail_closed still preserves all raw transcripts and identifies the first missing or ambiguous static fact
cuda_elf_static_evidence_hint=proves=exact ELF identity, Build ID and the static symbol/segment/relocation/disassembly facts directly present in the saved tool output
cuda_elf_static_evidence_hint=does_not_prove=that CUDA loads the ELF, that the caller selected every runtime registration-contributing DSO, a host stub is registered, a private table slot is called, the inferred function signature is correct, guest Type-2 works, Kimi is correct or TPS changes
cuda_elf_static_evidence_hint=next=use verified static addresses and call shape to configure one public Runtime trigger or read-only core/debugger capture; preserve the evidence directory with the immutable diagnostic result
"""


@dataclass(frozen=True)
class Symbol:
    address: int
    kind: str
    name: str

    def json(self) -> dict[str, Any]:
        return {"address": f"0x{self.address:x}", "kind": self.kind, "name": self.name}


@dataclass(frozen=True)
class ElfSection:
    index: int
    name: str
    section_type: str
    address: int
    offset: int
    size: int
    entry_size: int
    flags: str
    link: int
    info: int
    alignment: int

    def json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "type": self.section_type,
            "address": f"0x{self.address:x}",
            "offset": self.offset,
            "size_bytes": self.size,
            "entry_size_bytes": self.entry_size,
            "flags": self.flags,
            "link": self.link,
            "info": self.info,
            "alignment": self.alignment,
        }


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_nonnegative(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {text}") from error
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_named_regex(text: str) -> tuple[str, str]:
    name, separator, expression = text.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or not expression:
        raise argparse.ArgumentTypeError("named selector must be NAME=REGEX")
    try:
        re.compile(expression)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"invalid regex for {name}: {error}") from error
    return name, expression


def parse_named_window(text: str) -> tuple[str, int]:
    name, separator, value = text.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise argparse.ArgumentTypeError("disassembly selector must be NAME=BYTES")
    window = parse_nonnegative(value)
    if window == 0:
        raise argparse.ArgumentTypeError("disassembly window must be positive")
    return name, window


def named_mapping(items: list[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in items:
        if name in result:
            raise RuntimeError(f"duplicate {label} name: {name}")
        result[name] = value
    return result


def parse_nm(text: str) -> list[Symbol]:
    rows: list[Symbol] = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([0-9A-Fa-f]+)\s+([A-Za-z])\s+(.+)", line)
        if match:
            rows.append(Symbol(int(match.group(1), 16), match.group(2), match.group(3)))
    return rows


def parse_readelf_sections(text: str) -> list[ElfSection]:
    rows: list[ElfSection] = []
    row_pattern = re.compile(
        r"^\s*\[\s*(?P<index>\d+)\]\s+(?P<prefix>.*?)\s+"
        r"(?P<address>[0-9A-Fa-f]+)\s+(?P<offset>[0-9A-Fa-f]+)\s+"
        r"(?P<size>[0-9A-Fa-f]+)\s+(?P<entry_size>[0-9A-Fa-f]+)\s*"
        r"(?P<flags>[A-Za-z]*)\s+(?P<link>\d+)\s+(?P<info>\d+)\s+(?P<alignment>\d+)\s*$"
    )
    for line in text.splitlines():
        match = row_pattern.fullmatch(line)
        if not match:
            continue
        prefix = match.group("prefix").split()
        if len(prefix) == 1:
            name = ""
            section_type = prefix[0]
        elif len(prefix) == 2:
            name, section_type = prefix
        else:
            raise RuntimeError(f"unexpected readelf section prefix: {match.group('prefix')}")
        rows.append(
            ElfSection(
                index=int(match.group("index")),
                name=name,
                section_type=section_type,
                address=int(match.group("address"), 16),
                offset=int(match.group("offset"), 16),
                size=int(match.group("size"), 16),
                entry_size=int(match.group("entry_size"), 16),
                flags=match.group("flags"),
                link=int(match.group("link")),
                info=int(match.group("info")),
                alignment=int(match.group("alignment")),
            )
        )
    if not rows:
        raise RuntimeError("readelf produced no parseable section headers")
    indices = [row.index for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("readelf produced duplicate section indices")
    return rows


def fatbin_registration_facts(sections: list[ElfSection]) -> dict[str, Any] | None:
    # This count is exact for one hashed ELF and intentionally says nothing
    # about the completeness of a process-wide DSO set.  The Type-2 cuBLAS
    # probe closes that separate boundary by grouping observed library records
    # by dladdr(code), then comparing each group with per-ELF facts like this.
    matches = [section for section in sections if section.name == ".nvFatBinSegment"]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"expected at most one .nvFatBinSegment, got {len(matches)}")
    section = matches[0]
    count, remainder = divmod(section.size, CUDA_FATBIN_DESCRIPTOR_SIZE_BYTES)
    return {
        "name": section.name,
        "section_index": section.index,
        "size_bytes": section.size,
        "descriptor_size_bytes": CUDA_FATBIN_DESCRIPTOR_SIZE_BYTES,
        "descriptor_count": count,
        "remainder_bytes": remainder,
    }


class Collector:
    def __init__(self, output: pathlib.Path):
        self.output = output
        self.commands: dict[str, dict[str, Any]] = {}
        self.transcripts: dict[str, str] = {}

    def run(self, name: str, argv: list[str], required: bool = True) -> str:
        executable = shutil.which(argv[0])
        if not executable:
            raise RuntimeError(f"required tool is not on PATH: {argv[0]}")
        completed = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        transcript = self.output / f"{name}.txt"
        transcript.write_text(completed.stdout, encoding="utf-8")
        self.commands[name] = {"argv": argv, "exit_code": completed.returncode}
        self.transcripts[name] = transcript.name
        if required and completed.returncode != 0:
            raise RuntimeError(f"{argv[0]} failed with exit code {completed.returncode}; see {transcript}")
        return completed.stdout


def build_id(notes: str) -> str | None:
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
    return match.group(1).lower() if match else None


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> int:
    library = pathlib.Path(args.library).resolve(strict=True)
    output = pathlib.Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"output directory already exists: {output}")
    if not library.is_file():
        raise RuntimeError(f"library is not a regular file: {library}")
    output.mkdir(parents=True)

    collector = Collector(output)
    actual_sha256 = sha256(library)
    errors: list[str] = []
    if args.expected_sha256 and actual_sha256 != args.expected_sha256.lower():
        errors.append(f"library SHA256 differs: expected {args.expected_sha256.lower()}, got {actual_sha256}")

    try:
        collector.run("file", ["file", "--brief", str(library)])
        dynamic_nm = collector.run("nm-dynamic", ["nm", "-D", "--defined-only", str(library)])
        local_nm = collector.run("nm-local-demangled", ["nm", "-a", "-n", "-C", str(library)])
        notes = collector.run("readelf-notes", ["readelf", "-n", str(library)])
        collector.run("readelf-program-headers", ["readelf", "-W", "-l", str(library)])
        collector.run("readelf-dynamic", ["readelf", "-W", "-d", str(library)])
        sections_text = collector.run("readelf-sections", ["readelf", "-W", "-S", str(library)])
        collector.run("readelf-relocations", ["readelf", "-W", "-r", str(library)])
        collector.run("readelf-version-info", ["readelf", "-W", "--version-info", str(library)])
        collector.run("objdump-dynamic-symbols", ["objdump", "-T", str(library)])
        collector.run("objdump-dynamic-relocations", ["objdump", "-R", str(library)])
        dynamic_symbols = parse_nm(dynamic_nm)
        local_symbols = parse_nm(local_nm)
        sections = parse_readelf_sections(sections_text)
    except RuntimeError as error:
        errors.append(str(error))
        dynamic_symbols = []
        local_symbols = []
        sections = []
        notes = ""

    registration: dict[str, Any] | None = None
    if sections:
        try:
            registration = fatbin_registration_facts(sections)
        except RuntimeError as error:
            errors.append(str(error))
    if registration is not None and registration["remainder_bytes"] != 0:
        errors.append(
            ".nvFatBinSegment size is not divisible by the 24-byte CUDA fatbin registration descriptor"
        )

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump:
        collector.run("cuobjdump-list-elf", [cuobjdump, "--list-elf", str(library)], required=False)
    else:
        collector.commands["cuobjdump-list-elf"] = {"status": "unavailable", "reason": "cuobjdump is not on PATH"}
    if args.cuda_sass:
        if not cuobjdump:
            errors.append("--cuda-sass requested but cuobjdump is not on PATH")
        else:
            try:
                collector.run("cuobjdump-sass", [cuobjdump, "--dump-sass", str(library)])
            except RuntimeError as error:
                errors.append(str(error))
    if args.dwarf_decoded_line:
        try:
            collector.run("readelf-dwarf-decoded-line", ["readelf", "--debug-dump=decodedline", str(library)])
        except RuntimeError as error:
            errors.append(str(error))

    dynamic_queries = named_mapping(args.dynamic_symbol_query, "dynamic query")
    local_queries = named_mapping(args.local_symbol_query, "local query")
    all_queries: dict[str, dict[str, Any]] = {}
    for source, queries, symbols in (
        ("dynamic", dynamic_queries, dynamic_symbols),
        ("local_demangled", local_queries, local_symbols),
    ):
        for name, expression in queries.items():
            if name in all_queries:
                errors.append(f"query name appears in both sources: {name}")
                continue
            regex = re.compile(expression)
            matches = [symbol for symbol in symbols if regex.search(symbol.name)]
            all_queries[name] = {
                "source": source,
                "regex": expression,
                "match_count": len(matches),
                "matches": [symbol.json() for symbol in matches],
            }

    required_unique = set(args.require_unique)
    for name in sorted(required_unique):
        result = all_queries.get(name)
        if result is None:
            errors.append(f"required unique query is not declared: {name}")
        elif result["match_count"] != 1:
            errors.append(f"required unique query {name} matched {result['match_count']} symbols")

    disassembly_windows = named_mapping(args.disassemble, "disassembly")
    for name, window in disassembly_windows.items():
        query = all_queries.get(name)
        if query is None:
            errors.append(f"disassembly selector has no named query: {name}")
            continue
        if query["source"] != "local_demangled" or query["match_count"] != 1:
            errors.append(f"disassembly selector {name} needs one local demangled symbol")
            continue
        address = int(query["matches"][0]["address"], 16)
        try:
            collector.run(
                f"{name}-disassembly",
                ["objdump", "-d", "-C", f"--start-address=0x{address:x}", f"--stop-address=0x{address + window:x}", str(library)],
            )
        except RuntimeError as error:
            errors.append(str(error))

    text_expectations: list[dict[str, Any]] = []
    for name, expression in args.require_text:
        transcript = output / f"{name}.txt"
        expectation = {"transcript": f"{name}.txt", "regex": expression, "matched": False}
        if not transcript.is_file():
            errors.append(f"text predicate names no transcript: {name}")
            text_expectations.append(expectation)
            continue
        content = transcript.read_text(encoding="utf-8")
        expectation["matched"] = bool(re.search(expression, content, flags=re.MULTILINE))
        text_expectations.append(expectation)
        if not expectation["matched"]:
            errors.append(f"required text predicate {name} did not match")

    result = {
        "schema_version": SCHEMA_VERSION,
        "library": {"path": str(library), "sha256": actual_sha256, "build_id": build_id(notes)},
        "elf_sections": [section.json() for section in sections],
        "cuda_fatbin_registration": registration,
        "queries": all_queries,
        "commands": collector.commands,
        "transcripts": collector.transcripts,
        "required_unique": sorted(required_unique),
        "required_text": text_expectations,
        "errors": errors,
        "status": "pass" if not errors else "fail_closed",
    }
    write_json(output / "static-evidence.json", result)
    print(f"cuda_elf_static_evidence={output / 'static-evidence.json'}")
    return 0 if not errors else 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--library", required=True, help="要采集的 exact ELF；工具不会修改它")
    parser.add_argument("--expected-sha256", help="期望的 64 位 SHA256；不一致时保留现场并 fail closed")
    parser.add_argument("--output-dir", required=True, help="必须不存在的新证据目录")
    parser.add_argument(
        "--dynamic-symbol-query", action="append", type=parse_named_regex, default=[], metavar="NAME=REGEX",
        help="在完整 nm -D 输出中建立一个命名选择器；可重复",
    )
    parser.add_argument(
        "--local-symbol-query", action="append", type=parse_named_regex, default=[], metavar="NAME=REGEX",
        help="在完整 nm -a -n -C 输出中建立一个命名选择器；可重复",
    )
    parser.add_argument(
        "--require-unique", action="append", default=[], metavar="NAME",
        help="要求同名 selector 恰好解析到一个 symbol；可重复",
    )
    parser.add_argument(
        "--disassemble", action="append", type=parse_named_window, default=[], metavar="NAME=BYTES",
        help="对唯一 local selector 保存 NAME-disassembly.txt；可重复",
    )
    parser.add_argument(
        "--require-text", action="append", type=parse_named_regex, default=[], metavar="TRANSCRIPT=REGEX",
        help="要求已有 transcript 包含 regex；例如 wrapper-disassembly=cudaFuncSetAttribute@plt",
    )
    parser.add_argument(
        "--cuda-sass",
        action="store_true",
        help="额外运行 cuobjdump --dump-sass；适合需要确认 SASS/fatbin 的场景，缺工具会 fail closed",
    )
    parser.add_argument(
        "--dwarf-decoded-line",
        action="store_true",
        help="额外保存 readelf --debug-dump=decodedline；适合有 DWARF 的 source/PC 对照",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if argv == ["--hint"]:
        print(HINT, end="")
        return 0
    return collect(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as error:
        print(f"cuda_elf_static_evidence_error={error}", file=sys.stderr)
        raise SystemExit(2)

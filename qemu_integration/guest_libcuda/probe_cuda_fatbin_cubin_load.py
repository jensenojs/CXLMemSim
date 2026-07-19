#!/usr/bin/env python3
"""验证一个精确 CUDA library fatbin 中指定 CUBIN 是否能由真实 Driver 加载。

这个入口服务 Kimi Type-2 correctness 的低成本诊断。它从一个 SHA256 已冻结的 shared
library 中定位拥有声明 files_size 的唯一 fatbin，提取声明 SM 的 ELF CUBIN，记录输入与
解码身份，并可在当前 GPU 的真实 CUDA Driver 中调用 cuModuleLoadData。

它不启动 QEMU、guest、BAR2 或 Kimi，也不替代正式 same-VM paired correctness。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
from dataclasses import dataclass
from typing import Any


FATBIN_MAGIC = 0xBA55ED50
FATBIN_VERSION = 1
FATBIN_KIND_PTX = 1
FATBIN_KIND_ELF = 2
FATBIN_FLAG_COMPRESSED_LZ4 = 0x2000
FATBIN_FLAG_COMPRESSED_ZSTD = 0x8000

FATBIN_HEADER = struct.Struct("<IHHQ")
FATBIN_FILE_HEADER = struct.Struct("<HHIIIIIIIIIQQQ")

CUDA_SUCCESS = 0


class ProbeFailure(RuntimeError):
    """A validated diagnostic boundary that must be surfaced to the caller."""


@dataclass(frozen=True)
class FatbinFile:
    relative_offset: int
    kind: int
    version: int
    header_size: int
    payload_size: int
    compressed_size: int
    sm_version: int
    bit_width: int
    flags: int
    uncompressed_size: int
    payload_offset: int

    def as_json(self) -> dict[str, int | str]:
        return {
            "relative_offset": self.relative_offset,
            "kind": {FATBIN_KIND_ELF: "elf", FATBIN_KIND_PTX: "ptx"}.get(self.kind, "unknown"),
            "kind_value": self.kind,
            "version": self.version,
            "header_size": self.header_size,
            "payload_size": self.payload_size,
            "compressed_size": self.compressed_size,
            "sm_version": self.sm_version,
            "bit_width": self.bit_width,
            "flags": f"0x{self.flags:x}",
            "uncompressed_size": self.uncompressed_size,
        }


def hint(stream: Any) -> None:
    stream.write(
        "fatbin_cubin_load_probe_hint=self=qemu_integration/guest_libcuda/probe_cuda_fatbin_cubin_load.py\n"
        "fatbin_cubin_load_probe_hint=problem=Kimi baseline reached cuLibraryLoadData in libcublas; L40 sm_89 "
        "had no exact CUBIN and the shim selected first PTX sm_120, which the real Driver rejected with "
        "CUDA_ERROR_INVALID_PTX\n"
        "fatbin_cubin_load_probe_hint=mental_model=freeze the library SHA256 and fatbin shape, extract one declared "
        "ELF CUBIN without executing it, then ask the current CUDA Driver only whether cuModuleLoadData accepts it\n"
        "fatbin_cubin_load_probe_hint=use_when=Before changing guest fatbin candidate ranking, when an exact failed "
        "library contains a lower same-major CUBIN and a higher incompatible PTX\n"
        "fatbin_cubin_load_probe_hint=inputs=library path, expected SHA256, unique fatbin files_size, candidate SM, "
        "output directory; --load additionally requires expected device SM\n"
        "fatbin_cubin_load_probe_hint=outputs=fatbin-cubin-load.json and fatbin-cubin-load.transcript.txt in output "
        "directory; JSON carries library/image hashes, selected entry, Driver/device identity and load result\n"
        "fatbin_cubin_load_probe_hint=proves=A successful --load proves this exact CUBIN was accepted by this exact "
        "CUDA Driver and device after input identity checks\n"
        "fatbin_cubin_load_probe_hint=does_not_prove=It does not prove guest shim selection, BAR2, QEMU, HetGPU, "
        "Kimi baseline/concordia correctness or TPS\n"
        "fatbin_cubin_load_probe_hint=next=Use a successful L40 result to constrain the guest selector; then rebuild, "
        "fresh-pull and run a fresh Kimi same-VM paired task\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_nonnegative(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise argparse.ArgumentTypeError("must be a 64-character lowercase-or-uppercase SHA256 hex string")
    return normalized


def parse_fatbin_files(blob: mmap.mmap, fatbin_offset: int, files_size: int) -> list[FatbinFile]:
    files_start = fatbin_offset + FATBIN_HEADER.size
    files_end = files_start + files_size
    offset = files_start
    entries: list[FatbinFile] = []
    while offset < files_end:
        if files_end - offset < FATBIN_FILE_HEADER.size:
            raise ProbeFailure(f"fatbin at 0x{fatbin_offset:x} has trailing bytes smaller than a file header")
        (
            kind,
            version,
            header_size,
            payload_size,
            _unknown0,
            compressed_size,
            _unknown1,
            _unknown2,
            sm_version,
            bit_width,
            _unknown3,
            flags,
            _unknown5,
            uncompressed_size,
        ) = FATBIN_FILE_HEADER.unpack_from(blob, offset)
        if header_size < FATBIN_FILE_HEADER.size or header_size > 4096:
            raise ProbeFailure(f"fatbin entry at relative offset {offset - files_start} has invalid header size {header_size}")
        if payload_size > files_end - offset - header_size:
            raise ProbeFailure(f"fatbin entry at relative offset {offset - files_start} exceeds declared files_size")
        entries.append(
            FatbinFile(
                relative_offset=offset - files_start,
                kind=kind,
                version=version,
                header_size=header_size,
                payload_size=payload_size,
                compressed_size=compressed_size,
                sm_version=sm_version,
                bit_width=bit_width,
                flags=flags,
                uncompressed_size=uncompressed_size,
                payload_offset=offset + header_size,
            )
        )
        offset += header_size + payload_size
    if offset != files_end:
        raise ProbeFailure(f"fatbin at 0x{fatbin_offset:x} does not consume declared files_size")
    return entries


def locate_unique_fatbin(blob: mmap.mmap, expected_files_size: int) -> tuple[int, list[FatbinFile]]:
    magic = struct.pack("<I", FATBIN_MAGIC)
    positions: list[tuple[int, list[FatbinFile]]] = []
    search_from = 0
    while True:
        offset = blob.find(magic, search_from)
        if offset < 0:
            break
        search_from = offset + 1
        if len(blob) - offset < FATBIN_HEADER.size:
            continue
        found_magic, version, header_size, files_size = FATBIN_HEADER.unpack_from(blob, offset)
        if found_magic != FATBIN_MAGIC or version != FATBIN_VERSION or header_size != FATBIN_HEADER.size:
            continue
        if files_size != expected_files_size or files_size > len(blob) - offset - header_size:
            continue
        positions.append((offset, parse_fatbin_files(blob, offset, files_size)))
    if not positions:
        raise ProbeFailure(f"no fatbin with files_size={expected_files_size} exists in the exact library")
    if len(positions) != 1:
        found = ", ".join(f"0x{offset:x}" for offset, _ in positions)
        raise ProbeFailure(f"fatbin files_size={expected_files_size} is ambiguous at {found}")
    return positions[0]


def decode_entry(blob: mmap.mmap, entry: FatbinFile) -> bytes:
    compressed_lz4 = bool(entry.flags & FATBIN_FLAG_COMPRESSED_LZ4)
    compressed_zstd = bool(entry.flags & FATBIN_FLAG_COMPRESSED_ZSTD)
    if compressed_lz4 and compressed_zstd:
        raise ProbeFailure("selected CUBIN declares both LZ4 and Zstd compression")
    payload = bytes(blob[entry.payload_offset : entry.payload_offset + entry.payload_size])
    if not compressed_lz4 and not compressed_zstd:
        return payload
    if not entry.compressed_size or entry.compressed_size > len(payload) or not entry.uncompressed_size:
        raise ProbeFailure("selected CUBIN has invalid compressed or uncompressed length")
    compressed = payload[: entry.compressed_size]
    destination = ctypes.create_string_buffer(entry.uncompressed_size)
    source = ctypes.create_string_buffer(compressed, len(compressed))
    if compressed_lz4:
        lz4_path = ctypes.util.find_library("lz4")
        if not lz4_path:
            raise ProbeFailure("liblz4 is unavailable for the declared fatbin payload")
        lz4 = ctypes.CDLL(lz4_path)
        lz4.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lz4.LZ4_decompress_safe.restype = ctypes.c_int
        decoded_size = lz4.LZ4_decompress_safe(source, destination, len(compressed), entry.uncompressed_size)
        if decoded_size <= 0 or decoded_size > entry.uncompressed_size:
            raise ProbeFailure(f"LZ4 decompression failed with result {decoded_size}")
    else:
        zstd_path = ctypes.util.find_library("zstd")
        if not zstd_path:
            raise ProbeFailure("libzstd is unavailable for the declared fatbin payload")
        zstd = ctypes.CDLL(zstd_path)
        zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        zstd.ZSTD_decompress.restype = ctypes.c_size_t
        zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
        zstd.ZSTD_isError.restype = ctypes.c_uint
        decoded_size = zstd.ZSTD_decompress(destination, entry.uncompressed_size, source, len(compressed))
        if zstd.ZSTD_isError(decoded_size) or decoded_size > entry.uncompressed_size:
            raise ProbeFailure(f"Zstd decompression failed with result {decoded_size}")
    return destination.raw[:decoded_size]


def driver_load(image: bytes, expected_device_sm: int) -> dict[str, int | str]:
    cuda_path = ctypes.util.find_library("cuda")
    if not cuda_path:
        raise ProbeFailure("libcuda.so.1 is unavailable")
    cuda = ctypes.CDLL(cuda_path)
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDriverGetVersion.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceComputeCapability.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int

    init_result = cuda.cuInit(0)
    if init_result != CUDA_SUCCESS:
        raise ProbeFailure(f"cuInit returned {init_result}")
    driver_version = ctypes.c_int()
    version_result = cuda.cuDriverGetVersion(ctypes.byref(driver_version))
    if version_result != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDriverGetVersion returned {version_result}")
    device = ctypes.c_int()
    device_result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if device_result != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDeviceGet(0) returned {device_result}")
    major = ctypes.c_int()
    minor = ctypes.c_int()
    capability_result = cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)
    if capability_result != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDeviceComputeCapability returned {capability_result}")
    device_sm = major.value * 10 + minor.value
    if device_sm != expected_device_sm:
        raise ProbeFailure(f"device compute capability sm_{device_sm} differs from expected sm_{expected_device_sm}")
    context = ctypes.c_void_p()
    context_result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
    if context_result != CUDA_SUCCESS:
        raise ProbeFailure(f"cuCtxCreate_v2 returned {context_result}")
    module = ctypes.c_void_p()
    buffer = ctypes.create_string_buffer(image, len(image))
    try:
        load_result = cuda.cuModuleLoadData(ctypes.byref(module), buffer)
        unload_result: int | None = None
        if load_result == CUDA_SUCCESS:
            unload_result = cuda.cuModuleUnload(module)
        return {
            "driver_version": driver_version.value,
            "device_ordinal": device.value,
            "device_sm": device_sm,
            "context_create_result": context_result,
            "module_load_result": load_result,
            "module_unload_result": unload_result,
        }
    finally:
        destroy_result = cuda.cuCtxDestroy_v2(context)
        if destroy_result != CUDA_SUCCESS:
            raise ProbeFailure(f"cuCtxDestroy_v2 returned {destroy_result}")


def write_outputs(output_dir: Path, result: dict[str, Any], transcript: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fatbin-cubin-load.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "fatbin-cubin-load.transcript.txt").write_text("\n".join(transcript) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hint", action="store_true", help="print stable agent-oriented context and exit")
    parser.add_argument("--library", type=Path, help="exact CUDA shared library containing the target fatbin")
    parser.add_argument("--expected-sha256", type=parse_sha256, metavar="SHA256", help="required library SHA256")
    parser.add_argument("--fatbin-files-size", type=parse_nonnegative, help="required unique fatbin files_size")
    parser.add_argument("--cubin-sm", type=parse_nonnegative, help="ELF CUBIN SM version to extract")
    parser.add_argument("--output-dir", type=Path, help="new directory for JSON and transcript")
    parser.add_argument("--load", action="store_true", help="create a real CUDA context and call cuModuleLoadData")
    parser.add_argument(
        "--expected-device-sm",
        type=parse_nonnegative,
        help="required current Driver device SM; mandatory with --load to prevent cross-GPU claims",
    )
    args = parser.parse_args(argv)
    if args.hint:
        return args
    required = ("library", "expected_sha256", "fatbin_files_size", "cubin_sm", "output_dir")
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + name for name in missing))
    if args.load and args.expected_device_sm is None:
        parser.error("--load requires --expected-device-sm")
    if not args.load and args.expected_device_sm is not None:
        parser.error("--expected-device-sm is only meaningful with --load")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.hint:
        hint(sys.stdout)
        return 0
    assert args.library is not None
    assert args.output_dir is not None
    if args.output_dir.exists():
        print(f"output directory already exists: {args.output_dir}", file=sys.stderr)
        return 2
    transcript = [f"library={args.library}", f"expected_sha256={args.expected_sha256}"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "cuda-fatbin-cubin-load-probe",
        "status": "fail",
        "proves": "Exact input identity and one Driver cuModuleLoadData acceptance result when status=pass and load was requested.",
        "does_not_prove": "Guest shim selection, BAR2, QEMU, HetGPU, Kimi paired correctness, or TPS.",
    }
    try:
        library = args.library.resolve(strict=True)
        actual_sha256 = sha256_file(library)
        transcript.append(f"library_resolved={library}")
        transcript.append(f"library_sha256={actual_sha256}")
        if actual_sha256 != args.expected_sha256:
            raise ProbeFailure(f"library SHA256 mismatch: expected {args.expected_sha256}, got {actual_sha256}")
        with library.open("rb") as source:
            blob = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                fatbin_offset, entries = locate_unique_fatbin(blob, args.fatbin_files_size)
                candidates = [entry for entry in entries if entry.kind == FATBIN_KIND_ELF and entry.sm_version == args.cubin_sm]
                if len(candidates) != 1:
                    raise ProbeFailure(f"expected exactly one ELF sm_{args.cubin_sm} candidate; found {len(candidates)}")
                entry = candidates[0]
                image = decode_entry(blob, entry)
            finally:
                blob.close()
        if not image.startswith(b"\x7fELF"):
            raise ProbeFailure("decoded candidate does not begin with an ELF header")
        result.update(
            {
                "library": str(library),
                "library_sha256": actual_sha256,
                "fatbin": {
                    "offset": f"0x{fatbin_offset:x}",
                    "files_size": args.fatbin_files_size,
                    "entries": [entry.as_json() for entry in entries],
                },
                "selected_cubin": entry.as_json(),
                "selected_cubin_sha256": hashlib.sha256(image).hexdigest(),
                "selected_cubin_size": len(image),
                "load_requested": args.load,
            }
        )
        transcript.append(f"fatbin_offset=0x{fatbin_offset:x}")
        transcript.append(f"selected_cubin=sm_{entry.sm_version} relative_offset={entry.relative_offset}")
        transcript.append(f"selected_cubin_sha256={result['selected_cubin_sha256']}")
        if args.load:
            driver = driver_load(image, args.expected_device_sm)
            result["driver"] = driver
            transcript.append("driver_module_load_result=" + str(driver["module_load_result"]))
            if driver["module_load_result"] != CUDA_SUCCESS:
                raise ProbeFailure(f"cuModuleLoadData returned {driver['module_load_result']}")
        result["status"] = "pass"
        write_outputs(args.output_dir, result, transcript)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ProbeFailure, ValueError) as error:
        result["error"] = str(error)
        write_outputs(args.output_dir, result, transcript + ["error=" + str(error)])
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

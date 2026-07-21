#!/usr/bin/env python3
"""在真实 CUDA Driver 上加载一份精确重建的 CUDA fatbin library wrapper。

Kimi Type-2 baseline 的 guest shim 会把 Runtime 传来的 fatbin wrapper 保存为 CUlibrary record，
随后由 cuLibraryGetModule 请求 module materialization，并在当前 cuBLAS 初始化窗口继续查询一个
明确命名的 module global。这个入口按 manifest 声明的精确 fatbin 偏移，从同一份 SHA256 已冻结的
library bytes 重建 process-local CudartFatbincWrapper，再向真实 NVIDIA Driver 询问完整 library
container 的 load/get-module/global/unload 生命周期。

它不启动 guest、BAR2、QEMU、HetGPU 或 Kimi。`--get-module` 会在成功注册后调用真实
`cuLibraryGetModule`；`--module-global` 在 module 成功后调用 `cuModuleGetGlobal_v2`。输入身份、
fatbin layout、wrapper layout、library option、Driver 返回值和 handle 生命周期都会写入 output directory。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import mmap
from pathlib import Path
import struct
import sys
from typing import Any

from probe_cuda_fatbin_cubin_load import (
    CUDA_SUCCESS,
    FATBIN_HEADER,
    FATBIN_KIND_ELF,
    FATBIN_KIND_PTX,
    FatbinFile,
    ProbeFailure,
    parse_fatbin_files,
    parse_nonnegative,
    parse_sha256,
    sha256_file,
)


CUDART_FATBINC_MAGIC = 0x466243B1
CUDART_FATBINC_VERSION = 1
CU_LIBRARY_BINARY_IS_PRESERVED = 1


class CudartFatbincWrapper(ctypes.Structure):
    """64-bit C layout consumed by cuLibraryLoadData in this probe."""

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("filename_or_fatbins", ctypes.c_void_p),
    ]


def hint(stream: Any) -> None:
    stream.write(
        "fatbin_library_load_probe_hint=self=qemu_integration/guest_libcuda/probe_cuda_fatbin_library_load.py\n"
        "fatbin_library_load_probe_hint=problem=One library may contain many same-sized fatbins, so files_size cannot identify the Runtime wrapper that reached cuLibraryLoadData\n"
        "fatbin_library_load_probe_hint=question=Does the real L40 Driver accept the exact declared wrapper, and after acceptance what do cuLibraryGetModule and the named cuModuleGetGlobal return?\n"
        "fatbin_library_load_probe_hint=mental_model=Library SHA256 plus explicit fatbin offset, header hash, full-region hash and ordered entries identify one wrapper. The tool copies only that header and files region into a fresh process-local 24-byte wrapper and keeps both buffers alive through cuLibraryUnload.\n"
        "fatbin_library_load_probe_hint=inputs=library path; expected SHA256; explicit fatbin offset; files_size; header and full-region SHA256; ordered entries; output directory; --load also requires expected L40 SM\n"
        "fatbin_library_load_probe_hint=outputs=fatbin-library-load.json and fatbin-library-load.transcript.txt; JSON carries library/fatbin hashes, entry sequence, reconstructed wrapper, option 1 value 0x1, Driver load/get-module/global results, handles and unload result\n"
        "fatbin_library_load_probe_hint=interpret=driver-accepted means the Driver accepted this full library container; an optional get-module result is recorded without being rewritten as a library-load result\n"
        "fatbin_library_load_probe_hint=proves=One native Driver answer for this exact reconstructed wrapper only\n"
        "fatbin_library_load_probe_hint=does_not_prove=Guest shim behavior, BAR2, QEMU, HetGPU, Kimi correctness or TPS\n"
        "fatbin_library_load_probe_hint=next=accepted constrains guest library registration and lifetime semantics; rejected constrains why the guest reached this legacy submodule\n"
    )


def locate_fatbin_at_offset(blob: mmap.mmap, expected_offset: int, expected_files_size: int) -> tuple[int, list[FatbinFile]]:
    if expected_offset > len(blob) - FATBIN_HEADER.size:
        raise ProbeFailure(f"fatbin offset 0x{expected_offset:x} cannot contain a complete header")
    magic, version, header_size, files_size = FATBIN_HEADER.unpack_from(blob, expected_offset)
    if (magic, version, header_size) != (0xBA55ED50, 1, FATBIN_HEADER.size):
        raise ProbeFailure(f"fatbin offset 0x{expected_offset:x} does not contain the declared fatbin header")
    if files_size != expected_files_size:
        raise ProbeFailure(
            f"fatbin offset 0x{expected_offset:x} files_size differs: "
            f"expected {expected_files_size}, got {files_size}"
        )
    if files_size > len(blob) - expected_offset - header_size:
        raise ProbeFailure(f"fatbin offset 0x{expected_offset:x} files region exceeds the exact library")
    return expected_offset, parse_fatbin_files(blob, expected_offset, files_size)


def parse_expected_entry(value: str) -> tuple[str, int]:
    kind_text, separator, sm_text = value.partition(":")
    if separator != ":" or kind_text not in {"elf", "ptx"} or not sm_text:
        raise argparse.ArgumentTypeError("expected entry must be elf:SM or ptx:SM")
    try:
        return kind_text, parse_nonnegative(sm_text)
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(f"invalid expected entry {value!r}: {error}") from error


def entry_shape(entries: list[FatbinFile]) -> list[tuple[str, int]]:
    kinds = {FATBIN_KIND_ELF: "elf", FATBIN_KIND_PTX: "ptx"}
    return [(kinds.get(entry.kind, "unknown"), entry.sm_version) for entry in entries]


def display_handle(value: ctypes.c_void_p) -> str:
    return f"0x{(value.value or 0):x}"


def configure_driver() -> Any:
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
    cuda.cuLibraryLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
    ]
    cuda.cuLibraryLoadData.restype = ctypes.c_int
    cuda.cuLibraryGetModule.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuLibraryGetModule.restype = ctypes.c_int
    cuda.cuModuleGetGlobal_v2.argtypes = [
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_char_p,
    ]
    cuda.cuModuleGetGlobal_v2.restype = ctypes.c_int
    cuda.cuLibraryUnload.argtypes = [ctypes.c_void_p]
    cuda.cuLibraryUnload.restype = ctypes.c_int
    return cuda


def driver_load_wrapper(
    wrapper: CudartFatbincWrapper,
    expected_device_sm: int,
    get_module: bool,
    module_global: str | None,
) -> dict[str, int | str | bool | None]:
    cuda = configure_driver()
    if (init_result := cuda.cuInit(0)) != CUDA_SUCCESS:
        raise ProbeFailure(f"cuInit returned {init_result}")
    driver_version = ctypes.c_int()
    if (version_result := cuda.cuDriverGetVersion(ctypes.byref(driver_version))) != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDriverGetVersion returned {version_result}")
    device = ctypes.c_int()
    if (device_result := cuda.cuDeviceGet(ctypes.byref(device), 0)) != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDeviceGet(0) returned {device_result}")
    major, minor = ctypes.c_int(), ctypes.c_int()
    if (capability_result := cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)) != CUDA_SUCCESS:
        raise ProbeFailure(f"cuDeviceComputeCapability returned {capability_result}")
    device_sm = major.value * 10 + minor.value
    if device_sm != expected_device_sm:
        raise ProbeFailure(f"device compute capability sm_{device_sm} differs from expected sm_{expected_device_sm}")
    context = ctypes.c_void_p()
    if (context_result := cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)) != CUDA_SUCCESS:
        raise ProbeFailure(f"cuCtxCreate_v2 returned {context_result}")
    library = ctypes.c_void_p()
    module = ctypes.c_void_p()
    option = ctypes.c_int(CU_LIBRARY_BINARY_IS_PRESERVED)
    option_value = ctypes.c_void_p(1)
    try:
        load_result = cuda.cuLibraryLoadData(
            ctypes.byref(library), ctypes.byref(wrapper), None, None, 0,
            ctypes.byref(option), ctypes.byref(option_value), 1,
        )
        module_result: int | None = None
        module_called = False
        global_result: int | None = None
        global_called = False
        global_address = ctypes.c_uint64()
        global_size = ctypes.c_size_t()
        unload_result: int | None = None
        if load_result == CUDA_SUCCESS:
            if not library.value:
                raise ProbeFailure("cuLibraryLoadData returned CUDA_SUCCESS with a null CUlibrary handle")
            if get_module:
                module_called = True
                module_result = cuda.cuLibraryGetModule(ctypes.byref(module), library)
                if module_result == CUDA_SUCCESS and module_global is not None:
                    global_called = True
                    global_result = cuda.cuModuleGetGlobal_v2(
                        ctypes.byref(global_address), ctypes.byref(global_size), module, module_global.encode("ascii"),
                    )
            unload_result = cuda.cuLibraryUnload(library)
            if unload_result != CUDA_SUCCESS:
                raise ProbeFailure(f"cuLibraryUnload returned {unload_result}")
        return {
            "driver_version": driver_version.value,
            "device_ordinal": device.value,
            "device_sm": device_sm,
            "context_create_result": context_result,
            "library_load_result": load_result,
            "library_handle": display_handle(library),
            "library_handle_nonnull": bool(library.value),
            "library_get_module_requested": get_module,
            "library_get_module_called": module_called,
            "library_get_module_result": module_result,
            "library_module_handle": display_handle(module),
            "library_module_handle_nonnull": bool(module.value),
            "module_global_name": module_global,
            "module_global_called": global_called,
            "module_global_result": global_result,
            "module_global_address": f"0x{global_address.value:x}",
            "module_global_address_nonzero": bool(global_address.value),
            "module_global_size": global_size.value,
            "library_unload_called": load_result == CUDA_SUCCESS,
            "library_unload_result": unload_result,
        }
    finally:
        if (destroy_result := cuda.cuCtxDestroy_v2(context)) != CUDA_SUCCESS:
            raise ProbeFailure(f"cuCtxDestroy_v2 returned {destroy_result}")


def write_outputs(output_dir: Path, result: dict[str, Any], transcript: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fatbin-library-load.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "fatbin-library-load.transcript.txt").write_text("\n".join(transcript) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hint", action="store_true", help="print stable agent-oriented context and exit")
    parser.add_argument("--library", type=Path, help="exact CUDA shared library containing the target fatbin")
    parser.add_argument("--expected-sha256", type=parse_sha256, metavar="SHA256", help="required library SHA256")
    parser.add_argument("--fatbin-offset", type=parse_nonnegative, help="required exact fatbin offset in the library")
    parser.add_argument("--fatbin-files-size", type=parse_nonnegative, help="required fatbin files_size at the declared offset")
    parser.add_argument("--expected-header-sha256", type=parse_sha256, metavar="SHA256", help="required SHA256 of the 16-byte fatbin header")
    parser.add_argument("--expected-bytes-sha256", type=parse_sha256, metavar="SHA256", help="required SHA256 of the header and full files region")
    parser.add_argument("--expected-entry", type=parse_expected_entry, action="append", help="one ordered exact entry, formatted elf:SM or ptx:SM; repeat for the full fatbin sequence")
    parser.add_argument("--output-dir", type=Path, help="new directory for JSON and transcript")
    parser.add_argument("--load", action="store_true", help="create a real CUDA context and call cuLibraryLoadData")
    parser.add_argument("--get-module", action="store_true", help="after a successful load, call cuLibraryGetModule before unload")
    parser.add_argument("--module-global", help="after a successful module result, call cuModuleGetGlobal_v2 for this ASCII symbol")
    parser.add_argument("--expected-device-sm", type=parse_nonnegative, help="required current Driver device SM; mandatory with --load")
    args = parser.parse_args(argv)
    if args.hint:
        return args
    required = (
        "library", "expected_sha256", "fatbin_offset", "fatbin_files_size", "expected_header_sha256",
        "expected_bytes_sha256", "expected_entry", "output_dir",
    )
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + name for name in missing))
    if args.load and args.expected_device_sm is None:
        parser.error("--load requires --expected-device-sm")
    if args.get_module and not args.load:
        parser.error("--get-module requires --load")
    if args.module_global is not None and not args.get_module:
        parser.error("--module-global requires --get-module")
    if args.module_global is not None:
        try:
            args.module_global.encode("ascii")
        except UnicodeEncodeError:
            parser.error("--module-global must be ASCII")
    if not args.load and args.expected_device_sm is not None:
        parser.error("--expected-device-sm is only meaningful with --load")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.hint:
        hint(sys.stdout)
        return 0
    assert args.library is not None and args.output_dir is not None and args.expected_entry is not None
    if args.output_dir.exists():
        print(f"output directory already exists: {args.output_dir}", file=sys.stderr)
        return 2
    transcript = [
        f"library={args.library}",
        f"expected_sha256={args.expected_sha256}",
        f"expected_fatbin_offset=0x{args.fatbin_offset:x}",
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "cuda-fatbin-library-load-probe",
        "status": "error",
        "proves": "When status is driver-accepted or driver-rejected, the exact reconstructed wrapper reached the declared Driver.",
        "does_not_prove": "Guest shim selection, BAR2, QEMU, HetGPU, Kimi paired correctness, or TPS.",
    }
    try:
        library = args.library.resolve(strict=True)
        actual_sha256 = sha256_file(library)
        transcript.extend((f"library_resolved={library}", f"library_sha256={actual_sha256}"))
        if actual_sha256 != args.expected_sha256:
            raise ProbeFailure(f"library SHA256 mismatch: expected {args.expected_sha256}, got {actual_sha256}")
        with library.open("rb") as source:
            blob = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                fatbin_offset, entries = locate_fatbin_at_offset(blob, args.fatbin_offset, args.fatbin_files_size)
                observed_shape = entry_shape(entries)
                if observed_shape != args.expected_entry:
                    raise ProbeFailure(f"fatbin entry sequence differs: expected {args.expected_entry}, got {observed_shape}")
                fatbin_bytes = bytes(blob[fatbin_offset : fatbin_offset + FATBIN_HEADER.size + args.fatbin_files_size])
            finally:
                blob.close()
        if len(fatbin_bytes) != FATBIN_HEADER.size + args.fatbin_files_size:
            raise ProbeFailure("fatbin copy does not contain the declared header and full files region")
        magic, version, header_size, files_size = FATBIN_HEADER.unpack_from(fatbin_bytes)
        if (magic, version, header_size, files_size) != (0xBA55ED50, 1, FATBIN_HEADER.size, args.fatbin_files_size):
            raise ProbeFailure("copied fatbin header differs from its validated library form")
        header_sha256 = hashlib.sha256(fatbin_bytes[: FATBIN_HEADER.size]).hexdigest()
        bytes_sha256 = hashlib.sha256(fatbin_bytes).hexdigest()
        if header_sha256 != args.expected_header_sha256:
            raise ProbeFailure(
                f"fatbin header SHA256 mismatch: expected {args.expected_header_sha256}, got {header_sha256}"
            )
        if bytes_sha256 != args.expected_bytes_sha256:
            raise ProbeFailure(
                f"fatbin header-plus-files SHA256 mismatch: expected {args.expected_bytes_sha256}, got {bytes_sha256}"
            )
        fatbin_buffer = ctypes.create_string_buffer(fatbin_bytes, len(fatbin_bytes))
        wrapper = CudartFatbincWrapper(
            CUDART_FATBINC_MAGIC, CUDART_FATBINC_VERSION,
            ctypes.cast(fatbin_buffer, ctypes.c_void_p), None,
        )
        fatbin = {
            "offset": f"0x{fatbin_offset:x}",
            "header_sha256": header_sha256,
            "bytes_sha256": bytes_sha256,
            "files_size": args.fatbin_files_size,
            "entries": [entry.as_json() for entry in entries],
        }
        result.update({
            "library": str(library), "library_sha256": actual_sha256, "fatbin": fatbin,
            "wrapper": {
                "magic": f"0x{CUDART_FATBINC_MAGIC:x}", "version": CUDART_FATBINC_VERSION,
                "layout_size": ctypes.sizeof(CudartFatbincWrapper),
                "data_points_to_reconstructed_fatbin": True,
                "filename_or_fatbins_is_null": wrapper.filename_or_fatbins is None,
            },
            "library_options": {"option": CU_LIBRARY_BINARY_IS_PRESERVED, "value": "0x1", "code_preserved_until_unload": True},
            "load_requested": args.load,
            "get_module_requested": args.get_module,
            "module_global_requested": args.module_global,
        })
        transcript.extend((
            f"fatbin_offset=0x{fatbin_offset:x}", f"fatbin_header_sha256={fatbin['header_sha256']}",
            f"fatbin_bytes_sha256={fatbin['bytes_sha256']}",
            "entries=" + ",".join(f"{kind}:{sm}" for kind, sm in observed_shape),
            f"wrapper_layout_size={ctypes.sizeof(CudartFatbincWrapper)}", "library_option=1 value=0x1",
        ))
        if args.load:
            driver = driver_load_wrapper(wrapper, args.expected_device_sm, args.get_module, args.module_global)
            result["driver"] = driver
            transcript.extend((
                f"driver_library_load_result={driver['library_load_result']}",
                f"driver_library_handle={driver['library_handle']}",
                f"driver_library_get_module_called={driver['library_get_module_called']}",
                f"driver_library_get_module_result={driver['library_get_module_result']}",
                f"driver_library_module_handle={driver['library_module_handle']}",
                f"driver_module_global_name={driver['module_global_name']}",
                f"driver_module_global_called={driver['module_global_called']}",
                f"driver_module_global_result={driver['module_global_result']}",
                f"driver_module_global_address={driver['module_global_address']}",
                f"driver_module_global_size={driver['module_global_size']}",
                f"driver_library_unload_called={driver['library_unload_called']}",
                f"driver_library_unload_result={driver['library_unload_result']}",
            ))
            result["status"] = "driver-accepted" if driver["library_load_result"] == CUDA_SUCCESS else "driver-rejected"
        else:
            result["status"] = "static-validated"
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

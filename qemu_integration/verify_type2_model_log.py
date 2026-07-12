#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify observable facts in a Type-2 guest log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--expect-cubin-loads", type=int)
    parser.add_argument("--expect-function-queries", type=int)
    parser.add_argument("--require-global-query-missing", action="store_true")
    parser.add_argument("--require-module-global-success", action="store_true")
    parser.add_argument("--require-param-layout-launch", action="store_true")
    parser.add_argument("--require-tiny-result", action="store_true")
    parser.add_argument("--require-runtime-meminfo-roundtrip", action="store_true")
    parser.add_argument("--require-model-failure", action="store_true")
    parser.add_argument("--require-model-success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = args.log.read_bytes()
    lines = data.decode(errors="replace").splitlines()

    def locations(fragment: str) -> list[int]:
        return [number for number, line in enumerate(lines, 1) if fragment in line]

    cubin_selected = locations("library CUBIN load selected")
    cubin_modules = locations("cubin module=")
    function_queries = locations("[CXL-CUDA] cuModuleGetFunction(")
    global_queries = locations("cuGetProcAddress(symbol=cuModuleGetGlobal")
    global_missing = locations("cuGetProcAddress(cuModuleGetGlobal) -> pfn=NULL status=SYMBOL_NOT_FOUND")
    module_global_calls = locations("[CXL-CUDA] cuModuleGetGlobal_v2(")
    module_global_successes = []
    module_global_pattern = re.compile(r"Got global '([^']+)' -> 0x([0-9a-fA-F]+) size=(\d+)")
    for number, line in enumerate(lines, 1):
        match = module_global_pattern.search(line)
        if match:
            module_global_successes.append(
                {
                    "line": number,
                    "name": match.group(1),
                    "device_pointer": int(match.group(2), 16),
                    "size": int(match.group(3)),
                }
            )
    tiny_results = locations("kernel_probe result=1234 expected=1234")
    param_info_calls = locations("[CXL-CUDA] cuFuncGetParamInfo(")
    guest_launches = locations("[CXL-CUDA] cuLaunchKernel(")
    backend_launches = locations("CXL hetGPU: [dev0] Launching kernel")
    driver_meminfo_calls = locations("[CXL-CUDA] cuMemGetInfo")
    before_meminfo_success = locations("before_dlopen_meminfo cudaMemGetInfo err=0 name=cudaSuccess")
    after_meminfo_success = locations("after_dlopen_meminfo cudaMemGetInfo err=0 name=cudaSuccess")
    meminfo_calls = locations("cudaMemGetInfo(free, total)")
    unsupported = locations("CUDA error: API call is not supported in the installed CUDA driver")
    model_failures = locations("TYPE2_MODEL_1P5B_SMOKE_FAIL")
    model_successes = locations("TYPE2_MODEL_1P5B_SMOKE_PASS")
    model_rc_zero = locations("TYPE2_MODEL_RUN_RC=0")
    actual_offload = [
        number
        for number, line in enumerate(lines, 1)
        if re.search(r"offloaded [1-9][0-9]*/[1-9][0-9]* layers to GPU", line)
        and not line.lstrip().startswith("+")
    ]

    checks: dict[str, bool] = {
        "cubin_result_count_matches_load_count": len(cubin_selected) == len(cubin_modules),
    }
    if args.expect_cubin_loads is not None:
        checks["expected_cubin_loads"] = len(cubin_selected) == args.expect_cubin_loads
    if args.expect_function_queries is not None:
        checks["expected_function_queries"] = len(function_queries) == args.expect_function_queries
    if args.require_global_query_missing:
        checks["global_query_requested"] = bool(global_queries)
        checks["global_query_resolved_missing"] = bool(global_missing)
    if args.require_module_global_success:
        kmask_successes = [entry for entry in module_global_successes if entry["name"] == "kmask_iq2xs"]
        checks["module_global_called"] = bool(module_global_calls)
        checks["module_global_backend_returned_facts"] = bool(module_global_successes) and all(
            entry["device_pointer"] != 0 and entry["size"] != 0 for entry in module_global_successes
        )
        checks["kmask_iq2xs_resolved"] = bool(kmask_successes)
    if args.require_param_layout_launch:
        checks["function_parameter_layout_queried"] = bool(param_info_calls)
        checks["guest_launch_called"] = bool(guest_launches)
        checks["backend_launch_called"] = bool(backend_launches)
    if args.require_tiny_result:
        checks["tiny_result_1234"] = bool(tiny_results)
    if args.require_runtime_meminfo_roundtrip:
        checks["runtime_meminfo_before_fatbin"] = bool(before_meminfo_success)
        checks["runtime_meminfo_after_fatbin"] = bool(after_meminfo_success)
        checks["runtime_meminfo_reentered_driver"] = bool(
            len(driver_meminfo_calls) >= 2
            and before_meminfo_success
            and after_meminfo_success
            and driver_meminfo_calls[0] < before_meminfo_success[0]
            and driver_meminfo_calls[-1] < after_meminfo_success[-1]
        )
    if args.require_model_failure:
        checks["model_failure_visible"] = bool(meminfo_calls and unsupported and model_failures)
        checks["model_failure_order"] = bool(
            function_queries
            and meminfo_calls
            and unsupported
            and model_failures
            and function_queries[-1] < unsupported[0] <= meminfo_calls[0] < model_failures[-1]
        )
    if args.require_model_success:
        checks["model_process_returned_zero"] = bool(model_rc_zero)
        checks["model_gpu_offload_visible"] = bool(actual_offload)
        checks["model_success_marker_visible"] = bool(model_successes and not model_failures)

    report = {
        "log": str(args.log.resolve()),
        "log_sha256": hashlib.sha256(data).hexdigest(),
        "observations": {
            "cubin_loads": len(cubin_selected),
            "cubin_module_results": len(cubin_modules),
            "function_queries": len(function_queries),
            "global_query_lines": global_queries,
            "global_missing_lines": global_missing,
            "module_global_call_lines": module_global_calls,
            "module_global_successes": module_global_successes,
            "tiny_result_lines": tiny_results,
            "param_info_call_lines": param_info_calls,
            "guest_launch_lines": guest_launches,
            "backend_launch_lines": backend_launches,
            "driver_meminfo_call_lines": driver_meminfo_calls,
            "before_meminfo_success_lines": before_meminfo_success,
            "after_meminfo_success_lines": after_meminfo_success,
            "meminfo_call_lines": meminfo_calls,
            "unsupported_lines": unsupported,
            "model_failure_lines": model_failures,
            "model_success_lines": model_successes,
            "model_rc_zero_lines": model_rc_zero,
            "actual_offload_lines": actual_offload,
        },
        "checks": checks,
        "proves": [
            "the listed strings, counts, and ordering are present in this log",
            "cuModuleGetGlobal was requested from the resolver and resolved as missing when that check passes",
            "module global calls returned non-zero NVIDIA device pointers and sizes when that check passes",
            "cudaMemGetInfo re-entered the guest driver and returned success before and after fatbin registration when that check passes",
        ],
        "does_not_prove": [
            "which exact initrd or payload produced the log unless a run manifest accompanies it",
            "that a resolved module global was consumed correctly by a later kernel",
            "that the missing resolver entry caused cudaMemGetInfo to fail",
            "model correctness or GPU offload unless their own result markers are present",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

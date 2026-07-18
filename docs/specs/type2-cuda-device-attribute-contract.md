# Type-2 CUDA device attribute contract

## Background

The Type-2 device has two distinct memory facts. QEMU owns the simulated CXL
device-memory and Dynamic Capacity Device (DCD) state. The selected NVIDIA
backend owns CUDA device capabilities and the memory visible to CUDA
allocations. The guest CUDA shim currently exposes both through the same
minimal BAR2 device-information interface.

The first same-VM Kimi baseline reached a CUDA quantized matrix-multiplication
selection failure. The guest reported an L40 with 512 MiB, and llama aborted
after it could not select an MMQ tile. QEMU is configured with a 512 MiB Type-2
device-memory window and DCD enabled. The failure must first be traced through
the exact guest-visible CUDA attributes before changing the protocol.

## This slice

This component slice adds observability to the existing tiny CUDA runtime
probe. It records the CUDA Runtime device properties that llama consumes and
the direct Driver API results for the two shared-memory attributes used by MMQ:

```text
CUDA Runtime device properties
  totalGlobalMem, sharedMemPerBlock, sharedMemPerBlockOptin,
  warpSize, multiProcessorCount, maxThreadsPerBlock, compute capability

CUDA Driver API
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK (8)
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN (97)
```

The probe also records `cudaMemGetInfo`. It retains the existing dynamic DSO
load, CUDA allocation, kernel launch, synchronization, DtoH copy and value
check. This is a diagnostic result, not Kimi correctness.

## Boundaries

The probe must print values actually returned by the guest CUDA Runtime and
Driver API. It must not embed L40 capacity, shared-memory limits, compute
capability, or a success default.

This change does not add BAR2 registers, change allocation policy, enlarge the
512 MiB CXL memory window, or modify the QEMU device model. Those decisions
remain pending on the probe result.

## Expected result and decision

The current hypothesis predicts a successful Driver API return with value zero
for attribute 97. A zero `sharedMemPerBlockOptin` explains why llama rejects
every MMQ tile. A 512 MiB CUDA total-memory value demonstrates whether the DCD
capacity is being reused as CUDA memory.

If either observation differs, preserve the output and trace that value before
changing QEMU or the shim. If both observations match, the next cross-component
slice separates CUDA backend capability registers from CXL/DCD memory-state
registers and makes unsupported Driver API attributes fail explicitly.

## Acceptance

- The component payload still contains the existing tiny probe at the same
  path.
- Its output has one begin/end bounded device-property block.
- The block contains Runtime properties, `cudaMemGetInfo`, and Driver API
  results for attributes 8 and 97.
- A failed API call is explicit and causes the probe to fail.
- Existing tiny kernel result `1234` remains required.

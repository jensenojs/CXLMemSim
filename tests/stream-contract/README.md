# Stream-Contract Consistency Micro-Tests

# agent-context: problem=The gpu-w68.5.19 root cause (cuMemcpy2DAsync dropping hStream across the guest shim -> BAR2 -> QEMU chain, so the copy ran unordered against a non-blocking producer stream) was only visible as NaN in a full Kimi run; nothing local could expose that defect class in seconds.
# agent-context: mental_model=CUDA promises that work enqueued on one stream executes in issue order. Each case puts a producer kernel (with a clock64 delay loop sizing the race window) and one async copy on the same stream, then reads the destination back: contract-honoring stacks yield the expected pattern, a stream-dropping stack yields the recognizable stale/overwritten pattern. A legacy-stream twin case discriminates stream-semantics loss from general breakage.
# agent-context: role=Provide a deterministic, seconds-scale pass/fail gate for "async copy API stream semantics survive the forwarding chain", runnable both natively and inside the Type-2 guest with zero changes.
# agent-context: use_when=Use before/after any change to the shim copy paths, the BAR2 wire protocol, or the QEMU copy handlers; use as the acceptance gate for the dtod-stream-aware fix and as a regression gate afterwards.
# agent-context: inputs=An NVIDIA GPU visible to the process (host native or Type-2 guest via shim); nvcc + g++-14 for building (host only); optional env DELAY (clock64 cycles, default 3e9 ~ 1.7 s window) and BYTES (default 1 MiB).
# agent-context: outputs=key=value lines per case (test, stream, delay_cycles, bytes, word counts, first mismatch, diagnosis, verdict) plus run.sh case_result lines and a final matrix=PASS|FAIL against the declared surface expectation. Exit 0 iff observation matches expectation.
# agent-context: interpret=PASS on native proves the test itself is correct. FAIL with diagnosis=copy_observed_stale_source_stream_order_dropped (or copy_overwritten_by_unordered_producer_stream_order_dropped for HtoD) proves the copy executed without waiting for the same-stream producer - the convicted defect class. Only cuMemcpy2DAsync+nonblocking is expected to FAIL on the pre-fix stack.
# agent-context: proves=On native: the suite exercises real stream ordering and passes, so a later stack FAIL is attributable to the forwarding chain, not the test. On the Type-2 stack: whether the specific async copy API preserves same-stream ordering relative to a producer kernel.
# agent-context: does_not_prove=It does not measure performance, does not cover batched/graphed copies or cross-stream dependencies, and a stack PASS for DtoD/HtoD on current components reflects their existing stream-forwarding paths, not the 2D fix.
# agent-context: next=Run the same build/ inside the type2-kvm guest against the pre-fix shim (expect memcpy2d_async+nonblocking FAIL) and against the stream-aware shim + QEMU pair (expect all PASS); then register the matrix in CI per gpu-b6a2.

## Convicted mechanism (gpu-w68.5.19)

Guest shim `cuMemcpy2DAsync` dropped `hStream` (only used as a trace label);
the wire command `MEM_COPY_2D_DTOD` carried no stream; QEMU executed a
synchronous `cuMemcpy2D` on the host legacy default stream. The legacy stream
does not implicitly synchronize with non-blocking streams, so the copy could
overtake an in-flight producer kernel and copy not-yet-written source bytes.
Kernel launches do forward the stream, which is exactly why the race is
deterministic: the producer runs on a real non-blocking host stream while the
copy runs on the legacy stream.

Fix pairing status at suite creation (2026-08-09): shim side landed as
`ca0fb8f` (descriptor protocol v2, PARAM6 carries stream_wire); QEMU side
handler still executes the synchronous legacy-stream copy. `type2-fixed`
expects PASS only when both sides are in.

## Layout

| file | content |
| --- | --- |
| `stream_contract.cuh` | shared producer kernel, argument parsing, setup, deterministic validation |
| `memcpy2d_async.cu` | `cuMemcpy2DAsync` (the convicted API) |
| `memcpy_dtod_async.cu` | `cuMemcpyAsync` device-to-device (currently stream-forwarded; negative control + regression gate) |
| `memcpy_htod_async.cu` | `cuMemcpyHtoDAsync` (currently stream-forwarded; inverted dependency: copy must overwrite the earlier same-stream producer) |
| `build.sh` | nvcc build, env `NVCC`/`SM` (default `sm_86`)/`CCBIN` (default `g++-14`) |
| `run.sh` | one-command matrix: `--surface native|type2-prefix|type2-fixed` |

## Usage

```bash
./build.sh                              # produces build/<case> binaries
./run.sh --surface native               # host ground truth, all must PASS
DELAY=0 ./run.sh --surface native       # smoke: ordering is semantic, not timing
./build/memcpy2d_async --stream=nonblocking --delay-cycles=3000000000
```

In the Type-2 guest: copy `build/` plus `run.sh` in, then
`./run.sh --surface type2-prefix` (pre-fix components) or
`./run.sh --surface type2-fixed` (stream-aware shim + QEMU). No rebuild in the
guest; the binaries only need `libcuda.so.1` (the shim) and `libcudart`.

## Expected verdict matrix

| case | stream | native | type2-prefix | type2-fixed |
| --- | --- | --- | --- | --- |
| memcpy2d_async | legacy | PASS | PASS | PASS |
| memcpy2d_async | nonblocking | PASS | FAIL (convicted defect) | PASS |
| memcpy_dtod_async | legacy | PASS | PASS | PASS |
| memcpy_dtod_async | nonblocking | PASS | PASS | PASS |
| memcpy_htod_async | legacy | PASS | PASS | PASS |
| memcpy_htod_async | nonblocking | PASS | PASS | PASS |

The legacy-stream twin of the 2D case is the discriminating control: on the
pre-fix stack the producer (legacy stream, forwarded) and the copy (host
legacy stream) are ordered, so it must PASS; only the non-blocking case fails.
If the legacy case also fails, the stack is broken in a different way and the
matrix must not be read as the convicted defect.

## Determinism

The producer delays `DELAY` clock64 cycles before its first write; the
misordered copy has that whole window to overtake. With the default ~1.7 s
window, a stack that drops the stream fails every run, and the assertion is
exact (word counts over the full buffer), so PASS/FAIL is deterministic.
`DELAY` quantifies the race window width when a suspected fix only narrows it.

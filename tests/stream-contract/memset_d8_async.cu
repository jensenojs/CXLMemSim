// Case: cuMemsetD8Async must be ordered after a producer kernel that writes
// the SAME buffer earlier on the same stream.
//
// Inverted dependency (like the HtoD case): the memset must win. Producer
// writes PAT_FRESH into D on stream S; the memset on S then fills D with
// 0x00. Contract-honoring final content: 0x00. If the forwarding chain drops
// the stream, the fill executes immediately (the shim emulates MEM_SET with
// chunked HtoD copies) while the producer is still spinning, and the producer
// then overwrites it: final content PAT_FRESH -> FAIL.
//
// Component behavior: the pre-fix shim (984feab) is `(void)hStream` + an
// immediate synchronous fill. The fix (30795e2b) drains the caller's stream
// in the guest before filling, so this case is expected to PASS on the fix
// pair even though the Concordia backend still lacks the async copy entries:
// the drain needs no backend support.

#include "stream_contract.cuh"

int main(int argc, char **argv) {
    CaseArgs args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }

    Buffers bufs;
    int rc = setup_buffers(args, &bufs);
    if (rc) return rc;

    cudaStream_t stream;
    rc = make_stream(args.nonblocking, &stream);
    if (rc) return rc;

    // Producer: delay, then write FRESH into the destination buffer, on S.
    producer_kernel<<<1, 256, 0, stream>>>(
        (unsigned int *)(uintptr_t)bufs.dst, bufs.words, PAT_FRESH,
        args.delay_cycles);
    CUDA_RT_CHECK(cudaGetLastError());

    // Operation under test: on S it must execute after the producer and win.
    CUDA_DRV_CHECK(cuMemsetD8Async(bufs.dst, 0x00, args.bytes,
                                   (CUstream)stream));

    CUDA_DRV_CHECK(cuStreamSynchronize((CUstream)stream));

    bool pass = false;
    rc = validate("cuMemsetD8Async", args, bufs.dst, 0x00000000u, PAT_FRESH,
                  "memset_overwritten_by_unordered_producer_stream_order_dropped",
                  bufs.words, &pass);
    if (rc) return rc;

    cuMemFree_v2(bufs.src);
    cuMemFree_v2(bufs.dst);
    if (args.nonblocking) cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}

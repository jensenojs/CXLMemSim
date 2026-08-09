// Case: cuMemcpyHtoDAsync must be ordered after a producer kernel that writes
// the SAME destination buffer earlier on the same stream.
//
// The HtoD source is host memory, so the producer-consumer pattern of the DtoD
// cases does not apply. Instead this case inverts the dependency: the copy
// must win. Producer writes PAT_FRESH into D on stream S; the HtoD copy on S
// then overwrites D with PAT_HOST. Contract-honoring final content: PAT_HOST.
// If the forwarding chain drops the stream and runs the copy unordered against
// the (non-blocking) producer stream, the copy can land first and the producer
// then overwrites it: final content PAT_FRESH -> FAIL.
//
// The current shim forwards the stream for this API
// (CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC), so on-stack it is expected to PASS in
// both stream modes: negative control and regression gate.

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

    // Pinned host source with its own pattern, filled before enqueue.
    void *host_src = nullptr;
    CUDA_DRV_CHECK(cuMemAllocHost_v2(&host_src, args.bytes));
    unsigned int *words = (unsigned int *)host_src;
    for (size_t i = 0; i < bufs.words; i++) {
        words[i] = PAT_HOST;
    }

    // Copy under test: on S it must execute after the producer and win.
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync_v2(bufs.dst, host_src, args.bytes,
                                        (CUstream)stream));

    CUDA_DRV_CHECK(cuStreamSynchronize((CUstream)stream));

    bool pass = false;
    rc = validate("cuMemcpyHtoDAsync", args, bufs.dst, PAT_HOST, PAT_FRESH,
                  "copy_overwritten_by_unordered_producer_stream_order_dropped",
                  bufs.words, &pass);
    cuMemFreeHost(host_src);
    if (rc) return rc;

    cuMemFree_v2(bufs.src);
    cuMemFree_v2(bufs.dst);
    if (args.nonblocking) cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}

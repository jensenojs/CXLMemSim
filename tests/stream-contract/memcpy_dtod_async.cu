// Case: cuMemcpyAsync (device-to-device) must run after a producer kernel on
// the same stream.
//
// Same structure as the cuMemcpy2DAsync case, but the current shim forwards
// the stream for this API (CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC), so on-stack it
// is expected to PASS in both stream modes: it is the negative control that
// proves the suite discriminates stream-semantics loss instead of failing on
// anything, and it is the regression gate if this API ever loses its stream.

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

    producer_kernel<<<1, 256, 0, stream>>>(
        (unsigned int *)(uintptr_t)bufs.src, bufs.words, PAT_FRESH,
        args.delay_cycles);
    CUDA_RT_CHECK(cudaGetLastError());

    // cuMemcpyAsync with two device pointers resolves to DtoD.
    CUDA_DRV_CHECK(cuMemcpyAsync(bufs.dst, bufs.src, args.bytes,
                                 (CUstream)stream));

    CUDA_DRV_CHECK(cuStreamSynchronize((CUstream)stream));

    bool pass = false;
    rc = validate("cuMemcpyAsync-DtoD", args, bufs.dst, PAT_FRESH, PAT_STALE,
                  "copy_observed_stale_source_stream_order_dropped",
                  bufs.words, &pass);
    if (rc) return rc;

    cuMemFree_v2(bufs.src);
    cuMemFree_v2(bufs.dst);
    if (args.nonblocking) cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}

// Case: cuMemcpy2DAsync must run after a producer kernel on the same stream.
//
// The convicted Type-2 defect (gpu-w68.5.19): the guest shim drops hStream,
// the wire command MEM_COPY_2D_DTOD has no stream field, and the device model
// runs a synchronous cuMemcpy2D on the host legacy default stream, which does
// not synchronize with a non-blocking producer stream. The copy then reads
// the source before the producer wrote it.
//
// Expected: PASS natively (both stream modes); PASS on-stack in legacy mode;
// FAIL on-stack in nonblocking mode until the stream-aware fix lands on both
// shim and QEMU.

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

    // Producer: delay, then write FRESH into the whole source, on S.
    producer_kernel<<<1, 256, 0, stream>>>(
        (unsigned int *)(uintptr_t)bufs.src, bufs.words, PAT_FRESH,
        args.delay_cycles);
    CUDA_RT_CHECK(cudaGetLastError());

    // Copy under test: same stream S must order it after the producer.
    // Contiguous 2D region (pitch == width) over the full buffer.
    CUDA_MEMCPY2D copy;
    std::memset(&copy, 0, sizeof(copy));
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = bufs.src;
    copy.dstDevice = bufs.dst;
    copy.srcPitch = args.bytes;
    copy.dstPitch = args.bytes;
    copy.WidthInBytes = args.bytes;
    copy.Height = 1;
    CUDA_DRV_CHECK(cuMemcpy2DAsync_v2(&copy, (CUstream)stream));

    CUDA_DRV_CHECK(cuStreamSynchronize((CUstream)stream));

    bool pass = false;
    rc = validate("cuMemcpy2DAsync", args, bufs.dst, PAT_FRESH, PAT_STALE,
                  "copy_observed_stale_source_stream_order_dropped",
                  bufs.words, &pass);
    if (rc) return rc;

    cuMemFree_v2(bufs.src);
    cuMemFree_v2(bufs.dst);
    if (args.nonblocking) cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}

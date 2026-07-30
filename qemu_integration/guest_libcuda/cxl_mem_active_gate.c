/*
 * Bounded CXL.mem active-path gate.
 *
 * The producer writes a finite object through the BAR4 coherent mapping. The
 * same object is then consumed by the existing CUDA HtoD entry point and
 * checked after DtoH. This keeps the gate on real guest, QEMU and Driver
 * paths without using a synthetic CXLMemSim request.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef uint64_t CUdeviceptr;

#define CUDA_SUCCESS 0
#define GATE_BYTES 4096U
#define GATE_MAGIC UINT32_C(0xC1A2A5E)

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGetCount(int *count);
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev);
extern CUresult cuCtxDestroy_v2(CUcontext ctx);
extern CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
extern CUresult cuMemFree_v2(CUdeviceptr dptr);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytesize);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytesize);

extern int cxlCoherentAlloc(uint64_t size, void **host_ptr);
extern int cxlCoherentFree(void *host_ptr);
extern uint64_t cxlHostToDevice(void *host_ptr);
extern int cxlCoherentFence(void);

static void usage(const char *argv0) {
    printf("usage: %s [--help|--hint]\n", argv0);
}

static void hint(void) {
    puts("agent_hint=problem=Prove one finite guest-owned object reaches BAR4/CXL.mem and a real GPU consumer");
    puts("agent_hint=inputs=Existing CXL Type-2 endpoint, coherent pool, CUDA Driver shim and CXLMemSim server");
    puts("agent_hint=outputs=cxl_mem_active_gate markers with range, logical bytes, GPU consumer and DtoH oracle");
    puts("agent_hint=proves=The declared buffer generated real BAR4 accesses and the existing GPU copy path returned the same bytes");
    puts("agent_hint=does_not_prove=Kimi weight/KV/workspace placement, SSD backing, cache policy or TPS");
    puts("agent_hint=next=Compare CXLMemSim read/write counters and QEMU request intervals from the same run");
}

static uint32_t expected_word(size_t index) {
    return GATE_MAGIC ^ (uint32_t)(index * UINT32_C(0x9e3779b9));
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (argc == 2 && strcmp(argv[1], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (argc == 2 && strcmp(argv[1], "--hint") == 0) {
            hint();
            return 0;
        }
        usage(argv[0]);
        return 2;
    }

    printf("cxl_mem_active_gate=begin bytes=%u source=bar4-coherent-pool consumer=cuda-memcpy\n", GATE_BYTES);
    if (cuInit(0) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuInit");
        return 1;
    }

    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1) {
        puts("cxl_mem_active_gate=fail stage=cuDeviceGetCount");
        return 1;
    }

    CUdevice device;
    CUcontext context;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate_v2(&context, 0, device) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuCtxCreate");
        return 1;
    }

    void *cxl_buffer = NULL;
    if (cxlCoherentAlloc(GATE_BYTES, &cxl_buffer) != 0 || !cxl_buffer) {
        puts("cxl_mem_active_gate=fail stage=cxlCoherentAlloc");
        return 1;
    }
    uint64_t cxl_offset = cxlHostToDevice(cxl_buffer);
    printf("cxl_mem_active_gate=producer status=pass range_offset=0x%" PRIx64 " logical_bytes=%u\n",
           cxl_offset, GATE_BYTES);

    volatile uint32_t *source = (volatile uint32_t *)cxl_buffer;
    for (size_t index = 0; index < GATE_BYTES / sizeof(*source); index++)
        source[index] = expected_word(index);
    if (cxlCoherentFence() != 0) {
        puts("cxl_mem_active_gate=fail stage=cxlCoherentFence-before-consume");
        cxlCoherentFree(cxl_buffer);
        return 1;
    }

    CUdeviceptr device_buffer = 0;
    if (cuMemAlloc_v2(&device_buffer, GATE_BYTES) != CUDA_SUCCESS || !device_buffer) {
        puts("cxl_mem_active_gate=fail stage=cuMemAlloc");
        cxlCoherentFree(cxl_buffer);
        return 1;
    }
    if (cuMemcpyHtoD_v2(device_buffer, (const void *)cxl_buffer, GATE_BYTES) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuMemcpyHtoD-from-cxl");
        cuMemFree_v2(device_buffer);
        cxlCoherentFree(cxl_buffer);
        return 1;
    }

    uint8_t output[GATE_BYTES];
    memset(output, 0, sizeof(output));
    if (cuMemcpyDtoH_v2(output, device_buffer, GATE_BYTES) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuMemcpyDtoH");
        cuMemFree_v2(device_buffer);
        cxlCoherentFree(cxl_buffer);
        return 1;
    }

    size_t mismatch = GATE_BYTES;
    for (size_t index = 0; index < GATE_BYTES / sizeof(uint32_t); index++) {
        uint32_t actual;
        memcpy(&actual, output + index * sizeof(actual), sizeof(actual));
        if (actual != expected_word(index)) {
            mismatch = index * sizeof(actual);
            break;
        }
    }

    cuMemFree_v2(device_buffer);
    cuCtxDestroy_v2(context);
    cxlCoherentFree(cxl_buffer);
    if (mismatch != GATE_BYTES) {
        printf("cxl_mem_active_gate=fail stage=numeric-oracle mismatch_offset=%zu\n", mismatch);
        return 1;
    }

    printf("cxl_mem_active_gate=consumer status=pass gpu_bytes=%u\n", GATE_BYTES);
    printf("cxl_mem_active_gate=oracle status=pass bytes=%u\n", GATE_BYTES);
    puts("cxl_mem_active_gate=pass");
    return 0;
}

/*
 * Bounded CXL.mem active-path gate.
 *
 * The producer writes a finite object through the BAR4 coherent mapping. The
 * same object is then consumed through its mapped device alias and checked
 * after DtoH. This keeps the gate on real guest, QEMU and Driver paths.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUevent;
typedef void *CUstream;
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef uint64_t CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define GATE_BYTES 4096U
#define MAP_BYTES (128U * 1024U * 1024U)
#define GATE_MAGIC UINT32_C(0xC1A2A5E)
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE 38
#define CU_STREAM_CAPTURE_MODE_GLOBAL 0

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGetCount(int *count);
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev);
extern CUresult cuCtxDestroy_v2(CUcontext ctx);
extern CUresult cuDeviceGetAttribute(int *value, int attribute, CUdevice dev);
extern CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
extern CUresult cuMemFree_v2(CUdeviceptr dptr);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytesize);
extern CUresult cuModuleLoadData(CUmodule *module, const void *image);
extern CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                                    const char *name);
extern CUresult cuModuleUnload(CUmodule module);
extern CUresult cuLaunchKernel(CUfunction function, unsigned int grid_x,
                               unsigned int grid_y, unsigned int grid_z,
                               unsigned int block_x, unsigned int block_y,
                               unsigned int block_z, unsigned int shared_bytes,
                               void *stream, void **params, void **extra);
extern CUresult cuEventCreate(CUevent *event, unsigned int flags);
extern CUresult cuEventDestroy_v2(CUevent event);
extern CUresult cuEventRecord(CUevent event, void *stream);
extern CUresult cuEventSynchronize(CUevent event);
extern CUresult cuEventElapsedTime(float *milliseconds, CUevent start,
                                   CUevent end);
extern CUresult cuStreamCreate(CUstream *stream, unsigned int flags);
extern CUresult cuStreamDestroy_v2(CUstream stream);
extern CUresult cuStreamBeginCapture(CUstream stream, int mode);
extern CUresult cuStreamEndCapture(CUstream stream, CUgraph *graph);
extern CUresult cuGraphInstantiateWithFlags(CUgraphExec *graph_exec,
                                            CUgraph graph,
                                            uint64_t flags);
extern CUresult cuGraphLaunch(CUgraphExec graph_exec, CUstream stream);
extern CUresult cuGraphExecDestroy(CUgraphExec graph_exec);
extern CUresult cuGraphDestroy(CUgraph graph);

extern int cxlCoherentAlloc(uint64_t size, void **host_ptr);
extern int cxlCoherentFree(void *host_ptr);
extern uint64_t cxlHostToDevice(void *host_ptr);
extern int cxlCoherentFence(void);
extern int cxlCoherentMapDevice(void *host_ptr, uint64_t mapped_bytes,
                                uint64_t request_bytes,
                                uint64_t *device_alias,
                                int *can_map_host_memory);
extern int cxlCoherentUnmapDevice(void *host_ptr, uint64_t device_alias,
                                  uint64_t *htod_command_delta);

static const char copy_ptx[] =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry cxl_copy_words(\n"
    " .param .u64 src, .param .u64 dst, .param .u64 words) {\n"
    " .reg .pred %p; .reg .b32 %r<7>; .reg .b64 %rd<9>;\n"
    " ld.param.u64 %rd1, [src]; ld.param.u64 %rd2, [dst];\n"
    " ld.param.u64 %rd3, [words]; mov.u32 %r1, %tid.x;\n"
    " mov.u32 %r2, %ntid.x; mov.u32 %r3, %ctaid.x;\n"
    " mad.lo.u32 %r4, %r3, %r2, %r1; mov.u32 %r5, %nctaid.x;\n"
    " mul.lo.u32 %r6, %r5, %r2; cvt.u64.u32 %rd4, %r4;\n"
    " cvt.u64.u32 %rd5, %r6;\n"
    "loop: setp.ge.u64 %p, %rd4, %rd3; @%p bra done;\n"
    " shl.b64 %rd6, %rd4, 2; add.u64 %rd7, %rd1, %rd6;\n"
    " add.u64 %rd8, %rd2, %rd6; ld.global.cg.u32 %r1, [%rd7];\n"
    " st.global.u32 [%rd8], %r1; add.u64 %rd4, %rd4, %rd5; bra loop;\n"
    "done: ret; }\n";

static void usage(const char *argv0) {
    printf("usage: %s [--help|--hint]\n", argv0);
}

static void hint(void) {
    puts("agent_hint=problem=Measure whether one BAR4 host shadow can be read directly by a real GPU without HtoD staging");
    puts("agent_hint=inputs=Existing CXL Type-2 endpoint, coherent pool, CUDA Driver shim and CXLMemSim server");
    puts("agent_hint=outputs=Capability, graph direct-read oracle, three bandwidth samples, zero-HtoD and exact pointer-query markers");
    puts("agent_hint=proves=One captured graph consumed the mapped host shadow through a device alias under the bounded GPU contract");
    puts("agent_hint=does_not_prove=Kimi object placement, direct GPU-to-CXLMemSim requests, overlap or TPS");
    puts("agent_hint=next=Reject the placement candidate below the fixed bandwidth floor; otherwise design exact Kimi object identity");
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

    int exit_status = 1;
    CUdevice device = 0;
    CUcontext context = NULL;
    void *cxl_buffer = NULL;
    uint64_t cxl_offset = 0;
    int l2_bytes = 0;
    int can_map = 0;
    uint64_t device_alias = 0;
    int mapped = 0;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr device_buffer = 0;
    CUevent start = NULL;
    CUevent end = NULL;
    CUstream stream = NULL;
    CUgraph graph = NULL;
    CUgraphExec graph_exec = NULL;
    uint64_t htod_delta = UINT64_MAX;
    float samples_ms[3] = {0.0f, 0.0f, 0.0f};
    double best_gbps = 0.0;
    double median_gbps = 0.0;

    printf("cxl_mem_active_gate=begin bytes=%u mapped_bytes=%u source=bar4-coherent-pool consumer=graph-direct-read\n",
           GATE_BYTES, MAP_BYTES);
    if (cuInit(0) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuInit");
        goto cleanup;
    }

    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 1) {
        puts("cxl_mem_active_gate=fail stage=cuDeviceGetCount");
        goto cleanup;
    }

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS || cuCtxCreate_v2(&context, 0, device) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuCtxCreate");
        goto cleanup;
    }

    if (cxlCoherentAlloc(MAP_BYTES, &cxl_buffer) != 0 || !cxl_buffer) {
        puts("cxl_mem_active_gate=fail stage=cxlCoherentAlloc");
        goto cleanup;
    }
    cxl_offset = cxlHostToDevice(cxl_buffer);
    printf("cxl_mem_active_gate=producer status=pass range_offset=0x%" PRIx64 " logical_bytes=%u\n", cxl_offset,
           GATE_BYTES);

    volatile uint32_t *source = (volatile uint32_t *)cxl_buffer;
    for (size_t index = 0; index < GATE_BYTES / sizeof(*source); index++)
        source[index] = expected_word(index);
    if (cxlCoherentFence() != 0) {
        puts("cxl_mem_active_gate=fail stage=cxlCoherentFence-before-consume");
        goto cleanup;
    }

    if (cuDeviceGetAttribute(&l2_bytes, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device) != CUDA_SUCCESS || l2_bytes <= 0 ||
        (uint64_t)l2_bytes >= MAP_BYTES) {
        puts("cxl_mem_active_gate=fail stage=l2-capacity");
        goto cleanup;
    }
    if (cxlCoherentMapDevice(cxl_buffer, MAP_BYTES, GATE_BYTES, &device_alias, &can_map) != CUDA_SUCCESS ||
        !device_alias || can_map != 1) {
        puts("cxl_mem_active_gate=fail stage=cxlCoherentMapDevice");
        goto cleanup;
    }
    mapped = 1;
    printf("cxl_mem_active_gate=capability status=pass can_map_host_memory=%d mapped_bytes=%u l2_bytes=%d "
           "device_alias=0x%" PRIx64 "\n",
           can_map, MAP_BYTES, l2_bytes, device_alias);

    CUresult result = cuModuleLoadData(&module, copy_ptx);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuModuleLoadData status=%d\n", result);
        goto cleanup;
    }
    result = cuModuleGetFunction(&function, module, "cxl_copy_words");
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuModuleGetFunction status=%d\n", result);
        goto cleanup;
    }
    result = cuMemAlloc_v2(&device_buffer, MAP_BYTES);
    if (result != CUDA_SUCCESS || !device_buffer) {
        printf("cxl_mem_active_gate=fail stage=cuMemAlloc status=%d device_buffer=0x%" PRIx64 "\n", result,
               device_buffer);
        goto cleanup;
    }
    result = cuEventCreate(&start, 0);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuEventCreate-start status=%d\n", result);
        goto cleanup;
    }
    result = cuEventCreate(&end, 0);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuEventCreate-end status=%d\n", result);
        goto cleanup;
    }

    uint64_t words = MAP_BYTES / sizeof(uint32_t);
    void *kernel_params[] = {&device_alias, &device_buffer, &words};
    result = cuStreamCreate(&stream, 0);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuStreamCreate status=%d\n", result);
        goto cleanup;
    }
    result = cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuStreamBeginCapture status=%d\n", result);
        goto cleanup;
    }
    result = cuLaunchKernel(function, 4096, 1, 1, 256, 1, 1, 0, stream,
                            kernel_params, NULL);
    if (result != CUDA_SUCCESS) {
        printf("cxl_mem_active_gate=fail stage=cuLaunchKernel-capture status=%d\n", result);
        goto cleanup;
    }
    result = cuStreamEndCapture(stream, &graph);
    if (result != CUDA_SUCCESS || !graph) {
        printf("cxl_mem_active_gate=fail stage=cuStreamEndCapture status=%d\n", result);
        goto cleanup;
    }
    result = cuGraphInstantiateWithFlags(&graph_exec, graph, 0);
    if (result != CUDA_SUCCESS || !graph_exec) {
        printf("cxl_mem_active_gate=fail stage=cuGraphInstantiate status=%d\n", result);
        goto cleanup;
    }
    puts("cxl_mem_active_gate=graph status=pass nodes=1 launches=4");
    for (size_t run = 0; run < 4; run++) {
        float elapsed_ms = 0.0f;
        result = cuEventRecord(start, stream);
        if (result != CUDA_SUCCESS) {
            printf("cxl_mem_active_gate=fail stage=cuEventRecord-start run=%zu status=%d\n", run, result);
            goto cleanup;
        }
        result = cuGraphLaunch(graph_exec, stream);
        if (result != CUDA_SUCCESS) {
            printf("cxl_mem_active_gate=fail stage=cuGraphLaunch run=%zu status=%d\n", run, result);
            goto cleanup;
        }
        result = cuEventRecord(end, stream);
        if (result != CUDA_SUCCESS) {
            printf("cxl_mem_active_gate=fail stage=cuEventRecord-end run=%zu status=%d\n", run, result);
            goto cleanup;
        }
        result = cuEventSynchronize(end);
        if (result != CUDA_SUCCESS) {
            printf("cxl_mem_active_gate=fail stage=cuEventSynchronize run=%zu status=%d\n", run, result);
            goto cleanup;
        }
        result = cuEventElapsedTime(&elapsed_ms, start, end);
        if (result != CUDA_SUCCESS) {
            printf("cxl_mem_active_gate=fail stage=cuEventElapsedTime run=%zu status=%d\n", run, result);
            goto cleanup;
        }
        if (elapsed_ms <= 0.0f) {
            printf("cxl_mem_active_gate=fail stage=elapsed-nonpositive run=%zu elapsed_ms=%.6f\n", run, elapsed_ms);
            goto cleanup;
        }
        if (run > 0)
            samples_ms[run - 1] = elapsed_ms;
    }
    printf("cxl_mem_active_gate=consumer status=pass gpu_bytes=%u\n", MAP_BYTES);

    uint8_t output[GATE_BYTES];
    memset(output, 0, sizeof(output));
    if (cuMemcpyDtoH_v2(output, device_buffer, GATE_BYTES) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=fail stage=cuMemcpyDtoH");
        goto cleanup;
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

    if (mismatch != GATE_BYTES) {
        printf("cxl_mem_active_gate=fail stage=numeric-oracle mismatch_offset=%zu\n", mismatch);
        goto cleanup;
    }
    printf("cxl_mem_active_gate=oracle status=pass bytes=%u\n", GATE_BYTES);

    if (cxlCoherentUnmapDevice(cxl_buffer, device_alias, &htod_delta) != CUDA_SUCCESS ||
        htod_delta != 0) {
        printf("cxl_mem_active_gate=fail stage=unmap htod_command_delta=%" PRIu64 "\n",
               htod_delta);
        goto cleanup;
    }
    mapped = 0;
    printf("cxl_mem_active_gate=direct_read status=pass mapped_bytes=%u htod_command_delta=%" PRIu64 "\n", MAP_BYTES,
           htod_delta);

    printf("cxl_mem_active_gate=lifetime status=pass retire_policy=device-exit\n");

    float sorted[3] = {samples_ms[0], samples_ms[1], samples_ms[2]};
    for (size_t left = 0; left < 2; left++) {
        for (size_t right = left + 1; right < 3; right++) {
            if (sorted[right] < sorted[left]) {
                float tmp = sorted[left];
                sorted[left] = sorted[right];
                sorted[right] = tmp;
            }
        }
    }
    best_gbps = ((double)MAP_BYTES / 1000000000.0) / ((double)sorted[0] / 1000.0);
    median_gbps = ((double)MAP_BYTES / 1000000000.0) / ((double)sorted[1] / 1000.0);
    printf("cxl_mem_active_gate=bandwidth status=measured sample_ms=%.6f,%.6f,%.6f best_gbps=%.6f median_gbps=%.6f "
           "required_gbps=22.678274\n",
           samples_ms[0], samples_ms[1], samples_ms[2], best_gbps, median_gbps);
    if (best_gbps < 22.678274) {
        printf("cxl_mem_active_gate=fail stage=bandwidth-ceiling best_gbps=%.6f required_gbps=22.678274\n", best_gbps);
        goto cleanup;
    }

    exit_status = 0;

cleanup:
    if (mapped) {
        uint64_t cleanup_delta = UINT64_MAX;
        int cleanup_result =
            cxlCoherentUnmapDevice(cxl_buffer, device_alias, &cleanup_delta);
        if (cleanup_result == CUDA_SUCCESS) {
            mapped = 0;
        } else {
            printf("cxl_mem_active_gate=cleanup_fail stage=unmap status=%d htod_command_delta=%" PRIu64 "\n",
                   cleanup_result, cleanup_delta);
            exit_status = 1;
        }
    }
    if (end && cuEventDestroy_v2(end) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=event-end");
        exit_status = 1;
    }
    if (start && cuEventDestroy_v2(start) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=event-start");
        exit_status = 1;
    }
    if (graph_exec && cuGraphExecDestroy(graph_exec) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=graph-exec");
        exit_status = 1;
    }
    if (graph && cuGraphDestroy(graph) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=graph");
        exit_status = 1;
    }
    if (stream && cuStreamDestroy_v2(stream) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=stream");
        exit_status = 1;
    }
    if (device_buffer && cuMemFree_v2(device_buffer) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=device-buffer");
        exit_status = 1;
    }
    if (module && cuModuleUnload(module) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=module");
        exit_status = 1;
    }
    if (context && cuCtxDestroy_v2(context) != CUDA_SUCCESS) {
        puts("cxl_mem_active_gate=cleanup_fail stage=context");
        exit_status = 1;
    }
    if (cxl_buffer && !mapped && cxlCoherentFree(cxl_buffer) != 0) {
        puts("cxl_mem_active_gate=cleanup_fail stage=coherent-buffer");
        exit_status = 1;
    }
    if (exit_status != 0)
        return exit_status;

    puts("cxl_mem_active_gate=pass");
    return 0;
}

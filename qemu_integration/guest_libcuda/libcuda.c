/*
 * CXL Type 2 GPU - Guest libcuda.so Shim
 * Implements CUDA Driver API by communicating with CXL Type 2 device
 *
 * This library intercepts CUDA calls and forwards them to the CXL Type 2
 * device which runs the hetGPU backend on the host.
 *
 * Compile: gcc -shared -fPIC -o libcuda.so.1 libcuda.c -ldl
 * Usage: LD_PRELOAD=./libcuda.so.1 ./cuda_program
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cxl_gpu_cmd.h"
#include "cxl_gpu_context_state.h"
#include "cxl_gpu_transport.h"

/* These symbols are linked into the shim's declared runtime dependency set. */
extern int LZ4_decompress_safe(const char *src, char *dst, int compressed_size, int dst_capacity);
extern size_t ZSTD_decompress(void *dst, size_t dst_capacity, const void *src, size_t compressed_size);
extern unsigned int ZSTD_isError(size_t code);
extern void cxl_cuda_provenance_emit_first_stack(const char *symbol);

/* CUDA types */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUarray;
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;
typedef void *CUlinkState;
typedef int CUgraphNodeType;
typedef void *CUlibrary;
typedef void *CUkernel;
typedef int CUjit_option;
typedef int CUlibraryOption;
typedef int CUkernelNodeAttrID;
typedef int CUfunction_attribute;
typedef int CUfunc_cache;
typedef int CUsharedconfig;
typedef enum {
    CU_MODULE_EAGER_LOADING = 0x1,
    CU_MODULE_LAZY_LOADING = 0x2,
} CUmoduleLoadingMode;
typedef uint64_t CUdeviceptr;
typedef uint64_t cuuint64_t;
typedef uint32_t cuuint32_t;
typedef int CUlimit;
typedef int CUjitInputType;
typedef int CUstreamCaptureStatus;
typedef int CUstreamCaptureMode;
typedef int CUmemorytype;
typedef enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
} CUdriverProcAddressQueryResult;

typedef struct {
    unsigned char bytes[16];
} CUuuid;

typedef struct {
    size_t srcXInBytes;
    size_t srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2D;

typedef struct {
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
    CUkernel kern;
    CUcontext ctx;
} CUDA_KERNEL_NODE_PARAMS;

typedef union {
    int operation;
    uint64_t pad[6];
} CUstreamBatchMemOpParams;

typedef struct {
    void *functionTable;
    size_t functionWindowSize;
    void *dataTable;
    size_t dataWindowSize;
} CUlibraryHostUniversalFunctionAndDataTable;

typedef struct {
    uint32_t magic;
    uint32_t version;
    const void *data;
    const void *filename_or_fatbins;
} CudartFatbincWrapper;

typedef struct __attribute__((aligned(8))) {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t files_size;
} CudartFatbinHeader;

typedef struct {
    uint16_t kind;
    uint16_t version;
    uint32_t header_size;
    uint32_t payload_size;
    uint32_t unknown0;
    uint32_t compressed_size;
    uint32_t unknown1;
    uint32_t unknown2;
    uint32_t sm_version;
    uint32_t bit_width;
    uint32_t unknown3;
    uint64_t flags;
    uint64_t unknown5;
    uint64_t uncompressed_payload;
} CudartFatbinFileHeader;

#define CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE 0
#define CU_LIBRARY_BINARY_IS_PRESERVED 1
#define CUDART_FATBINC_MAGIC 0x466243B1U
#define CUDART_FATBINC_VERSION 0x1U
#define CUDART_FATBIN_MAGIC 0xBA55ED50U
#define CUDART_FATBIN_VERSION 0x1U
#define CUDART_FATBIN_KIND_PTX 0x1U
#define CUDART_FATBIN_KIND_ELF 0x2U
#define CUDART_FATBIN_FLAG_COMPRESSED_LZ4 0x2000ULL
#define CUDART_FATBIN_FLAG_COMPRESSED_ZSTD 0x8000ULL

/* CUDA error codes */
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_NO_BINARY_FOR_GPU 209
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_READY 600
#define CUDA_ERROR_LAUNCH_FAILED 700
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE 708
#define CUDA_ERROR_CONTEXT_IS_DESTROYED 709
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_ERROR_UNKNOWN 999

#define CU_MEMORYTYPE_DEVICE 0x02

/* CUDA device attributes */
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X 2
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y 3
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z 4
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X 5
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y 6
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z 7
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE 10
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

#define CXL_MAX_KERNEL_ARGS 64

/* Global state */
static CxlGpuTransport g_transport = CXL_GPU_TRANSPORT_INITIALIZER;
static int g_initialized = 0;
#ifdef CXL_GPU_CONTEXT_SHIM_TEST
static uint8_t g_test_bar2[CXL_GPU_CMD_REG_SIZE];
static CUresult (*g_test_execute_cmd)(uint32_t cmd);
#endif
static uintptr_t g_cudart_placeholder_module = 0x435844465442494eULL; /* "CXDFTBIN" diagnostic placeholder */
static uintptr_t g_cudart_placeholder_library = 0x4358444c49425259ULL; /* "CXDLIBRY" diagnostic placeholder */
static uintptr_t g_cudart_placeholder_kernel = 0x4358444b45524e4cULL; /* "CXDKERNL" diagnostic placeholder */
static uintptr_t g_cudart_placeholder_function = 0x43584446554e4354ULL; /* "CXDFUNCT" diagnostic placeholder */

typedef struct CXLGraphKernelNodeSnapshot {
    uint64_t node_id;
    CUDA_KERNEL_NODE_PARAMS params;
    void **kernel_params;
    uint8_t *param_bytes;
    struct CXLGraphKernelNodeSnapshot *next;
} CXLGraphKernelNodeSnapshot;

static CXLGraphKernelNodeSnapshot *g_graph_kernel_node_snapshots;
static void graph_kernel_node_snapshots_clear(void);

typedef struct CXLLinkOutput {
    uint64_t id;
    void *bytes;
    size_t size;
    struct CXLLinkOutput *next;
} CXLLinkOutput;

typedef struct CXLStreamCaptureSnapshot {
    uint64_t stream_wire;
    CUgraphNode *dependencies;
    size_t count;
    struct CXLStreamCaptureSnapshot *next;
} CXLStreamCaptureSnapshot;

static CXLLinkOutput *g_link_outputs;
static CXLStreamCaptureSnapshot *g_stream_capture_snapshots;

typedef struct CXLCudaErrorName {
    CUresult error;
    char *name;
    struct CXLCudaErrorName *next;
} CXLCudaErrorName;

static CXLCudaErrorName *g_cuda_error_names;
static CXLCudaErrorName *g_cuda_error_strings;
static pthread_mutex_t g_cuda_error_names_lock = PTHREAD_MUTEX_INITIALIZER;

#define CUDART_LIBRARY_RECORD_OPTION_CAP 4
#define CUDART_LIBRARY_RECORD_MAGIC 0x43584c4942524152ULL /* "CXLIBRAR" */

typedef struct CudartLibraryRecord {
    uint64_t magic;
    unsigned int id;
    int alive;
    const void *code;
    unsigned int num_jit_options;
    unsigned int num_library_options;
    unsigned int stored_library_options;
    int preserve_binary;
    const void *preserved_code;
    CUmodule module;
    CUlibraryOption options[CUDART_LIBRARY_RECORD_OPTION_CAP];
    void *option_values[CUDART_LIBRARY_RECORD_OPTION_CAP];
    struct CudartLibraryRecord *next;
} CudartLibraryRecord;

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuCtxGetCurrent(CUcontext *pctx);

static CudartLibraryRecord *g_cudart_library_records = NULL;
static unsigned int g_cudart_library_next_id = 1;

/* CUlibrary is the record address.  Unload marks a record dead but keeps that
 * address reserved until process exit, so an old opaque handle can never name
 * a later library after allocator reuse.  The test reset frees the whole list
 * because no handle may cross a test boundary.
 * knockout: handle lookup is linear in registered libraries; replace the
 * lookup index only if measured Kimi setup time shows this list on the critical
 * path, while retaining the address-stable tombstone lifetime. */

static CudartLibraryRecord *cudart_library_record_from_handle(CUlibrary library) {
    if (!library) {
        return NULL;
    }
    for (CudartLibraryRecord *record = g_cudart_library_records; record; record = record->next) {
        if ((CUlibrary)record == library && record->magic == CUDART_LIBRARY_RECORD_MAGIC) {
            return record;
        }
    }
    return NULL;
}

/* Debug logging */
static int g_debug = 0;
#define DLOG(...)                                                                                                      \
    do {                                                                                                               \
        if (g_debug)                                                                                                   \
            fprintf(stderr, "[CXL-CUDA] " __VA_ARGS__);                                                                \
    } while (0)

static void log_context_state(const char *api, CUresult result) {
    CxlCudaContextStateView state = cxl_cuda_context_state_view();

    DLOG("context_state api=%s result=%d token=%" PRIuPTR " mode=%u "
         "primary_retain_count=%u current_depth=%u\n",
         api, result, state.token, state.mode, state.primary_retain_count, state.current_depth);
}

static inline uint64_t maybe_bar4_offset(uint64_t value);
static inline uint64_t bar4_offset_of(void *host_ptr);

/* The CUDA shim and cxl-gpu-case share the transport implementation.  These
 * wrappers preserve the existing call sites while keeping BAR2 ownership in
 * cxl_gpu_transport.c. */
static inline uint32_t reg_read32(uint32_t offset) { return cxl_gpu_transport_read32(&g_transport, offset); }

static inline uint64_t reg_read64(uint32_t offset) { return cxl_gpu_transport_read64(&g_transport, offset); }

static inline void reg_write32(uint32_t offset, uint32_t value) {
    cxl_gpu_transport_write32(&g_transport, offset, value);
}

static inline void reg_write64(uint32_t offset, uint64_t value) {
    cxl_gpu_transport_write64(&g_transport, offset, value);
}

/* QEMU stores module/function handles as zero-based array indices. CUDA opaque
 * handles must keep NULL reserved for failure, so expose index+1 to callers
 * and decode it only at the BAR2 command boundary. */
static inline void *cxl_gpu_handle_from_id(uint64_t id) { return (void *)(uintptr_t)(id + 1); }

static inline uint64_t cxl_gpu_id_from_handle(const void *handle) { return (uint64_t)(uintptr_t)handle - 1; }

static bool cxl_gpu_handle_id(const void *handle, uint64_t *id) {
    uint64_t value;

    if (!handle || !id)
        return false;
    value = cxl_gpu_id_from_handle(handle);
    if (value > UINT32_MAX)
        return false;
    *id = value;
    return true;
}

#define CXL_GPU_STREAM_HANDLE_TAG (UINT64_C(1) << 32)

static inline CUstream cxl_gpu_stream_handle_from_id(uint64_t id) {
    return (CUstream)(uintptr_t)(CXL_GPU_STREAM_HANDLE_TAG | id);
}

static bool cxl_gpu_stream_handle_id(CUstream stream, uint64_t *id) {
    uint64_t value = (uint64_t)(uintptr_t)stream;

    if (!id || value < CXL_GPU_STREAM_HANDLE_TAG ||
        value > CXL_GPU_STREAM_HANDLE_TAG + UINT32_MAX)
        return false;
    *id = value - CXL_GPU_STREAM_HANDLE_TAG;
    return true;
}

static bool cxl_gpu_stream_wire(CUstream stream, uint64_t *wire) {
    uint64_t value;

    if (!wire)
        return false;
    if (!stream) {
        *wire = CXL_GPU_STREAM_WIRE_NULL;
        return true;
    }
    value = (uint64_t)(uintptr_t)stream;
    if (value == 1) {
        *wire = CXL_GPU_STREAM_WIRE_LEGACY;
        return true;
    }
    if (value == 2) {
        *wire = CXL_GPU_STREAM_WIRE_PER_THREAD;
        return true;
    }
    return cxl_gpu_stream_handle_id(stream, wire);
}

static CXLLinkOutput *link_output_get(uint64_t id, bool create) {
    for (CXLLinkOutput *output = g_link_outputs; output; output = output->next) {
        if (output->id == id)
            return output;
    }
    if (!create)
        return NULL;
    CXLLinkOutput *output = calloc(1, sizeof(*output));
    if (!output)
        return NULL;
    output->id = id;
    output->next = g_link_outputs;
    g_link_outputs = output;
    return output;
}

static void link_output_remove(uint64_t id) {
    CXLLinkOutput **cursor = &g_link_outputs;
    while (*cursor) {
        if ((*cursor)->id == id) {
            CXLLinkOutput *output = *cursor;
            *cursor = output->next;
            free(output->bytes);
            free(output);
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static CXLStreamCaptureSnapshot *stream_capture_snapshot_get(uint64_t wire) {
    for (CXLStreamCaptureSnapshot *snapshot = g_stream_capture_snapshots;
         snapshot; snapshot = snapshot->next) {
        if (snapshot->stream_wire == wire)
            return snapshot;
    }
    CXLStreamCaptureSnapshot *snapshot = calloc(1, sizeof(*snapshot));
    if (!snapshot)
        return NULL;
    snapshot->stream_wire = wire;
    snapshot->next = g_stream_capture_snapshots;
    g_stream_capture_snapshots = snapshot;
    return snapshot;
}

static inline void data_write(size_t offset, const void *src, size_t len) {
    cxl_gpu_transport_data_write(&g_transport, offset, src, len);
}

static inline void data_read(size_t offset, void *dst, size_t len) {
    cxl_gpu_transport_data_read(&g_transport, offset, dst, len);
}

/* Cross-process lock for command serialization.
 * Multiple processes sharing the same BAR2 MMIO registers must serialize
 * their full command sequences (write params  write cmd  poll status  read result).
 * Callers that write params before execute_cmd must call cmd_lock()/cmd_unlock(). */
static void cmd_lock(void) {
    if (cxl_gpu_transport_lock(&g_transport) != 0)
        DLOG("BAR2 command lock failed: %s\n", strerror(errno));
}

static void cmd_unlock(void) {
    if (cxl_gpu_transport_unlock(&g_transport) != 0)
        DLOG("BAR2 command unlock failed: %s\n", strerror(errno));
}

/* Execute command and wait for completion.
 * Caller MUST hold cmd_lock() if params were written before this call. */
static uint32_t g_api_chain_sequence;

static CUresult execute_cmd_traced(uint32_t cmd, const char *symbol) {
    uint32_t sequence = __sync_add_and_fetch(&g_api_chain_sequence, 1);
    uint64_t call_id = ((uint64_t)(uint32_t)getpid() << 32) | sequence;
    reg_write64(CXL_GPU_REG_CALL_ID, call_id);
    DLOG("api_chain event=guest-entry call_id=0x%016" PRIx64
         " symbol=%s command=0x%x\n",
         call_id, symbol, cmd);
#ifdef CXL_GPU_CONTEXT_SHIM_TEST
    if (g_test_execute_cmd) {
        CUresult result = g_test_execute_cmd(cmd);
        reg_write64(CXL_GPU_REG_CALL_ID, 0);
        DLOG("api_chain event=guest-return call_id=0x%016" PRIx64
             " symbol=%s command=0x%x result=%d\n",
             call_id, symbol, cmd, result);
        return result;
    }
#endif
    CUresult result = (CUresult)cxl_gpu_transport_execute(&g_transport, cmd);
    reg_write64(CXL_GPU_REG_CALL_ID, 0);
    DLOG("api_chain event=guest-return call_id=0x%016" PRIx64
         " symbol=%s command=0x%x result=%d\n",
         call_id, symbol, cmd, result);
    return result;
}

#define execute_cmd(cmd) execute_cmd_traced((cmd), __func__)

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
static void context_storage_test_reset(void);

void cxl_cuda_test_reset(void) {
    while (g_cudart_library_records) {
        CudartLibraryRecord *record = g_cudart_library_records;
        g_cudart_library_records = record->next;
        free(record);
    }
    g_cudart_library_next_id = 1;
    g_api_chain_sequence = 0;
    graph_kernel_node_snapshots_clear();
    while (g_link_outputs)
        link_output_remove(g_link_outputs->id);
    while (g_stream_capture_snapshots) {
        CXLStreamCaptureSnapshot *snapshot = g_stream_capture_snapshots;
        g_stream_capture_snapshots = snapshot->next;
        free(snapshot->dependencies);
        free(snapshot);
    }
    memset(g_test_bar2, 0, sizeof(g_test_bar2));
    g_transport = (CxlGpuTransport)CXL_GPU_TRANSPORT_INITIALIZER;
    g_transport.regs = (volatile uint32_t *)g_test_bar2;
    g_transport.data = (volatile uint8_t *)g_test_bar2 + CXL_GPU_DATA_OFFSET;
    g_transport.bar_size = sizeof(g_test_bar2);
    g_initialized = 1;
    g_test_execute_cmd = NULL;
    cxl_cuda_context_state_reset();
    context_storage_test_reset();
}

void cxl_cuda_test_set_executor(CUresult (*executor)(uint32_t cmd)) { g_test_execute_cmd = executor; }

uint64_t cxl_cuda_test_read_reg64(uint32_t offset) { return reg_read64(offset); }

void cxl_cuda_test_write_result(unsigned int index, uint64_t value) {
    static const uint32_t result_offsets[] = {
        CXL_GPU_REG_RESULT0,
        CXL_GPU_REG_RESULT1,
        CXL_GPU_REG_RESULT2,
        CXL_GPU_REG_RESULT3,
    };

    if (index < sizeof(result_offsets) / sizeof(result_offsets[0]))
        reg_write64(result_offsets[index], value);
}

void cxl_cuda_test_write_reg32(uint32_t offset, uint32_t value) { reg_write32(offset, value); }

void cxl_cuda_test_read_data(size_t offset, void *dst, size_t length) {
    cxl_gpu_transport_data_read(&g_transport, offset, dst, length);
}
#endif

/* Find and map CXL Type 2 device */
static int find_and_map_device(void) { return cxl_gpu_transport_open(&g_transport, g_debug); }

/* CUDA 12 runtime resolves most driver entry points through cuGetProcAddress.
 * Use the dynamic symbol table of this shim so missing APIs stay visible in
 * CXL_CUDA_DEBUG logs. */
static void *lookup_proc_address(const char *symbol) {
    if (!symbol) {
        return NULL;
    }
    if (getenv("CXL_CUDA_HIDE_LIBRARY_LOAD_DATA") && strcmp(symbol, "cuLibraryLoadData") == 0) {
        /* knockout: CUDA 12.9 registration path probe. This deliberately hides
         * cuLibraryLoadData from cuGetProcAddress without removing the exported
         * symbol, so we can test whether libcudart has an older fallback path.
         * Remove once the after-dlopen err=36 root cause is identified. */
        fprintf(stderr, "[CXL-CUDA] lookup_proc_address(symbol=%s) hidden by CXL_CUDA_HIDE_LIBRARY_LOAD_DATA\n",
                symbol);
        return NULL;
    }
    void *fn = dlsym(RTLD_DEFAULT, symbol);
    fprintf(stderr, "[CXL-CUDA] lookup_proc_address(symbol=%s) -> %p\n", symbol, fn);
    return fn;
}

static void log_proc_address_caller(const char *event, const char *symbol, const void *caller) {
    Dl_info info = {0};
    if (dladdr(caller, &info) != 0 && info.dli_fbase && info.dli_fname) {
        uintptr_t base = (uintptr_t)info.dli_fbase;
        fprintf(stderr,
                "[CXL-CUDA] api_provenance event=%s pid=%ld symbol=%s caller_status=resolved "
                "caller_file=%s caller_base=0x%llx caller_offset=0x%llx\n",
                event, (long)getpid(), symbol ? symbol : "(null)", info.dli_fname, (unsigned long long)base,
                (unsigned long long)((uintptr_t)caller - base));
    } else {
        fprintf(stderr,
                "[CXL-CUDA] api_provenance event=%s pid=%ld symbol=%s caller_status=unresolved caller_address=%p\n",
                event, (long)getpid(), symbol ? symbol : "(null)", caller);
    }
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                          CUdriverProcAddressQueryResult *symbolStatus) {
    const void *caller = __builtin_return_address(0);
    g_debug = (getenv("CXL_CUDA_DEBUG") != NULL);
    fprintf(stderr, "[CXL-CUDA] cuGetProcAddress(symbol=%s, version=%d, flags=0x%lx, pfn=%p, status=%p)\n",
            symbol ? symbol : "(null)", cudaVersion, (unsigned long)flags, (void *)pfn, (void *)symbolStatus);
    log_proc_address_caller("query", symbol, caller);
    (void)cudaVersion;
    (void)flags;

    if (!pfn) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    void *fn = lookup_proc_address(symbol);
    if (!fn) {
        *pfn = NULL;
        if (symbolStatus) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        }
        fprintf(stderr, "[CXL-CUDA] cuGetProcAddress(%s) -> pfn=NULL status=SYMBOL_NOT_FOUND result=CUDA_SUCCESS\n",
                symbol ? symbol : "(null)");
        log_proc_address_caller("unresolved", symbol, caller);
        /* CUDA's driver entry-point API reports unsupported symbols through
         * pfn=NULL and symbolStatus, while the call itself still succeeds.
         * libcudart probes large API tables during initialization and treats a
         * non-success return as a driver entry-point query failure rather than
         * as an optional missing symbol. */
        return CUDA_SUCCESS;
    }

    *pfn = fn;
    if (symbolStatus) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    fprintf(stderr, "[CXL-CUDA] cuGetProcAddress(%s) -> pfn=%p status=SUCCESS result=CUDA_SUCCESS\n", symbol, fn);
    log_proc_address_caller("resolved", symbol, caller);
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
    return cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

CUresult cuFuncGetParamInfo(CUfunction hfunc, size_t paramIndex, size_t *paramOffset,
                            size_t *paramSize);

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, void *value_out) {
    (void)hNode;
    (void)value_out;
    DLOG("cuGraphKernelNodeGetAttribute(attr=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", attr);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CXLGraphKernelNodeSnapshot *graph_kernel_node_snapshot_find(uint64_t node_id) {
    for (CXLGraphKernelNodeSnapshot *snapshot = g_graph_kernel_node_snapshots; snapshot;
         snapshot = snapshot->next) {
        if (snapshot->node_id == node_id)
            return snapshot;
    }
    return NULL;
}

static void graph_kernel_node_snapshots_clear(void) {
    while (g_graph_kernel_node_snapshots) {
        CXLGraphKernelNodeSnapshot *snapshot = g_graph_kernel_node_snapshots;

        g_graph_kernel_node_snapshots = snapshot->next;
        free(snapshot->kernel_params);
        free(snapshot->param_bytes);
        free(snapshot);
    }
}

CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    CXLGraphKernelNodeParamsWire wire = {0};
    uint64_t node_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hNode || !nodeParams)
        return CUDA_ERROR_INVALID_VALUE;
    node_id = cxl_gpu_id_from_handle(hNode);
    if (node_id > UINT32_MAX)
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    CXLGraphKernelNodeSnapshot *snapshot = graph_kernel_node_snapshot_find(node_id);
    if (snapshot) {
        *nodeParams = snapshot->params;
        cmd_unlock();
        return CUDA_SUCCESS;
    }
    reg_write64(CXL_GPU_REG_PARAM0, node_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_KERNEL_NODE_GET_PARAMS);
    if (result != CUDA_SUCCESS) {
        cmd_unlock();
        return result;
    }
    data_read(0, &wire, sizeof(wire));
    if (wire.function_id == UINT32_MAX || wire.reserved != 0 || wire.num_args > CXL_MAX_KERNEL_ARGS ||
        wire.num_args > (CXL_GPU_DATA_SIZE - sizeof(wire)) / sizeof(CXLGraphKernelNodeParamWire) ||
        (wire.num_args && wire.param_extent == 0) ||
        wire.param_extent > CXL_GPU_DATA_SIZE - sizeof(wire) -
            wire.num_args * sizeof(CXLGraphKernelNodeParamWire)) {
        cmd_unlock();
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint8_t *param_bytes = wire.param_extent ? malloc(wire.param_extent) : NULL;
    void **kernel_params = wire.num_args ? calloc(wire.num_args, sizeof(*kernel_params)) : NULL;
    if ((wire.param_extent && !param_bytes) || (wire.num_args && !kernel_params)) {
        free(param_bytes);
        free(kernel_params);
        cmd_unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (wire.param_extent)
        data_read(sizeof(wire) + wire.num_args * sizeof(CXLGraphKernelNodeParamWire),
                  param_bytes, wire.param_extent);
    CXLGraphKernelNodeParamWire param_wires[CXL_MAX_KERNEL_ARGS] = {0};
    if (wire.num_args)
        data_read(sizeof(wire), param_wires, wire.num_args * sizeof(*param_wires));
    CUfunction function = (CUfunction)cxl_gpu_handle_from_id(wire.function_id);
    for (uint32_t i = 0; i < wire.num_args; i++) {
        if (param_wires[i].offset > wire.param_extent ||
            param_wires[i].size > wire.param_extent - param_wires[i].offset) {
            free(param_bytes);
            free(kernel_params);
            cmd_unlock();
            return CUDA_ERROR_INVALID_VALUE;
        }
        kernel_params[i] = param_bytes + param_wires[i].offset;
    }
    snapshot = calloc(1, sizeof(*snapshot));
    if (!snapshot) {
        free(param_bytes);
        free(kernel_params);
        cmd_unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    snapshot->node_id = node_id;
    snapshot->kernel_params = kernel_params;
    snapshot->param_bytes = param_bytes;
    snapshot->params.func = function;
    snapshot->params.gridDimX = wire.grid_dim_x;
    snapshot->params.gridDimY = wire.grid_dim_y;
    snapshot->params.gridDimZ = wire.grid_dim_z;
    snapshot->params.blockDimX = wire.block_dim_x;
    snapshot->params.blockDimY = wire.block_dim_y;
    snapshot->params.blockDimZ = wire.block_dim_z;
    snapshot->params.sharedMemBytes = wire.shared_mem_bytes;
    snapshot->params.kernelParams = kernel_params;
    snapshot->params.extra = NULL;
    snapshot->params.kern = NULL;
    snapshot->params.ctx = NULL;
    snapshot->next = g_graph_kernel_node_snapshots;
    g_graph_kernel_node_snapshots = snapshot;
    *nodeParams = snapshot->params;
    cmd_unlock();
    return CUDA_SUCCESS;
}

CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                        const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    size_t param_offsets[CXL_MAX_KERNEL_ARGS];
    size_t param_sizes[CXL_MAX_KERNEL_ARGS];
    size_t param_extent = 0;
    uint32_t num_args = 0;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hGraphExec || !hNode || !nodeParams || !nodeParams->func)
        return CUDA_ERROR_INVALID_VALUE;
    if (!nodeParams->kernelParams && nodeParams->extra)
        return CUDA_ERROR_NOT_SUPPORTED;

    while (num_args < CXL_MAX_KERNEL_ARGS) {
        size_t offset = 0;
        size_t size = 0;
        CUresult result = cuFuncGetParamInfo(nodeParams->func, num_args, &offset, &size);

        if (result == CUDA_ERROR_INVALID_VALUE)
            break;
        if (result != CUDA_SUCCESS)
            return result;
        if (!nodeParams->kernelParams || !nodeParams->kernelParams[num_args] ||
            offset > CXL_GPU_DATA_SIZE || size > CXL_GPU_DATA_SIZE - offset)
            return CUDA_ERROR_INVALID_VALUE;
        param_offsets[num_args] = offset;
        param_sizes[num_args] = size;
        if (offset + size > param_extent)
            param_extent = offset + size;
        num_args++;
    }
    if (num_args == CXL_MAX_KERNEL_ARGS) {
        size_t offset = 0;
        size_t size = 0;
        if (cuFuncGetParamInfo(nodeParams->func, num_args, &offset, &size) == CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
    }

    uint8_t *param_buffer = NULL;
    if (param_extent) {
        param_buffer = calloc(1, param_extent);
        if (!param_buffer)
            return CUDA_ERROR_OUT_OF_MEMORY;
        for (uint32_t i = 0; i < num_args; i++)
            memcpy(param_buffer + param_offsets[i], nodeParams->kernelParams[i], param_sizes[i]);
    }

    cmd_lock();
    if (param_extent)
        data_write(0, param_buffer, param_extent);
    free(param_buffer);
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hGraphExec));
    reg_write64(CXL_GPU_REG_PARAM1, cxl_gpu_id_from_handle(hNode));
    reg_write64(CXL_GPU_REG_PARAM2, cxl_gpu_id_from_handle(nodeParams->func));
    reg_write64(CXL_GPU_REG_PARAM3, ((uint64_t)nodeParams->gridDimY << 32) | nodeParams->gridDimX);
    reg_write64(CXL_GPU_REG_PARAM4, ((uint64_t)nodeParams->blockDimX << 32) | nodeParams->gridDimZ);
    reg_write64(CXL_GPU_REG_PARAM5, ((uint64_t)nodeParams->blockDimZ << 32) | nodeParams->blockDimY);
    reg_write64(CXL_GPU_REG_PARAM6, ((uint64_t)num_args << 32) | nodeParams->sharedMemBytes);
    reg_write64(CXL_GPU_REG_PARAM7, param_extent);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS);
    cmd_unlock();
    return result;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    uint64_t graph_exec_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hGraphExec, &graph_exec_id))
        return CUDA_ERROR_INVALID_HANDLE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_exec_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_EXEC_DESTROY);
    cmd_unlock();
    return result;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    uint64_t graph_exec_id, stream_wire;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hGraphExec, &graph_exec_id))
        return CUDA_ERROR_INVALID_HANDLE;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_exec_id);
    reg_write64(CXL_GPU_REG_PARAM1, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_LAUNCH);
    cmd_unlock();
    return result;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
    uint64_t graph_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hGraph, &graph_id))
        return CUDA_ERROR_INVALID_HANDLE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_DESTROY);
    cmd_unlock();
    return result;
}

CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                            CUgraphNode *phErrorNode, char *logBuffer,
                            size_t bufferSize) {
    uint64_t graph_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!phGraphExec || !cxl_gpu_handle_id(hGraph, &graph_id))
        return CUDA_ERROR_INVALID_VALUE;
    if ((!logBuffer && bufferSize) || bufferSize > CXL_GPU_DATA_SIZE)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_id);
    reg_write64(CXL_GPU_REG_PARAM1, bufferSize);
    reg_write64(CXL_GPU_REG_PARAM2, phErrorNode ? 1 : 0);
    reg_write64(CXL_GPU_REG_PARAM3, logBuffer ? 1 : 0);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_INSTANTIATE);
    uint64_t graph_exec_id = reg_read64(CXL_GPU_REG_RESULT0);
    uint64_t error_node_id = reg_read64(CXL_GPU_REG_RESULT1);
    if (logBuffer && bufferSize)
        data_read(0, logBuffer, bufferSize);
    if (phErrorNode)
        *phErrorNode = error_node_id == UINT64_MAX ? NULL :
                       (CUgraphNode)cxl_gpu_handle_from_id(error_node_id);
    if (result == CUDA_SUCCESS) {
        if (graph_exec_id == UINT64_MAX || graph_exec_id > UINT32_MAX) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_HANDLE;
        }
        *phGraphExec = (CUgraphExec)cxl_gpu_handle_from_id(graph_exec_id);
    }
    cmd_unlock();
    return result;
}

CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                               CUgraphNode *phErrorNode, char *logBuffer,
                               size_t bufferSize) {
    return cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer,
                              bufferSize);
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
    uint64_t graph_id;
    size_t capacity;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!numNodes || !cxl_gpu_handle_id(hGraph, &graph_id))
        return CUDA_ERROR_INVALID_VALUE;
    capacity = nodes ? *numNodes : 0;
    if (capacity > CXL_GPU_DATA_SIZE / sizeof(uint64_t))
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_id);
    reg_write64(CXL_GPU_REG_PARAM1, capacity);
    reg_write64(CXL_GPU_REG_PARAM2, nodes ? 1 : 0);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_GET_NODES);
    if (result != CUDA_SUCCESS) {
        cmd_unlock();
        return result;
    }

    size_t count = reg_read64(CXL_GPU_REG_RESULT0);
    if (nodes) {
        if (count > capacity) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_VALUE;
        }
        uint64_t *node_ids = count ? malloc(count * sizeof(*node_ids)) : NULL;
        if (count && !node_ids) {
            cmd_unlock();
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (count)
            data_read(0, node_ids, count * sizeof(*node_ids));
        for (size_t i = 0; i < count; i++) {
            if (node_ids[i] > UINT32_MAX) {
                free(node_ids);
                cmd_unlock();
                return CUDA_ERROR_INVALID_HANDLE;
            }
            nodes[i] = (CUgraphNode)cxl_gpu_handle_from_id(node_ids[i]);
        }
        for (size_t i = count; i < capacity; i++)
            nodes[i] = NULL;
        free(node_ids);
    }
    *numNodes = count;
    cmd_unlock();
    return CUDA_SUCCESS;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
    uint64_t node_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!type)
        return CUDA_ERROR_INVALID_VALUE;
    if (!cxl_gpu_handle_id(hNode, &node_id))
        return CUDA_ERROR_INVALID_HANDLE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, node_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_NODE_GET_TYPE);
    if (result == CUDA_SUCCESS)
        *type = (CUgraphNodeType)(int32_t)reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return result;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);

static int uuid_equal(const CUuuid *uuid, const unsigned char bytes[16]) {
    return uuid && memcmp(uuid->bytes, bytes, 16) == 0;
}

static CUresult cudart_get_module_from_cubin(CUmodule *module, const void *fatbinc_wrapper) {
    fprintf(stderr, "[CXL-CUDA] CUDART_INTERFACE.get_module_from_cubin(module=%p, fatbin=%p) -> placeholder\n",
            (void *)module, fatbinc_wrapper);
    if (!module || !fatbinc_wrapper) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *module = (CUmodule)&g_cudart_placeholder_module;
    return CUDA_SUCCESS;
}

static CUresult cudart_get_primary_context(CUcontext *pctx, CUdevice dev) {
    DLOG("CUDART_INTERFACE.get_primary_context(dev=%d)\n", dev);
    return cuDevicePrimaryCtxRetain(pctx, dev);
}

static CUresult cudart_get_module_from_cubin_ext1(CUmodule *module, const void *fatbinc_wrapper, void *arg3, void *arg4,
                                                  uint32_t arg5) {
    fprintf(stderr,
            "[CXL-CUDA] CUDART_INTERFACE.get_module_from_cubin_ext1(module=%p, fatbin=%p, arg3=%p, arg4=%p, arg5=%u)\n",
            (void *)module, fatbinc_wrapper, arg3, arg4, arg5);
    if (arg3 || arg4) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return cudart_get_module_from_cubin(module, fatbinc_wrapper);
}

static CUresult cudart_interface_fn7(size_t arg1) {
    fprintf(stderr, "[CXL-CUDA] CUDART_INTERFACE.fn7(arg=%zu) -> CUDA_SUCCESS\n", arg1);
    return CUDA_SUCCESS;
}

static CUresult cudart_get_module_from_cubin_ext2(const void *fatbin_header, CUmodule *module, void *arg3, void *arg4,
                                                  uint32_t arg5) {
    fprintf(stderr,
            "[CXL-CUDA] CUDART_INTERFACE.get_module_from_cubin_ext2(fatbin=%p, module=%p, arg3=%p, arg4=%p, arg5=%u)\n",
            fatbin_header, (void *)module, arg3, arg4, arg5);
    if (arg5 != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!module || !fatbin_header) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *module = (CUmodule)&g_cudart_placeholder_module;
    return CUDA_SUCCESS;
}

static CUresult cudart_load_compilers(void) {
    DLOG("CUDART_INTERFACE.load_compilers -> CUDA_SUCCESS\n");
    return CUDA_SUCCESS;
}

typedef void (*context_storage_dtor_cb_t)(CUcontext context, void *key, void *value);

typedef struct {
    int in_use;
    CUcontext context;
    void *state_mgr;
    void *ctx_state;
    context_storage_dtor_cb_t dtor_cb;
} ContextStorageEntry;

#define CONTEXT_STORAGE_MAX_ENTRIES 128

static ContextStorageEntry g_context_storage[CONTEXT_STORAGE_MAX_ENTRIES];
static volatile int g_context_storage_lock = 0;
static unsigned int g_context_storage_put_count = 0;
static unsigned int g_context_storage_get_count = 0;

static void context_storage_lock(void) {
    while (__sync_lock_test_and_set(&g_context_storage_lock, 1)) {
    }
}

static void context_storage_unlock(void) { __sync_lock_release(&g_context_storage_lock); }

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
static void context_storage_test_reset(void) {
    context_storage_lock();
    memset(g_context_storage, 0, sizeof(g_context_storage));
    g_context_storage_put_count = 0;
    g_context_storage_get_count = 0;
    context_storage_unlock();
}
#endif

static uint64_t debug_hash_bytes(const void *ptr, size_t len) {
    const unsigned char *bytes = (const unsigned char *)ptr;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void debug_print_hex_span(const unsigned char *bytes, size_t start, size_t len) {
    fprintf(stderr, "[%zu..%zu]=", start, start + len - 1);
    for (size_t i = 0; i < len; i++) {
        fprintf(stderr, "%02x", bytes[start + i]);
    }
}

static void context_storage_log_bytes(const char *label, const void *ptr) {
    if (!g_debug) {
        return;
    }
    if (!ptr) {
        fprintf(stderr, "[CXL-CUDA] %s bytes=<null>\n", label);
        return;
    }
    const unsigned char *bytes = (const unsigned char *)ptr;
    uint64_t hash = debug_hash_bytes(ptr, 256);
    fprintf(stderr, "[CXL-CUDA] %s bytes256_hash=0x%016llx ", label, (unsigned long long)hash);
    debug_print_hex_span(bytes, 0, 64);
    fprintf(stderr, " ");
    debug_print_hex_span(bytes, 64, 64);
    fprintf(stderr, "\n");
    fprintf(stderr, "[CXL-CUDA] %s bytes256_tail ", label);
    debug_print_hex_span(bytes, 128, 64);
    fprintf(stderr, " ");
    debug_print_hex_span(bytes, 192, 64);
    fprintf(stderr, "\n");
}

static void context_storage_log_entries(const char *label) {
    ContextStorageEntry entries[CONTEXT_STORAGE_MAX_ENTRIES];
    int count = 0;

    context_storage_lock();
    for (int i = 0; i < CONTEXT_STORAGE_MAX_ENTRIES; i++) {
        if (!g_context_storage[i].in_use) {
            continue;
        }
        if (count < CONTEXT_STORAGE_MAX_ENTRIES) {
            entries[count++] = g_context_storage[i];
        }
    }
    context_storage_unlock();

    fprintf(stderr, "[CXL-CUDA] %s context_storage_entries=%d\n", label, count);
    for (int i = 0; i < count; i++) {
        fprintf(stderr, "[CXL-CUDA] %s entry[%d] context=%p state_mgr=%p ctx_state=%p dtor=%p\n", label, i,
                entries[i].context, entries[i].state_mgr, entries[i].ctx_state, (void *)entries[i].dtor_cb);
        context_storage_log_bytes(label, entries[i].ctx_state);
    }
}

static CUresult context_local_storage_put(CUcontext context, void *state_mgr, void *ctx_state,
                                          context_storage_dtor_cb_t dtor_cb) {
    unsigned int call_id = ++g_context_storage_put_count;
    DLOG("CONTEXT_LOCAL_STORAGE.ctor_like#%u(cu_ctx=%p state_mgr=%p ctx_state=%p dtor=%p)\n", call_id, context,
         state_mgr, ctx_state, (void *)dtor_cb);
    context_storage_log_bytes("CONTEXT_LOCAL_STORAGE.ctor_like state_mgr", state_mgr);
    context_storage_log_bytes("CONTEXT_LOCAL_STORAGE.ctor_like ctx_state", ctx_state);

    context_storage_lock();
    int free_slot = -1;
    for (int i = 0; i < CONTEXT_STORAGE_MAX_ENTRIES; i++) {
        if (g_context_storage[i].in_use) {
            if (g_context_storage[i].context == context && g_context_storage[i].state_mgr == state_mgr) {
                g_context_storage[i].state_mgr = state_mgr;
                g_context_storage[i].ctx_state = ctx_state;
                g_context_storage[i].dtor_cb = dtor_cb;
                context_storage_unlock();
                return CUDA_SUCCESS;
            }
        } else if (free_slot < 0) {
            free_slot = i;
        }
    }

    if (free_slot < 0) {
        context_storage_unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    g_context_storage[free_slot].in_use = 1;
    g_context_storage[free_slot].context = context;
    g_context_storage[free_slot].state_mgr = state_mgr;
    g_context_storage[free_slot].ctx_state = ctx_state;
    g_context_storage[free_slot].dtor_cb = dtor_cb;
    context_storage_unlock();
    return CUDA_SUCCESS;
}

static CUresult context_local_storage_delete(CUcontext context, void *state_mgr) {
    DLOG("CONTEXT_LOCAL_STORAGE.delete_like(cu_ctx=%p state_mgr=%p)\n", context, state_mgr);

    context_storage_lock();
    for (int i = 0; i < CONTEXT_STORAGE_MAX_ENTRIES; i++) {
        if (g_context_storage[i].in_use && g_context_storage[i].context == context &&
            g_context_storage[i].state_mgr == state_mgr) {
            memset(&g_context_storage[i], 0, sizeof(g_context_storage[i]));
            break;
        }
    }
    context_storage_unlock();

    return CUDA_ERROR_DEINITIALIZED;
}

static CUresult context_local_storage_get(void **ctx_state, CUcontext context, void *state_mgr) {
    unsigned int call_id = ++g_context_storage_get_count;
    DLOG("CONTEXT_LOCAL_STORAGE.get_state_like#%u(out=%p cu_ctx=%p state_mgr=%p)\n", call_id, (void *)ctx_state,
         context, state_mgr);

    if (!ctx_state) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ctx_state = NULL;

    context_storage_lock();
    for (int i = 0; i < CONTEXT_STORAGE_MAX_ENTRIES; i++) {
        if (g_context_storage[i].in_use && g_context_storage[i].context == context &&
            g_context_storage[i].state_mgr == state_mgr) {
            *ctx_state = g_context_storage[i].ctx_state;
            context_storage_unlock();
            DLOG("CONTEXT_LOCAL_STORAGE.get_state_like#%u -> ctx_state=%p\n", call_id, *ctx_state);
            context_storage_log_bytes("CONTEXT_LOCAL_STORAGE.get_state_like ctx_state", *ctx_state);
            return CUDA_SUCCESS;
        }
    }
    context_storage_unlock();

    uintptr_t current_context = 0;
    CUresult current_result = cxl_cuda_context_get_current(&current_context);
    if (current_result != CUDA_SUCCESS || current_context == 0) {
        DLOG("CONTEXT_LOCAL_STORAGE.get_state_like#%u -> CUDA_ERROR_INVALID_CONTEXT\n", call_id);
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    DLOG("CONTEXT_LOCAL_STORAGE.get_state_like#%u -> CUDA_ERROR_INVALID_HANDLE\n", call_id);
    return CUDA_ERROR_INVALID_HANDLE;
}

static void context_storage_clear_context(CUcontext context, int call_dtors) {
    ContextStorageEntry callbacks[CONTEXT_STORAGE_MAX_ENTRIES];
    int callback_count = 0;

    context_storage_lock();
    for (int i = 0; i < CONTEXT_STORAGE_MAX_ENTRIES; i++) {
        if (!g_context_storage[i].in_use)
            continue;
        if (context && g_context_storage[i].context != context)
            continue;

        if (call_dtors && g_context_storage[i].dtor_cb && callback_count < CONTEXT_STORAGE_MAX_ENTRIES) {
            callbacks[callback_count++] = g_context_storage[i];
        }
        memset(&g_context_storage[i], 0, sizeof(g_context_storage[i]));
    }
    context_storage_unlock();

    for (int i = 0; i < callback_count; i++) {
        DLOG("CONTEXT_LOCAL_STORAGE.dtor_callback(ctx=%p state_mgr=%p ctx_state=%p)\n", callbacks[i].context,
             callbacks[i].state_mgr, callbacks[i].ctx_state);
        callbacks[i].dtor_cb(callbacks[i].context, callbacks[i].state_mgr, callbacks[i].ctx_state);
    }
}

static CUresult context_check(CUcontext ctx_in, uint32_t *result1, const void **result2, uintptr_t arg4, uintptr_t arg5,
                              uintptr_t arg6) {
    DLOG("CONTEXT_CHECKS.context_check(ctx=%p result1=%p result2=%p arg4=%p arg5=0x%lx arg6=%p)\n", ctx_in,
         (void *)result1, (void *)result2, (void *)arg4, (unsigned long)arg5, (void *)arg6);
    if (result1) {
        *result1 = 0;
    }
    if (result2) {
        DLOG("CONTEXT_CHECKS.context_check -> result1=%u result2=%p\n", result1 ? *result1 : 0xffffffffU,
             result2 ? *result2 : NULL);
    }
    return CUDA_SUCCESS;
}

static uint32_t context_check_fn3(uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4, uintptr_t arg5,
                                  uintptr_t arg6) {
    DLOG("CONTEXT_CHECKS.check_fn3(arg1=%p arg2=%p arg3=%p arg4=%p arg5=%p arg6=%p) -> 0\n", (void *)arg1, (void *)arg2,
         (void *)arg3, (void *)arg4, (void *)arg5, (void *)arg6);
    return 0;
}

static CUresult context_checks_unknown_slot1(void) {
    DLOG("CONTEXT_CHECKS.slot1 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot4(void) {
    DLOG("CONTEXT_CHECKS.slot4 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot5(void) {
    DLOG("CONTEXT_CHECKS.slot5 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot6(void) {
    DLOG("CONTEXT_CHECKS.slot6 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot7(void) {
    DLOG("CONTEXT_CHECKS.slot7 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot8(void) {
    DLOG("CONTEXT_CHECKS.slot8 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot9(void) {
    DLOG("CONTEXT_CHECKS.slot9 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot10(void) {
    DLOG("CONTEXT_CHECKS.slot10 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot11(void) {
    DLOG("CONTEXT_CHECKS.slot11 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot12(void) {
    DLOG("CONTEXT_CHECKS.slot12 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot13(void) {
    DLOG("CONTEXT_CHECKS.slot13 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult context_checks_unknown_slot14(void) {
    DLOG("CONTEXT_CHECKS.slot14 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static const void *CUDART_INTERFACE_TABLE[13];
static const void *INTEGRITY_CHECK_TABLE[3];

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_check_table;
    const void *fn_address;
    uint64_t unix_seconds;
} IntegrityPass3Input;

typedef struct {
    CUuuid guid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} IntegrityDeviceHashInfo;

CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);
CUresult cuDeviceGetUuid(void *uuid, CUdevice dev);

static const uint8_t INTEGRITY_MIXING_TABLE[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13, 0x62, 0xa7, 0x05,
    0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca, 0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4,
    0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12, 0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0,
    0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a, 0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89,
    0x0b, 0x22, 0x5f, 0x21, 0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97,
    0x03, 0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6, 0x4f, 0xb8,
    0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1, 0x45, 0x9d, 0x70, 0x59, 0x64,
    0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02, 0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6,
    0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f, 0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9,
    0xce, 0xba, 0xc5, 0xea, 0x26, 0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81,
    0x4d, 0x52, 0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a, 0x78,
    0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39, 0xf2, 0xef, 0xb7, 0x0e,
    0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a, 0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f,
    0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

static void integrity_check_single_pass(uint8_t state[66], uint8_t input) {
    uint8_t cursor = state[0x40];
    state[cursor + 0x10] = input;
    uint8_t next_cursor = (uint8_t)((cursor + 1) & 0x0f);
    state[cursor + 0x20] = (uint8_t)(state[cursor] ^ input);
    uint8_t mixed = INTEGRITY_MIXING_TABLE[(uint8_t)(input ^ state[0x41])];
    uint8_t old = state[cursor + 0x30];
    state[cursor + 0x30] = (uint8_t)(mixed ^ old);
    state[0x41] = (uint8_t)(mixed ^ old);
    state[0x40] = next_cursor;

    if (next_cursor != 0) {
        return;
    }

    uint8_t temp = 0x29;
    uint8_t round = 0;
    for (;;) {
        temp = (uint8_t)(temp ^ state[0]);
        state[0] = temp;
        for (int i = 1; i < 0x30; i++) {
            temp = (uint8_t)(state[i] ^ INTEGRITY_MIXING_TABLE[temp]);
            state[i] = temp;
        }
        temp = (uint8_t)(temp + round);
        round = (uint8_t)(round + 1);
        if (round == 0x12) {
            break;
        }
        temp = INTEGRITY_MIXING_TABLE[temp];
    }
}

static void integrity_hash_pass(uint8_t state[66], const void *data, size_t len, uint8_t xor_mask) {
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        integrity_check_single_pass(state, (uint8_t)(bytes[i] ^ xor_mask));
    }
}

static void integrity_pass5(uint8_t state[66], uint64_t out[2]) {
    uint8_t temp = (uint8_t)(16 - state[0x40]);
    for (uint8_t i = 0; i < temp; i++) {
        integrity_check_single_pass(state, temp);
    }

    for (int i = 0x30; i < 0x40; i++) {
        integrity_check_single_pass(state, state[i]);
    }

    memcpy(&out[0], state, sizeof(uint64_t));
    memcpy(&out[1], state + sizeof(uint64_t), sizeof(uint64_t));
}

static CUresult integrity_device_hash_info(IntegrityDeviceHashInfo *info) {
    if (!info) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    memset(info, 0, sizeof(*info));
    CUresult err = cuDeviceGetUuid(&info->guid, 0);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    err = cuDeviceGetAttribute(&info->pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, 0);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    err = cuDeviceGetAttribute(&info->pci_device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, 0);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    return cuDeviceGetAttribute(&info->pci_domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, 0);
}

static int cxl_cuda_effective_driver_version(void) {
    return 12090; /* CUDA 12.9 */
}

static CUresult integrity_check(uint32_t version, uint64_t unix_seconds, uint64_t result[2]) {
    DLOG("INTEGRITY_CHECK.integrity_check(version=%u, unix_seconds=%llu)\n", version, (unsigned long long)unix_seconds);
    if (g_debug) {
        const void *caller = __builtin_return_address(0);
        Dl_info caller_info = {0};
        if (dladdr(caller, &caller_info) != 0) {
            uintptr_t base = (uintptr_t)caller_info.dli_fbase;
            DLOG("INTEGRITY_CHECK.caller address=%p file=%s base=%p offset=0x%" PRIxPTR "\n", caller,
                 caller_info.dli_fname ? caller_info.dli_fname : "<unknown>", caller_info.dli_fbase,
                 (uintptr_t)caller - base);
        } else {
            DLOG("INTEGRITY_CHECK.caller address=%p unresolved=1\n", caller);
        }
    }
    if (!result) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    switch (version % 10) {
    case 0:
        result[0] = 0x3341181c03cb675cULL;
        result[1] = 0x8ed383aa1f4cd1e8ULL;
        return CUDA_SUCCESS;
    case 1:
        result[0] = 0x1841181c03cb675cULL;
        result[1] = 0x8ed383aa1f4cd1e8ULL;
        return CUDA_SUCCESS;
    default:
        break;
    }

    static const uint8_t pass1_result[16] = {0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
                                             0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f};

    uint8_t state[66];
    memset(state, 0, sizeof(state));
    integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x36);

    uint32_t current_process = (uint32_t)getpid();
    uint32_t current_thread = (uint32_t)(uintptr_t)pthread_self();

    IntegrityPass3Input pass3 = {
        .driver_version = (uint32_t)cxl_cuda_effective_driver_version(),
        .version = version,
        .current_process = current_process,
        .current_thread = current_thread,
        .cudart_table = CUDART_INTERFACE_TABLE,
        .integrity_check_table = INTEGRITY_CHECK_TABLE,
        .fn_address = INTEGRITY_CHECK_TABLE[1],
        .unix_seconds = unix_seconds,
    };

    DLOG("INTEGRITY_CHECK.input driver_version=%u version=%u unix_seconds=%llu pid=%u tid=%u\n", pass3.driver_version,
         pass3.version, (unsigned long long)pass3.unix_seconds, pass3.current_process, pass3.current_thread);
    DLOG("INTEGRITY_CHECK.input tables cudart=%p cudart_size=%zu integrity=%p integrity_size=%zu fn=%p\n",
         (void *)pass3.cudart_table, sizeof(CUDART_INTERFACE_TABLE), (void *)pass3.integrity_check_table,
         sizeof(INTEGRITY_CHECK_TABLE), pass3.fn_address);
    integrity_hash_pass(state, &pass3, sizeof(pass3), 0);

    IntegrityDeviceHashInfo device_info;
    CUresult device_info_err = integrity_device_hash_info(&device_info);
    if (device_info_err != CUDA_SUCCESS) {
        return device_info_err;
    }
    DLOG("INTEGRITY_CHECK.input device_count=1 "
         "uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x pci=%d:%d:%d\n",
         device_info.guid.bytes[0], device_info.guid.bytes[1], device_info.guid.bytes[2], device_info.guid.bytes[3],
         device_info.guid.bytes[4], device_info.guid.bytes[5], device_info.guid.bytes[6], device_info.guid.bytes[7],
         device_info.guid.bytes[8], device_info.guid.bytes[9], device_info.guid.bytes[10], device_info.guid.bytes[11],
         device_info.guid.bytes[12], device_info.guid.bytes[13], device_info.guid.bytes[14], device_info.guid.bytes[15],
         device_info.pci_domain, device_info.pci_bus, device_info.pci_device);
    integrity_hash_pass(state, &device_info, sizeof(device_info), 0);

    uint64_t pass5_1[2];
    integrity_pass5(state, pass5_1);

    memset(state, 0, 16);
    memset(state + 0x30, 0, 18);
    integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x5c);
    integrity_hash_pass(state, pass5_1, sizeof(pass5_1), 0);
    integrity_pass5(state, result);

    DLOG("INTEGRITY_CHECK.integrity_check -> %016llx %016llx\n", (unsigned long long)result[0],
         (unsigned long long)result[1]);
    return CUDA_SUCCESS;
}

/* The real CUDA 12.9 INTEGRITY_CHECK export has a third, non-null slot.  Its
 * host implementation accepts a boolean and returns CUDA_SUCCESS.  Runtime
 * initialization probes the slot as part of the table ABI, so a NULL pointer
 * is an invalid table shape even though slot 1 owns the integrity digest. */
static CUresult integrity_check_set_enabled(int enabled) {
    DLOG("INTEGRITY_CHECK.set_enabled(enabled=%d) -> CUDA_SUCCESS\n", enabled != 0);
    return CUDA_SUCCESS;
}

static uint32_t TOOLS_RUNTIME_BUFFER1[1024];
static unsigned char TOOLS_RUNTIME_BUFFER2[14];

static void tools_get_buffer1(void **ptr, size_t *size) {
    DLOG("TOOLS_RUNTIME_CALLBACK_HOOKS.get_buffer1\n");
    if (ptr) {
        *ptr = TOOLS_RUNTIME_BUFFER1;
    }
    if (size) {
        *size = sizeof(TOOLS_RUNTIME_BUFFER1) / sizeof(TOOLS_RUNTIME_BUFFER1[0]);
    }
}

static void tools_get_buffer2(void **ptr, size_t *size) {
    DLOG("TOOLS_RUNTIME_CALLBACK_HOOKS.get_buffer2\n");
    if (ptr) {
        *ptr = TOOLS_RUNTIME_BUFFER2;
    }
    if (size) {
        *size = sizeof(TOOLS_RUNTIME_BUFFER2);
    }
}

static CUresult tools_runtime_callback_hook_slot4(const void *input, uint64_t *out) {
    DLOG("TOOLS_RUNTIME_CALLBACK_HOOKS.slot4(input=%p, out=%p)\n", input, (void *)out);
    if (!out) {
        return CUDA_ERROR_UNKNOWN;
    }
    // knockout: Kimi currently calls this slot with input=NULL; recover the non-NULL ABI from L40 evidence before use.
    if (input) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *out = 0;
    return CUDA_SUCCESS;
}

static CUresult tools_tls_get(void **out) {
    DLOG("TOOLS_TLS.get(out=%p)\n", (void *)out);
    if (!out) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *out = NULL;
    return CUDA_SUCCESS;
}

static const unsigned char CUDART_INTERFACE_UUID[16] = {0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
                                                        0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};

static const unsigned char TOOLS_RUNTIME_CALLBACK_HOOKS_UUID[16] = {0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
                                                                    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};

static const unsigned char TOOLS_TLS_UUID[16] = {0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
                                                 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc};

static const unsigned char CONTEXT_LOCAL_STORAGE_UUID[16] = {0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
                                                             0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};

static const unsigned char CONTEXT_CHECKS_UUID[16] = {0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
                                                      0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e};

static const unsigned char INTEGRITY_CHECK_UUID[16] = {0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
                                                       0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2};

static const unsigned char F8CFF951_EXPORT_UUID[16] = {0xf8, 0xcf, 0xf9, 0x51, 0x21, 0x46, 0x8b, 0x4e,
                                                       0xb9, 0xe2, 0xfb, 0x46, 0x9e, 0x7c, 0x0d, 0xd9};

static CUresult f8cff951_export_fn1(void) {
    DLOG("F8CFF951.fn1 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn2(void) {
    DLOG("F8CFF951.fn2 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn5(void) {
    DLOG("F8CFF951.fn5 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn7(void) {
    DLOG("F8CFF951.fn7 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn8(void) {
    DLOG("F8CFF951.fn8 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn9(void) {
    DLOG("F8CFF951.fn9 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn11(void) {
    DLOG("F8CFF951.fn11 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult f8cff951_export_fn15(void) {
    DLOG("F8CFF951.fn15 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cudart_interface_unknown_slot3(void) {
    DLOG("CUDART_INTERFACE.slot3 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cudart_interface_unknown_slot4(void) {
    DLOG("CUDART_INTERFACE.slot4 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cudart_interface_unknown_slot5(void) {
    DLOG("CUDART_INTERFACE.slot5 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cudart_interface_unknown_slot9(void) {
    DLOG("CUDART_INTERFACE.slot9 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult cudart_interface_unknown_slot11(void) {
    DLOG("CUDART_INTERFACE.slot11 -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* Diagnostic implementation of the host-observed f8cff951 internal export table.
 * Host NVIDIA 580.142 exposes 16 slots with size sentinels at 0,4,6,10,14.
 * The callback semantics are still unknown, so callbacks return NOT_SUPPORTED if
 * invoked.  This tests whether table presence alone gates CUDA 12.9 post-fatbin
 * context/memory/sync dispatch. */
static const void *F8CFF951_EXPORT_TABLE[16] = {
    (const void *)(uintptr_t)0x18,
    (const void *)f8cff951_export_fn1,
    (const void *)f8cff951_export_fn2,
    NULL,
    (const void *)(uintptr_t)0x10,
    (const void *)f8cff951_export_fn5,
    (const void *)(uintptr_t)0x20,
    (const void *)f8cff951_export_fn7,
    (const void *)f8cff951_export_fn8,
    (const void *)f8cff951_export_fn9,
    (const void *)(uintptr_t)0x10,
    (const void *)f8cff951_export_fn11,
    NULL,
    NULL,
    (const void *)(uintptr_t)0x38,
    (const void *)f8cff951_export_fn15,
};

static const void *CUDART_INTERFACE_TABLE[13] = {
    (const void *)(uintptr_t)(sizeof(CUDART_INTERFACE_TABLE)),
    (const void *)cudart_get_module_from_cubin,
    (const void *)cudart_get_primary_context,
    (const void *)cudart_interface_unknown_slot3,
    (const void *)cudart_interface_unknown_slot4,
    (const void *)cudart_interface_unknown_slot5,
    (const void *)cudart_get_module_from_cubin_ext1,
    (const void *)cudart_interface_fn7,
    (const void *)cudart_get_module_from_cubin_ext2,
    (const void *)cudart_interface_unknown_slot9,
    NULL,
    (const void *)cudart_interface_unknown_slot11,
    (const void *)cudart_load_compilers,
};

static const void *TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE[7] = {
    (const void *)(uintptr_t)(sizeof(TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE)),
    NULL,
    (const void *)tools_get_buffer1,
    NULL,
    (const void *)tools_runtime_callback_hook_slot4,
    NULL,
    (const void *)tools_get_buffer2,
};

static const void *TOOLS_TLS_TABLE[3] = {
    (const void *)(uintptr_t)(sizeof(TOOLS_TLS_TABLE)),
    NULL,
    (const void *)tools_tls_get,
};

static const void *CONTEXT_LOCAL_STORAGE_TABLE[4] = {
    (const void *)context_local_storage_put,
    (const void *)context_local_storage_delete,
    (const void *)context_local_storage_get,
    NULL,
};

static const void *CONTEXT_CHECKS_TABLE[15] = {
    (const void *)(uintptr_t)(sizeof(CONTEXT_CHECKS_TABLE)),
    (const void *)context_checks_unknown_slot1,
    (const void *)context_check,
    (const void *)context_check_fn3,
    (const void *)context_checks_unknown_slot4,
    (const void *)context_checks_unknown_slot5,
    (const void *)context_checks_unknown_slot6,
    (const void *)context_checks_unknown_slot7,
    (const void *)context_checks_unknown_slot8,
    (const void *)context_checks_unknown_slot9,
    (const void *)context_checks_unknown_slot10,
    (const void *)context_checks_unknown_slot11,
    (const void *)context_checks_unknown_slot12,
    (const void *)context_checks_unknown_slot13,
    (const void *)context_checks_unknown_slot14,
};

static const void *INTEGRITY_CHECK_TABLE[3] = {
    (const void *)(uintptr_t)(sizeof(INTEGRITY_CHECK_TABLE)),
    (const void *)integrity_check,
    (const void *)integrity_check_set_enabled,
};

CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    g_debug = (getenv("CXL_CUDA_DEBUG") != NULL);

    if (pExportTableId) {
        DLOG("cuGetExportTable(uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
             pExportTableId->bytes[0], pExportTableId->bytes[1], pExportTableId->bytes[2], pExportTableId->bytes[3],
             pExportTableId->bytes[4], pExportTableId->bytes[5], pExportTableId->bytes[6], pExportTableId->bytes[7],
             pExportTableId->bytes[8], pExportTableId->bytes[9], pExportTableId->bytes[10], pExportTableId->bytes[11],
             pExportTableId->bytes[12], pExportTableId->bytes[13], pExportTableId->bytes[14],
             pExportTableId->bytes[15]);
    } else {
        DLOG("cuGetExportTable(uuid=(null))\n");
    }

    if (!ppExportTable || !pExportTableId) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (uuid_equal(pExportTableId, F8CFF951_EXPORT_UUID)) {
        *ppExportTable = F8CFF951_EXPORT_TABLE;
        DLOG("cuGetExportTable -> F8CFF951_EXPORT_TABLE size=%zu\n", sizeof(F8CFF951_EXPORT_TABLE));
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, CUDART_INTERFACE_UUID)) {
        *ppExportTable = CUDART_INTERFACE_TABLE;
        DLOG("cuGetExportTable -> CUDART_INTERFACE_TABLE size=%zu\n", sizeof(CUDART_INTERFACE_TABLE));
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, TOOLS_RUNTIME_CALLBACK_HOOKS_UUID)) {
        *ppExportTable = TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE;
        DLOG("cuGetExportTable -> TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE size=%zu\n",
             sizeof(TOOLS_RUNTIME_CALLBACK_HOOKS_TABLE));
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, TOOLS_TLS_UUID)) {
        *ppExportTable = TOOLS_TLS_TABLE;
        DLOG("cuGetExportTable -> TOOLS_TLS_TABLE size=%zu\n", sizeof(TOOLS_TLS_TABLE));
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, CONTEXT_LOCAL_STORAGE_UUID)) {
        *ppExportTable = CONTEXT_LOCAL_STORAGE_TABLE;
        DLOG("cuGetExportTable -> CONTEXT_LOCAL_STORAGE_TABLE\n");
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, CONTEXT_CHECKS_UUID)) {
        *ppExportTable = CONTEXT_CHECKS_TABLE;
        DLOG("cuGetExportTable -> CONTEXT_CHECKS_TABLE size=%zu\n", sizeof(CONTEXT_CHECKS_TABLE));
        return CUDA_SUCCESS;
    }

    if (uuid_equal(pExportTableId, INTEGRITY_CHECK_UUID)) {
        *ppExportTable = INTEGRITY_CHECK_TABLE;
        DLOG("cuGetExportTable -> INTEGRITY_CHECK_TABLE size=%zu\n", sizeof(INTEGRITY_CHECK_TABLE));
        return CUDA_SUCCESS;
    }

    *ppExportTable = NULL;
    DLOG("cuGetExportTable -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ========================================================================
 * CUDA Driver API Implementation
 * ======================================================================== */

CUresult cuInit(unsigned int flags) {
    (void)flags;

    g_debug = (getenv("CXL_CUDA_DEBUG") != NULL);
    DLOG("cuInit(%u)\n", flags);

    if (g_initialized) {
        return CUDA_SUCCESS;
    }

    if (find_and_map_device() < 0) {
        return CUDA_ERROR_NO_DEVICE;
    }

    /* Check device status */
    uint32_t status = reg_read32(CXL_GPU_REG_STATUS);
    if (!(status & CXL_GPU_STATUS_READY)) {
        DLOG("Device not ready, status=0x%x\n", status);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    g_initialized = 1;
    DLOG("Initialization complete\n");
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *version) {
    DLOG("cuDriverGetVersion\n");
    if (!version)
        return CUDA_ERROR_INVALID_VALUE;
    /* The guest payload currently uses CUDA Runtime 12.9.  libcudart refuses to
     * initialize when the Driver API reports an older version, before it calls
     * into cuDeviceGetCount. */
    *version = cxl_cuda_effective_driver_version();
    DLOG("  version=%d\n", *version);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr) {
    CXLCudaErrorName *entry;
    CUresult result;
    uint64_t length;
    char *name;

    DLOG("cuGetErrorName(error=%d)\n", error);
    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    *pStr = NULL;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    pthread_mutex_lock(&g_cuda_error_names_lock);
    for (entry = g_cuda_error_names; entry; entry = entry->next) {
        if (entry->error == error) {
            *pStr = entry->name;
            pthread_mutex_unlock(&g_cuda_error_names_lock);
            return CUDA_SUCCESS;
        }
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(int64_t)error);
    result = execute_cmd(CXL_GPU_CMD_GET_ERROR_NAME);
    length = reg_read64(CXL_GPU_REG_RESULT0);
    if (result != CUDA_SUCCESS) {
        cmd_unlock();
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return result;
    }
    if (length >= CXL_GPU_DATA_SIZE) {
        cmd_unlock();
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }

    name = malloc((size_t)length + 1);
    entry = malloc(sizeof(*entry));
    if (!name || !entry) {
        free(name);
        free(entry);
        cmd_unlock();
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    data_read(0, name, (size_t)length + 1);
    cmd_unlock();
    if (name[length] != '\0') {
        free(name);
        free(entry);
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }

    entry->error = error;
    entry->name = name;
    entry->next = g_cuda_error_names;
    g_cuda_error_names = entry;
    *pStr = name;
    DLOG("  name=%s\n", name);
    pthread_mutex_unlock(&g_cuda_error_names_lock);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **pStr) {
    CXLCudaErrorName *entry;
    CUresult result;
    uint64_t length;
    char *string;

    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    *pStr = NULL;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    pthread_mutex_lock(&g_cuda_error_names_lock);
    for (entry = g_cuda_error_strings; entry; entry = entry->next) {
        if (entry->error == error) {
            *pStr = entry->name;
            pthread_mutex_unlock(&g_cuda_error_names_lock);
            return CUDA_SUCCESS;
        }
    }
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(int64_t)error);
    result = execute_cmd(CXL_GPU_CMD_GET_ERROR_STRING);
    length = reg_read64(CXL_GPU_REG_RESULT0);
    if (result != CUDA_SUCCESS || length >= CXL_GPU_DATA_SIZE) {
        cmd_unlock();
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return result != CUDA_SUCCESS ? result : CUDA_ERROR_INVALID_VALUE;
    }
    string = malloc((size_t)length + 1);
    entry = malloc(sizeof(*entry));
    if (!string || !entry) {
        free(string);
        free(entry);
        cmd_unlock();
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    data_read(0, string, (size_t)length + 1);
    cmd_unlock();
    if (string[length] != '\0') {
        free(string);
        free(entry);
        pthread_mutex_unlock(&g_cuda_error_names_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    entry->error = error;
    entry->name = string;
    entry->next = g_cuda_error_strings;
    g_cuda_error_strings = entry;
    *pStr = string;
    pthread_mutex_unlock(&g_cuda_error_names_lock);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    DLOG("cuDeviceGetCount\n");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!count)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_GET_DEVICE_COUNT);
    if (err == CUDA_SUCCESS) {
        *count = reg_read64(CXL_GPU_REG_RESULT0);
        DLOG("  count=%d\n", *count);
    }
    cmd_unlock();
    return err;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    DLOG("cuDeviceGet(ordinal=%d)\n", ordinal);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    if (ordinal != 0)
        return CUDA_ERROR_INVALID_DEVICE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, ordinal);
    CUresult err = execute_cmd(CXL_GPU_CMD_GET_DEVICE);
    if (err == CUDA_SUCCESS) {
        *device = reg_read64(CXL_GPU_REG_RESULT0);
    }
    cmd_unlock();
    return err;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    DLOG("cuDeviceGetName(dev=%d)\n", dev);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!name || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;

    /* Read device name from registers */
    size_t to_read = (len < 64) ? len : 64;
    for (size_t i = 0; i < to_read; i += 8) {
        uint64_t val = reg_read64(CXL_GPU_REG_DEV_NAME + i);
        memcpy(name + i, &val, (to_read - i < 8) ? to_read - i : 8);
    }
    name[len - 1] = '\0';
    DLOG("  name=%s\n", name);
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    DLOG("cuDeviceTotalMem(dev=%d)\n", dev);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!bytes)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;

    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_GET_TOTAL_MEM);
    if (err == CUDA_SUCCESS) {
        *bytes = reg_read64(CXL_GPU_REG_RESULT0);
        DLOG("  bytes=%zu\n", *bytes);
    }
    cmd_unlock();
    return err;
}

CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev) {
    DLOG("cuDeviceGetAttribute(attrib=%d, dev=%d)\n", attrib, dev);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(int64_t)(int32_t)attrib);
    CUresult err = execute_cmd(CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE);
    if (err == CUDA_SUCCESS) {
        *value = (int)(int32_t)reg_read64(CXL_GPU_REG_RESULT0);
        DLOG("  value=%d\n", *value);
    }
    cmd_unlock();
    return err;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    DLOG("cuModuleGetLoadingMode(mode=%p)\n", (void *)mode);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!mode)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_MODULE_GET_LOADING_MODE);
    if (err == CUDA_SUCCESS) {
        *mode = (CUmoduleLoadingMode)(uint32_t)reg_read64(CXL_GPU_REG_RESULT0);
        DLOG("  mode=%d\n", (int)*mode);
    }
    cmd_unlock();
    return err;
}

CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev) {
    DLOG("cuCtxCreate(flags=%u, dev=%d)\n", flags, dev);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!ctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    if (flags != 0)
        return CUDA_ERROR_NOT_SUPPORTED;

    CUresult state_err = cxl_cuda_context_prepare_ordinary_create();
    if (state_err != CUDA_SUCCESS)
        return state_err;

    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_CTX_CREATE);
    if (err == CUDA_SUCCESS) {
        *ctx = (CUcontext)(uintptr_t)reg_read64(CXL_GPU_REG_RESULT0);
        err = cxl_cuda_context_commit_ordinary_create((uintptr_t)*ctx);
        if (err == CUDA_SUCCESS)
            DLOG("  ctx=%p\n", *ctx);
    }
    cmd_unlock();
    log_context_state("cuCtxCreate_v2", err);
    return err;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    DLOG("cuCtxDestroy(ctx=%p)\n", ctx);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    CUresult state_err = cxl_cuda_context_prepare_destroy((uintptr_t)ctx);
    if (state_err != CUDA_SUCCESS)
        return state_err;

    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_CTX_DESTROY);
    if (err == CUDA_SUCCESS) {
        context_storage_clear_context(ctx, 1);
        err = cxl_cuda_context_commit_destroy((uintptr_t)ctx);
    }
    cmd_unlock();
    return err;
}

CUresult cuCtxSynchronize(void) {
    DLOG("cuCtxSynchronize\n");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_CTX_SYNC);
    cmd_unlock();
    return err;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    DLOG("cuMemAlloc(size=%zu)\n", bytesize);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dptr)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, bytesize);
    CUresult err = execute_cmd(CXL_GPU_CMD_MEM_ALLOC);
    if (err == CUDA_SUCCESS) {
        *dptr = reg_read64(CXL_GPU_REG_RESULT0);
        DLOG("  dptr=0x%lx\n", (unsigned long)*dptr);
    }
    cmd_unlock();
    return err;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    DLOG("cuMemFree(dptr=0x%lx)\n", (unsigned long)dptr);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, dptr);
    CUresult err = execute_cmd(CXL_GPU_CMD_MEM_FREE);
    cmd_unlock();
    return err;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount) {
    DLOG("cuMemcpyHtoD(dst=0x%lx, size=%zu)\n", (unsigned long)dstDevice, byteCount);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!srcHost)
        return CUDA_ERROR_INVALID_VALUE;

    /* Transfer in chunks that fit in data buffer */
    size_t offset = 0;
    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > CXL_GPU_DATA_SIZE) {
            chunk = CXL_GPU_DATA_SIZE;
        }

        cmd_lock();
        data_write(0, (const uint8_t *)srcHost + offset, chunk);
        reg_write64(CXL_GPU_REG_PARAM0, dstDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, chunk);

        CUresult err = execute_cmd(CXL_GPU_CMD_MEM_COPY_HTOD);
        cmd_unlock();
        if (err != CUDA_SUCCESS) {
            return err;
        }
        offset += chunk;
    }

    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount, CUstream hStream) {
    DLOG("cuMemcpyHtoDAsync(dst=0x%lx, size=%zu, stream=%p)\n", (unsigned long)dstDevice, byteCount, hStream);
    (void)hStream;
    /* knockout: the current Type-2 command path serializes transfers and kernel
     * launches. Completing the copy before return preserves correctness; add a
     * stream-aware BAR2 command only when concurrent stream execution is measured. */
    return cuMemcpyHtoD_v2(dstDevice, srcHost, byteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, hStream);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount) {
    DLOG("cuMemcpyDtoH(src=0x%lx, size=%zu)\n", (unsigned long)srcDevice, byteCount);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dstHost)
        return CUDA_ERROR_INVALID_VALUE;

    /* Transfer in chunks */
    size_t offset = 0;
    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > CXL_GPU_DATA_SIZE) {
            chunk = CXL_GPU_DATA_SIZE;
        }

        cmd_lock();
        reg_write64(CXL_GPU_REG_PARAM0, srcDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, chunk);

        CUresult err = execute_cmd(CXL_GPU_CMD_MEM_COPY_DTOH);
        if (err != CUDA_SUCCESS) {
            cmd_unlock();
            return err;
        }

        data_read(0, (uint8_t *)dstHost + offset, chunk);
        cmd_unlock();
        offset += chunk;
    }

    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    DLOG("cuMemcpyDtoHAsync(src=0x%lx, size=%zu, stream=%p)\n", (unsigned long)srcDevice, byteCount, hStream);
    (void)hStream;
    /* knockout: the current Type-2 command path serializes transfers and kernel
     * launches. Completing the copy before return preserves correctness; add a
     * stream-aware BAR2 command only when concurrent stream execution is measured. */
    return cuMemcpyDtoH_v2(dstHost, srcDevice, byteCount);
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, byteCount, hStream);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t byteCount);

static CUresult cxl_cuda_pointer_is_device(CUdeviceptr ptr, bool *is_device) {
    if (!is_device)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, ptr);
    CUresult result = execute_cmd(CXL_GPU_CMD_MEM_GET_POINTER_MEMORY_TYPE);
    if (result == CUDA_SUCCESS) {
        int memory_type = (int)reg_read64(CXL_GPU_REG_RESULT0);
        *is_device = memory_type == 2 || memory_type == 4;
    }
    cmd_unlock();

    if (result == CUDA_ERROR_INVALID_VALUE) {
        *is_device = false;
        return CUDA_SUCCESS;
    }
    return result;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t byteCount) {
    DLOG("cuMemcpy(dst=0x%lx, src=0x%lx, size=%zu)\n",
         (unsigned long)dst, (unsigned long)src, byteCount);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dst || !src)
        return CUDA_ERROR_INVALID_VALUE;

    bool dst_is_device = false;
    bool src_is_device = false;
    CUresult result = cxl_cuda_pointer_is_device(dst, &dst_is_device);
    if (result != CUDA_SUCCESS)
        return result;
    result = cxl_cuda_pointer_is_device(src, &src_is_device);
    if (result != CUDA_SUCCESS)
        return result;

    if (dst_is_device && src_is_device)
        return cuMemcpyDtoD_v2(dst, src, byteCount);
    if (dst_is_device)
        return cuMemcpyHtoD_v2(dst, (const void *)(uintptr_t)src, byteCount);
    if (src_is_device)
        return cuMemcpyDtoH_v2((void *)(uintptr_t)dst, src, byteCount);

    memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, byteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount,
                       CUstream hStream) {
    DLOG("cuMemcpyAsync(dst=0x%lx, src=0x%lx, size=%zu, stream=%p)\n",
         (unsigned long)dst, (unsigned long)src, byteCount, hStream);
    (void)hStream;
    /* knockout: the current Type-2 command path serializes copies and kernel
     * launches. Completing the UVA-directed copy before return preserves
     * correctness; add stream ordering only when concurrent execution exists. */
    return cuMemcpy(dst, src, byteCount);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    DLOG("cuModuleLoadData\n");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!module || !image)
        return CUDA_ERROR_INVALID_VALUE;

    /* Copy PTX to data buffer */
    size_t len = strlen((const char *)image) + 1;
    if (len > CXL_GPU_DATA_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    data_write(0, image, len);
    CUresult err = execute_cmd(CXL_GPU_CMD_MODULE_LOAD_PTX);
    if (err == CUDA_SUCCESS) {
        *module = (CUmodule)cxl_gpu_handle_from_id(reg_read64(CXL_GPU_REG_RESULT0));
        DLOG("  module=%p\n", *module);
    }
    return err;
}

static CUresult cxl_module_load_cubin(CUmodule *module, const void *image, size_t image_size, uint32_t encoding,
                                      size_t uncompressed_size) {
    DLOG("cxl_module_load_cubin(size=%zu, encoding=0x%x, uncompressed=%zu)\n", image_size, encoding, uncompressed_size);
    if (!g_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!module || !image || !image_size || image_size > CXL_GPU_DATA_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    data_write(0, image, image_size);
    reg_write64(CXL_GPU_REG_PARAM0, image_size);
    reg_write64(CXL_GPU_REG_PARAM1, encoding);
    reg_write64(CXL_GPU_REG_PARAM2, uncompressed_size);
    CUresult err = execute_cmd(CXL_GPU_CMD_MODULE_LOAD_CUBIN);
    if (err == CUDA_SUCCESS) {
        *module = (CUmodule)cxl_gpu_handle_from_id(reg_read64(CXL_GPU_REG_RESULT0));
        DLOG("  cubin module=%p\n", *module);
    }
    return err;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, void *options,
                            void **optionValues) {
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    (void)module;
    (void)fatCubin;
    DLOG("cuModuleLoadFatBinary -> CUDA_ERROR_NOT_SUPPORTED\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void cudart_log_fatbin_file_headers(const CudartFatbinHeader *header) {
    if (!header) {
        return;
    }
    if (header->magic != CUDART_FATBIN_MAGIC || header->version != CUDART_FATBIN_VERSION) {
        fprintf(stderr, "[CXL-CUDA]   fatbin files skipped: unexpected magic/version\n");
        return;
    }
    if (header->header_size < sizeof(CudartFatbinHeader) || header->header_size > 4096 ||
        header->files_size > (256ULL * 1024ULL * 1024ULL)) {
        fprintf(stderr, "[CXL-CUDA]   fatbin files skipped: unreasonable header_size=%u files_size=%llu\n",
                header->header_size, (unsigned long long)header->files_size);
        return;
    }

    const unsigned char *files = ((const unsigned char *)header) + header->header_size;
    uint64_t offset = 0;
    for (unsigned int i = 0; i < 4 && offset + sizeof(CudartFatbinFileHeader) <= header->files_size; i++) {
        const CudartFatbinFileHeader *file = (const CudartFatbinFileHeader *)(const void *)(files + offset);
        fprintf(stderr,
                "[CXL-CUDA]   fatbin_file[%u] offset=%llu kind=%u version=0x%x header_size=%u "
                "payload_size=%u compressed_size=%u flags=0x%llx sm_version=0x%x bit_width=%u "
                "uncompressed_payload=%llu\n",
                i, (unsigned long long)offset, file->kind, file->version, file->header_size, file->payload_size,
                file->compressed_size, (unsigned long long)file->flags, file->sm_version, file->bit_width,
                (unsigned long long)file->uncompressed_payload);

        if (file->header_size < sizeof(CudartFatbinFileHeader) || file->header_size > 4096) {
            fprintf(stderr, "[CXL-CUDA]   fatbin_file[%u] stop: unreasonable header_size=%u\n", i, file->header_size);
            break;
        }
        uint64_t step = (uint64_t)file->header_size + (uint64_t)file->payload_size;
        if (step == 0 || step > header->files_size - offset) {
            fprintf(stderr, "[CXL-CUDA]   fatbin_file[%u] stop: step=%llu remaining=%llu\n", i,
                    (unsigned long long)step, (unsigned long long)(header->files_size - offset));
            break;
        }
        offset += step;
    }
}

static void cudart_log_fatbin_headers(const void *code) {
    if (!code) {
        fprintf(stderr, "[CXL-CUDA]   fatbinc_wrapper=<null>\n");
        return;
    }

    const CudartFatbincWrapper *wrapper = (const CudartFatbincWrapper *)code;
    fprintf(stderr, "[CXL-CUDA]   fatbinc_wrapper magic=0x%08x version=0x%08x data=%p filename_or_fatbins=%p\n",
            wrapper->magic, wrapper->version, wrapper->data, wrapper->filename_or_fatbins);

    const CudartFatbinHeader *header = NULL;
    if (wrapper->magic == CUDART_FATBINC_MAGIC && wrapper->version == CUDART_FATBINC_VERSION && wrapper->data) {
        header = (const CudartFatbinHeader *)wrapper->data;
        fprintf(stderr, "[CXL-CUDA]   fatbinc_wrapper recognized; using wrapper->data as fatbin header\n");
    } else {
        header = (const CudartFatbinHeader *)code;
        fprintf(stderr,
                "[CXL-CUDA]   fatbinc_wrapper not recognized; also interpreting code as direct fatbin header\n");
    }

    fprintf(stderr, "[CXL-CUDA]   fatbin_header magic=0x%08x version=0x%04x header_size=%u files_size=%llu\n",
            header->magic, header->version, header->header_size, (unsigned long long)header->files_size);
    cudart_log_fatbin_file_headers(header);
}

static CUresult cudart_decode_fatbin_file(const CudartFatbinFileHeader *file, unsigned char **decoded_out,
                                          size_t *decoded_size_out) {
    if (!file || !decoded_out || !decoded_size_out) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *decoded_out = NULL;
    *decoded_size_out = 0;

    const unsigned char *payload = (const unsigned char *)file + file->header_size;
    bool lz4_compressed = (file->flags & CUDART_FATBIN_FLAG_COMPRESSED_LZ4) != 0;
    bool zstd_compressed = (file->flags & CUDART_FATBIN_FLAG_COMPRESSED_ZSTD) != 0;
    if (lz4_compressed && zstd_compressed) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (!lz4_compressed && !zstd_compressed) {
        if (!file->payload_size || file->payload_size > CXL_GPU_DATA_SIZE) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        unsigned char *decoded = (unsigned char *)malloc((size_t)file->payload_size + 1);
        if (!decoded) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(decoded, payload, file->payload_size);
        decoded[file->payload_size] = 0;
        *decoded_out = decoded;
        *decoded_size_out = file->payload_size;
        return CUDA_SUCCESS;
    }

    if (!file->compressed_size || file->compressed_size > file->payload_size || !file->uncompressed_payload ||
        file->uncompressed_payload > CXL_GPU_DATA_SIZE) {
        fprintf(stderr,
                "[CXL-CUDA]   library module decode reject: compressed=%u payload=%u uncompressed=%llu "
                "data_cap=%u\n",
                file->compressed_size, file->payload_size, (unsigned long long)file->uncompressed_payload,
                CXL_GPU_DATA_SIZE);
        return CUDA_ERROR_INVALID_VALUE;
    }

    size_t capacity = (size_t)file->uncompressed_payload;
    unsigned char *decoded = (unsigned char *)malloc(capacity + 1);
    if (!decoded) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    size_t decoded_size = 0;
    if (lz4_compressed) {
        int result =
            LZ4_decompress_safe((const char *)payload, (char *)decoded, (int)file->compressed_size, (int)capacity);
        if (result <= 0 || (size_t)result > capacity) {
            fprintf(stderr, "[CXL-CUDA]   library module decode reject: LZ4 result=%d\n", result);
            free(decoded);
            return CUDA_ERROR_INVALID_VALUE;
        }
        decoded_size = (size_t)result;
    } else {
        size_t result = ZSTD_decompress(decoded, capacity, payload, file->compressed_size);
        unsigned int failed = ZSTD_isError(result);
        if (failed || result > capacity) {
            fprintf(stderr, "[CXL-CUDA]   library module decode reject: Zstd result=%zu failed=%u\n", result, failed);
            free(decoded);
            return CUDA_ERROR_INVALID_VALUE;
        }
        decoded_size = result;
    }

    decoded[decoded_size] = 0;
    *decoded_out = decoded;
    *decoded_size_out = decoded_size;
    return CUDA_SUCCESS;
}

static CUresult cudart_load_module_from_fatbin(const void *code, CUmodule *module) {
    if (!code || !module) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *module = NULL;

    const CudartFatbincWrapper *wrapper = (const CudartFatbincWrapper *)code;
    const CudartFatbinHeader *header = NULL;
    if (wrapper->magic == CUDART_FATBINC_MAGIC && wrapper->version == CUDART_FATBINC_VERSION && wrapper->data) {
        header = (const CudartFatbinHeader *)wrapper->data;
    } else {
        header = (const CudartFatbinHeader *)code;
    }

    if (header->magic != CUDART_FATBIN_MAGIC || header->version != CUDART_FATBIN_VERSION ||
        header->header_size < sizeof(*header) || header->header_size > 4096 ||
        header->files_size > (256ULL * 1024ULL * 1024ULL)) {
        fprintf(stderr, "[CXL-CUDA]   library module load reject: invalid fatbin header\n");
        return CUDA_ERROR_INVALID_VALUE;
    }

    const unsigned char *files = (const unsigned char *)header + header->header_size;
    const CudartFatbinFileHeader *ptx_candidate = NULL;
    uint64_t ptx_offset = 0;
    const CudartFatbinFileHeader *elf_candidate = NULL;
    uint64_t elf_offset = 0;
    uint32_t target_sm = reg_read32(CXL_GPU_REG_CC_MAJOR) * 10U + reg_read32(CXL_GPU_REG_CC_MINOR);
    uint64_t offset = 0;
    while (offset + sizeof(CudartFatbinFileHeader) <= header->files_size) {
        const CudartFatbinFileHeader *file = (const CudartFatbinFileHeader *)(const void *)(files + offset);
        if (file->header_size < sizeof(*file) || file->header_size > 4096 ||
            file->header_size > header->files_size - offset ||
            file->payload_size > header->files_size - offset - file->header_size) {
            fprintf(stderr, "[CXL-CUDA]   library module load reject: malformed file at offset=%llu\n",
                    (unsigned long long)offset);
            return CUDA_ERROR_INVALID_VALUE;
        }

        if (file->kind == CUDART_FATBIN_KIND_PTX && file->sm_version <= target_sm &&
            (!ptx_candidate || file->sm_version > ptx_candidate->sm_version)) {
            ptx_candidate = file;
            ptx_offset = offset;
        }
        if (file->kind == CUDART_FATBIN_KIND_ELF && file->sm_version <= target_sm &&
            file->sm_version / 10U == target_sm / 10U &&
            (!elf_candidate || file->sm_version > elf_candidate->sm_version)) {
            elf_candidate = file;
            elf_offset = offset;
        }

        uint64_t step = (uint64_t)file->header_size + (uint64_t)file->payload_size;
        if (!step || step > header->files_size - offset) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        offset += step;
    }

    if (elf_candidate) {
        const unsigned char *payload = (const unsigned char *)elf_candidate + elf_candidate->header_size;
        bool zstd_compressed = (elf_candidate->flags & CUDART_FATBIN_FLAG_COMPRESSED_ZSTD) != 0;
        bool lz4_compressed = (elf_candidate->flags & CUDART_FATBIN_FLAG_COMPRESSED_LZ4) != 0;
        if (zstd_compressed && lz4_compressed) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (zstd_compressed || lz4_compressed) {
            if (!elf_candidate->compressed_size || elf_candidate->compressed_size > elf_candidate->payload_size ||
                elf_candidate->compressed_size > CXL_GPU_DATA_SIZE || !elf_candidate->uncompressed_payload) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            uint32_t encoding =
                zstd_compressed ? CXL_GPU_MODULE_DATA_ZSTD : CXL_GPU_MODULE_DATA_LZ4;
            const char *encoding_name = zstd_compressed ? "zstd" : "lz4";
            fprintf(stderr,
                    "[CXL-CUDA]   library CUBIN load selected offset=%llu sm=0x%x compressed_size=%u "
                    "uncompressed_size=%llu encoding=%s\n",
                    (unsigned long long)elf_offset, elf_candidate->sm_version, elf_candidate->compressed_size,
                    (unsigned long long)elf_candidate->uncompressed_payload, encoding_name);
            return cxl_module_load_cubin(module, payload, elf_candidate->compressed_size, encoding,
                                         (size_t)elf_candidate->uncompressed_payload);
        }

        unsigned char *cubin = NULL;
        size_t cubin_size = 0;
        CUresult decode_result = cudart_decode_fatbin_file(elf_candidate, &cubin, &cubin_size);
        if (decode_result != CUDA_SUCCESS) {
            return decode_result;
        }
        fprintf(stderr, "[CXL-CUDA]   library CUBIN load selected offset=%llu sm=0x%x decoded_size=%zu\n",
                (unsigned long long)elf_offset, elf_candidate->sm_version, cubin_size);
        CUresult result = cxl_module_load_cubin(module, cubin, cubin_size, 0, cubin_size);
        free(cubin);
        return result;
    }

    if (ptx_candidate) {
        unsigned char *ptx = NULL;
        size_t ptx_size = 0;
        CUresult decode_result = cudart_decode_fatbin_file(ptx_candidate, &ptx, &ptx_size);
        if (decode_result != CUDA_SUCCESS) {
            return decode_result;
        }

        while (ptx_size && ptx[ptx_size - 1] == '\0') {
            ptx_size--;
        }
        ptx[ptx_size] = '\0';
        fprintf(stderr, "[CXL-CUDA]   library PTX load selected offset=%llu sm=0x%x decoded_size=%zu\n",
                (unsigned long long)ptx_offset, ptx_candidate->sm_version, ptx_size);
        CUresult result = cuModuleLoadData(module, ptx);
        free(ptx);
        return result;
    }

    fprintf(stderr,
            "[CXL-CUDA]   library module load reject: no compatible CUBIN or PTX for sm_%u in selected fatbin "
            "submodule\n",
            target_sm);
    return CUDA_ERROR_NO_BINARY_FOR_GPU;
}

static CUresult cudart_library_materialize_module(CudartLibraryRecord *record) {
    if (!record || !record->alive) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (record->module) {
        return CUDA_SUCCESS;
    }

    CUresult result = cudart_load_module_from_fatbin(record->code, &record->module);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[CXL-CUDA] library_record id=%u module materialization failed result=%d\n", record->id,
                result);
        return result;
    }

    fprintf(stderr, "[CXL-CUDA] library_record id=%u module materialized module=%p\n", record->id, record->module);
    return CUDA_SUCCESS;
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
    void *caller = __builtin_return_address(0);
    Dl_info caller_info;
    if (caller && dladdr(caller, &caller_info) && caller_info.dli_fbase) {
        fprintf(stderr, "[CXL-CUDA] cuLibraryLoadData caller=%p file=%s file_offset=0x%lx\n", caller,
                caller_info.dli_fname ? caller_info.dli_fname : "(unknown)",
                (unsigned long)((uintptr_t)caller - (uintptr_t)caller_info.dli_fbase));
    }
    cxl_cuda_provenance_emit_first_stack("cuLibraryLoadData");
    context_storage_log_entries("cuLibraryLoadData:entry");
    fprintf(stderr,
            "[CXL-CUDA] cuLibraryLoadData(library=%p, code=%p, jitOptions=%p, jitOptionsValues=%p, "
            "numJitOptions=%u, libraryOptions=%p, libraryOptionValues=%p, numLibraryOptions=%u) -> library object\n",
            (void *)library, code, (void *)jitOptions, (void *)jitOptionsValues, numJitOptions, (void *)libraryOptions,
            (void *)libraryOptionValues, numLibraryOptions);
    cudart_log_fatbin_headers(code);

    if (library) {
        *library = NULL;
    }

    for (unsigned int i = 0; i < numLibraryOptions; i++) {
        if (!libraryOptions) {
            fprintf(stderr, "[CXL-CUDA]   libraryOptions[%u]=<null libraryOptions>\n", i);
            continue;
        }
        CUlibraryOption option = libraryOptions[i];
        void *option_value = libraryOptionValues ? libraryOptionValues[i] : NULL;
        fprintf(stderr, "[CXL-CUDA]   libraryOptions[%u]=%d libraryOptionValues[%u]=%p\n", i, option, i, option_value);
        if (option == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE && option_value) {
            CUlibraryHostUniversalFunctionAndDataTable *table =
                (CUlibraryHostUniversalFunctionAndDataTable *)option_value;
            fprintf(stderr,
                    "[CXL-CUDA]   host_universal_table functionTable=%p functionWindowSize=%zu dataTable=%p "
                    "dataWindowSize=%zu\n",
                    table->functionTable, table->functionWindowSize, table->dataTable, table->dataWindowSize);
        }
    }
    if (getenv("CXL_CUDA_LIBRARY_LOAD_DATA_NOT_SUPPORTED")) {
        fprintf(stderr, "[CXL-CUDA]   diagnostic override -> CUDA_ERROR_NOT_SUPPORTED\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (!library || !code) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (numLibraryOptions && !libraryOptions) {
        fprintf(stderr, "[CXL-CUDA]   library object reject: numLibraryOptions=%u with null libraryOptions\n",
                numLibraryOptions);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (numLibraryOptions > CUDART_LIBRARY_RECORD_OPTION_CAP) {
        fprintf(stderr, "[CXL-CUDA]   library object reject: numLibraryOptions=%u exceeds cap=%u\n", numLibraryOptions,
                CUDART_LIBRARY_RECORD_OPTION_CAP);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    int preserve_binary = 0;
    for (unsigned int i = 0; i < numLibraryOptions; i++) {
        CUlibraryOption option = libraryOptions[i];
        void *option_value = libraryOptionValues ? libraryOptionValues[i] : NULL;

        if (option == CU_LIBRARY_BINARY_IS_PRESERVED) {
            if (option_value) {
                preserve_binary = 1;
            }
            continue;
        }

        fprintf(stderr, "[CXL-CUDA]   library object reject: unsupported library option %d\n", option);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (!preserve_binary) {
        fprintf(stderr, "[CXL-CUDA]   library object reject: CU_LIBRARY_BINARY_IS_PRESERVED not asserted; "
                        "fatbin length is unknown, so this shim will not memcpy unknown code bytes\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    /* The code pointer names the registration wrapper stored in the DSO that
     * contributed this library object.  Caller attribution alone usually
     * points at libcudart, so record dladdr(code) as the runtime corpus fact.
     * The verifier fails if any record is unmapped or comes from a DSO absent
     * from the exact per-ELF prediction; this exposes an unexpected fifth
     * contributor without imposing a process-global library ceiling. */
    Dl_info code_info = {0};
    const char *code_file = "(unmapped)";
    uintptr_t code_base = 0;
    unsigned long code_offset = 0;
    if (dladdr(code, &code_info) && code_info.dli_fbase) {
        code_file = code_info.dli_fname ? code_info.dli_fname : "(unknown)";
        code_base = (uintptr_t)code_info.dli_fbase;
        code_offset = (unsigned long)((uintptr_t)code - code_base);
    }

    CudartLibraryRecord *record = calloc(1, sizeof(*record));
    if (!record) {
        fprintf(stderr, "[CXL-CUDA]   library_record allocation failed -> CUDA_ERROR_OUT_OF_MEMORY\n");
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    record->magic = CUDART_LIBRARY_RECORD_MAGIC;
    record->id = g_cudart_library_next_id++;
    record->alive = 1;
    record->code = code;
    record->preserved_code = code;
    record->num_jit_options = numJitOptions;
    record->num_library_options = numLibraryOptions;

    if (libraryOptions) {
        record->stored_library_options = numLibraryOptions;
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            record->options[i] = libraryOptions[i];
            record->option_values[i] = libraryOptionValues ? libraryOptionValues[i] : NULL;
            if (record->options[i] == CU_LIBRARY_BINARY_IS_PRESERVED && record->option_values[i]) {
                record->preserve_binary = 1;
            }
        }
    }

    record->next = g_cudart_library_records;
    g_cudart_library_records = record;

    *library = (CUlibrary)record;
    fprintf(stderr,
            "[CXL-CUDA]   library_record id=%u handle=%p code=%p code_file=%s code_base=0x%llx code_offset=0x%lx "
            "numJitOptions=%u numLibraryOptions=%u "
            "storedOptions=%u preserve_binary=%d module=%p alive=%d magic=0x%llx\n",
            record->id, (void *)*library, record->code, code_file, (unsigned long long)code_base, code_offset,
            record->num_jit_options,
            record->num_library_options, record->stored_library_options, record->preserve_binary, record->module,
            record->alive, (unsigned long long)record->magic);
    context_storage_log_entries("cuLibraryLoadData:success_exit");
    return CUDA_SUCCESS;
}

CUresult cuLibraryUnload(CUlibrary library) {
    CudartLibraryRecord *record = cudart_library_record_from_handle(library);
    if (!record) {
        fprintf(stderr, "[CXL-CUDA] cuLibraryUnload(library=%p) -> CUDA_ERROR_INVALID_HANDLE\n", library);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!record->alive) {
        fprintf(stderr, "[CXL-CUDA] cuLibraryUnload(library=%p id=%u) -> CUDA_ERROR_INVALID_HANDLE already dead\n",
                library, record->id);
        return CUDA_ERROR_INVALID_HANDLE;
    }

    record->alive = 0;
    fprintf(stderr, "[CXL-CUDA] cuLibraryUnload(library=%p id=%u) -> CUDA_SUCCESS\n", library, record->id);
    return CUDA_SUCCESS;
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
                               void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions,
                               void **libraryOptionValues, unsigned int numLibraryOptions) {
    (void)library;
    (void)jitOptions;
    (void)jitOptionsValues;
    (void)numJitOptions;
    (void)libraryOptions;
    (void)libraryOptionValues;
    (void)numLibraryOptions;
    fprintf(stderr, "[CXL-CUDA] cuLibraryLoadFromFile(fileName=%s) -> CUDA_ERROR_NOT_SUPPORTED\n",
            fileName ? fileName : "(null)");
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name) {
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetKernel(library=%p, name=%s) -> CUDA_ERROR_NOT_SUPPORTED\n", library,
            name ? name : "(null)");
    if (!pKernel || !library || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CudartLibraryRecord *record = cudart_library_record_from_handle(library);
    if (!record || !record->alive) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *pKernel = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
    if (!pMod || !library) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CudartLibraryRecord *record = cudart_library_record_from_handle(library);
    if (!record || !record->alive) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!record->module) {
        *pMod = NULL;
        CUresult result = cudart_library_materialize_module(record);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[CXL-CUDA] cuLibraryGetModule(library=%p id=%u) -> error=%d\n", library, record->id,
                    result);
            return result;
        }
    }
    *pMod = record->module;
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetModule(library=%p id=%u) -> module=%p CUDA_SUCCESS\n", library, record->id,
            record->module);
    return CUDA_SUCCESS;
}

CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    (void)dptr;
    (void)bytes;
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetGlobal(library=%p, name=%s) -> CUDA_ERROR_NOT_SUPPORTED\n", library,
            name ? name : "(null)");
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    (void)dptr;
    (void)bytes;
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetManaged(library=%p, name=%s) -> CUDA_ERROR_NOT_SUPPORTED\n", library,
            name ? name : "(null)");
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol) {
    (void)fptr;
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetUnifiedFunction(library=%p, symbol=%s) -> CUDA_ERROR_NOT_SUPPORTED\n",
            library, symbol ? symbol : "(null)");
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetKernelCount(unsigned int *count, CUlibrary lib) {
    (void)count;
    fprintf(stderr, "[CXL-CUDA] cuLibraryGetKernelCount(library=%p) -> CUDA_ERROR_NOT_SUPPORTED\n", lib);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryEnumerateKernels(CUkernel *kernels, unsigned int numKernels, CUlibrary lib) {
    (void)kernels;
    fprintf(stderr, "[CXL-CUDA] cuLibraryEnumerateKernels(library=%p, numKernels=%u) -> CUDA_ERROR_NOT_SUPPORTED\n",
            lib, numKernels);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
    if (!pFunc || !kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pFunc = NULL;

    Dl_info kernel_info;
    if (!dladdr(kernel, &kernel_info) || !kernel_info.dli_fbase || !kernel_info.dli_sname) {
        fprintf(stderr,
                "[CXL-CUDA] cuKernelGetFunction(kernel=%p) -> CUDA_ERROR_NOT_SUPPORTED "
                "reason=kernel-symbol-unavailable\n",
                kernel);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUfunction resolved = NULL;
    unsigned int matches = 0;
    for (CudartLibraryRecord *record = g_cudart_library_records; record; record = record->next) {
        Dl_info code_info;
        if (!record->alive || !record->module || !dladdr(record->code, &code_info) ||
            code_info.dli_fbase != kernel_info.dli_fbase) {
            continue;
        }

        CUfunction candidate = NULL;
        CUresult result = cuModuleGetFunction(&candidate, record->module, kernel_info.dli_sname);
        if (result == CUDA_ERROR_NOT_FOUND) {
            continue;
        }
        if (result != CUDA_SUCCESS) {
            fprintf(stderr,
                    "[CXL-CUDA] cuKernelGetFunction(kernel=%p symbol=%s library_id=%u module=%p) "
                    "-> error=%d\n",
                    kernel, kernel_info.dli_sname, record->id, record->module, result);
            return result;
        }
        resolved = candidate;
        matches++;
    }

    if (matches != 1) {
        fprintf(stderr,
                "[CXL-CUDA] cuKernelGetFunction(kernel=%p symbol=%s owner=%s) -> "
                "CUDA_ERROR_NOT_SUPPORTED reason=module-match-count count=%u\n",
                kernel, kernel_info.dli_sname, kernel_info.dli_fname ? kernel_info.dli_fname : "(unknown)", matches);
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    *pFunc = resolved;
    fprintf(stderr, "[CXL-CUDA] cuKernelGetFunction(kernel=%p symbol=%s owner=%s) -> function=%p CUDA_SUCCESS\n",
            kernel, kernel_info.dli_sname, kernel_info.dli_fname ? kernel_info.dli_fname : "(unknown)", resolved);
    return CUDA_SUCCESS;
}

CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel) {
    fprintf(stderr, "[CXL-CUDA] cuKernelGetAttribute(kernel=%p, attrib=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", kernel,
            attrib);
    if (!pi || !kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pi = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel) {
    fprintf(stderr, "[CXL-CUDA] cuKernelSetAttribute(kernel=%p, attrib=%d, val=%d) -> CUDA_ERROR_NOT_SUPPORTED\n",
            kernel, attrib, val);
    if (!kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config) {
    fprintf(stderr, "[CXL-CUDA] cuKernelSetCacheConfig(kernel=%p, config=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", kernel,
            config);
    if (!kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuKernelGetName(const char **name, CUkernel kernel) {
    fprintf(stderr, "[CXL-CUDA] cuKernelGetName(kernel=%p) -> CUDA_ERROR_NOT_SUPPORTED\n", kernel);
    if (!name || !kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
    fprintf(stderr, "[CXL-CUDA] cuKernelGetParamInfo(kernel=%p, index=%zu) -> CUDA_ERROR_NOT_SUPPORTED\n", kernel,
            paramIndex);
    if (!kernel || !paramOffset || !paramSize) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *paramOffset = 0;
    *paramSize = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    fprintf(stderr, "[CXL-CUDA] cuFuncSetCacheConfig(func=%p, config=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", hfunc, config);
    if (!hfunc) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    fprintf(stderr, "[CXL-CUDA] cuFuncSetSharedMemConfig(func=%p, config=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", hfunc,
            config);
    if (!hfunc) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    fprintf(stderr, "[CXL-CUDA] cuFuncGetAttribute(func=%p, attrib=%d) -> CUDA_ERROR_NOT_SUPPORTED\n", hfunc, attrib);
    if (!pi || !hfunc) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pi = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    DLOG("cuFuncSetAttribute(func=%p, attrib=%d, value=%d)\n", hfunc, attrib, value);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hfunc));
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)attrib);
    reg_write64(CXL_GPU_REG_PARAM2, (uint64_t)(int64_t)value);
    CUresult err = execute_cmd(CXL_GPU_CMD_FUNC_SET_ATTRIBUTE);
    cmd_unlock();
    return err;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction hfunc, int blockSize,
                                                              size_t dynamicSMemSize, unsigned int flags) {
    DLOG("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks=%p, func=%p, blockSize=%d, "
         "dynamicSMemSize=%zu, flags=%u)\n",
         (void *)numBlocks, hfunc, blockSize, dynamicSMemSize, flags);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!numBlocks || !hfunc)
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hfunc));
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)(int64_t)blockSize);
    reg_write64(CXL_GPU_REG_PARAM2, dynamicSMemSize);
    reg_write64(CXL_GPU_REG_PARAM3, flags);
    CUresult err = execute_cmd(CXL_GPU_CMD_FUNC_GET_OCCUPANCY);
    if (err == CUDA_SUCCESS)
        *numBlocks = (int)(int64_t)reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return err;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction hfunc, int blockSize,
                                                     size_t dynamicSMemSize) {
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, hfunc, blockSize, dynamicSMemSize, 0);
}

CUresult cuFuncGetName(const char **name, CUfunction hfunc) {
    fprintf(stderr, "[CXL-CUDA] cuFuncGetName(func=%p) -> CUDA_ERROR_NOT_SUPPORTED\n", hfunc);
    if (!name || !hfunc) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuFuncGetParamInfo(CUfunction hfunc, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
    DLOG("cuFuncGetParamInfo(func=%p, index=%zu)\n", hfunc, paramIndex);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc || !paramOffset) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hfunc));
    reg_write64(CXL_GPU_REG_PARAM1, paramIndex);
    CUresult err = execute_cmd(CXL_GPU_CMD_FUNC_GET_PARAM_INFO);
    if (err == CUDA_SUCCESS) {
        *paramOffset = reg_read64(CXL_GPU_REG_RESULT0);
        if (paramSize)
            *paramSize = reg_read64(CXL_GPU_REG_RESULT1);
    }
    cmd_unlock();
    return err;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    DLOG("cuModuleGetFunction(mod=%p, name=%s)\n", hmod, name);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc || !name)
        return CUDA_ERROR_INVALID_VALUE;

    if (!hmod) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    size_t len = strlen(name) + 1;
    if (len > CXL_GPU_DATA_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hmod));
    data_write(0, name, len);

    CUresult err = execute_cmd(CXL_GPU_CMD_FUNC_GET);
    if (err == CUDA_SUCCESS) {
        *hfunc = (CUfunction)cxl_gpu_handle_from_id(reg_read64(CXL_GPU_REG_RESULT0));
        DLOG("  func=%p\n", *hfunc);
    }
    cmd_unlock();
    return err;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    DLOG("cuModuleGetGlobal_v2(dptr=%p, bytes=%p, mod=%p, name=%s)\n", (void *)dptr, (void *)bytes, hmod,
         name ? name : "<null>");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if ((!dptr && !bytes) || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!hmod) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    size_t len = strlen(name) + 1;
    if (len > CXL_GPU_DATA_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(hmod));
    data_write(0, name, len);
    CUresult err = execute_cmd(CXL_GPU_CMD_MODULE_GET_GLOBAL);
    if (err == CUDA_SUCCESS) {
        uint64_t global = reg_read64(CXL_GPU_REG_RESULT0);
        uint64_t size = reg_read64(CXL_GPU_REG_RESULT1);
        if (dptr)
            *dptr = global;
        if (bytes)
            *bytes = size;
        DLOG("  global=%" PRIu64 " size=%" PRIu64 "\n", global, size);
    }
    cmd_unlock();
    return err;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    (void)extra;
    uint64_t stream_wire;

    DLOG("cuLaunchKernel(f=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u)\n", f, gridDimX, gridDimY, gridDimZ,
         blockDimX, blockDimY, blockDimZ, sharedMemBytes);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    if (!f) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;

    if (!kernelParams && extra) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    size_t param_offsets[CXL_MAX_KERNEL_ARGS];
    size_t param_sizes[CXL_MAX_KERNEL_ARGS];
    uint32_t num_args = 0;
    size_t param_extent = 0;
    while (num_args < CXL_MAX_KERNEL_ARGS) {
        size_t offset = 0;
        size_t size = 0;
        CUresult query = cuFuncGetParamInfo(f, num_args, &offset, &size);

        if (query == CUDA_ERROR_INVALID_VALUE) {
            break;
        }
        if (query != CUDA_SUCCESS) {
            return query;
        }
        if (!kernelParams || !kernelParams[num_args] || offset > CXL_GPU_DATA_SIZE ||
            size > CXL_GPU_DATA_SIZE - offset) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        param_offsets[num_args] = offset;
        param_sizes[num_args] = size;
        if (offset + size > param_extent)
            param_extent = offset + size;
        num_args++;
    }
    if (num_args == CXL_MAX_KERNEL_ARGS) {
        size_t offset = 0;
        size_t size = 0;
        if (cuFuncGetParamInfo(f, num_args, &offset, &size) == CUDA_SUCCESS) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    uint8_t *param_buffer = NULL;
    if (param_extent) {
        param_buffer = calloc(1, param_extent);
        if (!param_buffer)
            return CUDA_ERROR_OUT_OF_MEMORY;
        for (uint32_t i = 0; i < num_args; i++) {
            memcpy(param_buffer + param_offsets[i], kernelParams[i], param_sizes[i]);
        }
    }

    cmd_lock();
    if (param_extent)
        data_write(0, param_buffer, param_extent);
    free(param_buffer);
    reg_write64(CXL_GPU_REG_PARAM0, cxl_gpu_id_from_handle(f));
    reg_write64(CXL_GPU_REG_PARAM1, ((uint64_t)gridDimY << 32) | gridDimX);
    reg_write64(CXL_GPU_REG_PARAM2, ((uint64_t)blockDimX << 32) | gridDimZ);
    reg_write64(CXL_GPU_REG_PARAM3, ((uint64_t)blockDimZ << 32) | blockDimY);
    reg_write64(CXL_GPU_REG_PARAM4, ((uint64_t)num_args << 32) | sharedMemBytes);
    reg_write64(CXL_GPU_REG_PARAM5, param_extent);
    reg_write64(CXL_GPU_REG_PARAM6, stream_wire);
    CUresult err = execute_cmd(CXL_GPU_CMD_LAUNCH_KERNEL);
    cmd_unlock();
    return err;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                         void **optionValues, CUlinkState *stateOut) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!stateOut || (numOptions && (!options || !optionValues)))
        return CUDA_ERROR_INVALID_VALUE;
    if (numOptions)
        return CUDA_ERROR_NOT_SUPPORTED;
    cmd_lock();
    CUresult result = execute_cmd(CXL_GPU_CMD_LINK_CREATE);
    uint64_t id = reg_read64(CXL_GPU_REG_RESULT0);
    if (result == CUDA_SUCCESS) {
        if (id > UINT32_MAX) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_HANDLE;
        }
        *stateOut = (CUlinkState)cxl_gpu_handle_from_id(id);
    }
    cmd_unlock();
    return result;
}

CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                      void **optionValues, CUlinkState *stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data,
                          size_t size, const char *name, unsigned int numOptions,
                          CUjit_option *options, void **optionValues) {
    uint64_t id;
    size_t name_size = name ? strlen(name) + 1 : 0;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(state, &id) || !data || !size ||
        (numOptions && (!options || !optionValues)) ||
        size > CXL_GPU_DATA_SIZE || name_size > CXL_GPU_DATA_SIZE - size)
        return CUDA_ERROR_INVALID_VALUE;
    if (numOptions)
        return CUDA_ERROR_NOT_SUPPORTED;
    cmd_lock();
    data_write(0, data, size);
    if (name_size)
        data_write(size, name, name_size);
    reg_write64(CXL_GPU_REG_PARAM0, id);
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)(uint32_t)type);
    reg_write64(CXL_GPU_REG_PARAM2, size);
    reg_write64(CXL_GPU_REG_PARAM3, name_size);
    CUresult result = execute_cmd(CXL_GPU_CMD_LINK_ADD_DATA);
    cmd_unlock();
    return result;
}

CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                       size_t size, const char *name, unsigned int numOptions,
                       CUjit_option *options, void **optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                            optionValues);
}

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(state, &id) || !cubinOut)
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_LINK_COMPLETE);
    size_t size = reg_read64(CXL_GPU_REG_RESULT0);
    if (result == CUDA_SUCCESS) {
        if (size > CXL_GPU_DATA_SIZE) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_VALUE;
        }
        CXLLinkOutput *output = link_output_get(id, true);
        void *bytes = size ? malloc(size) : NULL;
        if (!output || (size && !bytes)) {
            free(bytes);
            cmd_unlock();
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (size)
            data_read(0, bytes, size);
        free(output->bytes);
        output->bytes = bytes;
        output->size = size;
        *cubinOut = bytes;
        if (sizeOut)
            *sizeOut = size;
    }
    cmd_unlock();
    return result;
}

CUresult cuLinkDestroy(CUlinkState state) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(state, &id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_LINK_DESTROY);
    if (result == CUDA_SUCCESS)
        link_output_remove(id);
    cmd_unlock();
    return result;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!phStream)
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, Flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_CREATE);
    uint64_t id = reg_read64(CXL_GPU_REG_RESULT0);
    if (result == CUDA_SUCCESS) {
        if (id > UINT32_MAX) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_HANDLE;
        }
        *phStream = cxl_gpu_stream_handle_from_id(id);
    }
    cmd_unlock();
    return result;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_handle_id(hStream, &id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_DESTROY);
    cmd_unlock();
    return result;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    uint64_t wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(hStream, &wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_SYNC);
    cmd_unlock();
    return result;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                           unsigned int Flags) {
    uint64_t stream_wire, event_id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire) ||
        !cxl_gpu_handle_id(hEvent, &event_id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    reg_write64(CXL_GPU_REG_PARAM1, event_id);
    reg_write64(CXL_GPU_REG_PARAM2, Flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_WAIT_EVENT);
    cmd_unlock();
    return result;
}

CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                             cuuint32_t value, unsigned int flags) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(stream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    reg_write64(CXL_GPU_REG_PARAM1, addr);
    reg_write64(CXL_GPU_REG_PARAM2, value);
    reg_write64(CXL_GPU_REG_PARAM3, flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_WAIT_VALUE32);
    cmd_unlock();
    return result;
}

CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count,
                            CUstreamBatchMemOpParams *paramArray,
                            unsigned int flags) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(stream, &stream_wire) || count >= 256 ||
        (count && !paramArray) || count > CXL_GPU_DATA_SIZE / sizeof(*paramArray))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    if (count)
        data_write(0, paramArray, count * sizeof(*paramArray));
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    reg_write64(CXL_GPU_REG_PARAM1, count);
    reg_write64(CXL_GPU_REG_PARAM2, flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_BATCH_MEM_OP);
    cmd_unlock();
    return result;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx || !cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_GET_CTX);
    if (result == CUDA_SUCCESS)
        result = cuCtxGetCurrent(pctx);
    cmd_unlock();
    return result;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)(uint32_t)mode);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_BEGIN_CAPTURE);
    cmd_unlock();
    return result;
}

CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    return cuStreamBeginCapture_v2(hStream, mode);
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!phGraph || !cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_END_CAPTURE);
    uint64_t graph_id = reg_read64(CXL_GPU_REG_RESULT0);
    if (result == CUDA_SUCCESS)
        *phGraph = graph_id == UINT64_MAX ? NULL :
                   (CUgraph)cxl_gpu_handle_from_id(graph_id);
    cmd_unlock();
    return result;
}

CUresult cuStreamIsCapturing(CUstream hStream,
                             CUstreamCaptureStatus *captureStatus) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!captureStatus || !cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_IS_CAPTURING);
    if (result == CUDA_SUCCESS)
        *captureStatus = (CUstreamCaptureStatus)reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return result;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream hStream,
                                   CUstreamCaptureStatus *captureStatus_out,
                                   cuuint64_t *id_out, CUgraph *graph_out,
                                   const CUgraphNode **dependencies_out,
                                   size_t *numDependencies_out) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!captureStatus_out || !cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_GET_CAPTURE_INFO);
    if (result == CUDA_SUCCESS) {
        size_t count = reg_read64(CXL_GPU_REG_RESULT2);
        uint64_t graph_id = reg_read64(CXL_GPU_REG_RESULT3);
        if (count > CXL_GPU_DATA_SIZE / sizeof(uint64_t)) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_VALUE;
        }
        CXLStreamCaptureSnapshot *snapshot = stream_capture_snapshot_get(stream_wire);
        CUgraphNode *dependencies = count ? calloc(count, sizeof(*dependencies)) : NULL;
        uint64_t *ids = count ? malloc(count * sizeof(*ids)) : NULL;
        if (!snapshot || (count && (!dependencies || !ids))) {
            free(dependencies);
            free(ids);
            cmd_unlock();
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (count)
            data_read(0, ids, count * sizeof(*ids));
        for (size_t i = 0; i < count; i++)
            dependencies[i] = (CUgraphNode)cxl_gpu_handle_from_id(ids[i]);
        free(ids);
        free(snapshot->dependencies);
        snapshot->dependencies = dependencies;
        snapshot->count = count;
        *captureStatus_out = (CUstreamCaptureStatus)reg_read64(CXL_GPU_REG_RESULT0);
        if (id_out)
            *id_out = reg_read64(CXL_GPU_REG_RESULT1);
        if (graph_out)
            *graph_out = graph_id == UINT64_MAX ? NULL :
                         (CUgraph)cxl_gpu_handle_from_id(graph_id);
        if (dependencies_out)
            *dependencies_out = snapshot->dependencies;
        if (numDependencies_out)
            *numDependencies_out = count;
    }
    cmd_unlock();
    return result;
}

CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                            CUdevice dstDevice, CUstream hStream) {
    uint64_t stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, devPtr);
    reg_write64(CXL_GPU_REG_PARAM1, count);
    reg_write64(CXL_GPU_REG_PARAM2, (uint64_t)(int64_t)dstDevice);
    reg_write64(CXL_GPU_REG_PARAM3, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_MEM_PREFETCH_ASYNC);
    cmd_unlock();
    return result;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    DLOG("cuMemGetInfo\n");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    uintptr_t token = 0;
    CUresult state_err = cxl_cuda_context_get_current_live(&token);
    if (state_err != CUDA_SUCCESS) {
        log_context_state("cuMemGetInfo_v2", state_err);
        return state_err;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, token);
    CUresult err = execute_cmd(CXL_GPU_CMD_MEM_GET_INFO);
    if (err == CUDA_SUCCESS) {
        if (free)
            *free = reg_read64(CXL_GPU_REG_RESULT0);
        if (total)
            *total = reg_read64(CXL_GPU_REG_RESULT1);
    }
    cmd_unlock();
    log_context_state("cuMemGetInfo_v2", err);
    return err;
}

/* Version compatibility aliases */
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) { return cuDeviceTotalMem_v2(bytes, dev); }

CUresult cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice dev) { return cuCtxCreate_v2(ctx, flags, dev); }

CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) { return cuMemAlloc_v2(dptr, bytesize); }

CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) { return cuMemcpyHtoD_v2(dst, src, bytes); }

CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) { return cuMemcpyDtoH_v2(dst, src, bytes); }

CUresult cuMemGetInfo(size_t *free, size_t *total) { return cuMemGetInfo_v2(free, total); }

CUresult cuStreamDestroy(CUstream hStream) { return cuStreamDestroy_v2(hStream); }

/* Additional API functions for comprehensive testing */

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    DLOG("cuCtxGetCurrent()\n");
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    uintptr_t token = 0;
    CUresult err = cxl_cuda_context_get_current(&token);
    if (err == CUDA_SUCCESS)
        *pctx = (CUcontext)token;
    return err;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    DLOG("cuCtxSetCurrent(%p)\n", ctx);
    return cxl_cuda_context_set_current((uintptr_t)ctx);
}

CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pvalue)
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(uint32_t)limit);
    CUresult result = execute_cmd(CXL_GPU_CMD_CTX_GET_LIMIT);
    if (result == CUDA_SUCCESS)
        *pvalue = reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return result;
}

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!canAccessPeer)
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(int64_t)dev);
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)(int64_t)peerDev);
    CUresult result = execute_cmd(CXL_GPU_CMD_DEVICE_CAN_ACCESS_PEER);
    if (result == CUDA_SUCCESS)
        *canAccessPeer = (int)reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return result;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!peerContext)
        return CUDA_ERROR_INVALID_CONTEXT;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(uintptr_t)peerContext);
    reg_write64(CXL_GPU_REG_PARAM1, Flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_CTX_ENABLE_PEER);
    cmd_unlock();
    return result;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!peerContext)
        return CUDA_ERROR_INVALID_CONTEXT;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, (uint64_t)(uintptr_t)peerContext);
    CUresult result = execute_cmd(CXL_GPU_CMD_CTX_DISABLE_PEER);
    cmd_unlock();
    return result;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    DLOG("cuCtxPushCurrent_v2(%p)\n", ctx);
    return cxl_cuda_context_push_current((uintptr_t)ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    DLOG("cuCtxPopCurrent_v2()\n");
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    uintptr_t token = 0;
    CUresult err = cxl_cuda_context_pop_current(&token);
    if (err == CUDA_SUCCESS)
        *pctx = (CUcontext)token;
    return err;
}

/* cuGetProcAddress receives the stable API names and selects the ABI version
 * from cudaVersion. CUDA 12.9 asks for these two names with version 4000;
 * exposing only the ELF `_v2` entry points leaves libcudart's context-enter
 * and context-leave slots bound to its local CUDA error 36 stubs. */
CUresult cuCtxPushCurrent(CUcontext ctx) { return cuCtxPushCurrent_v2(ctx); }

CUresult cuCtxPopCurrent(CUcontext *pctx) { return cuCtxPopCurrent_v2(pctx); }

CUresult cuCtxGetDevice(CUdevice *device) {
    DLOG("cuCtxGetDevice()\n");
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    uintptr_t token = 0;
    CUresult err = cxl_cuda_context_get_current_live(&token);
    if (err != CUDA_SUCCESS)
        return err;
    *device = 0; /* Currently only support device 0 */
    return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    DLOG("cuCtxGetApiVersion(ctx=%p)\n", ctx);
    (void)ctx;
    if (!version)
        return CUDA_ERROR_INVALID_VALUE;
    *version = 12090;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
    DLOG("cuCtxGetFlags()\n");
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    uintptr_t token = 0;
    CUresult err = cxl_cuda_context_get_current_live(&token);
    if (err != CUDA_SUCCESS)
        return err;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    DLOG("cuDevicePrimaryCtxRetain(dev=%d)\n", dev);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;

    bool needs_create = false;
    uintptr_t token = 0;
    CUresult state_err = cxl_cuda_context_prepare_primary_retain(&needs_create, &token);
    if (state_err != CUDA_SUCCESS)
        return state_err;
    if (needs_create) {
        cmd_lock();
        CUresult err = execute_cmd(CXL_GPU_CMD_CTX_CREATE);
        if (err == CUDA_SUCCESS) {
            token = reg_read64(CXL_GPU_REG_RESULT0);
            err = cxl_cuda_context_commit_primary_retain(token);
            if (err == CUDA_SUCCESS)
                DLOG("  primary_ctx=%p\n", (void *)token);
        }
        cmd_unlock();
        if (err != CUDA_SUCCESS)
            return err;
    }
    *pctx = (CUcontext)token;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    DLOG("cuDevicePrimaryCtxRelease(dev=%d)\n", dev);
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    return cxl_cuda_context_primary_release();
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) { return cuDevicePrimaryCtxRelease(dev); }

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    DLOG("cuDevicePrimaryCtxSetFlags(dev=%d, flags=%u)\n", dev, flags);
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    if (flags != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    return cxl_cuda_context_primary_set_zero_flags();
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    DLOG("cuDevicePrimaryCtxGetState(dev=%d)\n", dev);
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    if (!flags || !active)
        return CUDA_ERROR_INVALID_VALUE;
    return cxl_cuda_context_primary_get_state(flags, active);
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    DLOG("cuDevicePrimaryCtxReset(dev=%d)\n", dev);
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    return cxl_cuda_context_primary_reset();
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) { return cuDevicePrimaryCtxReset(dev); }

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    DLOG("cuDeviceGetPCIBusId(dev=%d)\n", dev);
    if (!pciBusId || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    snprintf(pciBusId, len, "%s", g_transport.pci_bdf[0] ? g_transport.pci_bdf : "0000:00:00.0");
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    DLOG("cuDeviceGetByPCIBusId(%s)\n", pciBusId ? pciBusId : "(null)");
    if (!dev || !pciBusId)
        return CUDA_ERROR_INVALID_VALUE;
    if (g_transport.pci_bdf[0] && strcmp(pciBusId, g_transport.pci_bdf) != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    *dev = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int *value, int attrib, CUdevice srcDevice, CUdevice dstDevice) {
    DLOG("cuDeviceGetP2PAttribute(attrib=%d, src=%d, dst=%d)\n", attrib, srcDevice, dstDevice);
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    if (srcDevice != 0 || dstDevice != 0)
        return CUDA_ERROR_INVALID_DEVICE;
    *value = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    DLOG("cuMemcpyDtoD_v2(dst=0x%lx, src=0x%lx, size=%zu)\n", (unsigned long)dstDevice, (unsigned long)srcDevice,
         byteCount);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dstDevice || !srcDevice)
        return CUDA_ERROR_INVALID_VALUE;

    /* For D2D copy, use intermediate host buffer via data region */
    size_t offset = 0;
    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > CXL_GPU_DATA_SIZE) {
            chunk = CXL_GPU_DATA_SIZE;
        }

        cmd_lock();
        /* Read from source device memory to data region */
        reg_write64(CXL_GPU_REG_PARAM0, srcDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, chunk);
        CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_DTOH);
        if (result != CUDA_SUCCESS) {
            cmd_unlock();
            return result;
        }

        /* Write from data region to destination device memory */
        reg_write64(CXL_GPU_REG_PARAM0, dstDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, chunk);
        result = execute_cmd(CXL_GPU_CMD_MEM_COPY_HTOD);
        cmd_unlock();
        if (result != CUDA_SUCCESS) {
            return result;
        }

        offset += chunk;
    }

    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    (void)hStream;
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, byteCount, hStream);
}

static bool cxl_memcpy2d_offset(CUdeviceptr base, size_t pitch, size_t x, size_t y, size_t width, CUdeviceptr *row) {
    if (x > pitch || width > pitch - x || (y != 0 && pitch > SIZE_MAX / y)) {
        return false;
    }

    size_t offset = y * pitch + x;
    if (offset < x || offset > UINT64_MAX - base) {
        return false;
    }

    *row = base + offset;
    if (width > UINT64_MAX - *row) {
        return false;
    }
    return true;
}

static CUresult cxl_memcpy2d_device_to_device(const CUDA_MEMCPY2D *copy) {
    if (!copy) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (copy->srcMemoryType != CU_MEMORYTYPE_DEVICE || copy->dstMemoryType != CU_MEMORYTYPE_DEVICE) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (copy->WidthInBytes == 0 || copy->Height == 0) {
        return CUDA_SUCCESS;
    }
    if (!copy->srcDevice || !copy->dstDevice) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    for (size_t row = 0; row < copy->Height; row++) {
        CUdeviceptr src;
        CUdeviceptr dst;
        if (copy->srcY > SIZE_MAX - row || copy->dstY > SIZE_MAX - row ||
            !cxl_memcpy2d_offset(copy->srcDevice, copy->srcPitch, copy->srcXInBytes, copy->srcY + row,
                                 copy->WidthInBytes, &src) ||
            !cxl_memcpy2d_offset(copy->dstDevice, copy->dstPitch, copy->dstXInBytes, copy->dstY + row,
                                 copy->WidthInBytes, &dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        CUresult result = cuMemcpyDtoD_v2(dst, src, copy->WidthInBytes);
        if (result != CUDA_SUCCESS) {
            return result;
        }
    }

    return CUDA_SUCCESS;
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *copy) {
    DLOG("cuMemcpy2D_v2(copy=%p)\n", (const void *)copy);
    return cxl_memcpy2d_device_to_device(copy);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D *copy) { return cuMemcpy2D_v2(copy); }

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *copy, CUstream hStream) {
    DLOG("cuMemcpy2DAsync_v2(copy=%p, stream=%p)\n", (const void *)copy, hStream);
    (void)hStream;
    /* knockout: Type-2 currently serializes transfer commands. Preserve the
     * existing async copy contract by completing this multidimensional copy
     * before return; add a stream-aware BAR2 protocol only after it is measured. */
    return cxl_memcpy2d_device_to_device(copy);
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *copy, CUstream hStream) { return cuMemcpy2DAsync_v2(copy, hStream); }

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    DLOG("cuMemsetD8_v2(dst=0x%lx, val=0x%02x, count=%zu)\n", (unsigned long)dstDevice, uc, N);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    /* Fill data region with value and copy in chunks */
    size_t chunk_size = N < CXL_GPU_DATA_SIZE ? N : CXL_GPU_DATA_SIZE;
    uint8_t *temp = (uint8_t *)malloc(chunk_size);
    if (!temp)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memset(temp, uc, chunk_size);

    size_t offset = 0;
    while (offset < N) {
        size_t to_copy = (N - offset) < chunk_size ? (N - offset) : chunk_size;
        data_write(0, temp, to_copy);
        reg_write64(CXL_GPU_REG_PARAM0, dstDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, to_copy);
        CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_HTOD);
        if (result != CUDA_SUCCESS) {
            free(temp);
            return result;
        }
        offset += to_copy;
    }

    free(temp);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) { return cuMemsetD8_v2(dstDevice, uc, N); }

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    (void)hStream;
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    DLOG("cuMemsetD32_v2(dst=0x%lx, val=0x%08x, count=%zu)\n", (unsigned long)dstDevice, ui, N);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    /* Fill data region with value and copy in chunks */
    size_t chunk_elements = CXL_GPU_DATA_SIZE / sizeof(unsigned int);
    size_t chunk_size = chunk_elements * sizeof(unsigned int);
    unsigned int *temp = (unsigned int *)malloc(chunk_size);
    if (!temp)
        return CUDA_ERROR_OUT_OF_MEMORY;
    for (size_t i = 0; i < chunk_elements; i++) {
        temp[i] = ui;
    }

    size_t elements_done = 0;
    while (elements_done < N) {
        size_t to_copy = (N - elements_done) < chunk_elements ? (N - elements_done) : chunk_elements;
        data_write(0, temp, to_copy * sizeof(unsigned int));
        reg_write64(CXL_GPU_REG_PARAM0, dstDevice + elements_done * sizeof(unsigned int));
        reg_write64(CXL_GPU_REG_PARAM1, to_copy * sizeof(unsigned int));
        CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_HTOD);
        if (result != CUDA_SUCCESS) {
            free(temp);
            return result;
        }
        elements_done += to_copy;
    }

    free(temp);
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    DLOG("cuMemGetAddressRange_v2(dptr=0x%lx)\n", (unsigned long)dptr);

    /* Return the pointer itself as base; size is unknown without tracking */
    if (pbase)
        *pbase = dptr;
    if (psize)
        *psize = 0; /* Unknown */
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

CUresult cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr) {
    DLOG("cuPointerGetAttribute(attr=%d, ptr=0x%lx)\n", attribute, (unsigned long)ptr);

    if (!data)
        return CUDA_ERROR_INVALID_VALUE;

    switch (attribute) {
    case 1: /* CU_POINTER_ATTRIBUTE_CONTEXT */
        /* Allocation owner is a property of ptr, not the calling thread's
         * current context.  This shim does not maintain allocation provenance. */
        return CUDA_ERROR_NOT_SUPPORTED;
    case 2: /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */
        {
            bool is_device = false;
            CUresult result = cxl_cuda_pointer_is_device(ptr, &is_device);
            if (result != CUDA_SUCCESS)
                return result;
            *(int *)data = is_device ? 2 : 1;
            return CUDA_SUCCESS;
        }
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, int *attributes,
                                void **data, CUdeviceptr ptr) {
    DLOG("cuPointerGetAttributes(count=%u, ptr=0x%lx)\n", numAttributes,
         (unsigned long)ptr);
    if ((numAttributes != 0 && (!attributes || !data)))
        return CUDA_ERROR_INVALID_VALUE;

    for (unsigned int index = 0; index < numAttributes; index++) {
        CUresult result = cuPointerGetAttribute(data[index], attributes[index],
                                                ptr);
        if (result != CUDA_SUCCESS)
            return result;
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule hmod) {
    DLOG("cuModuleUnload(%p)\n", hmod);
    /* Modules are managed by hetGPU backend */
    return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!phEvent)
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, Flags);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_CREATE);
    uint64_t id = reg_read64(CXL_GPU_REG_RESULT0);
    if (result == CUDA_SUCCESS) {
        if (id > UINT32_MAX) {
            cmd_unlock();
            return CUDA_ERROR_INVALID_HANDLE;
        }
        *phEvent = (CUevent)cxl_gpu_handle_from_id(id);
    }
    cmd_unlock();
    return result;
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hEvent, &id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_DESTROY);
    cmd_unlock();
    return result;
}

CUresult cuEventDestroy(CUevent hEvent) { return cuEventDestroy_v2(hEvent); }

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    uint64_t event_id, stream_wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hEvent, &event_id) ||
        !cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, event_id);
    reg_write64(CXL_GPU_REG_PARAM1, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_RECORD);
    cmd_unlock();
    return result;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hEvent, &id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_SYNC);
    cmd_unlock();
    return result;
}

CUresult cuEventQuery(CUevent hEvent) {
    uint64_t id;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hEvent, &id))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_QUERY);
    cmd_unlock();
    return result;
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    uint64_t start, end;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pMilliseconds || !cxl_gpu_handle_id(hStart, &start) ||
        !cxl_gpu_handle_id(hEnd, &end))
        return CUDA_ERROR_INVALID_VALUE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, start);
    reg_write64(CXL_GPU_REG_PARAM1, end);
    CUresult result = execute_cmd(CXL_GPU_CMD_EVENT_ELAPSED_TIME);
    if (result == CUDA_SUCCESS) {
        uint32_t bits = reg_read64(CXL_GPU_REG_RESULT0);
        memcpy(pMilliseconds, &bits, sizeof(bits));
    }
    cmd_unlock();
    return result;
}

CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd) {
    return cuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

CUresult cuDeviceGetUuid(void *uuid, CUdevice dev) {
    DLOG("cuDeviceGetUuid(dev=%d)\n", dev);
    if (!uuid)
        return CUDA_ERROR_INVALID_VALUE;
    /* Generate a deterministic UUID based on device number */
    memset(uuid, 0, 16);
    ((unsigned char *)uuid)[0] = 0xCE; /* CXL */
    ((unsigned char *)uuid)[1] = 0x10; /* Type 2 */
    ((unsigned char *)uuid)[15] = (unsigned char)dev;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * P2P DMA Functions - Transfer data between GPU and CXL Type 3 memory
 * ============================================================================ */

/* P2P peer info structure */
typedef struct {
    uint32_t peer_id;
    uint32_t peer_type; /* CXL_P2P_PEER_TYPE2 or CXL_P2P_PEER_TYPE3 */
    uint64_t mem_size;
    int coherent;
} CXLPeerInfo;

/* Discover P2P peer devices on the CXL fabric */
int cxl_p2p_discover_peers(int *num_peers) {
    DLOG("cxl_p2p_discover_peers()\n");

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_DISCOVER);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    if (num_peers) {
        *num_peers = (int)reg_read64(CXL_GPU_REG_RESULT0);
    }

    DLOG("  discovered %d peers\n", num_peers ? *num_peers : -1);
    return CUDA_SUCCESS;
}

/* Get peer device information */
int cxl_p2p_get_peer_info(uint32_t peer_id, CXLPeerInfo *info) {
    DLOG("cxl_p2p_get_peer_info(peer=%u)\n", peer_id);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!info)
        return CUDA_ERROR_INVALID_VALUE;

    reg_write64(CXL_GPU_REG_PARAM0, peer_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_GET_PEER_INFO);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    info->peer_id = peer_id;
    info->peer_type = (uint32_t)reg_read64(CXL_GPU_REG_RESULT0);
    info->mem_size = reg_read64(CXL_GPU_REG_RESULT1);
    info->coherent = (int)reg_read64(CXL_GPU_REG_RESULT2);

    DLOG("  peer %u: type=%u, size=%lu MB, coherent=%d\n", peer_id, info->peer_type, info->mem_size / (1024 * 1024),
         info->coherent);
    return CUDA_SUCCESS;
}

/* Transfer data from GPU memory to Type 3 CXL memory */
int cxl_p2p_gpu_to_mem(uint32_t t3_peer_id, uint64_t gpu_offset, uint64_t mem_offset, uint64_t size) {
    DLOG("cxl_p2p_gpu_to_mem(peer=%u, gpu_off=0x%lx, mem_off=0x%lx, size=%lu)\n", t3_peer_id, (unsigned long)gpu_offset,
         (unsigned long)mem_offset, (unsigned long)size);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (size == 0)
        return CUDA_SUCCESS;

    reg_write64(CXL_GPU_REG_PARAM0, t3_peer_id);
    reg_write64(CXL_GPU_REG_PARAM1, gpu_offset);
    reg_write64(CXL_GPU_REG_PARAM2, mem_offset);
    reg_write64(CXL_GPU_REG_PARAM3, size);

    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_GPU_TO_MEM);
    if (result != CUDA_SUCCESS) {
        DLOG("  P2P GPU->MEM transfer failed: %d\n", result);
    }
    return result;
}

/* Transfer data from Type 3 CXL memory to GPU memory */
int cxl_p2p_mem_to_gpu(uint32_t t3_peer_id, uint64_t mem_offset, uint64_t gpu_offset, uint64_t size) {
    DLOG("cxl_p2p_mem_to_gpu(peer=%u, mem_off=0x%lx, gpu_off=0x%lx, size=%lu)\n", t3_peer_id, (unsigned long)mem_offset,
         (unsigned long)gpu_offset, (unsigned long)size);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (size == 0)
        return CUDA_SUCCESS;

    reg_write64(CXL_GPU_REG_PARAM0, t3_peer_id);
    reg_write64(CXL_GPU_REG_PARAM1, mem_offset);
    reg_write64(CXL_GPU_REG_PARAM2, gpu_offset);
    reg_write64(CXL_GPU_REG_PARAM3, size);

    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_MEM_TO_GPU);
    if (result != CUDA_SUCCESS) {
        DLOG("  P2P MEM->GPU transfer failed: %d\n", result);
    }
    return result;
}

/* Transfer data between two Type 3 CXL memory devices */
int cxl_p2p_mem_to_mem(uint32_t src_peer_id, uint32_t dst_peer_id, uint64_t src_offset, uint64_t dst_offset,
                       uint64_t size) {
    DLOG("cxl_p2p_mem_to_mem(src=%u, dst=%u, src_off=0x%lx, dst_off=0x%lx, size=%lu)\n", src_peer_id, dst_peer_id,
         (unsigned long)src_offset, (unsigned long)dst_offset, (unsigned long)size);

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (size == 0)
        return CUDA_SUCCESS;

    reg_write64(CXL_GPU_REG_PARAM0, src_peer_id);
    reg_write64(CXL_GPU_REG_PARAM1, dst_peer_id);
    reg_write64(CXL_GPU_REG_PARAM2, src_offset);
    reg_write64(CXL_GPU_REG_PARAM3, dst_offset);
    reg_write64(CXL_GPU_REG_PARAM4, size);

    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_MEM_TO_MEM);
    if (result != CUDA_SUCCESS) {
        DLOG("  P2P MEM->MEM transfer failed: %d\n", result);
    }
    return result;
}

/* Wait for all pending P2P transfers to complete */
int cxl_p2p_sync(void) {
    DLOG("cxl_p2p_sync()\n");

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    return execute_cmd(CXL_GPU_CMD_P2P_SYNC);
}

/* Get P2P engine status and statistics */
int cxl_p2p_get_status(int *num_peers, uint64_t *transfers_completed, uint64_t *bytes_transferred) {
    DLOG("cxl_p2p_get_status()\n");

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;

    CUresult result = execute_cmd(CXL_GPU_CMD_P2P_GET_STATUS);
    if (result != CUDA_SUCCESS) {
        return result;
    }

    if (num_peers) {
        *num_peers = (int)reg_read64(CXL_GPU_REG_RESULT0);
    }
    if (transfers_completed) {
        *transfers_completed = reg_read64(CXL_GPU_REG_RESULT1);
    }
    if (bytes_transferred) {
        *bytes_transferred = reg_read64(CXL_GPU_REG_RESULT2);
    }

    return CUDA_SUCCESS;
}

/* ========================================================================
 * BAR4 Coherent Memory Support
 * ======================================================================== */

static int g_bar4_fd = -1;
static volatile uint8_t *g_bar4_ptr = NULL;
static size_t g_bar4_size = 0;
static uint64_t g_coh_offset = 0; /* bump allocator offset */

static volatile uint8_t *ensure_bar4(void) {
    if (g_bar4_ptr)
        return g_bar4_ptr;

    /* Find BAR4 for the device we already mapped */
    char path[256];
    /* Scan sysfs for the device whose BAR2 we have open */
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir)
        return NULL;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.')
            continue;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", ent->d_name);
        FILE *fp = fopen(path, "r");
        if (!fp)
            continue;
        uint64_t start, end, flags;
        /* Skip BAR0..BAR3 (4 lines) */
        for (int i = 0; i < 4; i++) {
            if (fscanf(fp, "0x%lx 0x%lx 0x%lx\n", &start, &end, &flags) != 3)
                break;
        }
        /* Read BAR4 */
        if (fscanf(fp, "0x%lx 0x%lx 0x%lx", &start, &end, &flags) == 3 && end > start) {
            g_bar4_size = end - start + 1;
        }
        fclose(fp);
        if (g_bar4_size == 0)
            continue;

        /* Check vendor/device match */
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        int fd = open(path, O_RDONLY);
        if (fd < 0)
            continue;
        char buf[32];
        int n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n <= 0)
            continue;
        buf[n] = '\0';
        if ((uint16_t)strtol(buf, NULL, 16) != CXL_GPU_PCI_VENDOR_ID) {
            g_bar4_size = 0;
            continue;
        }

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", ent->d_name);
        fd = open(path, O_RDONLY);
        if (fd < 0)
            continue;
        n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n <= 0)
            continue;
        buf[n] = '\0';
        if ((uint16_t)strtol(buf, NULL, 16) != CXL_GPU_PCI_DEVICE_ID) {
            g_bar4_size = 0;
            continue;
        }

        /* Check this is the same device we're using (status must be READY) */
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource2", ent->d_name);
        int bar2_fd = open(path, O_RDWR | O_SYNC);
        if (bar2_fd < 0) {
            g_bar4_size = 0;
            continue;
        }
        void *bar2_map = mmap(NULL, 4096, PROT_READ, MAP_SHARED, bar2_fd, 0);
        if (bar2_map == MAP_FAILED) {
            close(bar2_fd);
            g_bar4_size = 0;
            continue;
        }
        uint32_t magic = *(volatile uint32_t *)bar2_map;
        uint32_t status = *(volatile uint32_t *)((uint8_t *)bar2_map + 8);
        munmap(bar2_map, 4096);
        close(bar2_fd);
        if (magic != CXL_GPU_MAGIC || !(status & CXL_GPU_STATUS_READY)) {
            g_bar4_size = 0;
            continue;
        }

        /* Map BAR4 */
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource4", ent->d_name);
        g_bar4_fd = open(path, O_RDWR);
        if (g_bar4_fd < 0) {
            g_bar4_size = 0;
            continue;
        }
        void *b4 = mmap(NULL, g_bar4_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_bar4_fd, 0);
        if (b4 == MAP_FAILED) {
            close(g_bar4_fd);
            g_bar4_fd = -1;
            g_bar4_size = 0;
            continue;
        }
        g_bar4_ptr = (volatile uint8_t *)b4;
        DLOG("Mapped BAR4 for %s (%zu MB)\n", ent->d_name, g_bar4_size >> 20);
        closedir(dir);
        return g_bar4_ptr;
    }
    closedir(dir);
    return NULL;
}

static inline uint64_t maybe_bar4_offset(uint64_t value) {
    uintptr_t ptr = (uintptr_t)value;

    if (g_bar4_ptr && ptr >= (uintptr_t)g_bar4_ptr && ptr < (uintptr_t)g_bar4_ptr + g_bar4_size) {
        return (uint64_t)(ptr - (uintptr_t)g_bar4_ptr);
    }
    return value;
}

int cxlCoherentAlloc(uint64_t size, void **host_ptr) {
    DLOG("cxlCoherentAlloc(size=%lu)\n", (unsigned long)size);
    if (!host_ptr || size == 0)
        return 1;
    volatile uint8_t *bar4 = ensure_bar4();
    if (!bar4)
        return 3;

    if (g_transport.regs) {
        cmd_lock();
        reg_write64(CXL_GPU_REG_PARAM0, size);
        CUresult r = execute_cmd(CXL_GPU_CMD_COHERENT_ALLOC);
        if (r == CUDA_SUCCESS) {
            uint64_t offset = reg_read64(CXL_GPU_REG_RESULT0);
            cmd_unlock();
            if (offset < g_bar4_size) {
                *host_ptr = (void *)(bar4 + offset);
                return 0;
            }
            return 2;
        }
        cmd_unlock();
    }

    size = (size + 4095) & ~4095UL;
    if (g_coh_offset + size > g_bar4_size)
        return 2;
    *host_ptr = (void *)(bar4 + g_coh_offset);
    g_coh_offset += size;
    return 0;
}

int cxlCoherentFree(void *host_ptr) {
    if (g_transport.regs && host_ptr) {
        uint64_t offset = bar4_offset_of(host_ptr);

        cmd_lock();
        reg_write64(CXL_GPU_REG_PARAM0, offset);
        CUresult r = execute_cmd(CXL_GPU_CMD_COHERENT_FREE);
        cmd_unlock();
        return r == CUDA_SUCCESS ? 0 : 1;
    }

    return 0;
}

void *cxlDeviceToHost(uint64_t dev_offset) {
    volatile uint8_t *bar4 = ensure_bar4();
    if (!bar4)
        return NULL;
    return (void *)(bar4 + dev_offset);
}

int cxlCoherentFence(void) {
    __sync_synchronize();
    return 0;
}

static inline uint64_t bar4_offset_of(void *host_ptr) {
    if (!host_ptr || !g_bar4_ptr)
        return 0;
    return (uint64_t)((uint8_t *)host_ptr - (uint8_t *)g_bar4_ptr);
}

int cxlSetBias(void *host_ptr, uint64_t size, int bias_mode) {
    if (!g_transport.regs)
        return 1;
    uint64_t addr = bar4_offset_of(host_ptr);
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, addr);
    reg_write64(CXL_GPU_REG_PARAM1, size);
    reg_write64(CXL_GPU_REG_PARAM2, (uint64_t)bias_mode);
    CUresult r = execute_cmd(CXL_GPU_CMD_SET_BIAS);
    cmd_unlock();
    return r == CUDA_SUCCESS ? 0 : 1;
}

int cxlGetBias(void *host_ptr, int *bias_mode) {
    if (!g_transport.regs || !bias_mode)
        return 1;
    uint64_t addr = bar4_offset_of(host_ptr);
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, addr);
    CUresult r = execute_cmd(CXL_GPU_CMD_GET_BIAS);
    *bias_mode = (int)reg_read64(CXL_GPU_REG_RESULT0);
    cmd_unlock();
    return r == CUDA_SUCCESS ? 0 : 1;
}

int cxlBiasFlip(void *host_ptr, uint64_t size, int new_bias) {
    if (!g_transport.regs)
        return 1;
    uint64_t addr = bar4_offset_of(host_ptr);
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, addr);
    reg_write64(CXL_GPU_REG_PARAM1, size);
    reg_write64(CXL_GPU_REG_PARAM2, (uint64_t)new_bias);
    CUresult r = execute_cmd(CXL_GPU_CMD_BIAS_FLIP);
    cmd_unlock();
    return r == CUDA_SUCCESS ? 0 : 1;
}

typedef struct {
    uint64_t snoop_hits;
    uint64_t snoop_misses;
    uint64_t coherency_requests;
    uint64_t back_invalidations;
    uint64_t writebacks;
    uint64_t evictions;
    uint64_t bias_flips;
    uint64_t device_bias_hits;
    uint64_t host_bias_hits;
    uint64_t upgrades;
    uint64_t downgrades;
    uint64_t directory_entries;
} CXLCoherencyStats;

int cxlGetCoherencyStats(CXLCoherencyStats *stats) {
    if (!stats)
        return 1;
    if (!g_transport.regs) {
        memset(stats, 0, sizeof(*stats));
        return 2;
    }

    cmd_lock();
    CUresult r = execute_cmd(CXL_GPU_CMD_COH_GET_STATS);
    if (r != CUDA_SUCCESS) {
        cmd_unlock();
        memset(stats, 0, sizeof(*stats));
        return 3;
    }
    stats->snoop_hits = reg_read64(CXL_GPU_REG_RESULT0);
    stats->snoop_misses = reg_read64(CXL_GPU_REG_RESULT1);
    stats->coherency_requests = reg_read64(CXL_GPU_REG_RESULT2);
    stats->back_invalidations = reg_read64(CXL_GPU_REG_RESULT3);

    uint64_t ext[8];
    data_read(0, ext, sizeof(ext));
    stats->writebacks = ext[0];
    stats->evictions = ext[1];
    stats->bias_flips = ext[2];
    stats->device_bias_hits = ext[3];
    stats->host_bias_hits = ext[4];
    stats->upgrades = ext[5];
    stats->downgrades = ext[6];
    stats->directory_entries = ext[7];
    cmd_unlock();
    return 0;
}

int cxlResetCoherencyStats(void) {
    if (!g_transport.regs)
        return 1;
    cmd_lock();
    CUresult r = execute_cmd(CXL_GPU_CMD_COH_RESET_STATS);
    cmd_unlock();
    return r == CUDA_SUCCESS ? 0 : 1;
}

CUresult cuCxlGetCoherentBase(CUdeviceptr *base, size_t *size, CUdevice dev) {
    (void)dev;
    volatile uint8_t *bar4 = ensure_bar4();
    if (base)
        *base = bar4 ? (CUdeviceptr)(uintptr_t)bar4 : 0;
    if (size)
        *size = g_bar4_size;
    return 0;
}

/* Library initialization/cleanup */
__attribute__((constructor)) static void libcuda_init(void) { DLOG("libcuda.so loaded (CXL Type 2 shim)\n"); }

__attribute__((destructor)) static void libcuda_cleanup(void) {
    CXLCudaErrorName *error_name;

    DLOG("libcuda.so unloading\n");
    graph_kernel_node_snapshots_clear();
    context_storage_clear_context(NULL, 0);
    if (g_bar4_ptr) {
        munmap((void *)g_bar4_ptr, g_bar4_size);
        g_bar4_ptr = NULL;
    }
    if (g_bar4_fd >= 0) {
        close(g_bar4_fd);
        g_bar4_fd = -1;
    }
    cxl_gpu_transport_close(&g_transport);
    while ((error_name = g_cuda_error_names) != NULL) {
        g_cuda_error_names = error_name->next;
        free(error_name->name);
        free(error_name);
    }
    g_initialized = 0;
}

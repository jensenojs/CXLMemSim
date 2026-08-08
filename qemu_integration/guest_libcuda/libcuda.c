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
#include <elf.h>
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
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "cxl_gpu_cmd.h"
#include "cxl_gpu_context_state.h"
#include "cxl_gpu_transport.h"
#include "cxl_cuda_observation.h"
#include "include/linux/cxl_type2_accel.h"

/* These symbols are linked into the shim's declared runtime dependency set. */
extern int LZ4_decompress_safe(const char *src, char *dst, int compressed_size, int dst_capacity);
extern size_t ZSTD_compress(void *dst, size_t dst_capacity, const void *src, size_t src_size, int compression_level);
extern size_t ZSTD_compressBound(size_t src_size);
extern size_t ZSTD_decompress(void *dst, size_t dst_capacity, const void *src, size_t compressed_size);
extern const char *ZSTD_getErrorName(size_t code);
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
typedef int CUgraphExecUpdateResult;
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
typedef int CUmemLocationType;
typedef int CUmemcpySrcAccessOrder;
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
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct {
    CUmemcpySrcAccessOrder srcAccessOrder;
    CUmemLocation srcLocHint;
    CUmemLocation dstLocHint;
    unsigned int flags;
} CUmemcpyAttributes;

_Static_assert(sizeof(CUmemLocation) == 8,
               "CUDA CUmemLocation ABI size mismatch");
_Static_assert(sizeof(CUmemcpyAttributes) == 24,
               "CUDA CUmemcpyAttributes ABI size mismatch");
_Static_assert(offsetof(CUmemcpyAttributes, flags) == 20,
               "CUDA CUmemcpyAttributes flags offset mismatch");

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

typedef struct {
    CUgraphExecUpdateResult result;
    CUgraphNode errorNode;
    CUgraphNode errorFromNode;
} CUgraphExecUpdateResultInfo;

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
#define CU_MEMCPY_SRC_ACCESS_ORDER_ANY 3
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
static int g_direct_source_enabled = 0;
#ifdef CXL_GPU_CONTEXT_SHIM_TEST
static uint8_t g_test_bar2[CXL_GPU_CMD_REG_SIZE];
static CUresult (*g_test_execute_cmd)(uint32_t cmd);
static CUresult (*g_test_direct_source_lease_releaser)(uint64_t lease_handle);
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

typedef struct CXLDirectSourcePending {
    uint64_t source_id;
    uint64_t lease_handle;
    uint64_t stream_wire;
    struct CXLDirectSourcePending *next;
} CXLDirectSourcePending;

static CXLDirectSourcePending *g_direct_source_pending;

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
#define CUDART_KERNEL_RECORD_MAGIC 0x43584c4b45524e4cULL /* "CXLKERNL" */
#define CUDART_DIRECT_ELF_MAX_SIZE (64ULL * 1024ULL * 1024ULL)

typedef enum CudartLibraryCodeKind {
    CUDART_LIBRARY_CODE_FATBIN = 1,
    CUDART_LIBRARY_CODE_DIRECT_ELF = 2,
} CudartLibraryCodeKind;

struct CudartLibraryRecord;

typedef struct CudartKernelRecord {
    uint64_t magic;
    int alive;
    struct CudartLibraryRecord *library;
    CUfunction function;
    char *name;
    struct CudartKernelRecord *next;
} CudartKernelRecord;

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
    size_t preserved_code_size;
    CudartLibraryCodeKind code_kind;
    CUmodule module;
    CudartKernelRecord *kernels;
    CUlibraryOption options[CUDART_LIBRARY_RECORD_OPTION_CAP];
    void *option_values[CUDART_LIBRARY_RECORD_OPTION_CAP];
    struct CudartLibraryRecord *next;
} CudartLibraryRecord;

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuStreamSynchronize(CUstream hStream);
static CUresult cxl_module_load_image(CUmodule *module, const void *image);
static CUresult direct_source_errno_result(int error);
static CUresult direct_sources_complete_locked(uint64_t stream_wire,
                                               bool all_streams);

static CudartLibraryRecord *g_cudart_library_records = NULL;
static unsigned int g_cudart_library_next_id = 1;

typedef struct FunctionParamLayout {
    CUfunction function;
    uint32_t num_args;
    size_t extent;
    size_t offsets[CXL_MAX_KERNEL_ARGS];
    size_t sizes[CXL_MAX_KERNEL_ARGS];
} FunctionParamLayout;

#define FUNCTION_PARAM_LAYOUT_PAGE_BITS 16
#define FUNCTION_PARAM_LAYOUT_PAGE_SIZE (1U << FUNCTION_PARAM_LAYOUT_PAGE_BITS)
static FunctionParamLayout **g_function_param_layout_pages[FUNCTION_PARAM_LAYOUT_PAGE_SIZE];
static pthread_mutex_t g_function_param_layouts_lock = PTHREAD_MUTEX_INITIALIZER;
static void function_param_layouts_clear(const char *reason);
static CUresult function_param_layout_copy(CUfunction function,
                                           size_t offsets[CXL_MAX_KERNEL_ARGS],
                                           size_t sizes[CXL_MAX_KERNEL_ARGS],
                                           uint32_t *num_args, size_t *extent);

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

static CudartKernelRecord *cudart_kernel_record_from_handle(CUkernel kernel) {
    if (!kernel) {
        return NULL;
    }
    for (CudartLibraryRecord *library = g_cudart_library_records; library; library = library->next) {
        for (CudartKernelRecord *record = library->kernels; record; record = record->next) {
            if ((CUkernel)record == kernel && record->magic == CUDART_KERNEL_RECORD_MAGIC) {
                return record;
            }
        }
    }
    return NULL;
}

/* Debug logging */
static int g_debug = 0;
static FILE *g_observation_stream = NULL;
static char *g_observation_buffer = NULL;
#define DLOG(...)                                                                                                      \
    do {                                                                                                               \
        if (g_debug)                                                                                                   \
            fprintf(stderr, "[CXL-CUDA] " __VA_ARGS__);                                                                \
    } while (0)
#define OLOG(...)                                                                                                      \
    do {                                                                                                               \
        if (g_observation_stream)                                                                                      \
            fprintf(g_observation_stream, "[CXL-CUDA] " __VA_ARGS__);                                                \
        else if (g_debug)                                                                                              \
            fprintf(stderr, "[CXL-CUDA] " __VA_ARGS__);                                                              \
    } while (0)

#define CXL_OBSERVATION_CATEGORY_COUNT 8
#define CXL_OBSERVATION_PAIR_COUNT \
    ((CXL_OBSERVATION_CATEGORY_COUNT * (CXL_OBSERVATION_CATEGORY_COUNT - 1)) / 2)
#define CXL_OBSERVATION_PUBLIC_CALL 0
#define CXL_OBSERVATION_OPEN_TOKEN_COUNT 4096
#define CXL_OBSERVATION_GAP_BOUNDARY CXL_OBSERVATION_CATEGORY_COUNT
#define CXL_OBSERVATION_GAP_IDENTITY_COUNT (CXL_OBSERVATION_CATEGORY_COUNT + 1)
#define CXL_OBSERVATION_OPERATION_GAP_CAPACITY 1024
#define CXL_OBSERVATION_CATEGORY_INTERVAL_CAPACITY 65536

typedef struct CXLObservationIdentity {
    bool valid;
    uint64_t sequence;
    const char *owner;
    const char *category;
    const char *operation;
    uint32_t category_index;
    uint32_t command;
} CXLObservationIdentity;

typedef struct CXLObservationGapAggregate {
    uint64_t count;
    uint64_t total_duration_ns;
    uint64_t max_duration_ns;
} CXLObservationGapAggregate;

typedef struct CXLObservationOperationGapAggregate {
    bool occupied;
    CXLObservationIdentity previous;
    CXLObservationIdentity next;
    CXLObservationGapAggregate gap;
} CXLObservationOperationGapAggregate;

typedef struct CXLObservationAggregate {
    uint64_t interval_count;
    uint64_t total_duration_ns;
    uint64_t union_duration_ns;
    uint64_t active_begin_ns;
    uint64_t last_union_end_ns;
    uint32_t active_depth;
    bool have_last_union_end;
    uint64_t largest_gap_begin_ns;
    uint64_t largest_gap_end_ns;
    bool have_largest_gap;
    CXLObservationIdentity last_end;
    CXLObservationIdentity largest_gap_previous;
    CXLObservationIdentity largest_gap_next;
} CXLObservationAggregate;

typedef struct CXLObservationPairAggregate {
    uint64_t union_duration_ns;
    uint64_t active_begin_ns;
    bool active;
} CXLObservationPairAggregate;

typedef struct CXLObservationCategoryInterval {
    uint64_t begin_ns;
    uint64_t end_ns;
    uint32_t category;
} CXLObservationCategoryInterval;

typedef struct CXLObservationOpenToken {
    uint64_t token;
    uint64_t begin_ns;
    uint64_t graph_ordinal;
    uint64_t operation_sequence;
    uint32_t category;
    uint32_t command;
    CXLObservationIdentity identity;
} CXLObservationOpenToken;

typedef struct CXLObservationClockAnchor {
    bool attempted;
    uint32_t result;
    uint64_t guest_before_ns;
    uint64_t guest_after_ns;
    uint64_t host_monotonic_ns;
    uint64_t host_realtime_ns;
    uint64_t host_sample_uncertainty_ns;
} CXLObservationClockAnchor;

enum {
    CXL_OBSERVATION_BURST_1,
    CXL_OBSERVATION_BURST_2,
    CXL_OBSERVATION_BURST_3_TO_4,
    CXL_OBSERVATION_BURST_5_TO_8,
    CXL_OBSERVATION_BURST_9_TO_16,
    CXL_OBSERVATION_BURST_OVER_16,
    CXL_OBSERVATION_BURST_BUCKET_COUNT,
};

enum {
    CXL_OBSERVATION_BARRIER_CTX_SYNC,
    CXL_OBSERVATION_BARRIER_GRAPH_LAUNCH,
    CXL_OBSERVATION_BARRIER_KERNEL_LAUNCH,
    CXL_OBSERVATION_BARRIER_STREAM_SYNC,
    CXL_OBSERVATION_BARRIER_COUNT,
};

typedef struct CXLObservationCommandBarrierBursts {
    uint64_t current_fused_calls;
    uint64_t fused_calls;
    uint64_t terminated_bursts;
    uint64_t terminated_fused_calls;
    uint64_t max_burst;
    uint64_t histogram[CXL_OBSERVATION_BURST_BUCKET_COUNT];
    uint64_t barriers[CXL_OBSERVATION_BARRIER_COUNT];
} CXLObservationCommandBarrierBursts;

typedef struct CXLObservationLedger {
    pthread_mutex_t lock;
    bool active;
    bool incomplete;
    uint64_t case_epoch;
    uint64_t span_begin_ns;
    uint64_t last_clock_ns;
    uint64_t next_token;
    uint64_t next_sequence;
    uint32_t open_count;
    const char *first_error;
    CXLObservationAggregate categories[CXL_OBSERVATION_CATEGORY_COUNT];
    CXLObservationPairAggregate
        category_pairs[CXL_OBSERVATION_CATEGORY_COUNT]
                      [CXL_OBSERVATION_CATEGORY_COUNT];
    CXLObservationAggregate all_known;
    uint64_t command_calls[256];
    uint64_t command_total_duration_ns[256];
    uint64_t command_status_poll_count[256];
    CXLObservationCommandBarrierBursts command_barrier_bursts;
    CXLObservationGapAggregate
        all_known_gaps[CXL_OBSERVATION_GAP_IDENTITY_COUNT]
                      [CXL_OBSERVATION_GAP_IDENTITY_COUNT];
    CXLObservationOperationGapAggregate
        operation_gaps[CXL_OBSERVATION_OPERATION_GAP_CAPACITY];
    uint64_t category_interval_count;
    uint64_t stored_category_interval_count;
    bool category_intervals_overflow;
    CXLObservationCategoryInterval
        category_intervals[CXL_OBSERVATION_CATEGORY_INTERVAL_CAPACITY];
    CXLObservationClockAnchor clock_anchors[2];
    CXLObservationOpenToken open_tokens[CXL_OBSERVATION_OPEN_TOKEN_COUNT];
} CXLObservationLedger;

static CXLObservationLedger g_observation_ledger = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
};
static int g_observation_fast_active;

static const char *const g_observation_category_names[CXL_OBSERVATION_CATEGORY_COUNT] = {
    "cuda_public_call",
    "selected_range_plan",
    "source_lease",
    "source_materialization",
    "resident_lookup",
    "resident_to_compute",
    "cuda_graph_prepare",
    "host_result_read",
};

static uint64_t observation_clock_ns_locked(void) {
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static void observation_fail_locked(const char *error) {
    g_observation_ledger.incomplete = true;
    if (!g_observation_ledger.first_error)
        g_observation_ledger.first_error = error;
}

static bool observation_add_locked(uint64_t *value, uint64_t increment) {
    if (increment > UINT64_MAX - *value) {
        observation_fail_locked("duration-overflow");
        return false;
    }
    *value += increment;
    return true;
}

static int observation_barrier_index(uint32_t command) {
    switch (command) {
    case CXL_GPU_CMD_CTX_SYNC:
        return CXL_OBSERVATION_BARRIER_CTX_SYNC;
    case CXL_GPU_CMD_GRAPH_LAUNCH:
        return CXL_OBSERVATION_BARRIER_GRAPH_LAUNCH;
    case CXL_GPU_CMD_LAUNCH_KERNEL:
        return CXL_OBSERVATION_BARRIER_KERNEL_LAUNCH;
    case CXL_GPU_CMD_STREAM_SYNC:
        return CXL_OBSERVATION_BARRIER_STREAM_SYNC;
    default:
        return -1;
    }
}

static uint32_t observation_burst_bucket(uint64_t calls) {
    if (calls == 1)
        return CXL_OBSERVATION_BURST_1;
    if (calls == 2)
        return CXL_OBSERVATION_BURST_2;
    if (calls <= 4)
        return CXL_OBSERVATION_BURST_3_TO_4;
    if (calls <= 8)
        return CXL_OBSERVATION_BURST_5_TO_8;
    if (calls <= 16)
        return CXL_OBSERVATION_BURST_9_TO_16;
    return CXL_OBSERVATION_BURST_OVER_16;
}

static void observation_command_barrier_record_locked(uint32_t command) {
    CXLObservationCommandBarrierBursts *bursts =
        &g_observation_ledger.command_barrier_bursts;

    if (command == CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC) {
        observation_add_locked(&bursts->current_fused_calls, 1);
        observation_add_locked(&bursts->fused_calls, 1);
        return;
    }

    const int barrier = observation_barrier_index(command);
    if (barrier < 0 || bursts->current_fused_calls == 0)
        return;

    const uint64_t calls = bursts->current_fused_calls;
    observation_add_locked(&bursts->terminated_bursts, 1);
    observation_add_locked(&bursts->terminated_fused_calls, calls);
    observation_add_locked(&bursts->histogram[observation_burst_bucket(calls)], 1);
    observation_add_locked(&bursts->barriers[barrier], 1);
    if (calls > bursts->max_burst)
        bursts->max_burst = calls;
    bursts->current_fused_calls = 0;
}

static void observation_update_category_pairs_locked(uint32_t category,
                                                     uint64_t now_ns) {
    for (uint32_t other = 0; other < CXL_OBSERVATION_CATEGORY_COUNT; other++) {
        if (other == category)
            continue;
        const uint32_t category_a = category < other ? category : other;
        const uint32_t category_b = category < other ? other : category;
        CXLObservationPairAggregate *pair =
            &g_observation_ledger.category_pairs[category_a][category_b];
        const bool active =
            g_observation_ledger.categories[category_a].active_depth != 0 &&
            g_observation_ledger.categories[category_b].active_depth != 0;
        if (active == pair->active)
            continue;
        if (active) {
            pair->active = true;
            pair->active_begin_ns = now_ns;
            continue;
        }
        if (now_ns < pair->active_begin_ns) {
            observation_fail_locked("pair-clock-reversal");
        } else {
            observation_add_locked(&pair->union_duration_ns,
                                   now_ns - pair->active_begin_ns);
        }
        pair->active = false;
    }
}

static uint32_t observation_gap_identity_index_locked(
    CXLObservationIdentity identity) {
    if (!identity.valid)
        return CXL_OBSERVATION_GAP_BOUNDARY;
    if (identity.category_index >= CXL_OBSERVATION_CATEGORY_COUNT) {
        observation_fail_locked("gap-category-out-of-range");
        return CXL_OBSERVATION_GAP_BOUNDARY;
    }
    return identity.category_index;
}

static uint64_t observation_gap_identity_hash(CXLObservationIdentity identity) {
    uint64_t hash = UINT64_C(1469598103934665603);

    hash ^= identity.valid ? 1 : 0;
    hash *= UINT64_C(1099511628211);
    if (!identity.valid)
        return hash;
    hash ^= identity.category_index;
    hash *= UINT64_C(1099511628211);
    hash ^= identity.command;
    hash *= UINT64_C(1099511628211);
    for (const unsigned char *cursor = (const unsigned char *)identity.operation;
         cursor && *cursor; cursor++) {
        hash ^= *cursor;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool observation_gap_identity_equal(CXLObservationIdentity left,
                                           CXLObservationIdentity right) {
    if (left.valid != right.valid)
        return false;
    if (!left.valid)
        return true;
    return left.category_index == right.category_index &&
           left.command == right.command &&
           strcmp(left.operation, right.operation) == 0;
}

static void observation_record_operation_gap_locked(
    uint64_t duration_ns, CXLObservationIdentity previous,
    CXLObservationIdentity next) {
    uint64_t hash = observation_gap_identity_hash(previous);
    hash ^= observation_gap_identity_hash(next) + UINT64_C(0x9e3779b97f4a7c15) +
            (hash << 6) + (hash >> 2);

    for (uint32_t probe = 0;
         probe < CXL_OBSERVATION_OPERATION_GAP_CAPACITY; probe++) {
        CXLObservationOperationGapAggregate *entry =
            &g_observation_ledger.operation_gaps[
                (hash + probe) & (CXL_OBSERVATION_OPERATION_GAP_CAPACITY - 1)];
        if (!entry->occupied) {
            entry->occupied = true;
            entry->previous = previous;
            entry->next = next;
        } else if (!observation_gap_identity_equal(entry->previous, previous) ||
                   !observation_gap_identity_equal(entry->next, next)) {
            continue;
        }
        if (entry->gap.count == UINT64_MAX) {
            observation_fail_locked("operation-gap-count-overflow");
            return;
        }
        entry->gap.count++;
        if (!observation_add_locked(&entry->gap.total_duration_ns, duration_ns))
            return;
        if (duration_ns > entry->gap.max_duration_ns)
            entry->gap.max_duration_ns = duration_ns;
        return;
    }
    observation_fail_locked("operation-gap-capacity");
}

static void observation_record_all_known_gap_locked(
    uint64_t begin_ns, uint64_t end_ns, CXLObservationIdentity previous,
    CXLObservationIdentity next) {
    if (end_ns <= begin_ns)
        return;
    const uint32_t previous_index =
        observation_gap_identity_index_locked(previous);
    const uint32_t next_index = observation_gap_identity_index_locked(next);
    CXLObservationGapAggregate *gap =
        &g_observation_ledger.all_known_gaps[previous_index][next_index];
    if (gap->count == UINT64_MAX) {
        observation_fail_locked("gap-count-overflow");
        return;
    }
    gap->count++;
    const uint64_t duration_ns = end_ns - begin_ns;
    if (!observation_add_locked(&gap->total_duration_ns, duration_ns))
        return;
    if (duration_ns > gap->max_duration_ns)
        gap->max_duration_ns = duration_ns;
    observation_record_operation_gap_locked(duration_ns, previous, next);
}

static void observation_consider_gap_locked(CXLObservationAggregate *aggregate,
                                            uint64_t begin_ns, uint64_t end_ns,
                                            CXLObservationIdentity previous,
                                            CXLObservationIdentity next) {
    if (end_ns < begin_ns) {
        observation_fail_locked("gap-boundary");
        return;
    }
    if (!aggregate->have_largest_gap ||
        end_ns - begin_ns > aggregate->largest_gap_end_ns - aggregate->largest_gap_begin_ns) {
        aggregate->have_largest_gap = true;
        aggregate->largest_gap_begin_ns = begin_ns;
        aggregate->largest_gap_end_ns = end_ns;
        aggregate->largest_gap_previous = previous;
        aggregate->largest_gap_next = next;
    }
}

static void observation_aggregate_begin_locked(CXLObservationAggregate *aggregate,
                                               uint64_t now_ns,
                                               CXLObservationIdentity identity) {
    if (aggregate->active_depth == 0) {
        const uint64_t gap_begin = aggregate->have_last_union_end
            ? aggregate->last_union_end_ns : g_observation_ledger.span_begin_ns;
        observation_consider_gap_locked(aggregate, gap_begin, now_ns,
                                        aggregate->last_end, identity);
        if (aggregate == &g_observation_ledger.all_known)
            observation_record_all_known_gap_locked(
                gap_begin, now_ns, aggregate->last_end, identity);
        aggregate->active_begin_ns = now_ns;
    }
    if (aggregate->active_depth == UINT32_MAX) {
        observation_fail_locked("active-depth-overflow");
        return;
    }
    aggregate->active_depth++;
    observation_add_locked(&aggregate->interval_count, 1);
}

static void observation_aggregate_end_locked(CXLObservationAggregate *aggregate,
                                             uint64_t begin_ns, uint64_t end_ns,
                                             CXLObservationIdentity identity) {
    if (aggregate->active_depth == 0) {
        observation_fail_locked("active-depth-underflow");
        return;
    }
    if (end_ns < begin_ns) {
        observation_fail_locked("interval-clock-reversal");
        end_ns = begin_ns;
    }
    observation_add_locked(&aggregate->total_duration_ns, end_ns - begin_ns);
    aggregate->active_depth--;
    if (aggregate->active_depth == 0) {
        if (end_ns < aggregate->active_begin_ns) {
            observation_fail_locked("union-clock-reversal");
        } else {
            observation_add_locked(&aggregate->union_duration_ns,
                                   end_ns - aggregate->active_begin_ns);
        }
        aggregate->last_union_end_ns = end_ns;
        aggregate->have_last_union_end = true;
        aggregate->last_end = identity;
    }
}

static void observation_record_category_interval_locked(uint32_t category,
                                                        uint64_t begin_ns,
                                                        uint64_t end_ns) {
    g_observation_ledger.category_interval_count++;
    if (g_observation_ledger.category_intervals_overflow)
        return;
    if (g_observation_ledger.stored_category_interval_count >=
        CXL_OBSERVATION_CATEGORY_INTERVAL_CAPACITY) {
        g_observation_ledger.category_intervals_overflow = true;
        return;
    }
    CXLObservationCategoryInterval *interval =
        &g_observation_ledger.category_intervals[
            g_observation_ledger.stored_category_interval_count++];
    interval->begin_ns = begin_ns;
    interval->end_ns = end_ns;
    interval->category = category;
}

static uint64_t observation_span_begin_locked(uint32_t category,
                                              uint64_t graph_ordinal,
                                              uint64_t operation_sequence,
                                              uint32_t command,
                                              const char *owner,
                                              const char *operation,
                                              uint64_t *begin_ns) {
    if (!g_observation_ledger.active)
        return 0;
    if (category >= CXL_OBSERVATION_CATEGORY_COUNT) {
        observation_fail_locked("unknown-category");
        return 0;
    }

    uint64_t now_ns = observation_clock_ns_locked();
    if (!now_ns) {
        observation_fail_locked("clock-read");
        return 0;
    }
    if (now_ns < g_observation_ledger.span_begin_ns ||
        now_ns < g_observation_ledger.last_clock_ns) {
        observation_fail_locked("clock-reversal");
        return 0;
    }
    g_observation_ledger.last_clock_ns = now_ns;

    if (g_observation_ledger.next_token == UINT64_MAX) {
        observation_fail_locked("token-overflow");
        return 0;
    }
    const uint64_t token = ++g_observation_ledger.next_token;
    CXLObservationOpenToken *open =
        &g_observation_ledger.open_tokens[token % CXL_OBSERVATION_OPEN_TOKEN_COUNT];
    if (open->token != 0) {
        observation_fail_locked("open-token-capacity");
        return 0;
    }
    if (g_observation_ledger.next_sequence == UINT64_MAX) {
        observation_fail_locked("sequence-overflow");
        return 0;
    }
    CXLObservationIdentity identity = {
        .valid = true,
        .sequence = ++g_observation_ledger.next_sequence,
        .owner = owner,
        .category = g_observation_category_names[category],
        .operation = operation,
        .category_index = category,
        .command = command,
    };
    *open = (CXLObservationOpenToken) {
        .token = token,
        .begin_ns = now_ns,
        .graph_ordinal = graph_ordinal,
        .operation_sequence = operation_sequence,
        .category = category,
        .command = command,
        .identity = identity,
    };
    g_observation_ledger.open_count++;
    CXLObservationAggregate *category_aggregate =
        &g_observation_ledger.categories[category];
    const bool category_became_active = category_aggregate->active_depth == 0;
    observation_aggregate_begin_locked(category_aggregate, now_ns, identity);
    if (category_became_active && category_aggregate->active_depth != 0)
        observation_update_category_pairs_locked(category, now_ns);
    observation_aggregate_begin_locked(&g_observation_ledger.all_known,
                                       now_ns, identity);
    if (begin_ns)
        *begin_ns = now_ns;
    return token;
}

static CUresult observation_span_end_locked(uint64_t token,
                                            int32_t operation_status,
                                            uint64_t *end_ns) {
    (void)operation_status;
    if (!g_observation_ledger.active || token == 0) {
        observation_fail_locked("span-end-without-decode");
        return CUDA_ERROR_INVALID_VALUE;
    }
    CXLObservationOpenToken *open =
        &g_observation_ledger.open_tokens[token % CXL_OBSERVATION_OPEN_TOKEN_COUNT];
    if (open->token != token) {
        observation_fail_locked("span-pairing");
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint64_t now_ns = observation_clock_ns_locked();
    if (!now_ns) {
        observation_fail_locked("clock-read");
        return CUDA_ERROR_UNKNOWN;
    }
    if (now_ns < g_observation_ledger.last_clock_ns)
        observation_fail_locked("clock-reversal");
    g_observation_ledger.last_clock_ns = now_ns;
    CXLObservationAggregate *category_aggregate =
        &g_observation_ledger.categories[open->category];
    const bool category_will_become_inactive = category_aggregate->active_depth == 1;
    observation_aggregate_end_locked(category_aggregate, open->begin_ns,
                                     now_ns, open->identity);
    if (category_will_become_inactive && category_aggregate->active_depth == 0) {
        if (open->category != CXL_OBSERVATION_PUBLIC_CALL)
            observation_record_category_interval_locked(
                open->category, category_aggregate->active_begin_ns, now_ns);
        observation_update_category_pairs_locked(open->category, now_ns);
    }
    observation_aggregate_end_locked(&g_observation_ledger.all_known,
                                     open->begin_ns, now_ns, open->identity);
    if (open->category == CXL_OBSERVATION_PUBLIC_CALL && open->command < 256) {
        observation_add_locked(
            &g_observation_ledger.command_calls[open->command], 1);
        observation_add_locked(
            &g_observation_ledger.command_total_duration_ns[open->command],
            now_ns - open->begin_ns);
    }
    memset(open, 0, sizeof(*open));
    g_observation_ledger.open_count--;
    if (end_ns)
        *end_ns = now_ns;
    return CUDA_SUCCESS;
}

static const char *observation_identity_field(CXLObservationIdentity identity,
                                              const char *field) {
    if (!identity.valid)
        return "null";
    if (strcmp(field, "owner") == 0)
        return identity.owner;
    if (strcmp(field, "category") == 0)
        return identity.category;
    return identity.operation;
}

static void observation_emit_summary_locked(const char *category,
                                            const char *owner,
                                            CXLObservationAggregate *aggregate,
                                            uint64_t span_end_ns) {
    CXLObservationIdentity empty = {0};
    if (aggregate->active_depth != 0)
        observation_fail_locked("open-span-at-terminal");
    if (aggregate->active_depth == 0) {
        const uint64_t gap_begin = aggregate->have_last_union_end
            ? aggregate->last_union_end_ns : g_observation_ledger.span_begin_ns;
        observation_consider_gap_locked(aggregate, gap_begin, span_end_ns,
                                        aggregate->last_end, empty);
        if (aggregate == &g_observation_ledger.all_known)
            observation_record_all_known_gap_locked(
                gap_begin, span_end_ns, aggregate->last_end, empty);
    }

    const uint64_t span_duration = span_end_ns >= g_observation_ledger.span_begin_ns
        ? span_end_ns - g_observation_ledger.span_begin_ns : 0;
    if (span_end_ns < g_observation_ledger.span_begin_ns)
        observation_fail_locked("terminal-boundary");
    if (aggregate->union_duration_ns > span_duration)
        observation_fail_locked("union-outside-span");
    if (aggregate->total_duration_ns < aggregate->union_duration_ns)
        observation_fail_locked("overlap-underflow");
    const uint64_t overlap = aggregate->total_duration_ns >= aggregate->union_duration_ns
        ? aggregate->total_duration_ns - aggregate->union_duration_ns : 0;
    const uint64_t gap = span_duration >= aggregate->union_duration_ns
        ? span_duration - aggregate->union_duration_ns : 0;
    if (aggregate == &g_observation_ledger.all_known) {
        uint64_t recorded_gap_ns = 0;
        uint64_t recorded_operation_gap_ns = 0;
        for (uint32_t previous = 0;
             previous < CXL_OBSERVATION_GAP_IDENTITY_COUNT; previous++) {
            for (uint32_t next = 0;
                 next < CXL_OBSERVATION_GAP_IDENTITY_COUNT; next++) {
                observation_add_locked(
                    &recorded_gap_ns,
                    g_observation_ledger.all_known_gaps[previous][next]
                        .total_duration_ns);
            }
        }
        if (recorded_gap_ns != gap)
            observation_fail_locked("gap-total-mismatch");
        for (uint32_t index = 0;
             index < CXL_OBSERVATION_OPERATION_GAP_CAPACITY; index++) {
            if (g_observation_ledger.operation_gaps[index].occupied)
                observation_add_locked(
                    &recorded_operation_gap_ns,
                    g_observation_ledger.operation_gaps[index]
                        .gap.total_duration_ns);
        }
        if (recorded_operation_gap_ns != gap)
            observation_fail_locked("operation-gap-total-mismatch");
    }

    fprintf(stderr,
            "[CXL-CUDA] interval_summary schema=interval-summary-v1 producer=guest-shim"
            " clock_domain=guest-monotonic case_epoch=%" PRIu64
            " scope=decode category=%s owner=%s status=%s"
            " span_begin_ns=%" PRIu64 " span_end_ns=%" PRIu64
            " interval_count=%" PRIu64 " total_duration_ns=%" PRIu64
            " union_duration_ns=%" PRIu64 " overlap_duration_ns=%" PRIu64
            " gap_duration_ns=%" PRIu64,
            g_observation_ledger.case_epoch, category, owner,
            g_observation_ledger.incomplete ? "incomplete" : "complete",
            g_observation_ledger.span_begin_ns, span_end_ns,
            aggregate->interval_count, aggregate->total_duration_ns,
            aggregate->union_duration_ns, overlap, gap);
    if (aggregate->have_largest_gap) {
        fprintf(stderr,
                " largest_gap_begin_ns=%" PRIu64 " largest_gap_end_ns=%" PRIu64,
                aggregate->largest_gap_begin_ns, aggregate->largest_gap_end_ns);
    } else {
        fputs(" largest_gap_begin_ns=null largest_gap_end_ns=null", stderr);
    }
    if (aggregate->largest_gap_previous.valid)
        fprintf(stderr, " previous_sequence=%" PRIu64,
                aggregate->largest_gap_previous.sequence);
    else
        fputs(" previous_sequence=null", stderr);
    fprintf(stderr,
            " previous_owner=%s previous_category=%s previous_operation=%s",
            observation_identity_field(aggregate->largest_gap_previous, "owner"),
            observation_identity_field(aggregate->largest_gap_previous, "category"),
            observation_identity_field(aggregate->largest_gap_previous, "operation"));
    if (aggregate->largest_gap_next.valid)
        fprintf(stderr, " next_sequence=%" PRIu64,
                aggregate->largest_gap_next.sequence);
    else
        fputs(" next_sequence=null", stderr);
    fprintf(stderr,
            " next_owner=%s next_category=%s next_operation=%s first_error=%s\n",
            observation_identity_field(aggregate->largest_gap_next, "owner"),
            observation_identity_field(aggregate->largest_gap_next, "category"),
            observation_identity_field(aggregate->largest_gap_next, "operation"),
            g_observation_ledger.first_error
                ? g_observation_ledger.first_error : "none");
}

static const char *observation_gap_category_name(uint32_t index) {
    return index == CXL_OBSERVATION_GAP_BOUNDARY
        ? "decode_boundary" : g_observation_category_names[index];
}

static void observation_emit_gap_summaries_locked(void) {
    for (uint32_t previous = 0;
         previous < CXL_OBSERVATION_GAP_IDENTITY_COUNT; previous++) {
        for (uint32_t next = 0;
             next < CXL_OBSERVATION_GAP_IDENTITY_COUNT; next++) {
            const CXLObservationGapAggregate *gap =
                &g_observation_ledger.all_known_gaps[previous][next];
            if (gap->count == 0)
                continue;
            fprintf(stderr,
                    "[CXL-CUDA] gap_summary schema=gap-summary-v1"
                    " producer=guest-shim clock_domain=guest-monotonic"
                    " case_epoch=%" PRIu64 " scope=decode"
                    " previous_category=%s next_category=%s"
                    " gap_count=%" PRIu64 " total_duration_ns=%" PRIu64
                    " max_duration_ns=%" PRIu64 "\n",
                    g_observation_ledger.case_epoch,
                    observation_gap_category_name(previous),
                    observation_gap_category_name(next), gap->count,
                    gap->total_duration_ns, gap->max_duration_ns);
        }
    }
    for (uint32_t index = 0;
         index < CXL_OBSERVATION_OPERATION_GAP_CAPACITY; index++) {
        const CXLObservationOperationGapAggregate *entry =
            &g_observation_ledger.operation_gaps[index];
        if (!entry->occupied)
            continue;
        fprintf(stderr,
                "[CXL-CUDA] operation_gap_summary"
                " schema=operation-gap-summary-v1 producer=guest-shim"
                " clock_domain=guest-monotonic case_epoch=%" PRIu64
                " scope=decode previous_category=%s previous_operation=%s",
                g_observation_ledger.case_epoch,
                entry->previous.valid ? entry->previous.category
                                      : "decode_boundary",
                entry->previous.valid ? entry->previous.operation
                                      : "decode_boundary");
        if (entry->previous.valid &&
            entry->previous.category_index == CXL_OBSERVATION_PUBLIC_CALL)
            fprintf(stderr, " previous_command=0x%x", entry->previous.command);
        else
            fputs(" previous_command=null", stderr);
        fprintf(stderr, " next_category=%s next_operation=%s",
                entry->next.valid ? entry->next.category : "decode_boundary",
                entry->next.valid ? entry->next.operation : "decode_boundary");
        if (entry->next.valid &&
            entry->next.category_index == CXL_OBSERVATION_PUBLIC_CALL)
            fprintf(stderr, " next_command=0x%x", entry->next.command);
        else
            fputs(" next_command=null", stderr);
        fprintf(stderr,
                " gap_count=%" PRIu64 " total_duration_ns=%" PRIu64
                " max_duration_ns=%" PRIu64 "\n",
                entry->gap.count, entry->gap.total_duration_ns,
                entry->gap.max_duration_ns);
    }
}

static void observation_validate_terminal_gaps_locked(uint64_t span_end_ns) {
    const CXLObservationAggregate *aggregate = &g_observation_ledger.all_known;
    if (span_end_ns < g_observation_ledger.span_begin_ns ||
        aggregate->union_duration_ns >
            span_end_ns - g_observation_ledger.span_begin_ns) {
        observation_fail_locked("gap-terminal-boundary");
        return;
    }
    uint64_t recorded_gap_ns = 0;
    for (uint32_t previous = 0;
         previous < CXL_OBSERVATION_GAP_IDENTITY_COUNT; previous++) {
        for (uint32_t next = 0;
             next < CXL_OBSERVATION_GAP_IDENTITY_COUNT; next++) {
            observation_add_locked(
                &recorded_gap_ns,
                g_observation_ledger.all_known_gaps[previous][next]
                    .total_duration_ns);
        }
    }
    const uint64_t final_gap_begin = aggregate->have_last_union_end
        ? aggregate->last_union_end_ns : g_observation_ledger.span_begin_ns;
    if (span_end_ns < final_gap_begin ||
        !observation_add_locked(&recorded_gap_ns,
                                span_end_ns - final_gap_begin)) {
        observation_fail_locked("gap-terminal-boundary");
        return;
    }
    const uint64_t expected_gap_ns =
        span_end_ns - g_observation_ledger.span_begin_ns -
        aggregate->union_duration_ns;
    if (recorded_gap_ns != expected_gap_ns)
        observation_fail_locked("gap-total-mismatch");
}

static void observation_validate_command_totals_locked(void) {
    uint64_t calls = 0;
    uint64_t duration_ns = 0;

    for (uint32_t command = 0; command < 256; command++) {
        if (!observation_add_locked(
                &calls, g_observation_ledger.command_calls[command]) ||
            !observation_add_locked(
                &duration_ns,
                g_observation_ledger.command_total_duration_ns[command]))
            return;
    }
    if (calls != g_observation_ledger.categories[CXL_OBSERVATION_PUBLIC_CALL]
                     .interval_count ||
        duration_ns !=
            g_observation_ledger.categories[CXL_OBSERVATION_PUBLIC_CALL]
                .total_duration_ns)
        observation_fail_locked("command-total-mismatch");
}

static void observation_validate_category_pairs_locked(void) {
    uint32_t pair_count = 0;

    for (uint32_t category_a = 0;
         category_a < CXL_OBSERVATION_CATEGORY_COUNT; category_a++) {
        for (uint32_t category_b = category_a + 1;
             category_b < CXL_OBSERVATION_CATEGORY_COUNT; category_b++) {
            const CXLObservationPairAggregate *pair =
                &g_observation_ledger.category_pairs[category_a][category_b];
            pair_count++;
            if (pair->active) {
                observation_fail_locked("pair-active-at-terminal");
                continue;
            }
            if (pair->union_duration_ns >
                    g_observation_ledger.categories[category_a].union_duration_ns ||
                pair->union_duration_ns >
                    g_observation_ledger.categories[category_b].union_duration_ns)
                observation_fail_locked("pair-union-exceeds-category");
        }
    }
    if (pair_count != CXL_OBSERVATION_PAIR_COUNT)
        observation_fail_locked("pair-count-mismatch");
}

static void observation_emit_category_pairs_locked(uint64_t span_end_ns) {
    for (uint32_t category_a = 0;
         category_a < CXL_OBSERVATION_CATEGORY_COUNT; category_a++) {
        for (uint32_t category_b = category_a + 1;
             category_b < CXL_OBSERVATION_CATEGORY_COUNT; category_b++) {
            const uint64_t category_a_union_ns =
                g_observation_ledger.categories[category_a].union_duration_ns;
            const uint64_t category_b_union_ns =
                g_observation_ledger.categories[category_b].union_duration_ns;
            const uint64_t intersection_union_ns =
                g_observation_ledger.category_pairs[category_a][category_b]
                    .union_duration_ns;
            fprintf(stderr,
                    "[CXL-CUDA] interval_pair_summary"
                    " schema=interval-pair-summary-v1 producer=guest-shim"
                    " clock_domain=guest-monotonic case_epoch=%" PRIu64
                    " scope=decode category_a=%s category_b=%s status=%s"
                    " span_begin_ns=%" PRIu64 " span_end_ns=%" PRIu64
                    " intersection_union_ns=%" PRIu64
                    " category_a_union_ns=%" PRIu64
                    " category_b_union_ns=%" PRIu64
                    " category_a_exclusive_ns=%" PRIu64
                    " category_b_exclusive_ns=%" PRIu64
                    " first_error=%s\n",
                    g_observation_ledger.case_epoch,
                    g_observation_category_names[category_a],
                    g_observation_category_names[category_b],
                    g_observation_ledger.incomplete ? "incomplete" : "complete",
                    g_observation_ledger.span_begin_ns, span_end_ns,
                    intersection_union_ns, category_a_union_ns,
                    category_b_union_ns,
                    category_a_union_ns - intersection_union_ns,
                    category_b_union_ns - intersection_union_ns,
                    g_observation_ledger.first_error
                        ? g_observation_ledger.first_error : "none");
        }
    }
}

static void observation_emit_category_intervals_locked(uint64_t span_end_ns) {
    const bool available = !g_observation_ledger.category_intervals_overflow;

    if (available) {
        uint64_t category_duration_ns[CXL_OBSERVATION_CATEGORY_COUNT] = {0};
        for (uint64_t index = 0;
             index < g_observation_ledger.stored_category_interval_count;
             index++) {
            const CXLObservationCategoryInterval *interval =
                &g_observation_ledger.category_intervals[index];
            if (interval->category == CXL_OBSERVATION_PUBLIC_CALL ||
                interval->category >= CXL_OBSERVATION_CATEGORY_COUNT ||
                interval->begin_ns < g_observation_ledger.span_begin_ns ||
                interval->end_ns < interval->begin_ns ||
                interval->end_ns > span_end_ns) {
                observation_fail_locked("category-interval-invalid");
                continue;
            }
            observation_add_locked(&category_duration_ns[interval->category],
                                   interval->end_ns - interval->begin_ns);
            fprintf(stderr,
                    "[CXL-CUDA] category_interval"
                    " schema=category-interval-v1 producer=guest-shim"
                    " clock_domain=guest-monotonic case_epoch=%" PRIu64
                    " scope=decode sequence=%" PRIu64 " category=%s"
                    " begin_ns=%" PRIu64 " end_ns=%" PRIu64 "\n",
                    g_observation_ledger.case_epoch, index + 1,
                    g_observation_category_names[interval->category],
                    interval->begin_ns, interval->end_ns);
        }
        for (uint32_t category = CXL_CUDA_OBS_SELECTED_RANGE_PLAN;
             category <= CXL_CUDA_OBS_HOST_RESULT_READ; category++) {
            if (category_duration_ns[category] !=
                g_observation_ledger.categories[category].union_duration_ns)
                observation_fail_locked("category-interval-union-mismatch");
        }
    }

    fprintf(stderr,
            "[CXL-CUDA] category_interval_terminal"
            " schema=category-interval-terminal-v1 producer=guest-shim"
            " clock_domain=guest-monotonic case_epoch=%" PRIu64
            " scope=decode status=%s span_begin_ns=%" PRIu64
            " span_end_ns=%" PRIu64 " interval_count=%" PRIu64
            " emitted_count=%" PRIu64 " capacity=%u reason=%s\n",
            g_observation_ledger.case_epoch,
            available ? "available" : "unavailable",
            g_observation_ledger.span_begin_ns, span_end_ns,
            g_observation_ledger.category_interval_count,
            available ? g_observation_ledger.stored_category_interval_count : 0,
            CXL_OBSERVATION_CATEGORY_INTERVAL_CAPACITY,
            available ? "none" : "capacity-exceeded");
}

static const char *observation_anchor_phase_name(uint32_t phase) {
    return phase == CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN ? "begin" : "end";
}

static void observation_capture_clock_anchor_locked(uint32_t phase) {
    const uint32_t index = phase == CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN ? 0 : 1;
    CXLObservationClockAnchor *anchor = &g_observation_ledger.clock_anchors[index];

    anchor->attempted = true;
    anchor->result = CXL_GPU_ERROR_UNKNOWN;
    anchor->guest_before_ns = observation_clock_ns_locked();
    if (!anchor->guest_before_ns || !g_transport.descriptor)
        goto complete;
    if (cxl_gpu_transport_lock(&g_transport) != 0)
        goto complete;
    cxl_gpu_transport_write64(&g_transport, CXL_GPU_REG_PARAM0,
                              CXL_GPU_OBSERVATION_ANCHOR_VERSION);
    cxl_gpu_transport_write64(&g_transport, CXL_GPU_REG_PARAM1, phase);
    cxl_gpu_transport_write64(&g_transport, CXL_GPU_REG_PARAM2,
                              g_observation_ledger.case_epoch);
    anchor->result = cxl_gpu_transport_execute(
        &g_transport, CXL_GPU_CMD_OBSERVATION_ANCHOR, NULL);
    if (anchor->result == CXL_GPU_SUCCESS) {
        anchor->host_monotonic_ns = cxl_gpu_transport_read64(
            &g_transport, CXL_GPU_REG_RESULT0);
        anchor->host_realtime_ns = cxl_gpu_transport_read64(
            &g_transport, CXL_GPU_REG_RESULT1);
        anchor->host_sample_uncertainty_ns = cxl_gpu_transport_read64(
            &g_transport, CXL_GPU_REG_RESULT2);
    }
    if (cxl_gpu_transport_unlock(&g_transport) != 0)
        anchor->result = CXL_GPU_ERROR_UNKNOWN;

complete:
    anchor->guest_after_ns = observation_clock_ns_locked();
}

static void observation_emit_clock_anchors_locked(void) {
    for (uint32_t index = 0; index < 2; index++) {
        const CXLObservationClockAnchor *anchor =
            &g_observation_ledger.clock_anchors[index];
        const uint32_t phase = index == 0
            ? CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN
            : CXL_GPU_OBSERVATION_ANCHOR_DECODE_END;
        const bool available =
            anchor->attempted && anchor->result == CXL_GPU_SUCCESS &&
            anchor->guest_before_ns && anchor->guest_after_ns &&
            anchor->guest_after_ns >= anchor->guest_before_ns &&
            anchor->host_monotonic_ns && anchor->host_realtime_ns;

        fprintf(stderr,
                "[CXL-CUDA] clock_anchor schema=observation-anchor-v1"
                " producer=guest-shim case_epoch=%" PRIu64
                " scope=decode phase=%s status=%s result=%u"
                " guest_before_ns=%" PRIu64 " guest_after_ns=%" PRIu64
                " host_monotonic_ns=%" PRIu64 " host_realtime_ns=%" PRIu64
                " host_sample_uncertainty_ns=%" PRIu64 " reason=%s\n",
                g_observation_ledger.case_epoch,
                observation_anchor_phase_name(phase),
                available ? "available" : "unavailable", anchor->result,
                anchor->guest_before_ns, anchor->guest_after_ns,
                anchor->host_monotonic_ns, anchor->host_realtime_ns,
                anchor->host_sample_uncertainty_ns,
                available ? "none" :
                    (!g_transport.descriptor ? "transport-unavailable" :
                     "anchor-command-failed"));
    }
}

static void observation_emit_command_barrier_bursts_locked(
    bool decode_terminal_observed) {
    const CXLObservationCommandBarrierBursts *bursts =
        &g_observation_ledger.command_barrier_bursts;

    fprintf(stderr,
            "[CXL-CUDA] command_barrier_burst_summary"
            " schema=guest-command-barrier-burst-summary-v1"
            " producer=guest-shim order_domain=guest-bar2-command"
            " case_epoch=%" PRIu64 " scope=decode status=%s"
            " source_command=0x%x fused_calls=%" PRIu64
            " terminated_bursts=%" PRIu64
            " terminated_fused_calls=%" PRIu64
            " trailing_fused_calls=%" PRIu64
            " max_terminated_burst=%" PRIu64
            " singleton_bursts=%" PRIu64 " multi_bursts=%" PRIu64
            " multi_fused_calls=%" PRIu64
            " burst_1=%" PRIu64 " burst_2=%" PRIu64
            " burst_3_4=%" PRIu64 " burst_5_8=%" PRIu64
            " burst_9_16=%" PRIu64 " burst_gt_16=%" PRIu64
            " ctx_sync_bursts=%" PRIu64
            " graph_launch_bursts=%" PRIu64
            " kernel_launch_bursts=%" PRIu64
            " stream_sync_bursts=%" PRIu64 " reason=%s\n",
            g_observation_ledger.case_epoch,
            decode_terminal_observed ? "complete" : "incomplete",
            CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC,
            bursts->fused_calls, bursts->terminated_bursts,
            bursts->terminated_fused_calls, bursts->current_fused_calls,
            bursts->max_burst,
            bursts->histogram[CXL_OBSERVATION_BURST_1],
            bursts->terminated_bursts
                - bursts->histogram[CXL_OBSERVATION_BURST_1],
            bursts->terminated_fused_calls
                - bursts->histogram[CXL_OBSERVATION_BURST_1],
            bursts->histogram[CXL_OBSERVATION_BURST_1],
            bursts->histogram[CXL_OBSERVATION_BURST_2],
            bursts->histogram[CXL_OBSERVATION_BURST_3_TO_4],
            bursts->histogram[CXL_OBSERVATION_BURST_5_TO_8],
            bursts->histogram[CXL_OBSERVATION_BURST_9_TO_16],
            bursts->histogram[CXL_OBSERVATION_BURST_OVER_16],
            bursts->barriers[CXL_OBSERVATION_BARRIER_CTX_SYNC],
            bursts->barriers[CXL_OBSERVATION_BARRIER_GRAPH_LAUNCH],
            bursts->barriers[CXL_OBSERVATION_BARRIER_KERNEL_LAUNCH],
            bursts->barriers[CXL_OBSERVATION_BARRIER_STREAM_SYNC],
            decode_terminal_observed ? "none" : "decode-terminal-missing");
}

static void observation_emit_terminal_locked(uint64_t span_end_ns,
                                             bool decode_terminal_observed) {
    observation_validate_terminal_gaps_locked(span_end_ns);
    observation_validate_command_totals_locked();
    observation_validate_category_pairs_locked();
    for (uint32_t category = 0;
         category <= CXL_CUDA_OBS_HOST_RESULT_READ; category++) {
        observation_emit_summary_locked(g_observation_category_names[category],
                                        category == 0 ? "guest-shim" : "llama",
                                        &g_observation_ledger.categories[category],
                                        span_end_ns);
    }
    observation_emit_summary_locked("all_known", "guest-shim",
                                    &g_observation_ledger.all_known,
                                    span_end_ns);
    observation_emit_category_pairs_locked(span_end_ns);
    observation_emit_category_intervals_locked(span_end_ns);
    observation_emit_command_barrier_bursts_locked(decode_terminal_observed);
    for (uint32_t command = 0; command < 256; command++) {
        if (g_observation_ledger.command_calls[command] == 0)
            continue;
        fprintf(stderr,
                "[CXL-CUDA] command_summary schema=guest-command-summary-v2"
                " producer=guest-shim clock_domain=guest-monotonic"
                " case_epoch=%" PRIu64 " scope=decode command=0x%x"
                " calls=%" PRIu64 " total_duration_ns=%" PRIu64
                " status_poll_count=%" PRIu64 "\n",
                g_observation_ledger.case_epoch, command,
                g_observation_ledger.command_calls[command],
                g_observation_ledger.command_total_duration_ns[command],
                g_observation_ledger.command_status_poll_count[command]);
    }
    observation_emit_gap_summaries_locked();
    observation_emit_clock_anchors_locked();
    fflush(stderr);
}

CUresult cuCxlObservationDecodeBeginV1(uint64_t case_epoch) {
    pthread_mutex_lock(&g_observation_ledger.lock);
    if (g_observation_ledger.active || case_epoch == 0) {
        if (g_observation_ledger.active)
            observation_fail_locked("decode-begin-while-active");
        pthread_mutex_unlock(&g_observation_ledger.lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    memset((char *)&g_observation_ledger + offsetof(CXLObservationLedger, active),
           0, sizeof(g_observation_ledger) - offsetof(CXLObservationLedger, active));
    g_observation_ledger.case_epoch = case_epoch;
    observation_capture_clock_anchor_locked(
        CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN);
    g_observation_ledger.span_begin_ns = observation_clock_ns_locked();
    g_observation_ledger.last_clock_ns = g_observation_ledger.span_begin_ns;
    if (!g_observation_ledger.span_begin_ns)
        observation_fail_locked("clock-read");
    g_observation_ledger.active = true;
    __atomic_store_n(&g_observation_fast_active, 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return g_observation_ledger.span_begin_ns ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuCxlObservationSpanBeginV1(uint32_t category,
                                    uint64_t graph_ordinal,
                                    uint64_t operation_sequence,
                                    uint64_t *token) {
    if (!token) {
        pthread_mutex_lock(&g_observation_ledger.lock);
        if (g_observation_ledger.active)
            observation_fail_locked("null-token-output");
        pthread_mutex_unlock(&g_observation_ledger.lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *token = 0;
    if (category < CXL_CUDA_OBS_SELECTED_RANGE_PLAN ||
        category > CXL_CUDA_OBS_HOST_RESULT_READ) {
        pthread_mutex_lock(&g_observation_ledger.lock);
        if (g_observation_ledger.active)
            observation_fail_locked("unknown-category");
        pthread_mutex_unlock(&g_observation_ledger.lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_observation_ledger.lock);
    *token = observation_span_begin_locked(category, graph_ordinal,
                                           operation_sequence, UINT32_MAX, "llama",
                                           g_observation_category_names[category],
                                           NULL);
    CUresult result = *token ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return result;
}

CUresult cuCxlObservationSpanEndV1(uint64_t token,
                                  int32_t operation_status) {
    pthread_mutex_lock(&g_observation_ledger.lock);
    CUresult result = observation_span_end_locked(token, operation_status, NULL);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return result;
}

CUresult cuCxlObservationDecodeEndV1(uint64_t case_epoch) {
    pthread_mutex_lock(&g_observation_ledger.lock);
    if (!g_observation_ledger.active ||
        case_epoch != g_observation_ledger.case_epoch) {
        if (g_observation_ledger.active)
            observation_fail_locked("decode-epoch-mismatch");
        pthread_mutex_unlock(&g_observation_ledger.lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint64_t span_end_ns = observation_clock_ns_locked();
    if (!span_end_ns) {
        observation_fail_locked("clock-read");
        span_end_ns = g_observation_ledger.last_clock_ns;
    }
    observation_capture_clock_anchor_locked(
        CXL_GPU_OBSERVATION_ANCHOR_DECODE_END);
    if (g_observation_ledger.open_count != 0)
        observation_fail_locked("open-span-at-terminal");
    observation_emit_terminal_locked(span_end_ns, true);
    g_observation_ledger.active = false;
    __atomic_store_n(&g_observation_fast_active, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return CUDA_SUCCESS;
}

static void observation_abandon_active_decode(void) {
    pthread_mutex_lock(&g_observation_ledger.lock);
    if (g_observation_ledger.active) {
        observation_fail_locked("decode-terminal-missing");
        uint64_t span_end_ns = observation_clock_ns_locked();
        if (!span_end_ns)
            span_end_ns = g_observation_ledger.last_clock_ns;
        observation_emit_terminal_locked(span_end_ns, false);
        g_observation_ledger.active = false;
        __atomic_store_n(&g_observation_fast_active, 0, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&g_observation_ledger.lock);
}

static void function_param_layouts_clear(const char *reason) {
    uint64_t cleared = 0;

    pthread_mutex_lock(&g_function_param_layouts_lock);
    for (size_t page_index = 0; page_index < FUNCTION_PARAM_LAYOUT_PAGE_SIZE; page_index++) {
        FunctionParamLayout **page = g_function_param_layout_pages[page_index];
        if (!page)
            continue;
        for (size_t slot = 0; slot < FUNCTION_PARAM_LAYOUT_PAGE_SIZE; slot++) {
            if (page[slot]) {
                free(page[slot]);
                cleared++;
            }
        }
        free(page);
        g_function_param_layout_pages[page_index] = NULL;
    }
    pthread_mutex_unlock(&g_function_param_layouts_lock);
    DLOG("function_param_cache event=clear reason=%s layouts=%" PRIu64 "\n",
         reason, cleared);
}

static void log_context_state(const char *api, CUresult result) {
    CxlCudaContextStateView state = cxl_cuda_context_state_view();

    DLOG("context_state api=%s result=%d token=%" PRIuPTR " mode=%u "
         "primary_retain_count=%u current_depth=%u\n",
         api, result, state.token, state.mode, state.primary_retain_count, state.current_depth);
}

static inline uint64_t maybe_bar4_offset(uint64_t value);
static inline uint64_t bar4_offset_of(void *host_ptr);
static bool bar4_pointer_range(const void *host_ptr, size_t size, uint64_t *offset);

typedef enum CXLBar4RangeKind {
    CXL_BAR4_RANGE_ORDINARY,
    CXL_BAR4_RANGE_CONTAINED,
    CXL_BAR4_RANGE_PARTIAL,
} CXLBar4RangeKind;

typedef struct CXLHtoDRoute {
    bool enabled;
    bool full_transfer;
    size_t minimum_transfer_bytes;
    size_t prefix_bytes_per_transfer;
    size_t total_bytes;
    size_t remaining_bytes;
    size_t routed_calls;
    size_t routed_bytes;
    size_t fallback_count;
    void *staging;
    uint64_t staging_offset;
} CXLHtoDRoute;

static CXLHtoDRoute g_htod_route;
static CXLBar4RangeKind bar4_range_kind(const void *host_ptr, size_t size,
                                        uint64_t *offset);
static CUresult htod_route_allocate_locked(void);

static const char *htod_route_mode(void) {
    if (!g_htod_route.enabled)
        return "disabled";
    return g_htod_route.full_transfer ? "cxlmem-bounded-full-transfer"
                                      : "cxlmem-bounded-prefix";
}

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

_Static_assert(CXL_GPU_STREAM_HANDLE_TAG > 2,
               "owned stream handles must not overlap CUDA special streams");
_Static_assert(CXL_GPU_STREAM_HANDLE_TAG + UINT32_MAX <
                   CXL_GPU_STREAM_WIRE_PER_THREAD,
               "owned stream handles must not overlap BAR2 stream sentinels");

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

static CUresult cublas_private_context_token(CUcontext context,
                                             uintptr_t *token) {
    uintptr_t current;
    CUresult result;

    if (!token)
        return CUDA_ERROR_INVALID_VALUE;
    result = cxl_cuda_context_get_current_live(&current);
    if (result != CUDA_SUCCESS)
        return result;
    if ((uintptr_t)context != current)
        return CUDA_ERROR_INVALID_CONTEXT;
    *token = current;
    return CUDA_SUCCESS;
}

static CUresult cublas_private_context_key(CUcontext context,
                                           uint64_t *key) {
    uintptr_t token;
    CUresult result = cublas_private_context_token(context, &token);

    if (result != CUDA_SUCCESS) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot4(context=%p, key=%p) -> %d\n",
             context, (void *)key, result);
        return result;
    }
    *key = (uint64_t)token;
    DLOG("CUBLAS_CONTEXT_STREAM.slot4(context=%p) -> key=%" PRIu64
         " CUDA_SUCCESS\n",
         context, *key);
    return CUDA_SUCCESS;
}

static CUresult cublas_private_stream_from_public(CUcontext context,
                                                  CUstream stream,
                                                  void **private_stream,
                                                  int per_thread) {
    uintptr_t token;
    uint64_t wire;
    CUresult result;

    (void)per_thread;
    if (!private_stream) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot51(context=%p, stream=%p) -> CUDA_ERROR_INVALID_VALUE\n",
             context, stream);
        return CUDA_ERROR_INVALID_VALUE;
    }
    result = cublas_private_context_token(context, &token);
    if (result != CUDA_SUCCESS) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot51(context=%p, stream=%p) -> %d\n",
             context, stream, result);
        return result;
    }
    (void)token;
    if (!cxl_gpu_stream_wire(stream, &wire)) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot51(context=%p, stream=%p) -> CUDA_ERROR_INVALID_HANDLE\n",
             context, stream);
        return CUDA_ERROR_INVALID_HANDLE;
    }

    if ((uintptr_t)stream <= 2)
        *private_stream = (void *)(uintptr_t)wire;
    else
        *private_stream = stream;
    DLOG("CUBLAS_CONTEXT_STREAM.slot51(context=%p, stream=%p, per_thread=%d) -> private_stream=%p CUDA_SUCCESS\n",
         context, stream, per_thread, *private_stream);
    return CUDA_SUCCESS;
}

static CUresult cublas_private_stream_identity(CUcontext context,
                                               const void *private_stream,
                                               uint64_t *identity) {
    uintptr_t token;
    uint64_t value;
    uint64_t id;
    CUresult result;

    if (!private_stream || !identity) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot39(context=%p, private_stream=%p, identity=%p) -> CUDA_ERROR_INVALID_VALUE\n",
             context, private_stream, (void *)identity);
        return CUDA_ERROR_INVALID_VALUE;
    }
    result = cublas_private_context_token(context, &token);
    if (result != CUDA_SUCCESS) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot39(context=%p, private_stream=%p) -> %d\n",
             context, private_stream, result);
        return result;
    }
    (void)token;

    value = (uint64_t)(uintptr_t)private_stream;
    if (value != CXL_GPU_STREAM_WIRE_NULL &&
        value != CXL_GPU_STREAM_WIRE_LEGACY &&
        value != CXL_GPU_STREAM_WIRE_PER_THREAD &&
        !cxl_gpu_stream_handle_id((CUstream)private_stream, &id)) {
        DLOG("CUBLAS_CONTEXT_STREAM.slot39(context=%p, private_stream=%p) -> CUDA_ERROR_INVALID_HANDLE\n",
             context, private_stream);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *identity = value;
    DLOG("CUBLAS_CONTEXT_STREAM.slot39(context=%p, private_stream=%p) -> identity=%" PRIu64
         " CUDA_SUCCESS\n",
         context, private_stream, *identity);
    return CUDA_SUCCESS;
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

static inline int batch_data_write(size_t offset, const void *src, size_t len) {
    return cxl_gpu_transport_batch_write(&g_transport, offset, src, len);
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
static uint64_t g_async_copy_sequence;

typedef struct CXLAsyncCopyTrace {
    bool active;
    uint64_t public_sequence;
    const char *api;
    size_t total_bytes;
    size_t range_count;
    uint64_t stream_wire;
    bool stream_wire_valid;
    uint32_t command_index;
    uint32_t last_command_index;
    uint64_t last_call_id;
    const char *implementation;
} CXLAsyncCopyTrace;

static __thread CXLAsyncCopyTrace g_async_copy_trace;

static uint64_t guest_monotonic_ns(void) {
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static uint64_t observation_cuda_call_begin(const char *symbol,
                                            uint32_t command,
                                            uint64_t call_id,
                                            uint64_t *begin_ns) {
    if (!__atomic_load_n(&g_observation_fast_active, __ATOMIC_ACQUIRE))
        return 0;
    pthread_mutex_lock(&g_observation_ledger.lock);
    uint64_t token = observation_span_begin_locked(
        CXL_OBSERVATION_PUBLIC_CALL, 0, call_id, command,
        "guest-shim", symbol, begin_ns);
    observation_command_barrier_record_locked(command);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return token;
}

static uint64_t observation_guest_span_begin(uint32_t category,
                                             const char *operation) {
    uint64_t token;

    if (!__atomic_load_n(&g_observation_fast_active, __ATOMIC_ACQUIRE))
        return 0;
    pthread_mutex_lock(&g_observation_ledger.lock);
    token = observation_span_begin_locked(
        category, 0,
        g_async_copy_trace.active ? g_async_copy_trace.public_sequence : 0,
        UINT32_MAX, "guest-shim", operation, NULL);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return token;
}

static void observation_guest_span_end(uint64_t token, CUresult result) {
    if (!token)
        return;
    pthread_mutex_lock(&g_observation_ledger.lock);
    (void)observation_span_end_locked(token, result, NULL);
    pthread_mutex_unlock(&g_observation_ledger.lock);
}

static uint64_t observation_cuda_call_end(uint64_t token, CUresult result,
                                          uint64_t fallback_ns) {
    if (!token)
        return fallback_ns;
    uint64_t end_ns = fallback_ns;
    pthread_mutex_lock(&g_observation_ledger.lock);
    (void)observation_span_end_locked(token, result, &end_ns);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return end_ns;
}

static void observation_command_status_polls(uint32_t command,
                                             uint32_t poll_count) {
    if (!__atomic_load_n(&g_observation_fast_active, __ATOMIC_ACQUIRE) ||
        command >= 256)
        return;
    pthread_mutex_lock(&g_observation_ledger.lock);
    if (g_observation_ledger.active)
        observation_add_locked(
            &g_observation_ledger.command_status_poll_count[command],
            poll_count);
    pthread_mutex_unlock(&g_observation_ledger.lock);
}

static void async_copy_trace_begin(const char *api, size_t total_bytes,
                                   size_t range_count, CUstream stream,
                                   const char *implementation) {
    uint64_t stream_wire = 0;
    bool stream_wire_valid = cxl_gpu_stream_wire(stream, &stream_wire);
    uint64_t public_sequence = __sync_add_and_fetch(&g_async_copy_sequence, 1);

    g_async_copy_trace = (CXLAsyncCopyTrace){
        .active = true,
        .public_sequence = public_sequence,
        .api = api,
        .total_bytes = total_bytes,
        .range_count = range_count,
        .stream_wire = stream_wire,
        .stream_wire_valid = stream_wire_valid,
        .implementation = implementation,
    };
    OLOG("async_copy event=public-entry public_sequence=%" PRIu64
         " api=%s total_bytes=%zu ranges=%zu guest_stream=%p stream_wire=%s0x%016" PRIx64
         " implementation=%s stream_forwarded=%u\n",
         public_sequence, api, total_bytes, range_count, stream,
         stream_wire_valid ? "" : "invalid:", stream_wire, implementation,
         strcmp(implementation, "async-enqueue") == 0 ||
             strcmp(implementation, "batch-command") == 0);
}

static CUresult async_copy_trace_end(CUresult result) {
    OLOG("async_copy event=public-return public_sequence=%" PRIu64
         " api=%s total_bytes=%zu ranges=%zu commands=%u result=%d implementation=%s\n",
         g_async_copy_trace.public_sequence, g_async_copy_trace.api,
         g_async_copy_trace.total_bytes, g_async_copy_trace.range_count,
         g_async_copy_trace.command_index,
         result, g_async_copy_trace.implementation);
    memset(&g_async_copy_trace, 0, sizeof(g_async_copy_trace));
    return result;
}

static CUresult execute_cmd_traced(uint32_t cmd, const char *symbol) {
    uint32_t sequence = __sync_add_and_fetch(&g_api_chain_sequence, 1);
    uint64_t call_id = ((uint64_t)(uint32_t)getpid() << 32) | sequence;
    uint64_t start_ns = 0;
    uint64_t observation_token = observation_cuda_call_begin(
        symbol, cmd, call_id, &start_ns);
    if (!start_ns)
        start_ns = guest_monotonic_ns();
    reg_write64(CXL_GPU_REG_CALL_ID, call_id);
    OLOG("api_chain event=guest-entry call_id=0x%016" PRIx64
         " symbol=%s command=0x%x guest_ns=%" PRIu64 "\n",
         call_id, symbol, cmd, start_ns);
    uint32_t async_command_index = 0;
    if (g_async_copy_trace.active) {
        async_command_index = ++g_async_copy_trace.command_index;
        g_async_copy_trace.last_command_index = async_command_index;
        g_async_copy_trace.last_call_id = call_id;
        OLOG("async_copy event=command-entry public_sequence=%" PRIu64
             " command_index=%u call_id=0x%016" PRIx64
             " api=%s command=0x%x p0=0x%016" PRIx64 " bytes=%" PRIu64
             " stream_wire=%s0x%016" PRIx64 " implementation=%s\n",
             g_async_copy_trace.public_sequence, async_command_index, call_id,
             g_async_copy_trace.api, cmd, reg_read64(CXL_GPU_REG_PARAM0),
             reg_read64(CXL_GPU_REG_PARAM1),
             g_async_copy_trace.stream_wire_valid ? "" : "invalid:",
             g_async_copy_trace.stream_wire,
             g_async_copy_trace.implementation);
    }
#ifdef CXL_GPU_CONTEXT_SHIM_TEST
    if (g_test_execute_cmd) {
        CUresult result = g_test_execute_cmd(cmd);
        observation_command_status_polls(cmd, 0);
        uint64_t end_ns = observation_cuda_call_end(
            observation_token, result, guest_monotonic_ns());
        reg_write64(CXL_GPU_REG_CALL_ID, 0);
        OLOG("api_chain event=guest-return call_id=0x%016" PRIx64
             " symbol=%s command=0x%x guest_ns=%" PRIu64
             " duration_ns=%" PRIu64 " status_poll_count=0 result=%d\n",
             call_id, symbol, cmd, end_ns,
             end_ns >= start_ns ? end_ns - start_ns : 0, result);
        if (g_async_copy_trace.active) {
            OLOG("async_copy event=command-return public_sequence=%" PRIu64
                 " command_index=%u call_id=0x%016" PRIx64
                 " api=%s command=0x%x result=%d implementation=%s\n",
                 g_async_copy_trace.public_sequence, async_command_index,
                 call_id, g_async_copy_trace.api, cmd, result,
                 g_async_copy_trace.implementation);
        }
        return result;
    }
#endif
    uint32_t poll_count = 0;
    CUresult result = (CUresult)cxl_gpu_transport_execute(&g_transport, cmd, &poll_count);
    observation_command_status_polls(cmd, poll_count);
    uint64_t end_ns = observation_cuda_call_end(
        observation_token, result, guest_monotonic_ns());
    reg_write64(CXL_GPU_REG_CALL_ID, 0);
    OLOG("api_chain event=guest-return call_id=0x%016" PRIx64
         " symbol=%s command=0x%x guest_ns=%" PRIu64
         " duration_ns=%" PRIu64 " status_poll_count=%u result=%d\n",
         call_id, symbol, cmd, end_ns,
         end_ns >= start_ns ? end_ns - start_ns : 0,
         poll_count, result);
    if (g_async_copy_trace.active) {
        OLOG("async_copy event=command-return public_sequence=%" PRIu64
             " command_index=%u call_id=0x%016" PRIx64
             " api=%s command=0x%x result=%d implementation=%s\n",
             g_async_copy_trace.public_sequence, async_command_index, call_id,
             g_async_copy_trace.api, cmd, result,
             g_async_copy_trace.implementation);
    }
    return result;
}

#define execute_cmd(cmd) execute_cmd_traced((cmd), __func__)

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
static void context_storage_test_reset(void);

void cxl_cuda_test_reset(void) {
    function_param_layouts_clear("test-reset");
    while (g_cudart_library_records) {
        CudartLibraryRecord *record = g_cudart_library_records;
        g_cudart_library_records = record->next;
        while (record->kernels) {
            CudartKernelRecord *kernel = record->kernels;
            record->kernels = kernel->next;
            free(kernel->name);
            free(kernel);
        }
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
    while (g_direct_source_pending) {
        CXLDirectSourcePending *pending = g_direct_source_pending;
        g_direct_source_pending = pending->next;
        free(pending);
    }
    memset(g_test_bar2, 0, sizeof(g_test_bar2));
    g_transport = (CxlGpuTransport)CXL_GPU_TRANSPORT_INITIALIZER;
    g_transport.regs = (volatile uint32_t *)g_test_bar2;
    g_transport.data = (volatile uint8_t *)g_test_bar2 + CXL_GPU_DATA_OFFSET;
    g_transport.descriptor = (volatile CXLGPURAMCommandDescriptor *)
        ((volatile uint8_t *)g_test_bar2 + CXL_GPU_DESCRIPTOR_OFFSET);
    g_transport.batch_data =
        (volatile uint8_t *)g_test_bar2 + CXL_GPU_BATCH_DATA_OFFSET;
    g_transport.descriptor->device_generation = 1;
    g_transport.bar_size = sizeof(g_test_bar2);
    g_initialized = 1;
    g_test_execute_cmd = NULL;
    g_test_direct_source_lease_releaser = NULL;
    cxl_cuda_context_state_reset();
    context_storage_test_reset();
}

int cxl_cuda_test_command_barrier_bursts(void) {
    pthread_mutex_lock(&g_observation_ledger.lock);
    memset((char *)&g_observation_ledger + offsetof(CXLObservationLedger, active),
           0, sizeof(g_observation_ledger) - offsetof(CXLObservationLedger, active));
    g_observation_ledger.active = true;

    observation_command_barrier_record_locked(
        CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
    observation_command_barrier_record_locked(CXL_GPU_CMD_GRAPH_LAUNCH);
    for (uint32_t index = 0; index < 2; index++)
        observation_command_barrier_record_locked(
            CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
    observation_command_barrier_record_locked(CXL_GPU_CMD_LAUNCH_KERNEL);
    for (uint32_t index = 0; index < 3; index++)
        observation_command_barrier_record_locked(
            CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
    observation_command_barrier_record_locked(CXL_GPU_CMD_MEM_GET_INFO);
    observation_command_barrier_record_locked(CXL_GPU_CMD_STREAM_SYNC);
    for (uint32_t index = 0; index < 5; index++)
        observation_command_barrier_record_locked(
            CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
    observation_command_barrier_record_locked(CXL_GPU_CMD_CTX_SYNC);
    for (uint32_t index = 0; index < 17; index++)
        observation_command_barrier_record_locked(
            CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);

    const CXLObservationCommandBarrierBursts *bursts =
        &g_observation_ledger.command_barrier_bursts;
    const bool passed =
        bursts->fused_calls == 28 &&
        bursts->terminated_bursts == 4 &&
        bursts->terminated_fused_calls == 11 &&
        bursts->current_fused_calls == 17 &&
        bursts->max_burst == 5 &&
        bursts->histogram[CXL_OBSERVATION_BURST_1] == 1 &&
        bursts->histogram[CXL_OBSERVATION_BURST_2] == 1 &&
        bursts->histogram[CXL_OBSERVATION_BURST_3_TO_4] == 1 &&
        bursts->histogram[CXL_OBSERVATION_BURST_5_TO_8] == 1 &&
        bursts->histogram[CXL_OBSERVATION_BURST_9_TO_16] == 0 &&
        bursts->histogram[CXL_OBSERVATION_BURST_OVER_16] == 0 &&
        bursts->barriers[CXL_OBSERVATION_BARRIER_CTX_SYNC] == 1 &&
        bursts->barriers[CXL_OBSERVATION_BARRIER_GRAPH_LAUNCH] == 1 &&
        bursts->barriers[CXL_OBSERVATION_BARRIER_KERNEL_LAUNCH] == 1 &&
        bursts->barriers[CXL_OBSERVATION_BARRIER_STREAM_SYNC] == 1;

    memset((char *)&g_observation_ledger + offsetof(CXLObservationLedger, active),
           0, sizeof(g_observation_ledger) - offsetof(CXLObservationLedger, active));
    __atomic_store_n(&g_observation_fast_active, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_observation_ledger.lock);
    return passed;
}

void cxl_cuda_test_set_executor(CUresult (*executor)(uint32_t cmd)) { g_test_execute_cmd = executor; }

void cxl_cuda_test_set_direct_source_lease_releaser(
    CUresult (*releaser)(uint64_t lease_handle)) {
    g_test_direct_source_lease_releaser = releaser;
}

void cxl_cuda_test_set_initialized(int initialized) { g_initialized = initialized; }

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

void cxl_cuda_test_write_data(size_t offset, const void *src, size_t length) {
    cxl_gpu_transport_data_write(&g_transport, offset, src, length);
}

void cxl_cuda_test_read_batch_data(size_t offset, void *dst, size_t length) {
    cxl_gpu_transport_batch_read(&g_transport, offset, dst, length);
}
#endif

/* Find and map CXL Type 2 device */
static int find_and_map_device(void) {
    if (g_transport.regs)
        return 0;
    return cxl_gpu_transport_open(&g_transport, g_debug);
}

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
    OLOG("lookup_proc_address(symbol=%s) -> %p\n", symbol, fn);
    return fn;
}

static void log_proc_address_caller(const char *event, const char *symbol, const void *caller) {
    Dl_info info = {0};
    if (dladdr(caller, &info) != 0 && info.dli_fbase && info.dli_fname) {
        uintptr_t base = (uintptr_t)info.dli_fbase;
        OLOG("api_provenance event=%s pid=%ld symbol=%s caller_status=resolved "
             "caller_file=%s caller_base=0x%llx caller_offset=0x%llx\n",
             event, (long)getpid(), symbol ? symbol : "(null)", info.dli_fname, (unsigned long long)base,
             (unsigned long long)((uintptr_t)caller - base));
    } else {
        OLOG("api_provenance event=%s pid=%ld symbol=%s caller_status=unresolved caller_address=%p\n", event,
             (long)getpid(), symbol ? symbol : "(null)", caller);
    }
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                          CUdriverProcAddressQueryResult *symbolStatus) {
    const void *caller = __builtin_return_address(0);
    OLOG("cuGetProcAddress(symbol=%s, version=%d, flags=0x%lx, pfn=%p, status=%p)\n",
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
        OLOG("cuGetProcAddress(%s) -> pfn=NULL status=SYMBOL_NOT_FOUND result=CUDA_SUCCESS\n",
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
    OLOG("cuGetProcAddress(%s) -> pfn=%p status=SUCCESS result=CUDA_SUCCESS\n", symbol, fn);
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

    CUresult layout_result = function_param_layout_copy(
        nodeParams->func, param_offsets, param_sizes, &num_args, &param_extent);
    if (layout_result != CUDA_SUCCESS)
        return layout_result;
    for (uint32_t i = 0; i < num_args; i++) {
        if (!nodeParams->kernelParams || !nodeParams->kernelParams[i])
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

CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     unsigned long long flags) {
    if (flags != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    return cuGraphInstantiate(phGraphExec, hGraph, NULL, NULL, 0);
}

CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                           CUgraphExecUpdateResultInfo *resultInfo) {
    uint64_t graph_exec_id, graph_id;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!resultInfo || !cxl_gpu_handle_id(hGraphExec, &graph_exec_id) ||
        !cxl_gpu_handle_id(hGraph, &graph_id))
        return CUDA_ERROR_INVALID_VALUE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, graph_exec_id);
    reg_write64(CXL_GPU_REG_PARAM1, graph_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_GRAPH_EXEC_UPDATE);
    resultInfo->result = (CUgraphExecUpdateResult)(int32_t)reg_read64(CXL_GPU_REG_RESULT0);
    uint64_t error_node_id = reg_read64(CXL_GPU_REG_RESULT1);
    uint64_t error_from_node_id = reg_read64(CXL_GPU_REG_RESULT2);
    resultInfo->errorNode = error_node_id == UINT64_MAX ? NULL :
                            (CUgraphNode)cxl_gpu_handle_from_id(error_node_id);
    resultInfo->errorFromNode = error_from_node_id == UINT64_MAX ? NULL :
                                (CUgraphNode)cxl_gpu_handle_from_id(error_from_node_id);
    cmd_unlock();
    return result;
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
    if (!g_debug && !g_observation_stream) {
        return;
    }
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
    /* cuDriverGetVersion describes the virtual Driver contract consumed by
     * guest CUDA userland.  The physical Driver version remains available in
     * the BAR2 register and QEMU trace, but advertising it here would claim
     * CUDA API semantics that this shim does not implement. */
    return 12090;
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
    uintptr_t token;
    CUresult result;

    DLOG("TOOLS_TLS.get(out=%p)\n", (void *)out);
    if (!out) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *out = NULL;

    result = cxl_cuda_context_get_current_live(&token);
    if (result != CUDA_SUCCESS) {
        DLOG("TOOLS_TLS.get -> %d\n", result);
        return result;
    }
    *out = (void *)token;
    DLOG("TOOLS_TLS.get -> context=%p CUDA_SUCCESS\n", *out);
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

static const unsigned char CUBLAS_CONTEXT_STREAM_UUID[16] = {
    0x21, 0x31, 0x8c, 0x60, 0x97, 0x14, 0x32, 0x48,
    0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2,
};

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

static const void *CUBLAS_CONTEXT_STREAM_TABLE[93] = {
    [0] = (const void *)(uintptr_t)(sizeof(CUBLAS_CONTEXT_STREAM_TABLE)),
    [4] = (const void *)cublas_private_context_key,
    [39] = (const void *)cublas_private_stream_identity,
    [51] = (const void *)cublas_private_stream_from_public,
};

CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
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

    if (uuid_equal(pExportTableId, CUBLAS_CONTEXT_STREAM_UUID)) {
        *ppExportTable = CUBLAS_CONTEXT_STREAM_TABLE;
        DLOG("cuGetExportTable -> CUBLAS_CONTEXT_STREAM_TABLE size=%zu\n",
             sizeof(CUBLAS_CONTEXT_STREAM_TABLE));
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

    DLOG("cuInit(%u)\n", flags);

    if (g_initialized) {
        return CUDA_SUCCESS;
    }

    if (find_and_map_device() < 0) {
        return CUDA_ERROR_NO_DEVICE;
    }
    if (g_direct_source_enabled &&
        cxl_gpu_transport_open_source(&g_transport) != 0) {
        DLOG("direct source device unavailable: %s\n", strerror(errno));
        return direct_source_errno_result(errno);
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

    /* CUDA userland queries the virtual Driver contract before cuInit.
     * Mapping BAR2 is transport discovery only. */
    if (!g_initialized && find_and_map_device() < 0)
        return CUDA_ERROR_NO_DEVICE;
    *version = cxl_cuda_effective_driver_version();
    if (*version <= 0)
        return CUDA_ERROR_NOT_INITIALIZED;
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
    bool context_destroyed = err == CUDA_SUCCESS;
    if (err == CUDA_SUCCESS) {
        context_storage_clear_context(ctx, 1);
        err = cxl_cuda_context_commit_destroy((uintptr_t)ctx);
    }
    cmd_unlock();
    if (context_destroyed)
        function_param_layouts_clear("context-destroy");
    return err;
}

CUresult cuCtxSynchronize(void) {
    DLOG("cuCtxSynchronize\n");
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    cmd_lock();
    CUresult err = execute_cmd(CXL_GPU_CMD_CTX_SYNC);
    if (err == CUDA_SUCCESS && g_direct_source_enabled)
        err = direct_sources_complete_locked(0, true);
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
    OLOG("cuMemcpyHtoD(dst=0x%lx, size=%zu)\n", (unsigned long)dstDevice, byteCount);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!srcHost)
        return CUDA_ERROR_INVALID_VALUE;

    uint64_t bar4_offset = 0;
    if (bar4_pointer_range(srcHost, byteCount, &bar4_offset)) {
        size_t offset = 0;
        while (offset < byteCount) {
            size_t chunk = byteCount - offset;
            if (chunk > CXL_GPU_BULK_TRANSFER_SIZE)
                chunk = CXL_GPU_BULK_TRANSFER_SIZE;
            cmd_lock();
            reg_write64(CXL_GPU_REG_PARAM0, bar4_offset + offset);
            reg_write64(CXL_GPU_REG_PARAM1, dstDevice + offset);
            reg_write64(CXL_GPU_REG_PARAM2, chunk);
            CUresult err = execute_cmd(CXL_GPU_CMD_BULK_HTOD);
            cmd_unlock();
            if (err != CUDA_SUCCESS)
                return err;
            offset += chunk;
        }
        return CUDA_SUCCESS;
    }

    /* Transfer in chunks that fit in data buffer */
    size_t offset = 0;
    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > CXL_GPU_DATA_SIZE) {
            chunk = CXL_GPU_DATA_SIZE;
        }

        cmd_lock();
        uint64_t data_write_start_ns = guest_monotonic_ns();
        data_write(0, (const uint8_t *)srcHost + offset, chunk);
        uint64_t data_write_end_ns = guest_monotonic_ns();
        DLOG("copy_transport event=data-write transport=ram-bulk api=%s public_sequence=%" PRIu64
             " chunk_index=%u offset=%zu bytes=%zu duration_ns=%" PRIu64 "\n",
             g_async_copy_trace.active ? g_async_copy_trace.api : "cuMemcpyHtoD",
             g_async_copy_trace.active ? g_async_copy_trace.public_sequence : 0,
             g_async_copy_trace.command_index + 1, offset, chunk,
             data_write_end_ns >= data_write_start_ns ?
                 data_write_end_ns - data_write_start_ns : 0);
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
    uint64_t stream_wire;
    CXLBar4RangeKind source_kind;

    OLOG("cuMemcpyHtoDAsync(dst=0x%lx, size=%zu, stream=%p)\n", (unsigned long)dstDevice, byteCount, hStream);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!srcHost || !byteCount)
        return CUDA_ERROR_INVALID_VALUE;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;

    source_kind = bar4_range_kind(srcHost, byteCount, NULL);
    if (source_kind == CXL_BAR4_RANGE_PARTIAL)
        return CUDA_ERROR_INVALID_VALUE;

    async_copy_trace_begin("cuMemcpyHtoDAsync", byteCount, 1, hStream,
                           "async-enqueue");
    size_t offset = 0;
    if (g_htod_route.enabled && source_kind == CXL_BAR4_RANGE_ORDINARY &&
        byteCount >= g_htod_route.minimum_transfer_bytes) {
        size_t route_target;
        CUresult result;

        cmd_lock();
        route_target = g_htod_route.full_transfer ? byteCount
                                                  : g_htod_route.prefix_bytes_per_transfer;
        if (route_target > byteCount)
            route_target = byteCount;
        if (route_target > g_htod_route.remaining_bytes) {
            if (g_htod_route.full_transfer) {
                cmd_unlock();
                return async_copy_trace_end(CUDA_ERROR_OUT_OF_MEMORY);
            }
            route_target = g_htod_route.remaining_bytes;
        }
        while (offset < route_target) {
            size_t routed = route_target - offset;
            if (routed > g_htod_route.prefix_bytes_per_transfer)
                routed = g_htod_route.prefix_bytes_per_transfer;
            result = htod_route_allocate_locked();
            if (result == CUDA_SUCCESS) {
                memcpy(g_htod_route.staging, (const uint8_t *)srcHost + offset,
                       routed);
                __sync_synchronize();
                reg_write64(CXL_GPU_REG_PARAM0, g_htod_route.staging_offset);
                reg_write64(CXL_GPU_REG_PARAM1, dstDevice + offset);
                reg_write64(CXL_GPU_REG_PARAM2, routed);
                reg_write64(CXL_GPU_REG_PARAM3, stream_wire);
                result = execute_cmd(CXL_GPU_CMD_BULK_HTOD_ASYNC);
                if (result == CUDA_SUCCESS) {
                    uint64_t case_epoch = g_transport.descriptor
                        ? __atomic_load_n(&g_transport.descriptor->active_case_epoch,
                                          __ATOMIC_ACQUIRE)
                        : 0;

                    OLOG("weight_source event=route-mapping process_id=%ld"
                         " case_epoch=%" PRIu64
                         " device_bdf=%s bar_index=4 public_sequence=%" PRIu64
                         " command_index=%u call_id=0x%016" PRIx64
                         " original_source_start=0x%016" PRIxPTR
                         " original_source_bytes=%zu bar4_range_start=0x%016" PRIx64
                         " bar4_range_bytes=%zu destination_cuda_start=0x%016" PRIx64
                         " destination_cuda_bytes=%zu guest_ns=%" PRIu64 "\n",
                         (long)getpid(), case_epoch,
                         g_transport.pci_bdf[0] ? g_transport.pci_bdf : "unavailable",
                         g_async_copy_trace.public_sequence,
                         g_async_copy_trace.last_command_index,
                         g_async_copy_trace.last_call_id,
                         (uintptr_t)srcHost + offset,
                         routed, g_htod_route.staging_offset, routed,
                         (uint64_t)dstDevice + offset, routed,
                         guest_monotonic_ns());
                }
            }
            if (result != CUDA_SUCCESS) {
                cmd_unlock();
                return async_copy_trace_end(result);
            }
            g_htod_route.remaining_bytes -= routed;
            g_htod_route.routed_calls++;
            g_htod_route.routed_bytes += routed;
            DLOG("copy_transport event=data-write transport=cxlmem-bar4 "
                 "api=%s public_sequence=%" PRIu64 " offset=0 bytes=%zu "
                 "bar4_offset=%" PRIu64 " stream_wire=%" PRIu64
                 " cumulative_routed_bytes=%zu budget_bytes=%zu\n",
                 g_async_copy_trace.api, g_async_copy_trace.public_sequence,
                 routed, g_htod_route.staging_offset, stream_wire,
                 g_htod_route.total_bytes - g_htod_route.remaining_bytes,
                 g_htod_route.total_bytes);
            offset += routed;
        }
        cmd_unlock();
    }
    while (offset < byteCount) {
        size_t chunk = byteCount - offset;
        if (chunk > CXL_GPU_DATA_SIZE)
            chunk = CXL_GPU_DATA_SIZE;

        cmd_lock();
        uint64_t data_write_start_ns = guest_monotonic_ns();
        data_write(0, (const uint8_t *)srcHost + offset, chunk);
        uint64_t data_write_end_ns = guest_monotonic_ns();
        DLOG("copy_transport event=data-write transport=ram-bulk api=%s public_sequence=%" PRIu64
             " chunk_index=%u offset=%zu bytes=%zu duration_ns=%" PRIu64 "\n",
             g_async_copy_trace.api, g_async_copy_trace.public_sequence,
             g_async_copy_trace.command_index + 1, offset, chunk,
             data_write_end_ns >= data_write_start_ns ?
                 data_write_end_ns - data_write_start_ns : 0);
        reg_write64(CXL_GPU_REG_PARAM0, dstDevice + offset);
        reg_write64(CXL_GPU_REG_PARAM1, chunk);
        reg_write64(CXL_GPU_REG_PARAM2, stream_wire);
        CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC);
        cmd_unlock();
        if (result != CUDA_SUCCESS)
            return async_copy_trace_end(result);
        offset += chunk;
    }
    return async_copy_trace_end(CUDA_SUCCESS);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, hStream);
}

typedef struct CXLBatchHtoDPlanEntry {
    CXLGPUBatchHtoDRange wire;
    const void *source;
} CXLBatchHtoDPlanEntry;

static CUresult direct_source_errno_result(int error) {
    switch (error) {
    case EINVAL:
    case EFAULT:
        return CUDA_ERROR_INVALID_VALUE;
    case ENOMEM:
        return CUDA_ERROR_OUT_OF_MEMORY;
    case ENODEV:
    case ENOTTY:
    case EOPNOTSUPP:
        return CUDA_ERROR_NOT_SUPPORTED;
    default:
        return CUDA_ERROR_UNKNOWN;
    }
}

static CUresult direct_source_unregister_locked(uint64_t source_id) {
    reg_write64(CXL_GPU_REG_PARAM0, source_id);
    return execute_cmd(CXL_GPU_CMD_SOURCE_UNREGISTER);
}

static CUresult direct_source_lease_release(uint64_t lease_handle) {
    uint64_t observation_token = observation_guest_span_begin(
        CXL_CUDA_OBS_SOURCE_LEASE, "source_lease_release");
    CUresult result;

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
    if (g_test_direct_source_lease_releaser) {
        result = g_test_direct_source_lease_releaser(lease_handle);
        observation_guest_span_end(observation_token, result);
        return result;
    }
#endif
    struct cxl_type2_source_release_v1 release = {
        .version = CXL_TYPE2_SOURCE_UAPI_VERSION,
        .lease_handle = lease_handle,
    };
    result = ioctl(g_transport.source_fd, CXL_TYPE2_SOURCE_RELEASE,
                   &release) == 0
                 ? CUDA_SUCCESS
                 : direct_source_errno_result(errno);
    observation_guest_span_end(observation_token, result);
    return result;
}

static CXLDirectSourcePending *direct_source_pending_new(uint64_t source_id, uint64_t lease_handle,
                                                         uint64_t stream_wire) {
    CXLDirectSourcePending *pending = malloc(sizeof(*pending));

    if (pending) {
        *pending = (CXLDirectSourcePending){
            .source_id = source_id,
            .lease_handle = lease_handle,
            .stream_wire = stream_wire,
        };
    }
    return pending;
}

static CUresult direct_sources_complete_locked(uint64_t stream_wire,
                                               bool all_streams) {
    CXLDirectSourcePending **link = &g_direct_source_pending;

    while (*link) {
        CXLDirectSourcePending *pending = *link;
        if (!all_streams && pending->stream_wire != stream_wire) {
            link = &pending->next;
            continue;
        }
        CUresult result;
        if (pending->source_id) {
            result = direct_source_unregister_locked(pending->source_id);
            if (result != CUDA_SUCCESS)
                return result;
            pending->source_id = 0;
        }
        result = direct_source_lease_release(pending->lease_handle);
        if (result != CUDA_SUCCESS)
            return result;
        *link = pending->next;
        free(pending);
    }
    return CUDA_SUCCESS;
}

static CUresult cuMemcpyBatchDirectAsync(CUdeviceptr *dsts,
                                         CUdeviceptr *srcs,
                                         size_t *sizes, size_t count,
                                         size_t *failIdx,
                                         uint64_t stream_wire) {
    struct cxl_type2_source_range_v1 *kernel_ranges = NULL;
    struct cxl_type2_source_run_v1 *kernel_runs = NULL;
    CXLGPUSourceRangeV1 *wire_ranges = NULL;
    CXLGPUSourceRunV1 *wire_runs = NULL;
    CXLGPUDirectRangeV1 *direct_ranges = NULL;
    CXLDirectSourcePending *pending = NULL;
    struct cxl_type2_source_acquire_v1 acquire = {0};
    struct cxl_type2_source_release_v1 release = {0};
    size_t register_bytes;
    size_t direct_bytes;
    CUresult result = CUDA_SUCCESS;

    if (g_transport.source_fd < 0 || count > CXL_TYPE2_SOURCE_MAX_RANGES)
        return CUDA_ERROR_NOT_SUPPORTED;
    kernel_ranges = malloc(count * sizeof(*kernel_ranges));
    kernel_runs = malloc(CXL_TYPE2_SOURCE_MAX_RUNS * sizeof(*kernel_runs));
    wire_ranges = malloc(count * sizeof(*wire_ranges));
    direct_ranges = malloc(count * sizeof(*direct_ranges));
    if (!kernel_ranges || !kernel_runs || !wire_ranges || !direct_ranges) {
        result = CUDA_ERROR_OUT_OF_MEMORY;
        goto out;
    }
    for (size_t index = 0; index < count; index++) {
        if (!srcs[index] || !dsts[index] || !sizes[index] ||
            srcs[index] > UINTPTR_MAX ||
            sizes[index] > UINTPTR_MAX - (uintptr_t)srcs[index] ||
            sizes[index] > UINT64_MAX - dsts[index]) {
            *failIdx = index;
            result = CUDA_ERROR_INVALID_VALUE;
            goto out;
        }
        kernel_ranges[index] = (struct cxl_type2_source_range_v1) {
            .user_address = srcs[index],
            .length = sizes[index],
        };
    }
    acquire.version = CXL_TYPE2_SOURCE_UAPI_VERSION;
    acquire.ranges_ptr = (uintptr_t)kernel_ranges;
    acquire.runs_ptr = (uintptr_t)kernel_runs;
    acquire.range_count = count;
    acquire.run_capacity = CXL_TYPE2_SOURCE_MAX_RUNS;
    uint64_t lease_observation_token = observation_guest_span_begin(
        CXL_CUDA_OBS_SOURCE_LEASE, "source_lease_acquire");
    if (ioctl(g_transport.source_fd, CXL_TYPE2_SOURCE_ACQUIRE, &acquire) != 0) {
        int source_errno = errno;
        observation_guest_span_end(
            lease_observation_token, direct_source_errno_result(source_errno));
        fprintf(stderr,
                "[CXL-CUDA] direct_source_acquire_failed errno=%d"
                " range_count=%zu\n",
                source_errno, count);
        for (size_t index = 0; index < count; index++) {
            fprintf(stderr,
                    "[CXL-CUDA] direct_source_acquire_range index=%zu"
                    " source=0x%" PRIx64 " length=%zu\n",
                    index, (uint64_t)srcs[index], sizes[index]);
        }
        result = direct_source_errno_result(source_errno);
        goto out;
    }
    observation_guest_span_end(lease_observation_token, CUDA_SUCCESS);
    release.version = CXL_TYPE2_SOURCE_UAPI_VERSION;
    release.lease_handle = acquire.lease_handle;

    if (!acquire.run_count || acquire.run_count > CXL_TYPE2_SOURCE_MAX_RUNS ||
        count > (SIZE_MAX - sizeof(CXLGPUSourceRegisterV1)) /
                    sizeof(*wire_ranges)) {
        result = CUDA_ERROR_UNKNOWN;
        goto release_lease;
    }
    register_bytes = sizeof(CXLGPUSourceRegisterV1) +
                     count * sizeof(*wire_ranges);
    if (acquire.run_count >
            (CXL_GPU_BATCH_DATA_SIZE - register_bytes) / sizeof(*wire_runs)) {
        result = CUDA_ERROR_NOT_SUPPORTED;
        goto release_lease;
    }
    register_bytes += acquire.run_count * sizeof(*wire_runs);
    direct_bytes = count * sizeof(*direct_ranges);
    _Static_assert(sizeof(*kernel_runs) == sizeof(*wire_runs),
                   "source run wire layout mismatch");
    wire_runs = (CXLGPUSourceRunV1 *)kernel_runs;
    pending = direct_source_pending_new(0, release.lease_handle, stream_wire);
    if (!pending) {
        result = CUDA_ERROR_OUT_OF_MEMORY;
        goto release_lease;
    }

    CXLGPUSourceRegisterV1 header = {
        .range_count = count,
        .run_count = acquire.run_count,
        .lease_handle = acquire.lease_handle,
        .logical_bytes = acquire.logical_bytes,
        .unique_dmap_bytes = acquire.unique_dmap_bytes,
    };
    for (size_t index = 0; index < count; index++) {
        wire_ranges[index] = (CXLGPUSourceRangeV1){
            .first_run = kernel_ranges[index].first_run,
            .run_count = kernel_ranges[index].run_count,
            .first_run_byte_offset =
                kernel_ranges[index].first_run_byte_offset,
            .length = kernel_ranges[index].length,
        };
        direct_ranges[index] = (CXLGPUDirectRangeV1){
            .destination = dsts[index],
            .size = sizes[index],
            .source_range = index,
        };
    }
    cmd_lock();
    if (direct_bytes > CXL_GPU_BATCH_DATA_SIZE - register_bytes ||
        batch_data_write(0, &header, sizeof(header)) != 0 ||
        batch_data_write(sizeof(header), wire_ranges,
                         count * sizeof(*wire_ranges)) != 0 ||
        batch_data_write(sizeof(header) + count * sizeof(*wire_ranges),
                         wire_runs, acquire.run_count * sizeof(*wire_runs)) != 0 ||
        batch_data_write(register_bytes, direct_ranges, direct_bytes) != 0) {
        result = CUDA_ERROR_UNKNOWN;
        goto unlock_release;
    }
    reg_write64(CXL_GPU_REG_PARAM0, register_bytes);
    reg_write64(CXL_GPU_REG_PARAM1, count);
    reg_write64(CXL_GPU_REG_PARAM2, stream_wire);
    result = execute_cmd(CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
    if (result != CUDA_SUCCESS) {
        uint64_t failed = reg_read64(CXL_GPU_REG_RESULT0);
        uint64_t fragments_enqueued = reg_read64(CXL_GPU_REG_RESULT2);

        if (failed < count)
            *failIdx = failed;
        if (fragments_enqueued) {
            pending->next = g_direct_source_pending;
            g_direct_source_pending = pending;
            pending = NULL;
            release.lease_handle = 0;
            cmd_unlock();
            goto out;
        }
        goto unlock_release;
    }
    pending->next = g_direct_source_pending;
    g_direct_source_pending = pending;
    pending = NULL;
    release.lease_handle = 0;
    cmd_unlock();
    goto out;

unlock_release:
    cmd_unlock();
release_lease:
    if (release.lease_handle) {
        CUresult release_result =
            direct_source_lease_release(release.lease_handle);

        if (release_result != CUDA_SUCCESS && result == CUDA_SUCCESS)
            result = release_result;
    }
out:
    free(pending);
    free(direct_ranges);
    free(wire_ranges);
    free(kernel_runs);
    free(kernel_ranges);
    return result;
}

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
void cxl_cuda_test_add_direct_source_pending(uint64_t source_id, uint64_t lease_handle, uint64_t stream_wire) {
    CXLDirectSourcePending *pending = direct_source_pending_new(source_id, lease_handle, stream_wire);
    if (!pending)
        abort();
    pending->next = g_direct_source_pending;
    g_direct_source_pending = pending;
}

CUresult cxl_cuda_test_complete_direct_sources(uint64_t stream_wire,
                                               bool all_streams) {
    cmd_lock();
    CUresult result = direct_sources_complete_locked(stream_wire, all_streams);
    cmd_unlock();
    return result;
}

size_t cxl_cuda_test_direct_source_pending_count(void) {
    size_t count = 0;
    for (CXLDirectSourcePending *pending = g_direct_source_pending; pending;
         pending = pending->next)
        count++;
    return count;
}
#endif

CUresult cuMemcpyBatchAsync(CUdeviceptr *dsts, CUdeviceptr *srcs,
                            size_t *sizes, size_t count,
                            CUmemcpyAttributes *attrs, size_t *attrsIdxs,
                            size_t numAttrs, size_t *failIdx,
                            CUstream hStream) {
    CXLBatchHtoDPlanEntry *plan = NULL;
    uint64_t stream_wire;
    size_t table_end;
    size_t payload_bytes;
    size_t source_bytes = 0;
    uint64_t lock_wait_start_ns;
    uint64_t lock_acquired_ns;
    uint64_t materialize_end_ns = 0;
    uint64_t result_fail_idx = UINT64_MAX;
    uint64_t successfully_enqueued = 0;
    CUresult result = CUDA_SUCCESS;

    if (failIdx)
        *failIdx = SIZE_MAX;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dsts || !srcs || !sizes || !attrs || !attrsIdxs || !failIdx ||
        count == 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;
    if (stream_wire == CXL_GPU_STREAM_WIRE_NULL ||
        stream_wire == CXL_GPU_STREAM_WIRE_LEGACY)
        return CUDA_ERROR_INVALID_VALUE;
    if (numAttrs != 1 || attrsIdxs[0] != 0 ||
        attrs[0].srcAccessOrder != CU_MEMCPY_SRC_ACCESS_ORDER_ANY ||
        attrs[0].flags != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (count > UINT32_MAX ||
        count > (SIZE_MAX - sizeof(CXLGPUBatchHtoDHeader)) /
                    sizeof(CXLGPUBatchHtoDRange))
        return CUDA_ERROR_NOT_SUPPORTED;

    if (g_direct_source_enabled) {
        for (size_t index = 0; index < count; index++) {
            if (source_bytes > SIZE_MAX - sizes[index]) {
                *failIdx = index;
                return CUDA_ERROR_INVALID_VALUE;
            }
            source_bytes += sizes[index];
        }
        async_copy_trace_begin("cuMemcpyBatchAsync", source_bytes, count,
                               hStream, "direct-source");
        result = cuMemcpyBatchDirectAsync(dsts, srcs, sizes, count, failIdx,
                                          stream_wire);
        OLOG("batch_htod_direct public_sequence=%" PRIu64
             " result=%d fail_idx=%zu ranges=%zu source_bytes=%zu"
             " payload_bytes=0 stream_wire=%" PRIu64 "\n",
             g_async_copy_trace.public_sequence, result, *failIdx, count,
             source_bytes, stream_wire);
        return async_copy_trace_end(result);
    }

    table_end = sizeof(CXLGPUBatchHtoDHeader) +
                count * sizeof(CXLGPUBatchHtoDRange);
    if (table_end > CXL_GPU_BATCH_DATA_SIZE ||
        table_end > SIZE_MAX - 63)
        return CUDA_ERROR_NOT_SUPPORTED;
    payload_bytes = (table_end + 63) & ~(size_t)63;
    if (payload_bytes > CXL_GPU_BATCH_DATA_SIZE)
        return CUDA_ERROR_NOT_SUPPORTED;

    plan = calloc(count, sizeof(*plan));
    if (!plan)
        return CUDA_ERROR_OUT_OF_MEMORY;

    for (size_t index = 0; index < count; index++) {
        size_t size = sizes[index];
        CUdeviceptr source = srcs[index];
        CUdeviceptr destination = dsts[index];

        if (!source || !destination || !size ||
            source > UINTPTR_MAX ||
            size > UINTPTR_MAX - (uintptr_t)source ||
            size > UINT64_MAX - destination) {
            *failIdx = index;
            result = CUDA_ERROR_INVALID_VALUE;
            goto out;
        }
        if (size > CXL_GPU_BATCH_DATA_SIZE - payload_bytes) {
            *failIdx = index;
            result = CUDA_ERROR_NOT_SUPPORTED;
            goto out;
        }

        plan[index].wire = (CXLGPUBatchHtoDRange){
            .source_offset = payload_bytes,
            .destination = destination,
            .size = size,
        };
        plan[index].source = (const void *)(uintptr_t)source;
        payload_bytes += size;
        source_bytes += size;
    }

    CXLGPUBatchHtoDHeader header = {
        .header_size = sizeof(header),
        .range_count = (uint32_t)count,
        .range_size = sizeof(CXLGPUBatchHtoDRange),
        .payload_bytes = payload_bytes,
    };

    async_copy_trace_begin("cuMemcpyBatchAsync", source_bytes, count,
                           hStream, "batch-command");
    lock_wait_start_ns = guest_monotonic_ns();
    cmd_lock();
    lock_acquired_ns = guest_monotonic_ns();
    if (batch_data_write(0, &header, sizeof(header)) != 0) {
        result = CUDA_ERROR_UNKNOWN;
        goto unlock;
    }
    for (size_t index = 0; index < count; index++) {
        size_t range_offset = sizeof(header) +
                              index * sizeof(CXLGPUBatchHtoDRange);
        if (batch_data_write(range_offset, &plan[index].wire,
                             sizeof(plan[index].wire)) != 0 ||
            batch_data_write(plan[index].wire.source_offset,
                             plan[index].source,
                             (size_t)plan[index].wire.size) != 0) {
            result = CUDA_ERROR_UNKNOWN;
            goto unlock;
        }
    }
    materialize_end_ns = guest_monotonic_ns();

    reg_write64(CXL_GPU_REG_PARAM0, count);
    reg_write64(CXL_GPU_REG_PARAM1, payload_bytes);
    reg_write64(CXL_GPU_REG_PARAM2, stream_wire);
    result = execute_cmd(CXL_GPU_CMD_BATCH_HTOD_ASYNC);
    {
        result_fail_idx = reg_read64(CXL_GPU_REG_RESULT0);
        successfully_enqueued = reg_read64(CXL_GPU_REG_RESULT1);

        if (result == CUDA_SUCCESS) {
            if (result_fail_idx != UINT64_MAX ||
                successfully_enqueued != count) {
                result = CUDA_ERROR_UNKNOWN;
            }
        } else if (result_fail_idx < count) {
            if (successfully_enqueued != result_fail_idx) {
                result = CUDA_ERROR_UNKNOWN;
            } else {
                *failIdx = (size_t)result_fail_idx;
            }
        } else if (result_fail_idx != UINT64_MAX ||
                   successfully_enqueued != 0) {
            result = CUDA_ERROR_UNKNOWN;
        }
    }

unlock:
    if (materialize_end_ns == 0)
        materialize_end_ns = guest_monotonic_ns();
    OLOG("batch_htod public_sequence=%" PRIu64
         " result=%d fail_idx=%" PRIu64
         " successfully_enqueued=%" PRIu64
         " ranges=%zu source_bytes=%zu payload_bytes=%zu"
         " lock_wait_duration_ns=%" PRIu64
         " materialize_duration_ns=%" PRIu64 "\n",
         g_async_copy_trace.public_sequence, result, result_fail_idx,
         successfully_enqueued, count, source_bytes, payload_bytes,
         lock_acquired_ns >= lock_wait_start_ns
             ? lock_acquired_ns - lock_wait_start_ns
             : 0,
         materialize_end_ns >= lock_acquired_ns
             ? materialize_end_ns - lock_acquired_ns
             : 0);
    cmd_unlock();
    result = async_copy_trace_end(result);
out:
    free(plan);
    return result;
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount) {
    OLOG("cuMemcpyDtoH(src=0x%lx, size=%zu)\n", (unsigned long)srcDevice, byteCount);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dstHost)
        return CUDA_ERROR_INVALID_VALUE;

    uint64_t bar4_offset = 0;
    if (bar4_pointer_range(dstHost, byteCount, &bar4_offset)) {
        size_t offset = 0;
        while (offset < byteCount) {
            size_t chunk = byteCount - offset;
            if (chunk > CXL_GPU_BULK_TRANSFER_SIZE)
                chunk = CXL_GPU_BULK_TRANSFER_SIZE;
            cmd_lock();
            reg_write64(CXL_GPU_REG_PARAM0, srcDevice + offset);
            reg_write64(CXL_GPU_REG_PARAM1, bar4_offset + offset);
            reg_write64(CXL_GPU_REG_PARAM2, chunk);
            CUresult err = execute_cmd(CXL_GPU_CMD_BULK_DTOH);
            cmd_unlock();
            if (err != CUDA_SUCCESS)
                return err;
            offset += chunk;
        }
        return CUDA_SUCCESS;
    }

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

        uint64_t data_read_start_ns = guest_monotonic_ns();
        data_read(0, (uint8_t *)dstHost + offset, chunk);
        uint64_t data_read_end_ns = guest_monotonic_ns();
        DLOG("copy_transport event=data-read transport=ram-bulk api=%s "
             "public_sequence=%" PRIu64
             " chunk_index=%u offset=%zu bytes=%zu duration_ns=%" PRIu64 "\n",
             g_async_copy_trace.active ? g_async_copy_trace.api : "cuMemcpyDtoH",
             g_async_copy_trace.active ? g_async_copy_trace.public_sequence : 0,
             g_async_copy_trace.command_index, offset, chunk,
             data_read_end_ns >= data_read_start_ns ?
                 data_read_end_ns - data_read_start_ns : 0);
        cmd_unlock();
        offset += chunk;
    }

    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    OLOG("cuMemcpyDtoHAsync(src=0x%lx, size=%zu, stream=%p)\n", (unsigned long)srcDevice, byteCount, hStream);
    /* knockout: DtoH uses a blocking Type-2 command. Synchronize the source
     * stream before the copy; add an async BAR2 command only when concurrent
     * stream execution is measured. */
    async_copy_trace_begin("cuMemcpyDtoHAsync", byteCount, 1, hStream,
                           "blocking");
    CUresult result = cuStreamSynchronize(hStream);
    if (result != CUDA_SUCCESS)
        return async_copy_trace_end(result);
    return async_copy_trace_end(cuMemcpyDtoH_v2(dstHost, srcDevice, byteCount));
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
    return cxl_module_load_image(module, image);
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

static bool cudart_size_add(size_t left, size_t right, size_t *sum) {
    return sum && !__builtin_add_overflow(left, right, sum);
}

static bool cudart_size_multiply(size_t left, size_t right, size_t *product) {
    return product && !__builtin_mul_overflow(left, right, product);
}

static bool cudart_range_within(size_t extent, size_t offset, size_t length) {
    size_t end;

    return cudart_size_add(offset, length, &end) && end <= extent;
}

static bool cudart_readable_mapping_extent(const void *pointer, size_t *extent) {
    FILE *maps;
    char line[1024];
    uintptr_t target;
    uintptr_t readable_end = 0;
    bool found = false;

    if (!pointer || !extent) {
        return false;
    }
    target = (uintptr_t)pointer;
    maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        return false;
    }

    while (fgets(line, sizeof(line), maps)) {
        uintptr_t start;
        uintptr_t end;
        char permissions[5] = {0};

        if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %4s", &start, &end, permissions) != 3) {
            continue;
        }
        if (!found) {
            if (target < start) {
                break;
            }
            if (target >= start && target < end) {
                if (permissions[0] != 'r') {
                    break;
                }
                readable_end = end;
                found = true;
            }
            continue;
        }
        if (start != readable_end || permissions[0] != 'r') {
            break;
        }
        readable_end = end;
    }
    fclose(maps);
    if (!found || readable_end <= target) {
        return false;
    }
    *extent = (size_t)(readable_end - target);
    return true;
}

static CUresult cudart_direct_elf_size(const void *code, size_t *elf_size) {
    const unsigned char *bytes = code;
    size_t readable_extent;
    Elf64_Ehdr header;
    size_t phdr_bytes = 0;
    size_t shdr_bytes = 0;
    size_t max_end;

    if (!code || !elf_size || !cudart_readable_mapping_extent(code, &readable_extent) ||
        readable_extent < sizeof(header)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy(&header, bytes, sizeof(header));
    if (memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_ident[EI_VERSION] != EV_CURRENT ||
        header.e_machine != EM_CUDA || header.e_ehsize != sizeof(header) || header.e_phnum == PN_XNUM ||
        (header.e_phnum && header.e_phentsize != sizeof(Elf64_Phdr)) ||
        (header.e_shnum && header.e_shentsize != sizeof(Elf64_Shdr))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!cudart_size_multiply(header.e_phnum, header.e_phentsize, &phdr_bytes) ||
        !cudart_size_multiply(header.e_shnum, header.e_shentsize, &shdr_bytes) ||
        !cudart_range_within(readable_extent, header.e_phoff, phdr_bytes) ||
        !cudart_range_within(readable_extent, header.e_shoff, shdr_bytes)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    max_end = sizeof(header);
    if (phdr_bytes) {
        size_t table_end = (size_t)header.e_phoff + phdr_bytes;
        if (table_end > max_end) {
            max_end = table_end;
        }
    }
    if (shdr_bytes) {
        size_t table_end = (size_t)header.e_shoff + shdr_bytes;
        if (table_end > max_end) {
            max_end = table_end;
        }
    }

    for (size_t index = 0; index < header.e_phnum; index++) {
        Elf64_Phdr program;
        size_t offset = (size_t)header.e_phoff + index * sizeof(program);
        size_t end;

        memcpy(&program, bytes + offset, sizeof(program));
        if (!cudart_size_add((size_t)program.p_offset, (size_t)program.p_filesz, &end) || end > readable_extent) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (end > max_end) {
            max_end = end;
        }
    }
    for (size_t index = 0; index < header.e_shnum; index++) {
        Elf64_Shdr section;
        size_t offset = (size_t)header.e_shoff + index * sizeof(section);
        size_t end;

        memcpy(&section, bytes + offset, sizeof(section));
        if (section.sh_type == SHT_NOBITS) {
            continue;
        }
        if (!cudart_size_add((size_t)section.sh_offset, (size_t)section.sh_size, &end) || end > readable_extent) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (end > max_end) {
            max_end = end;
        }
    }
    if (!max_end || max_end > CUDART_DIRECT_ELF_MAX_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *elf_size = max_end;
    return CUDA_SUCCESS;
}

static CUresult cudart_library_code_shape(const void *code, CudartLibraryCodeKind *kind, size_t *size) {
    size_t readable_extent;
    uint32_t magic;

    if (!code || !kind || !size || !cudart_readable_mapping_extent(code, &readable_extent) ||
        readable_extent < sizeof(magic)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy(&magic, code, sizeof(magic));
    if (magic == CUDART_FATBINC_MAGIC) {
        CudartFatbincWrapper wrapper;
        CudartFatbinHeader header;
        size_t header_extent;
        size_t total_size;

        if (readable_extent < sizeof(wrapper)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(&wrapper, code, sizeof(wrapper));
        if (wrapper.version != CUDART_FATBINC_VERSION || !wrapper.data ||
            !cudart_readable_mapping_extent(wrapper.data, &header_extent) || header_extent < sizeof(header)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(&header, wrapper.data, sizeof(header));
        if (header.magic != CUDART_FATBIN_MAGIC || header.version != CUDART_FATBIN_VERSION ||
            header.header_size < sizeof(header) || header.header_size > 4096 ||
            header.files_size > 256ULL * 1024ULL * 1024ULL ||
            !cudart_size_add(header.header_size, (size_t)header.files_size, &total_size) ||
            total_size > header_extent) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *kind = CUDART_LIBRARY_CODE_FATBIN;
        *size = 0;
        return CUDA_SUCCESS;
    }
    if (magic == CUDART_FATBIN_MAGIC) {
        CudartFatbinHeader header;
        size_t total_size;

        if (readable_extent < sizeof(header)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(&header, code, sizeof(header));
        if (header.version != CUDART_FATBIN_VERSION || header.header_size < sizeof(header) ||
            header.header_size > 4096 || header.files_size > 256ULL * 1024ULL * 1024ULL ||
            !cudart_size_add(header.header_size, (size_t)header.files_size, &total_size) ||
            total_size > readable_extent) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        *kind = CUDART_LIBRARY_CODE_FATBIN;
        *size = 0;
        return CUDA_SUCCESS;
    }
    if (readable_extent >= SELFMAG && memcmp(code, ELFMAG, SELFMAG) == 0) {
        CUresult result = cudart_direct_elf_size(code, size);
        if (result == CUDA_SUCCESS) {
            *kind = CUDART_LIBRARY_CODE_DIRECT_ELF;
        }
        return result;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

#ifdef CXL_GPU_CONTEXT_SHIM_TEST
CUresult cxl_cuda_test_direct_elf_size(const void *code, size_t *elf_size) {
    return cudart_direct_elf_size(code, elf_size);
}
#endif

static void cudart_log_fatbin_file_headers(const CudartFatbinHeader *header) {
    if (!g_debug && !g_observation_stream) {
        return;
    }
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
    if (!g_debug && !g_observation_stream) {
        return;
    }
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

static CUresult cxl_module_load_direct_elf(CUmodule *module, const void *image, size_t image_size) {
    if (image_size <= CXL_GPU_DATA_SIZE) {
        OLOG("direct ELF load raw_size=%zu encoding=raw\n", image_size);
        return cxl_module_load_cubin(module, image, image_size, 0, image_size);
    }

    size_t bound = ZSTD_compressBound(image_size);
    void *compressed = malloc(bound);
    if (!compressed) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    size_t compressed_size = ZSTD_compress(compressed, bound, image, image_size, 1);
    if (ZSTD_isError(compressed_size) || compressed_size > CXL_GPU_DATA_SIZE) {
        if (ZSTD_isError(compressed_size)) {
            fprintf(stderr, "[CXL-CUDA] direct ELF Zstd compression failed: %s\n",
                    ZSTD_getErrorName(compressed_size));
        } else {
            fprintf(stderr,
                    "[CXL-CUDA] direct ELF compressed payload exceeds BAR2 raw_size=%zu compressed_size=%zu "
                    "limit=%u\n",
                    image_size, compressed_size, CXL_GPU_DATA_SIZE);
        }
        free(compressed);
        return CUDA_ERROR_INVALID_VALUE;
    }

    OLOG("direct ELF load raw_size=%zu compressed_size=%zu encoding=zstd\n", image_size, compressed_size);
    CUresult result = cxl_module_load_cubin(module, compressed, compressed_size, CXL_GPU_MODULE_DATA_ZSTD, image_size);
    free(compressed);
    return result;
}

static CUresult cxl_module_load_image(CUmodule *module, const void *image) {
    uint32_t magic;
    memcpy(&magic, image, sizeof(magic));
    if (magic == CUDART_FATBINC_MAGIC || magic == CUDART_FATBIN_MAGIC || magic == UINT32_C(0x464c457f)) {
        CudartLibraryCodeKind kind;
        size_t image_size = 0;
        CUresult shape_result = cudart_library_code_shape(image, &kind, &image_size);
        if (shape_result != CUDA_SUCCESS) {
            return shape_result;
        }
        if (kind == CUDART_LIBRARY_CODE_DIRECT_ELF) {
            return cxl_module_load_direct_elf(module, image, image_size);
        }
        return cudart_load_module_from_fatbin(image, module);
    }

    size_t ptx_size = strnlen((const char *)image, CXL_GPU_DATA_SIZE);
    if (ptx_size == CXL_GPU_DATA_SIZE) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    data_write(0, image, ptx_size + 1);
    CUresult result = execute_cmd(CXL_GPU_CMD_MODULE_LOAD_PTX);
    if (result == CUDA_SUCCESS) {
        *module = (CUmodule)cxl_gpu_handle_from_id(reg_read64(CXL_GPU_REG_RESULT0));
        DLOG("  PTX module=%p\n", *module);
    }
    return result;
}

static CUresult cudart_library_materialize_module(CudartLibraryRecord *record) {
    if (!record || !record->alive) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (record->module) {
        return CUDA_SUCCESS;
    }

    CUresult result;
    if (record->code_kind == CUDART_LIBRARY_CODE_DIRECT_ELF) {
        if (!record->preserved_code_size || record->preserved_code_size > CUDART_DIRECT_ELF_MAX_SIZE) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        result = cxl_module_load_direct_elf(&record->module, record->preserved_code, record->preserved_code_size);
    } else {
        result = cudart_load_module_from_fatbin(record->code, &record->module);
    }
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[CXL-CUDA] library_record id=%u module materialization failed result=%d\n", record->id,
                result);
        return result;
    }

    OLOG("library_record id=%u module materialized module=%p\n", record->id, record->module);
    return CUDA_SUCCESS;
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
    void *caller = __builtin_return_address(0);
    Dl_info caller_info;
    if (caller && dladdr(caller, &caller_info) && caller_info.dli_fbase) {
        OLOG("cuLibraryLoadData caller=%p file=%s file_offset=0x%lx\n", caller,
             caller_info.dli_fname ? caller_info.dli_fname : "(unknown)",
             (unsigned long)((uintptr_t)caller - (uintptr_t)caller_info.dli_fbase));
    }
    cxl_cuda_provenance_emit_first_stack("cuLibraryLoadData");
    context_storage_log_entries("cuLibraryLoadData:entry");
    OLOG("cuLibraryLoadData(library=%p, code=%p, jitOptions=%p, jitOptionsValues=%p, numJitOptions=%u, "
         "libraryOptions=%p, libraryOptionValues=%p, numLibraryOptions=%u) -> library object\n",
         (void *)library, code, (void *)jitOptions, (void *)jitOptionsValues, numJitOptions, (void *)libraryOptions,
         (void *)libraryOptionValues, numLibraryOptions);
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
        OLOG("  libraryOptions[%u]=%d libraryOptionValues[%u]=%p\n", i, option, i, option_value);
        if (option == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE && option_value) {
            CUlibraryHostUniversalFunctionAndDataTable *table =
                (CUlibraryHostUniversalFunctionAndDataTable *)option_value;
            OLOG("  host_universal_table functionTable=%p functionWindowSize=%zu dataTable=%p dataWindowSize=%zu\n",
                 table->functionTable, table->functionWindowSize, table->dataTable, table->dataWindowSize);
        }
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

        if (option == CU_LIBRARY_BINARY_IS_PRESERVED) {
            preserve_binary = 1;
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

    CudartLibraryCodeKind code_kind;
    size_t code_size = 0;
    CUresult shape_result = cudart_library_code_shape(code, &code_kind, &code_size);
    if (shape_result != CUDA_SUCCESS) {
        fprintf(stderr, "[CXL-CUDA]   library object reject: unsupported or invalid code image\n");
        return shape_result;
    }
    if (code_kind == CUDART_LIBRARY_CODE_FATBIN) {
        cudart_log_fatbin_headers(code);
    } else {
        OLOG("  direct ELF validated size=%zu\n", code_size);
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
    record->preserved_code_size = code_size;
    record->code_kind = code_kind;
    record->num_jit_options = numJitOptions;
    record->num_library_options = numLibraryOptions;

    if (libraryOptions) {
        record->stored_library_options = numLibraryOptions;
        for (unsigned int i = 0; i < numLibraryOptions; i++) {
            record->options[i] = libraryOptions[i];
            record->option_values[i] = libraryOptionValues ? libraryOptionValues[i] : NULL;
            if (record->options[i] == CU_LIBRARY_BINARY_IS_PRESERVED) {
                record->preserve_binary = 1;
            }
        }
    }

    record->next = g_cudart_library_records;
    g_cudart_library_records = record;

    *library = (CUlibrary)record;
    OLOG("  library_record id=%u handle=%p code=%p code_file=%s code_base=0x%llx code_offset=0x%lx "
         "numJitOptions=%u numLibraryOptions=%u storedOptions=%u preserve_binary=%d code_kind=%u code_size=%zu "
         "module=%p alive=%d magic=0x%llx\n",
         record->id, (void *)*library, record->code, code_file, (unsigned long long)code_base, code_offset,
         record->num_jit_options, record->num_library_options, record->stored_library_options,
         record->preserve_binary, (unsigned int)record->code_kind, record->preserved_code_size, record->module,
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

    if (record->module) {
        CUresult result = cuModuleUnload(record->module);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "[CXL-CUDA] cuLibraryUnload(library=%p id=%u module=%p) -> error=%d\n", library,
                    record->id, record->module, result);
            return result;
        }
        record->module = NULL;
    }
    for (CudartKernelRecord *kernel = record->kernels; kernel; kernel = kernel->next) {
        kernel->alive = 0;
    }
    record->alive = 0;
    OLOG("cuLibraryUnload(library=%p id=%u) -> CUDA_SUCCESS\n", library, record->id);
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
    if (!pKernel || !library || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pKernel = NULL;
    CudartLibraryRecord *record = cudart_library_record_from_handle(library);
    if (!record || !record->alive) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUresult result = cudart_library_materialize_module(record);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[CXL-CUDA] cuLibraryGetKernel(library=%p id=%u name=%s) materialize -> error=%d\n",
                library, record->id, name, result);
        return result;
    }

    CUfunction function = NULL;
    result = cuModuleGetFunction(&function, record->module, name);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[CXL-CUDA] cuLibraryGetKernel(library=%p id=%u name=%s module=%p) -> error=%d\n",
                library, record->id, name, record->module, result);
        return result;
    }

    CudartKernelRecord *kernel = calloc(1, sizeof(*kernel));
    if (!kernel) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    kernel->name = strdup(name);
    if (!kernel->name) {
        free(kernel);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    kernel->magic = CUDART_KERNEL_RECORD_MAGIC;
    kernel->alive = 1;
    kernel->library = record;
    kernel->function = function;
    kernel->next = record->kernels;
    record->kernels = kernel;
    *pKernel = (CUkernel)kernel;
    OLOG("cuLibraryGetKernel(library=%p id=%u name=%s module=%p) -> kernel=%p function=%p CUDA_SUCCESS\n",
         library, record->id, name, record->module, (void *)*pKernel, function);
    return CUDA_SUCCESS;
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
    OLOG("cuLibraryGetModule(library=%p id=%u) -> module=%p CUDA_SUCCESS\n", library, record->id, record->module);
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

    CudartKernelRecord *record = cudart_kernel_record_from_handle(kernel);
    if (!record || !record->alive || !record->library || !record->library->alive || !record->function) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *pFunc = record->function;
    OLOG("cuKernelGetFunction(kernel=%p name=%s) -> function=%p CUDA_SUCCESS\n", kernel, record->name,
         record->function);
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
    uint64_t function_id;

    DLOG("cuFuncGetAttribute(func=%p, attrib=%d)\n", hfunc, attrib);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pi) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!cxl_gpu_handle_id(hfunc, &function_id)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, function_id);
    reg_write64(CXL_GPU_REG_PARAM1, (uint64_t)attrib);
    CUresult result = execute_cmd(CXL_GPU_CMD_FUNC_GET_ATTRIBUTE);
    if (result == CUDA_SUCCESS) {
        *pi = (int)(int64_t)reg_read64(CXL_GPU_REG_RESULT0);
    }
    cmd_unlock();
    return result;
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

static FunctionParamLayout *function_param_layout_find(CUfunction function) {
    uint64_t function_id;

    if (!cxl_gpu_handle_id(function, &function_id))
        return NULL;
    FunctionParamLayout **page =
        g_function_param_layout_pages[function_id >> FUNCTION_PARAM_LAYOUT_PAGE_BITS];
    return page ? page[function_id & (FUNCTION_PARAM_LAYOUT_PAGE_SIZE - 1)] : NULL;
}

static bool function_param_layout_insert(FunctionParamLayout *layout) {
    uint64_t function_id;

    if (!cxl_gpu_handle_id(layout->function, &function_id))
        return false;
    size_t page_index = function_id >> FUNCTION_PARAM_LAYOUT_PAGE_BITS;
    FunctionParamLayout **page = g_function_param_layout_pages[page_index];
    if (!page) {
        page = calloc(FUNCTION_PARAM_LAYOUT_PAGE_SIZE, sizeof(*page));
        if (!page)
            return false;
        g_function_param_layout_pages[page_index] = page;
    }
    page[function_id & (FUNCTION_PARAM_LAYOUT_PAGE_SIZE - 1)] = layout;
    return true;
}

static CUresult function_param_layout_copy(CUfunction function,
                                           size_t offsets[CXL_MAX_KERNEL_ARGS],
                                           size_t sizes[CXL_MAX_KERNEL_ARGS],
                                           uint32_t *num_args, size_t *extent) {
    uint32_t backend_queries = 0;
    uint32_t layout_commands = 0;
    uint64_t function_id;

    if (!cxl_gpu_handle_id(function, &function_id))
        return CUDA_ERROR_INVALID_HANDLE;

    pthread_mutex_lock(&g_function_param_layouts_lock);
    FunctionParamLayout *layout = function_param_layout_find(function);
    if (!layout) {
        FunctionParamLayout candidate = {.function = function};
        CXLFunctionParamLayoutWire wire;

        cmd_lock();
        reg_write64(CXL_GPU_REG_PARAM0, function_id);
        layout_commands = 1;
        CUresult result = execute_cmd(CXL_GPU_CMD_FUNC_GET_PARAM_LAYOUT);
        if (result == CUDA_SUCCESS) {
            data_read(0, &wire, sizeof(wire));
            backend_queries = reg_read64(CXL_GPU_REG_RESULT0);
        }
        cmd_unlock();
        if (result != CUDA_SUCCESS) {
            pthread_mutex_unlock(&g_function_param_layouts_lock);
            return result;
        }
        if (wire.reserved != 0 || wire.num_args > CXL_MAX_KERNEL_ARGS ||
            wire.extent > CXL_GPU_DATA_SIZE) {
            pthread_mutex_unlock(&g_function_param_layouts_lock);
            return CUDA_ERROR_INVALID_VALUE;
        }
        candidate.num_args = wire.num_args;
        candidate.extent = wire.extent;
        for (uint32_t i = 0; i < wire.num_args; i++) {
            if (wire.params[i].offset > wire.extent ||
                wire.params[i].size > wire.extent - wire.params[i].offset) {
                pthread_mutex_unlock(&g_function_param_layouts_lock);
                return CUDA_ERROR_INVALID_VALUE;
            }
            candidate.offsets[i] = wire.params[i].offset;
            candidate.sizes[i] = wire.params[i].size;
        }

        layout = calloc(1, sizeof(*layout));
        if (!layout) {
            pthread_mutex_unlock(&g_function_param_layouts_lock);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        *layout = candidate;
        if (!function_param_layout_insert(layout)) {
            free(layout);
            pthread_mutex_unlock(&g_function_param_layouts_lock);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    memcpy(offsets, layout->offsets, sizeof(layout->offsets));
    memcpy(sizes, layout->sizes, sizeof(layout->sizes));
    *num_args = layout->num_args;
    *extent = layout->extent;
    pthread_mutex_unlock(&g_function_param_layouts_lock);

    OLOG("function_param_layout event=%s function=%p args=%u extent=%zu backend_queries=%u layout_commands=%u\n",
         layout_commands ? "miss" : "hit", function, *num_args, *extent,
         backend_queries, layout_commands);
    return CUDA_SUCCESS;
}

CUresult cuFuncGetParamInfo(CUfunction hfunc, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
    size_t offsets[CXL_MAX_KERNEL_ARGS];
    size_t sizes[CXL_MAX_KERNEL_ARGS];
    size_t extent;
    uint32_t num_args;

    OLOG("cuFuncGetParamInfo(func=%p, index=%zu)\n", hfunc, paramIndex);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!hfunc || !paramOffset)
        return CUDA_ERROR_INVALID_VALUE;

    CUresult result = function_param_layout_copy(
        hfunc, offsets, sizes, &num_args, &extent);
    if (result != CUDA_SUCCESS)
        return result;
    if (paramIndex >= num_args)
        return CUDA_ERROR_INVALID_VALUE;

    *paramOffset = offsets[paramIndex];
    if (paramSize)
        *paramSize = sizes[paramIndex];
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    OLOG("cuModuleGetFunction(mod=%p, name=%s)\n", hmod, name);
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
        OLOG("  func=%p\n", *hfunc);
    }
    cmd_unlock();
    return err;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    OLOG("cuModuleGetGlobal_v2(dptr=%p, bytes=%p, mod=%p, name=%s)\n", (void *)dptr, (void *)bytes, hmod,
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

    OLOG("cuLaunchKernel(f=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u)\n", f, gridDimX, gridDimY, gridDimZ,
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
    CUresult layout_result = function_param_layout_copy(
        f, param_offsets, param_sizes, &num_args, &param_extent);
    if (layout_result != CUDA_SUCCESS)
        return layout_result;
    for (uint32_t i = 0; i < num_args; i++) {
        if (!kernelParams || !kernelParams[i])
            return CUDA_ERROR_INVALID_VALUE;
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
    uint64_t id, wire;
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_stream_handle_id(hStream, &id) ||
        !cxl_gpu_stream_wire(hStream, &wire))
        return CUDA_ERROR_INVALID_HANDLE;
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, id);
    CUresult result = execute_cmd(CXL_GPU_CMD_STREAM_DESTROY);
    if (result == CUDA_SUCCESS && g_direct_source_enabled)
        result = direct_sources_complete_locked(wire, false);
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
    if (result == CUDA_SUCCESS && g_direct_source_enabled)
        result = direct_sources_complete_locked(wire, false);
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
    CUresult result = cxl_cuda_context_primary_release();
    if (result == CUDA_SUCCESS)
        function_param_layouts_clear("primary-context-release");
    return result;
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
    CUresult result = cxl_cuda_context_primary_reset();
    if (result == CUDA_SUCCESS)
        function_param_layouts_clear("primary-context-reset");
    return result;
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

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, dstDevice);
    reg_write64(CXL_GPU_REG_PARAM1, srcDevice);
    reg_write64(CXL_GPU_REG_PARAM2, byteCount);
    CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_DTOD);
    cmd_unlock();
    return result;
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
    uint64_t stream_wire;

    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!dstDevice || !srcDevice)
        return CUDA_ERROR_INVALID_VALUE;
    if (!cxl_gpu_stream_wire(hStream, &stream_wire))
        return CUDA_ERROR_INVALID_HANDLE;

    async_copy_trace_begin("cuMemcpyDtoDAsync", byteCount, 1, hStream,
                           "stream-forwarded");
    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, dstDevice);
    reg_write64(CXL_GPU_REG_PARAM1, srcDevice);
    reg_write64(CXL_GPU_REG_PARAM2, byteCount);
    reg_write64(CXL_GPU_REG_PARAM3, stream_wire);
    CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC);
    cmd_unlock();
    return async_copy_trace_end(result);
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

    CUdeviceptr src;
    CUdeviceptr dst;
    size_t last_row = copy->Height - 1;
    if (copy->srcY > SIZE_MAX - last_row || copy->dstY > SIZE_MAX - last_row ||
        !cxl_memcpy2d_offset(copy->srcDevice, copy->srcPitch, copy->srcXInBytes, copy->srcY, copy->WidthInBytes,
                             &src) ||
        !cxl_memcpy2d_offset(copy->dstDevice, copy->dstPitch, copy->dstXInBytes, copy->dstY, copy->WidthInBytes,
                             &dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    CUdeviceptr ignored;
    if (!cxl_memcpy2d_offset(src, copy->srcPitch, 0, last_row, copy->WidthInBytes, &ignored) ||
        !cxl_memcpy2d_offset(dst, copy->dstPitch, 0, last_row, copy->WidthInBytes, &ignored)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, dst);
    reg_write64(CXL_GPU_REG_PARAM1, src);
    reg_write64(CXL_GPU_REG_PARAM2, copy->dstPitch);
    reg_write64(CXL_GPU_REG_PARAM3, copy->srcPitch);
    reg_write64(CXL_GPU_REG_PARAM4, copy->WidthInBytes);
    reg_write64(CXL_GPU_REG_PARAM5, copy->Height);
    CUresult result = execute_cmd(CXL_GPU_CMD_MEM_COPY_2D_DTOD);
    cmd_unlock();
    return result;
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *copy) {
    DLOG("cuMemcpy2D_v2(copy=%p)\n", (const void *)copy);
    return cxl_memcpy2d_device_to_device(copy);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D *copy) { return cuMemcpy2D_v2(copy); }

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *copy, CUstream hStream) {
    DLOG("cuMemcpy2DAsync_v2(copy=%p, stream=%p)\n", (const void *)copy, hStream);
    /* knockout: Type-2 currently serializes transfer commands. Preserve the
     * existing async copy contract by completing this multidimensional copy
     * before return; add a stream-aware BAR2 protocol only after it is measured. */
    size_t total_bytes = 0;
    if (copy && (copy->Height == 0 || copy->WidthInBytes <= SIZE_MAX / copy->Height))
        total_bytes = copy->WidthInBytes * copy->Height;
    async_copy_trace_begin("cuMemcpy2DAsync", total_bytes, 1, hStream,
                           "blocking");
    return async_copy_trace_end(cxl_memcpy2d_device_to_device(copy));
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
    uint64_t module_id;

    DLOG("cuModuleUnload(%p)\n", hmod);
    if (!g_initialized)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!cxl_gpu_handle_id(hmod, &module_id))
        return CUDA_ERROR_INVALID_HANDLE;

    cmd_lock();
    reg_write64(CXL_GPU_REG_PARAM0, module_id);
    CUresult result = execute_cmd(CXL_GPU_CMD_MODULE_UNLOAD);
    cmd_unlock();
    if (result == CUDA_SUCCESS)
        function_param_layouts_clear("module-unload");
    return result;
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

static bool bar4_pointer_range(const void *host_ptr, size_t size, uint64_t *offset) {
    uintptr_t ptr = (uintptr_t)host_ptr;
    uintptr_t base = (uintptr_t)g_bar4_ptr;

    if (!host_ptr || !offset || !g_bar4_ptr || ptr < base || ptr - base > g_bar4_size)
        return false;
    if (size > g_bar4_size - (ptr - base))
        return false;
    *offset = (uint64_t)(ptr - base);
    return true;
}

static CXLBar4RangeKind bar4_range_kind(const void *host_ptr, size_t size,
                                        uint64_t *offset) {
    uintptr_t ptr = (uintptr_t)host_ptr;
    uintptr_t base = (uintptr_t)g_bar4_ptr;
    uintptr_t end;
    uintptr_t bar4_end;

    if (!host_ptr || !size || !g_bar4_ptr)
        return CXL_BAR4_RANGE_ORDINARY;
    if (size - 1 > UINTPTR_MAX - ptr || g_bar4_size - 1 > UINTPTR_MAX - base)
        return CXL_BAR4_RANGE_PARTIAL;
    end = ptr + size - 1;
    bar4_end = base + g_bar4_size - 1;
    if (ptr >= base && end <= bar4_end) {
        if (offset)
            *offset = (uint64_t)(ptr - base);
        return CXL_BAR4_RANGE_CONTAINED;
    }
    if (ptr <= bar4_end && end >= base)
        return CXL_BAR4_RANGE_PARTIAL;
    return CXL_BAR4_RANGE_ORDINARY;
}

static bool parse_positive_size(const char *name, const char *value,
                                size_t *result) {
    char *end = NULL;
    uintmax_t parsed;

    if (!value || !value[0] || value[0] == '-' || value[0] == '+')
        return false;
    errno = 0;
    parsed = strtoumax(value, &end, 10);
    if (errno || !end || *end || !parsed || parsed > SIZE_MAX) {
        fprintf(stderr, "[CXL-CUDA] invalid %s: %s\n", name,
                value ? value : "<unset>");
        return false;
    }
    *result = (size_t)parsed;
    return true;
}

static void htod_route_parse(void) {
    const char *mode = getenv("CXL_CUDA_HTOD_ROUTE_MODE");
    const char *minimum = getenv("CXL_CUDA_HTOD_ROUTE_MIN_BYTES");
    const char *prefix = getenv("CXL_CUDA_HTOD_ROUTE_PREFIX_BYTES");
    const char *total = getenv("CXL_CUDA_HTOD_ROUTE_TOTAL_BYTES");

    if (!mode || strcmp(mode, "disabled") == 0) {
        if (minimum || prefix || total) {
            fprintf(stderr, "[CXL-CUDA] disabled HtoD route has numeric fields\n");
            abort();
        }
        return;
    }
    bool prefix_mode = strcmp(mode, "cxlmem-bounded-prefix") == 0;
    bool full_mode = strcmp(mode, "cxlmem-bounded-full-transfer") == 0;
    if ((!prefix_mode && !full_mode) ||
        !parse_positive_size("CXL_CUDA_HTOD_ROUTE_MIN_BYTES", minimum,
                             &g_htod_route.minimum_transfer_bytes) ||
        !parse_positive_size("CXL_CUDA_HTOD_ROUTE_PREFIX_BYTES", prefix,
                             &g_htod_route.prefix_bytes_per_transfer) ||
        !parse_positive_size("CXL_CUDA_HTOD_ROUTE_TOTAL_BYTES", total,
                             &g_htod_route.total_bytes) ||
        g_htod_route.prefix_bytes_per_transfer > CXL_GPU_BULK_TRANSFER_SIZE ||
        (prefix_mode && g_htod_route.minimum_transfer_bytes <
                            g_htod_route.prefix_bytes_per_transfer)) {
        fprintf(stderr, "[CXL-CUDA] invalid bounded HtoD route contract\n");
        abort();
    }
    g_htod_route.enabled = true;
    g_htod_route.full_transfer = full_mode;
    g_htod_route.remaining_bytes = g_htod_route.total_bytes;
}

/* Caller holds cmd_lock. Route staging must be acknowledged by QEMU; the
 * allocator's local bump fallback is not valid for this transport. */
static CUresult htod_route_allocate_locked(void) {
    volatile uint8_t *bar4;
    uint64_t offset;
    CUresult result;

    if (g_htod_route.staging)
        return CUDA_SUCCESS;
    if (!g_transport.regs)
        return CUDA_ERROR_NOT_INITIALIZED;
    bar4 = ensure_bar4();
    if (!bar4)
        return CUDA_ERROR_NOT_INITIALIZED;
    reg_write64(CXL_GPU_REG_PARAM0, g_htod_route.prefix_bytes_per_transfer);
    result = execute_cmd(CXL_GPU_CMD_COHERENT_ALLOC);
    if (result != CUDA_SUCCESS)
        return result;
    offset = reg_read64(CXL_GPU_REG_RESULT0);
    if (offset > g_bar4_size ||
        g_htod_route.prefix_bytes_per_transfer > g_bar4_size - offset)
        return CUDA_ERROR_INVALID_VALUE;
    g_htod_route.staging_offset = offset;
    g_htod_route.staging = (void *)(bar4 + offset);
    return CUDA_SUCCESS;
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

uint64_t cxlHostToDevice(void *host_ptr) {
    return bar4_offset_of(host_ptr);
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
__attribute__((constructor)) static void libcuda_init(void) {
    static const size_t observation_buffer_size = 1024 * 1024;
    const char *observation_log = getenv("CXL_CUDA_OBSERVATION_LOG");
    const char *direct_source = getenv("CXL_CUDA_DIRECT_SOURCE");

    g_debug = (getenv("CXL_CUDA_DEBUG") != NULL);
    if (direct_source) {
        if (strcmp(direct_source, "0") == 0)
            g_direct_source_enabled = 0;
        else if (strcmp(direct_source, "1") == 0)
            g_direct_source_enabled = 1;
        else {
            fprintf(stderr,
                    "[CXL-CUDA] CXL_CUDA_DIRECT_SOURCE must be 0 or 1\n");
            abort();
        }
    }
    if (observation_log) {
        if (observation_log[0] != '/') {
            fprintf(stderr, "[CXL-CUDA] CXL_CUDA_OBSERVATION_LOG must be an absolute path\n");
            abort();
        }
        g_observation_stream = fopen(observation_log, "w");
        if (!g_observation_stream) {
            fprintf(stderr, "[CXL-CUDA] failed to open observation log %s: %s\n", observation_log, strerror(errno));
            abort();
        }
        g_observation_buffer = malloc(observation_buffer_size);
        if (!g_observation_buffer ||
            setvbuf(g_observation_stream, g_observation_buffer, _IOFBF, observation_buffer_size) != 0) {
            fprintf(stderr, "[CXL-CUDA] failed to buffer observation log %s\n", observation_log);
            fclose(g_observation_stream);
            g_observation_stream = NULL;
            free(g_observation_buffer);
            g_observation_buffer = NULL;
            abort();
        }
    }
    htod_route_parse();
    fprintf(stderr,
            "[CXL-CUDA] htod_route_config mode=%s minimum_transfer_bytes=%zu "
            "prefix_bytes_per_transfer=%zu total_bytes=%zu\n",
            htod_route_mode(),
            g_htod_route.minimum_transfer_bytes,
            g_htod_route.prefix_bytes_per_transfer, g_htod_route.total_bytes);
    fprintf(stderr, "[CXL-CUDA] direct_source_config enabled=%d\n",
            g_direct_source_enabled);
    DLOG("libcuda.so loaded (CXL Type 2 shim)\n");
}

__attribute__((destructor)) static void libcuda_cleanup(void) {
    CXLCudaErrorName *error_name;

    DLOG("libcuda.so unloading\n");
    observation_abandon_active_decode();
    function_param_layouts_clear("process-exit");
    graph_kernel_node_snapshots_clear();
    context_storage_clear_context(NULL, 0);
    if (g_htod_route.staging && cxlCoherentFree(g_htod_route.staging) != 0) {
        fprintf(stderr, "[CXL-CUDA] failed to release HtoD route staging\n");
        abort();
    }
    g_htod_route.staging = NULL;
    fprintf(stderr,
            "[CXL-CUDA] htod_route_summary mode=%s routed_calls=%zu "
            "routed_bytes=%zu remaining_bytes=%zu fallback_count=%zu\n",
            htod_route_mode(),
            g_htod_route.routed_calls, g_htod_route.routed_bytes,
            g_htod_route.remaining_bytes, g_htod_route.fallback_count);
    if (g_bar4_ptr) {
        munmap((void *)g_bar4_ptr, g_bar4_size);
        g_bar4_ptr = NULL;
    }
    if (g_bar4_fd >= 0) {
        close(g_bar4_fd);
        g_bar4_fd = -1;
    }
    cxl_gpu_transport_close(&g_transport);
    if (g_observation_stream) {
        if (fclose(g_observation_stream) != 0) {
            fprintf(stderr, "[CXL-CUDA] failed to flush observation log: %s\n", strerror(errno));
            abort();
        }
        g_observation_stream = NULL;
        free(g_observation_buffer);
        g_observation_buffer = NULL;
    }
    while ((error_name = g_cuda_error_names) != NULL) {
        g_cuda_error_names = error_name->next;
        free(error_name->name);
        free(error_name);
    }
    g_initialized = 0;
}

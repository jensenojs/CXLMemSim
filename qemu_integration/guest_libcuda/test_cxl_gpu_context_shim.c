#include "cxl_gpu_cmd.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUarray;
typedef void *CUlibrary;
typedef int CUdriverProcAddressQueryResult;
typedef int CUmemorytype;
typedef int CUlibraryOption;

typedef struct {
    unsigned char bytes[16];
} CUuuid;

typedef struct {
    size_t srcXInBytes;
    size_t srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    uint64_t srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    uint64_t dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2D;

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

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_NO_BINARY_FOR_GPU 209
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE 708
#define CUDA_ERROR_CONTEXT_IS_DESTROYED 709
#define CUDA_ERROR_NOT_SUPPORTED 801

#define CU_MEMORYTYPE_DEVICE 0x02
#define CU_LIBRARY_BINARY_IS_PRESERVED 1

#define CUDART_FATBIN_MAGIC 0xBA55ED50U
#define CUDART_FATBIN_VERSION 0x1U
#define CUDART_FATBIN_KIND_PTX 0x1U
#define CUDART_FATBIN_KIND_ELF 0x2U

#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50

void cxl_cuda_test_reset(void);
void cxl_cuda_test_set_executor(CUresult (*executor)(uint32_t cmd));
uint64_t cxl_cuda_test_read_reg64(uint32_t offset);
void cxl_cuda_test_write_result(unsigned int index, uint64_t value);
void cxl_cuda_test_write_reg32(uint32_t offset, uint32_t value);
void cxl_cuda_test_read_data(size_t offset, void *dst, size_t length);

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult cuDevicePrimaryCtxReset(CUdevice dev);
CUresult cuPointerGetAttribute(void *data, int attribute, uint64_t ptr);
CUresult cuGetExportTable(const void **table, const CUuuid *uuid);
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, uint64_t flags,
                          CUdriverProcAddressQueryResult *symbolStatus);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize,
                                                              size_t dynamicSMemSize, unsigned int flags);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize,
                                                     size_t dynamicSMemSize);
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *copy, CUstream stream);
CUresult cuLibraryLoadData(CUlibrary *library, const void *code, void *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions);
CUresult cuLibraryUnload(CUlibrary library);
CUresult cuLibraryGetModule(void **module, CUlibrary library);
CUresult cuStreamCreate(CUstream *stream, unsigned int flags);

#define CHECK(expr)                                                                                                    \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);                                                 \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static uint32_t commands[32];
static unsigned int command_count;
static uint64_t issued_token = 41;
static int identity_pci_bus = 131;
static int identity_pci_device;
static int identity_pci_domain;
static CUresult occupancy_result = CUDA_SUCCESS;
static uint64_t occupancy_expected_flags;
static uint64_t memcpy2d_src_rows[8];
static uint64_t memcpy2d_dst_rows[8];
static unsigned int memcpy2d_row_count;
static unsigned int memcpy2d_phase;
static uint64_t memcpy2d_width;
static unsigned int cubin_load_count;

static const CUuuid integrity_check_uuid = {
    .bytes = {0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2},
};

static const CUuuid context_local_storage_uuid = {
    .bytes = {0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93},
};

static const CUuuid context_checks_uuid = {
    .bytes = {0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43, 0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e},
};

static const CUuuid tools_tls_uuid = {
    .bytes = {0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc},
};

static const CUuuid cublas_context_stream_uuid = {
    .bytes = {0x21, 0x31, 0x8c, 0x60, 0x97, 0x14, 0x32, 0x48, 0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2},
};

typedef struct DestroyedContextThread {
    pthread_barrier_t attached;
    pthread_barrier_t destroyed;
    CUcontext token;
    CUcontext current;
    CUdevice device;
    size_t free_bytes;
    size_t total_bytes;
    CUresult set_current_result;
    CUresult get_current_result;
    CUresult get_device_result;
    CUresult mem_info_result;
} DestroyedContextThread;

static CUresult fake_execute(uint32_t command) {
    commands[command_count++] = command;
    switch (command) {
    case CXL_GPU_CMD_CTX_CREATE:
        cxl_cuda_test_write_result(0, issued_token);
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_STREAM_CREATE:
        cxl_cuda_test_write_result(0, 7);
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_MEM_GET_INFO:
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == issued_token);
        cxl_cuda_test_write_result(0, UINT64_C(0x100000000));
        cxl_cuda_test_write_result(1, UINT64_C(0x200000000));
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_GET_TOTAL_MEM:
        cxl_cuda_test_write_result(0, UINT64_C(0x300000000));
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE:
        switch ((int)cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0)) {
        case 97:
            cxl_cuda_test_write_result(0, 49152);
            return CUDA_SUCCESS;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
            cxl_cuda_test_write_result(0, (uint64_t)(int64_t)identity_pci_bus);
            return CUDA_SUCCESS;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
            cxl_cuda_test_write_result(0, (uint64_t)(int64_t)identity_pci_device);
            return CUDA_SUCCESS;
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
            cxl_cuda_test_write_result(0, (uint64_t)(int64_t)identity_pci_domain);
            return CUDA_SUCCESS;
        default:
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    case CXL_GPU_CMD_CTX_DESTROY:
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_FUNC_GET_OCCUPANCY:
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == 4);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM1) == 128);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM2) == 73728);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM3) == occupancy_expected_flags);
        if (occupancy_result == CUDA_SUCCESS) {
            cxl_cuda_test_write_result(0, 3);
        }
        return occupancy_result;
    case CXL_GPU_CMD_MEM_COPY_DTOH:
        CHECK(memcpy2d_phase / 2 < memcpy2d_row_count);
        CHECK(memcpy2d_phase % 2 == 0);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == memcpy2d_src_rows[memcpy2d_phase / 2]);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM1) == memcpy2d_width);
        memcpy2d_phase++;
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_MEM_COPY_HTOD:
        CHECK(memcpy2d_phase / 2 < memcpy2d_row_count);
        CHECK(memcpy2d_phase % 2 == 1);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == memcpy2d_dst_rows[memcpy2d_phase / 2]);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM1) == memcpy2d_width);
        memcpy2d_phase++;
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_MODULE_LOAD_CUBIN: {
        unsigned char observed[8] = {0};
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == sizeof(observed));
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM1) == 0);
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM2) == sizeof(observed));
        cxl_cuda_test_read_data(0, observed, sizeof(observed));
        CHECK(observed[0] == 0x80);
        cubin_load_count++;
        cxl_cuda_test_write_result(0, issued_token);
        return CUDA_SUCCESS;
    }
    default:
        return CUDA_ERROR_INVALID_CONTEXT;
    }
}

static int test_query_and_context_sequence(void) {
    CUcontext context = NULL;
    CUcontext current = (CUcontext)1;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    size_t device_total = 0;
    int attribute = 0;
    int memory_type = 0;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;

    CHECK(cuMemGetInfo_v2(&free_bytes, &total_bytes) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(command_count == 0);
    CHECK(cuCtxSetCurrent((CUcontext)(uintptr_t)99) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(command_count == 0);
    CHECK(cuCtxCreate_v2(&context, 1, 0) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(command_count == 0);
    CHECK(cuCtxCreate_v2(&context, 0, 0) == CUDA_SUCCESS);
    CHECK(context == (CUcontext)(uintptr_t)issued_token);
    CHECK(command_count == 1 && commands[0] == CXL_GPU_CMD_CTX_CREATE);
    CHECK(cuMemGetInfo_v2(&free_bytes, &total_bytes) == CUDA_SUCCESS);
    CHECK(free_bytes == UINT64_C(0x100000000));
    CHECK(total_bytes == UINT64_C(0x200000000));
    CHECK(command_count == 2 && commands[1] == CXL_GPU_CMD_MEM_GET_INFO);
    CHECK(cuDeviceTotalMem_v2(&device_total, 0) == CUDA_SUCCESS);
    CHECK(device_total == UINT64_C(0x300000000));
    CHECK(cuDeviceGetAttribute(&attribute, 97, 0) == CUDA_SUCCESS);
    CHECK(attribute == 49152);
    CHECK(cuPointerGetAttribute(&memory_type, 1, 0) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(command_count == 4);
    CHECK(cuCtxDestroy_v2(context) == CUDA_SUCCESS);
    CHECK(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == NULL);
    return 0;
}

static int test_primary_retain_does_not_become_current(void) {
    CUcontext primary = NULL;
    CUcontext current = (CUcontext)1;
    unsigned int flags = 1;
    int active = -1;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    CHECK(cuDevicePrimaryCtxRetain(&primary, 0) == CUDA_SUCCESS);
    CHECK(primary == (CUcontext)(uintptr_t)issued_token);
    CHECK(command_count == 1 && commands[0] == CXL_GPU_CMD_CTX_CREATE);
    CHECK(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == NULL);
    CHECK(cuDevicePrimaryCtxRetain(&primary, 0) == CUDA_SUCCESS);
    CHECK(command_count == 1);
    CHECK(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(flags == 0 && active == 1);
    CHECK(cuDevicePrimaryCtxSetFlags(0, 0) == CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
    CHECK(cuDevicePrimaryCtxSetFlags(0, 1) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(command_count == 1);
    CHECK(cuDevicePrimaryCtxRelease(0) == CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxRelease(0) == CUDA_SUCCESS);
    CHECK(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS);
    CHECK(flags == 0 && active == 0);
    CHECK(cuDevicePrimaryCtxReset(0) == CUDA_SUCCESS);
    CHECK(command_count == 1);
    return 0;
}

static void *thread_with_destroyed_context(void *opaque) {
    DestroyedContextThread *thread = opaque;

    thread->set_current_result = cuCtxSetCurrent(thread->token);
    if (thread->set_current_result != CUDA_SUCCESS)
        return NULL;
    thread->get_current_result = cuCtxGetCurrent(&thread->current);
    pthread_barrier_wait(&thread->attached);
    pthread_barrier_wait(&thread->destroyed);
    thread->get_current_result = cuCtxGetCurrent(&thread->current);
    thread->get_device_result = cuCtxGetDevice(&thread->device);
    thread->mem_info_result = cuMemGetInfo_v2(&thread->free_bytes, &thread->total_bytes);
    return NULL;
}

static int test_destroy_keeps_other_thread_token_without_transport(void) {
    DestroyedContextThread thread = {0};
    CUcontext context = NULL;
    pthread_t worker;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    CHECK(cuCtxCreate_v2(&context, 0, 0) == CUDA_SUCCESS);
    CHECK(context == (CUcontext)(uintptr_t)issued_token);
    thread.token = context;
    CHECK(pthread_barrier_init(&thread.attached, NULL, 2) == 0);
    CHECK(pthread_barrier_init(&thread.destroyed, NULL, 2) == 0);
    CHECK(pthread_create(&worker, NULL, thread_with_destroyed_context, &thread) == 0);
    pthread_barrier_wait(&thread.attached);
    CHECK(thread.set_current_result == CUDA_SUCCESS);
    CHECK(cuCtxDestroy_v2(context) == CUDA_SUCCESS);
    pthread_barrier_wait(&thread.destroyed);
    CHECK(pthread_join(worker, NULL) == 0);
    CHECK(thread.get_current_result == CUDA_SUCCESS);
    CHECK(thread.current == context);
    CHECK(thread.get_device_result == CUDA_ERROR_CONTEXT_IS_DESTROYED);
    CHECK(thread.mem_info_result == CUDA_ERROR_CONTEXT_IS_DESTROYED);
    CHECK(command_count == 2);
    CHECK(commands[0] == CXL_GPU_CMD_CTX_CREATE);
    CHECK(commands[1] == CXL_GPU_CMD_CTX_DESTROY);
    pthread_barrier_destroy(&thread.destroyed);
    pthread_barrier_destroy(&thread.attached);
    return 0;
}

static int test_integrity_export_table_shape(void) {
    typedef CUresult (*integrity_check_t)(uint32_t version, uint64_t unix_seconds, uint64_t result[2]);
    typedef CUresult (*integrity_enable_t)(int enabled);
    const void *table = NULL;
    uint64_t result[2] = {0, 0};

    CHECK(cuGetExportTable(&table, &integrity_check_uuid) == CUDA_SUCCESS);
    CHECK(table != NULL);
    const void *const *slots = table;
    CHECK((uintptr_t)slots[0] == 3 * sizeof(void *));
    CHECK(slots[1] != NULL);
    CHECK(slots[2] != NULL);
    CHECK(((integrity_enable_t)slots[2])(1) == CUDA_SUCCESS);
    CHECK(((integrity_enable_t)slots[2])(0) == CUDA_SUCCESS);
    CHECK(((integrity_check_t)slots[1])(12090, UINT64_C(1784320000), result) == CUDA_SUCCESS);
    CHECK(result[0] == UINT64_C(0x3341181c03cb675c));
    CHECK(result[1] == UINT64_C(0x8ed383aa1f4cd1e8));
    return 0;
}

static int test_context_local_storage_keeps_managers_separate(void) {
    typedef CUresult (*context_storage_put_t)(CUcontext context, void *state_mgr, void *ctx_state, void *dtor);
    typedef CUresult (*context_storage_delete_t)(CUcontext context, void *state_mgr);
    typedef CUresult (*context_storage_get_t)(void **ctx_state, CUcontext context, void *state_mgr);
    const void *table = NULL;
    void *manager_a = (void *)(uintptr_t)0xa1;
    void *manager_b = (void *)(uintptr_t)0xb2;
    void *state_a = (void *)(uintptr_t)0xaaa1;
    void *state_b = (void *)(uintptr_t)0xbbb2;
    void *replacement_a = (void *)(uintptr_t)0xaaa3;
    void *out = NULL;
    CUcontext current = NULL;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    CHECK(cuGetExportTable(&table, &context_local_storage_uuid) == CUDA_SUCCESS);
    CHECK(table != NULL);
    const void *const *slots = table;
    context_storage_put_t put = (context_storage_put_t)slots[0];
    context_storage_delete_t delete = (context_storage_delete_t)slots[1];
    context_storage_get_t get = (context_storage_get_t)slots[2];
    CHECK(put != NULL && delete != NULL && get != NULL);

    out = (void *)(uintptr_t)0x1;
    CHECK(get(&out, NULL, manager_a) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(out == NULL);

    CHECK(cuCtxCreate_v2(&current, 0, 0) == CUDA_SUCCESS);
    CHECK(current == (CUcontext)(uintptr_t)issued_token);
    CHECK(command_count == 1 && commands[0] == CXL_GPU_CMD_CTX_CREATE);
    out = (void *)(uintptr_t)0x1;
    CHECK(get(&out, NULL, manager_a) == CUDA_ERROR_INVALID_HANDLE);
    CHECK(out == NULL);

    CHECK(put(NULL, manager_a, state_a, NULL) == CUDA_SUCCESS);
    CHECK(put(NULL, manager_b, state_b, NULL) == CUDA_SUCCESS);
    CHECK(get(&out, NULL, manager_a) == CUDA_SUCCESS);
    CHECK(out == state_a);
    out = NULL;
    CHECK(get(&out, NULL, manager_b) == CUDA_SUCCESS);
    CHECK(out == state_b);

    CHECK(delete(NULL, manager_a) == CUDA_ERROR_DEINITIALIZED);
    out = (void *)(uintptr_t)0x1;
    CHECK(get(&out, NULL, manager_a) == CUDA_ERROR_INVALID_HANDLE);
    CHECK(out == NULL);
    CHECK(get(&out, NULL, manager_b) == CUDA_SUCCESS);
    CHECK(out == state_b);

    CHECK(put(NULL, manager_a, replacement_a, NULL) == CUDA_SUCCESS);
    out = NULL;
    CHECK(get(&out, NULL, manager_a) == CUDA_SUCCESS);
    CHECK(out == replacement_a);
    out = NULL;
    CHECK(get(&out, NULL, manager_b) == CUDA_SUCCESS);
    CHECK(out == state_b);
    return 0;
}

static int test_context_check_preserves_result2(void) {
    typedef CUresult (*context_check_t)(CUcontext context, uint32_t *result1, const void **result2, uintptr_t arg4,
                                        uintptr_t arg5, uintptr_t arg6);
    struct ContextCheckCall {
        uintptr_t arg4;
        uintptr_t arg5;
        const void *result2;
    } calls[] = {
        {0, 1, (const void *)(uintptr_t)UINT64_C(0x0000100020003000)},
        {UINT64_C(0x0000200030004000), UINT32_MAX, (const void *)(uintptr_t)UINT64_C(0x0000400050006000)},
        {0, 1, (const void *)(uintptr_t)UINT64_C(0x0000700080009000)},
    };
    const void *table = NULL;

    CHECK(cuGetExportTable(&table, &context_checks_uuid) == CUDA_SUCCESS);
    CHECK(table != NULL);
    const void *const *slots = table;
    CHECK((uintptr_t)slots[0] == 15 * sizeof(void *));
    context_check_t check = (context_check_t)slots[2];
    CHECK(check != NULL);

    for (size_t index = 0; index < sizeof(calls) / sizeof(calls[0]); index++) {
        uint32_t result1 = UINT32_MAX;
        const void *result2 = calls[index].result2;

        CHECK(check((CUcontext)(uintptr_t)1, &result1, &result2, calls[index].arg4, calls[index].arg5, 0) ==
              CUDA_SUCCESS);
        CHECK(result1 == 0);
        CHECK(result2 == calls[index].result2);
    }
    return 0;
}

static int test_cublas_context_stream_export_table(void) {
    typedef CUresult (*tools_tls_get_t)(void **context);
    typedef CUresult (*context_key_t)(CUcontext context, uint64_t *key);
    typedef CUresult (*stream_from_public_t)(CUcontext context, CUstream stream, void **private_stream,
                                             int per_thread);
    typedef CUresult (*stream_identity_t)(CUcontext context, const void *private_stream, uint64_t *identity);
    const void *tools_tls_table = NULL;
    const void *context_stream_table = NULL;
    CUcontext context = NULL;
    void *private_context = (void *)(uintptr_t)1;
    CUstream stream = NULL;
    void *private_stream = NULL;
    uint64_t key = 0;
    uint64_t identity = 0;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    CHECK(cuGetExportTable(&tools_tls_table, &tools_tls_uuid) == CUDA_SUCCESS);
    CHECK(tools_tls_table != NULL);
    const void *const *tools_tls_slots = tools_tls_table;
    CHECK((uintptr_t)tools_tls_slots[0] == 3 * sizeof(void *));
    CHECK(tools_tls_slots[2] != NULL);
    tools_tls_get_t tools_tls_get = (tools_tls_get_t)tools_tls_slots[2];
    CHECK(tools_tls_get(&private_context) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(private_context == NULL);

    CHECK(cuCtxCreate_v2(&context, 0, 0) == CUDA_SUCCESS);
    CHECK(tools_tls_get(&private_context) == CUDA_SUCCESS);
    CHECK(private_context == context);

    CHECK(cuGetExportTable(&context_stream_table, &cublas_context_stream_uuid) == CUDA_SUCCESS);
    CHECK(context_stream_table != NULL);
    const void *const *slots = context_stream_table;
    CHECK((uintptr_t)slots[0] == 93 * sizeof(void *));
    CHECK(slots[4] != NULL && slots[39] != NULL && slots[51] != NULL);

    context_key_t context_key = (context_key_t)slots[4];
    stream_identity_t stream_identity = (stream_identity_t)slots[39];
    stream_from_public_t stream_from_public = (stream_from_public_t)slots[51];
    CHECK(context_key((CUcontext)private_context, &key) == CUDA_SUCCESS);
    CHECK(key == issued_token);
    CHECK(context_key((CUcontext)(uintptr_t)(issued_token + 1), &key) == CUDA_ERROR_INVALID_CONTEXT);

    CHECK(stream_from_public(context, NULL, &private_stream, 0) == CUDA_SUCCESS);
    CHECK((uintptr_t)private_stream == CXL_GPU_STREAM_WIRE_NULL);
    CHECK(stream_identity(context, private_stream, &identity) == CUDA_SUCCESS);
    CHECK(identity == CXL_GPU_STREAM_WIRE_NULL);
    CHECK(stream_from_public(context, (CUstream)(uintptr_t)1, &private_stream, 0) == CUDA_SUCCESS);
    CHECK((uintptr_t)private_stream == CXL_GPU_STREAM_WIRE_LEGACY);
    CHECK(stream_from_public(context, (CUstream)(uintptr_t)2, &private_stream, 1) == CUDA_SUCCESS);
    CHECK((uintptr_t)private_stream == CXL_GPU_STREAM_WIRE_PER_THREAD);

    CHECK(cuStreamCreate(&stream, 0) == CUDA_SUCCESS);
    CHECK(stream != NULL);
    CHECK(stream_from_public(context, stream, &private_stream, 0) == CUDA_SUCCESS);
    CHECK(private_stream == stream);
    CHECK(stream_identity(context, private_stream, &identity) == CUDA_SUCCESS);
    CHECK(identity == (uint64_t)(uintptr_t)stream);
    CHECK(stream_from_public(context, (CUstream)(uintptr_t)3, &private_stream, 0) == CUDA_ERROR_INVALID_HANDLE);
    CHECK(stream_from_public(context, stream, NULL, 0) == CUDA_ERROR_INVALID_VALUE);
    CHECK(stream_identity(context, NULL, &identity) == CUDA_ERROR_INVALID_VALUE);
    CHECK(command_count == 2);
    CHECK(commands[0] == CXL_GPU_CMD_CTX_CREATE);
    CHECK(commands[1] == CXL_GPU_CMD_STREAM_CREATE);
    CHECK(cuCtxDestroy_v2(context) == CUDA_SUCCESS);
    private_context = (void *)(uintptr_t)1;
    CHECK(tools_tls_get(&private_context) == CUDA_ERROR_INVALID_CONTEXT);
    CHECK(private_context == NULL);
    CHECK(command_count == 3 && commands[2] == CXL_GPU_CMD_CTX_DESTROY);
    return 0;
}

static int test_integrity_uses_runtime_device_identity(void) {
    typedef CUresult (*integrity_check_t)(uint32_t version, uint64_t unix_seconds, uint64_t result[2]);
    const void *table = NULL;
    uint64_t first[2] = {0, 0};
    uint64_t second[2] = {0, 0};

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    identity_pci_device = 0;
    identity_pci_domain = 0;

    CHECK(cuGetExportTable(&table, &integrity_check_uuid) == CUDA_SUCCESS);
    integrity_check_t callback = (integrity_check_t)((const void *const *)table)[1];

    identity_pci_bus = 131;
    CHECK(callback(12092, UINT64_C(1784320000), first) == CUDA_SUCCESS);
    identity_pci_bus = 132;
    CHECK(callback(12092, UINT64_C(1784320000), second) == CUDA_SUCCESS);

    CHECK(command_count == 6);
    for (unsigned int index = 0; index < command_count; index++) {
        CHECK(commands[index] == CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE);
    }
    CHECK(memcmp(first, second, sizeof(first)) != 0);
    return 0;
}

static int test_occupancy_driver_api_route(void) {
    CUdriverProcAddressQueryResult symbol_status = -1;
    void *resolved = NULL;
    int num_blocks = -1;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    occupancy_result = CUDA_SUCCESS;
    occupancy_expected_flags = 1;

    CHECK(cuGetProcAddress("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", &resolved, 7000, 0,
                           &symbol_status) == CUDA_SUCCESS);
    CHECK(resolved == (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    CHECK(symbol_status == 0);
    CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&num_blocks, (CUfunction)(uintptr_t)5, 128, 73728, 1) ==
          CUDA_SUCCESS);
    CHECK(num_blocks == 3);
    CHECK(command_count == 1 && commands[0] == CXL_GPU_CMD_FUNC_GET_OCCUPANCY);

    occupancy_result = CUDA_ERROR_NOT_SUPPORTED;
    occupancy_expected_flags = 0;
    num_blocks = 77;
    CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, (CUfunction)(uintptr_t)5, 128, 73728) ==
          CUDA_ERROR_NOT_SUPPORTED);
    CHECK(num_blocks == 77);
    CHECK(command_count == 2 && commands[1] == CXL_GPU_CMD_FUNC_GET_OCCUPANCY);
    CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(NULL, (CUfunction)(uintptr_t)5, 128, 73728, 1) ==
          CUDA_ERROR_INVALID_VALUE);
    CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&num_blocks, NULL, 128, 73728, 1) ==
          CUDA_ERROR_INVALID_VALUE);
    CHECK(command_count == 2);
    return 0;
}

static int test_memcpy2d_device_route(void) {
    CUdriverProcAddressQueryResult symbol_status = -1;
    CUDA_MEMCPY2D copy = {
        .srcXInBytes = 7,
        .srcY = 3,
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = UINT64_C(0x100000),
        .srcPitch = 1024,
        .dstXInBytes = 11,
        .dstY = 5,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstDevice = UINT64_C(0x200000),
        .dstPitch = 2048,
        .WidthInBytes = 64,
        .Height = 3,
    };
    void *resolved = NULL;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    command_count = 0;
    memcpy2d_row_count = copy.Height;
    memcpy2d_width = copy.WidthInBytes;
    memcpy2d_phase = 0;
    for (size_t row = 0; row < copy.Height; row++) {
        memcpy2d_src_rows[row] = copy.srcDevice + (copy.srcY + row) * copy.srcPitch + copy.srcXInBytes;
        memcpy2d_dst_rows[row] = copy.dstDevice + (copy.dstY + row) * copy.dstPitch + copy.dstXInBytes;
    }

    CHECK(cuGetProcAddress("cuMemcpy2DAsync_v2", &resolved, 12090, 0, &symbol_status) == CUDA_SUCCESS);
    CHECK(resolved == (void *)cuMemcpy2DAsync_v2);
    CHECK(symbol_status == 0);
    CHECK(cuMemcpy2DAsync_v2(&copy, NULL) == CUDA_SUCCESS);
    CHECK(memcpy2d_phase == 2 * copy.Height);
    CHECK(command_count == 2 * copy.Height);

    copy.WidthInBytes = copy.srcPitch - copy.srcXInBytes + 1;
    CHECK(cuMemcpy2DAsync_v2(&copy, NULL) == CUDA_ERROR_INVALID_VALUE);
    CHECK(command_count == 2 * memcpy2d_row_count);
    copy.WidthInBytes = memcpy2d_width;
    copy.srcMemoryType = 1;
    CHECK(cuMemcpy2DAsync_v2(&copy, NULL) == CUDA_ERROR_NOT_SUPPORTED);
    CHECK(command_count == 2 * memcpy2d_row_count);
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcY = SIZE_MAX;
    CHECK(cuMemcpy2DAsync_v2(&copy, NULL) == CUDA_ERROR_INVALID_VALUE);
    CHECK(command_count == 2 * memcpy2d_row_count);
    copy.srcY = 0;
    copy.srcDevice = UINT64_MAX - 15;
    CHECK(cuMemcpy2DAsync_v2(&copy, NULL) == CUDA_ERROR_INVALID_VALUE);
    CHECK(command_count == 2 * memcpy2d_row_count);
    return 0;
}

static int test_library_fatbin_prefers_highest_compatible_cubin(void) {
    unsigned char fatbin[sizeof(CudartFatbinHeader) + 3 * (sizeof(CudartFatbinFileHeader) + 8)] = {0};
    CudartFatbinHeader header = {
        .magic = CUDART_FATBIN_MAGIC,
        .version = CUDART_FATBIN_VERSION,
        .header_size = sizeof(CudartFatbinHeader),
        .files_size = sizeof(fatbin) - sizeof(CudartFatbinHeader),
    };
    CudartFatbinFileHeader sm80 = {
        .kind = CUDART_FATBIN_KIND_ELF,
        .version = 0x101,
        .header_size = sizeof(CudartFatbinFileHeader),
        .payload_size = 8,
        .sm_version = 80,
        .uncompressed_payload = 8,
    };
    CudartFatbinFileHeader sm90 = sm80;
    CudartFatbinFileHeader ptx120 = sm80;
    CUlibrary library = NULL;
    CUlibraryOption options[] = {CU_LIBRARY_BINARY_IS_PRESERVED};
    void *option_values[] = {(void *)(uintptr_t)1};
    size_t offset = sizeof(header);

    sm90.sm_version = 90;
    ptx120.kind = CUDART_FATBIN_KIND_PTX;
    ptx120.sm_version = 120;
    memcpy(fatbin, &header, sizeof(header));
    memcpy(fatbin + offset, &sm80, sizeof(sm80));
    fatbin[offset + sizeof(sm80)] = 0x80;
    offset += sizeof(sm80) + 8;
    memcpy(fatbin + offset, &sm90, sizeof(sm90));
    fatbin[offset + sizeof(sm90)] = 0x90;
    offset += sizeof(sm90) + 8;
    memcpy(fatbin + offset, &ptx120, sizeof(ptx120));
    memcpy(fatbin + offset + sizeof(ptx120), ".versio", 7);

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    cxl_cuda_test_write_reg32(CXL_GPU_REG_CC_MAJOR, 8);
    cxl_cuda_test_write_reg32(CXL_GPU_REG_CC_MINOR, 9);
    command_count = 0;
    cubin_load_count = 0;

    CHECK(cuLibraryLoadData(&library, fatbin, NULL, NULL, 0, options, option_values, 1) == CUDA_SUCCESS);
    CHECK(library != NULL);
    CHECK(cubin_load_count == 0);
    CHECK(command_count == 0);
    void *module = NULL;
    CHECK(cuLibraryGetModule(&module, library) == CUDA_SUCCESS);
    CHECK(module != NULL);
    CHECK(cubin_load_count == 1);
    CHECK(command_count == 1 && commands[0] == CXL_GPU_CMD_MODULE_LOAD_CUBIN);
    CHECK(cuLibraryUnload(library) == CUDA_SUCCESS);
    return 0;
}

static int test_library_legacy_only_fatbin_registers_without_module_load(void) {
    unsigned char fatbin[sizeof(CudartFatbinHeader) + 3 * (sizeof(CudartFatbinFileHeader) + 8)] = {0};
    CudartFatbinHeader header = {
        .magic = CUDART_FATBIN_MAGIC,
        .version = CUDART_FATBIN_VERSION,
        .header_size = sizeof(CudartFatbinHeader),
        .files_size = sizeof(fatbin) - sizeof(CudartFatbinHeader),
    };
    CudartFatbinFileHeader sm50 = {
        .kind = CUDART_FATBIN_KIND_ELF,
        .version = 0x101,
        .header_size = sizeof(CudartFatbinFileHeader),
        .payload_size = 8,
        .sm_version = 50,
        .uncompressed_payload = 8,
    };
    CudartFatbinFileHeader sm60 = sm50;
    CudartFatbinFileHeader sm61 = sm50;
    CUlibrary library = NULL;
    CUlibraryOption options[] = {CU_LIBRARY_BINARY_IS_PRESERVED};
    void *option_values[] = {(void *)(uintptr_t)1};
    size_t offset = sizeof(header);

    sm60.sm_version = 60;
    sm61.sm_version = 61;
    memcpy(fatbin, &header, sizeof(header));
    memcpy(fatbin + offset, &sm50, sizeof(sm50));
    fatbin[offset + sizeof(sm50)] = 0x50;
    offset += sizeof(sm50) + 8;
    memcpy(fatbin + offset, &sm60, sizeof(sm60));
    fatbin[offset + sizeof(sm60)] = 0x60;
    offset += sizeof(sm60) + 8;
    memcpy(fatbin + offset, &sm61, sizeof(sm61));
    fatbin[offset + sizeof(sm61)] = 0x61;

    cxl_cuda_test_reset();
    cxl_cuda_test_set_executor(fake_execute);
    cxl_cuda_test_write_reg32(CXL_GPU_REG_CC_MAJOR, 8);
    cxl_cuda_test_write_reg32(CXL_GPU_REG_CC_MINOR, 9);
    command_count = 0;
    cubin_load_count = 0;

    CHECK(cuLibraryLoadData(&library, fatbin, NULL, NULL, 0, options, option_values, 1) == CUDA_SUCCESS);
    CHECK(library != NULL);
    CHECK(cubin_load_count == 0);
    CHECK(command_count == 0);
    void *module = NULL;
    CHECK(cuLibraryGetModule(&module, library) == CUDA_ERROR_NO_BINARY_FOR_GPU);
    CHECK(module == NULL);
    CHECK(cubin_load_count == 0);
    CHECK(command_count == 0);
    CHECK(cuLibraryUnload(library) == CUDA_SUCCESS);
    return 0;
}

static int test_library_registration_exceeds_the_previous_fixed_capacity(void) {
    enum { library_count = 513 };
    unsigned char preserved_code[16] = {0};
    CUlibrary libraries[library_count];
    CUlibraryOption options[] = {CU_LIBRARY_BINARY_IS_PRESERVED};
    void *option_values[] = {(void *)(uintptr_t)1};

    cxl_cuda_test_reset();
    for (unsigned int i = 0; i < library_count; i++) {
        libraries[i] = NULL;
        CHECK(cuLibraryLoadData(&libraries[i], preserved_code, NULL, NULL, 0, options, option_values, 1) ==
              CUDA_SUCCESS);
        CHECK(libraries[i] != NULL);
    }
    for (unsigned int i = 0; i < library_count; i++) {
        CHECK(cuLibraryUnload(libraries[i]) == CUDA_SUCCESS);
        CHECK(cuLibraryUnload(libraries[i]) == CUDA_ERROR_INVALID_HANDLE);
        void *module = NULL;
        CHECK(cuLibraryGetModule(&module, libraries[i]) == CUDA_ERROR_INVALID_HANDLE);
    }

    CUlibrary later_library = NULL;
    CHECK(cuLibraryLoadData(&later_library, preserved_code, NULL, NULL, 0, options, option_values, 1) ==
          CUDA_SUCCESS);
    CHECK(later_library != NULL);
    CHECK(later_library != libraries[0]);
    CHECK(later_library != libraries[library_count - 1]);
    CHECK(cuLibraryUnload(later_library) == CUDA_SUCCESS);
    return 0;
}

int main(void) {
    return test_query_and_context_sequence() || test_primary_retain_does_not_become_current() ||
           test_destroy_keeps_other_thread_token_without_transport() || test_integrity_export_table_shape() ||
           test_context_local_storage_keeps_managers_separate() || test_context_check_preserves_result2() ||
           test_cublas_context_stream_export_table() ||
           test_integrity_uses_runtime_device_identity() || test_occupancy_driver_api_route() ||
           test_memcpy2d_device_route() || test_library_fatbin_prefers_highest_compatible_cubin() ||
           test_library_legacy_only_fatbin_registers_without_module_load() ||
           test_library_registration_exceeds_the_previous_fixed_capacity();
}

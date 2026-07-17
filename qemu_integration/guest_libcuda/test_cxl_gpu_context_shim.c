#include "cxl_gpu_cmd.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE 708
#define CUDA_ERROR_CONTEXT_IS_DESTROYED 709
#define CUDA_ERROR_NOT_SUPPORTED 801

void cxl_cuda_test_reset(void);
void cxl_cuda_test_set_executor(CUresult (*executor)(uint32_t cmd));
uint64_t cxl_cuda_test_read_reg64(uint32_t offset);
void cxl_cuda_test_write_result(unsigned int index, uint64_t value);

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

#define CHECK(expr)                                                           \
    do {                                                                      \
        if (!(expr)) {                                                        \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static uint32_t commands[16];
static unsigned int command_count;
static uint64_t issued_token = 41;

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

static CUresult fake_execute(uint32_t command)
{
    commands[command_count++] = command;
    switch (command) {
    case CXL_GPU_CMD_CTX_CREATE:
        cxl_cuda_test_write_result(0, issued_token);
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
        CHECK(cxl_cuda_test_read_reg64(CXL_GPU_REG_PARAM0) == 97);
        cxl_cuda_test_write_result(0, 49152);
        return CUDA_SUCCESS;
    case CXL_GPU_CMD_CTX_DESTROY:
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_INVALID_CONTEXT;
    }
}

static int test_query_and_context_sequence(void)
{
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

static int test_primary_retain_does_not_become_current(void)
{
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

static void *thread_with_destroyed_context(void *opaque)
{
    DestroyedContextThread *thread = opaque;

    thread->set_current_result = cuCtxSetCurrent(thread->token);
    if (thread->set_current_result != CUDA_SUCCESS)
        return NULL;
    thread->get_current_result = cuCtxGetCurrent(&thread->current);
    pthread_barrier_wait(&thread->attached);
    pthread_barrier_wait(&thread->destroyed);
    thread->get_current_result = cuCtxGetCurrent(&thread->current);
    thread->get_device_result = cuCtxGetDevice(&thread->device);
    thread->mem_info_result = cuMemGetInfo_v2(&thread->free_bytes,
                                               &thread->total_bytes);
    return NULL;
}

static int test_destroy_keeps_other_thread_token_without_transport(void)
{
    DestroyedContextThread thread = { 0 };
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

int main(void)
{
    return test_query_and_context_sequence() ||
           test_primary_retain_does_not_become_current() ||
           test_destroy_keeps_other_thread_token_without_transport();
}

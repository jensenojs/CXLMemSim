#include "cxl_gpu_transport.h"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    CxlGpuTransport *transport;
    uint32_t result;
} DelayedCompletion;

static void *complete_command(void *opaque) {
    DelayedCompletion *completion = opaque;
    struct timespec delay = {.tv_nsec = 1000000};

    nanosleep(&delay, NULL);
    cxl_gpu_transport_write32(completion->transport, CXL_GPU_REG_CMD_RESULT,
                              completion->result);
    cxl_gpu_transport_write32(completion->transport, CXL_GPU_REG_CMD_STATUS,
                              CXL_GPU_CMD_STATUS_COMPLETE);
    return NULL;
}

int main(void) {
    uint8_t *bar = calloc(1, CXL_GPU_CMD_REG_SIZE);
    assert(bar != NULL);

    CxlGpuTransport transport = CXL_GPU_TRANSPORT_INITIALIZER;
    transport.regs = (volatile uint32_t *)bar;
    transport.data = bar + CXL_GPU_DATA_OFFSET;
    transport.bar_size = CXL_GPU_CMD_REG_SIZE;

    cxl_gpu_transport_write32(&transport, CXL_GPU_REG_PARAM0, 0x12345678U);
    assert(cxl_gpu_transport_read32(&transport, CXL_GPU_REG_PARAM0) == 0x12345678U);
    cxl_gpu_transport_write64(&transport, CXL_GPU_REG_PARAM1, UINT64_C(0x123456789abcdef0));
    assert(cxl_gpu_transport_read64(&transport, CXL_GPU_REG_PARAM1) == UINT64_C(0x123456789abcdef0));

    const uint8_t expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    uint8_t actual[sizeof(expected)] = {0};
    cxl_gpu_transport_data_write(&transport, 3, expected, sizeof(expected));
    cxl_gpu_transport_data_read(&transport, 3, actual, sizeof(actual));
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    cxl_gpu_transport_write32(&transport, CXL_GPU_REG_CMD_STATUS, CXL_GPU_CMD_STATUS_COMPLETE);
    cxl_gpu_transport_write32(&transport, CXL_GPU_REG_CMD_RESULT, CXL_GPU_SUCCESS);
    uint32_t immediate_polls = 0;
    assert(cxl_gpu_transport_execute(&transport, CXL_GPU_CMD_CASE_BEGIN,
                                     &immediate_polls) == CXL_GPU_SUCCESS);
    assert(immediate_polls == 1);
    assert(cxl_gpu_transport_read32(&transport, CXL_GPU_REG_CMD) == CXL_GPU_CMD_CASE_BEGIN);

    cxl_gpu_transport_write32(&transport, CXL_GPU_REG_CMD_STATUS,
                              CXL_GPU_CMD_STATUS_IDLE);
    DelayedCompletion completion = {
        .transport = &transport,
        .result = CXL_GPU_ERROR_INVALID_VALUE,
    };
    pthread_t completion_thread;
    assert(pthread_create(&completion_thread, NULL, complete_command,
                          &completion) == 0);
    uint32_t polls = 0;
    assert(cxl_gpu_transport_execute(&transport, CXL_GPU_CMD_CASE_END, &polls) ==
           CXL_GPU_ERROR_INVALID_VALUE);
    assert(polls > 1);
    assert(pthread_join(completion_thread, NULL) == 0);

    free(bar);
    return 0;
}

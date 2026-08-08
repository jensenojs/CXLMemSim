#include "cxl_gpu_transport.h"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    CxlGpuTransport *transport;
    uint64_t expected_submission;
    uint64_t expected_command;
    uint64_t expected_epoch;
    uint64_t completion_status;
    uint32_t result;
} DescriptorCompletion;

static void *complete_command(void *opaque) {
    DescriptorCompletion *completion = opaque;
    volatile CXLGPURAMCommandDescriptor *descriptor = completion->transport->descriptor;
    volatile uint32_t *doorbell = (volatile uint32_t *)
        ((volatile uint8_t *)completion->transport->regs + CXL_GPU_DESCRIPTOR_DOORBELL_OFFSET);
    struct timespec pause = {.tv_nsec = 1000};

    while (*doorbell != CXL_GPU_DESCRIPTOR_DOORBELL_VALUE)
        nanosleep(&pause, NULL);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    assert(descriptor->protocol_version == CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION);
    assert(descriptor->descriptor_size == CXL_GPU_DESCRIPTOR_WIRE_SIZE);
    assert(descriptor->request_submission == completion->expected_submission);
    assert(descriptor->request_device_generation == 1);
    assert(descriptor->request_command == completion->expected_command);
    assert(descriptor->request_case_epoch == completion->expected_epoch);

    descriptor->completion_submission = descriptor->request_submission;
    descriptor->completion_device_generation = descriptor->request_device_generation;
    descriptor->result = completion->result;
    descriptor->results[0] = UINT64_C(0xfeedface);
    *doorbell = 0;
    __atomic_store_n(&descriptor->completion_status, completion->completion_status,
                     __ATOMIC_RELEASE);
    return NULL;
}

static uint32_t execute_with_completion(CxlGpuTransport *transport,
                                        DescriptorCompletion *completion,
                                        uint32_t *polls) {
    pthread_t thread;
    assert(pthread_create(&thread, NULL, complete_command, completion) == 0);
    uint32_t result = cxl_gpu_transport_execute(
        transport, (uint32_t)completion->expected_command, polls);
    assert(pthread_join(thread, NULL) == 0);
    return result;
}

int main(void) {
    uint8_t *bar = calloc(1, CXL_GPU_CMD_REG_SIZE);
    assert(bar != NULL);

    CxlGpuTransport transport = CXL_GPU_TRANSPORT_INITIALIZER;
    transport.regs = (volatile uint32_t *)bar;
    transport.data = bar + CXL_GPU_DATA_OFFSET;
    transport.descriptor = (volatile CXLGPURAMCommandDescriptor *)
        (bar + CXL_GPU_DESCRIPTOR_OFFSET);
    transport.batch_data = bar + CXL_GPU_BATCH_DATA_OFFSET;
    transport.bar_size = CXL_GPU_CMD_REG_SIZE;
    transport.descriptor->device_generation = 1;

    transport.descriptor->active_case_epoch = 17;
    transport.descriptor->sync_hint_device_generation = 1;
    transport.descriptor->sync_hint_case_epoch = 17;
    transport.descriptor->sync_hint_stream_wire = 23;
    __atomic_store_n(&transport.descriptor->sync_hint_valid, 1,
                     __ATOMIC_RELEASE);
    assert(cxl_gpu_transport_try_elide_stream_sync(&transport, 24) == 0);
    transport.descriptor->sync_hint_case_epoch = 18;
    assert(cxl_gpu_transport_try_elide_stream_sync(&transport, 23) == 0);
    transport.descriptor->sync_hint_case_epoch = 17;
    transport.descriptor->sync_hint_device_generation = 2;
    assert(cxl_gpu_transport_try_elide_stream_sync(&transport, 23) == 0);
    transport.descriptor->sync_hint_device_generation = 1;
    assert(cxl_gpu_transport_try_elide_stream_sync(&transport, 23) == 1);
    assert(transport.descriptor->guest_elided_stream_syncs == 1);
    __atomic_store_n(&transport.descriptor->sync_hint_valid, 0,
                     __ATOMIC_RELEASE);
    assert(cxl_gpu_transport_try_elide_stream_sync(&transport, 23) == 0);

    cxl_gpu_transport_write64(&transport, CXL_GPU_REG_PARAM1,
                              UINT64_C(0x123456789abcdef0));
    cxl_gpu_transport_write64(&transport, CXL_GPU_REG_CALL_ID, 42);
    assert(cxl_gpu_transport_read64(&transport, CXL_GPU_REG_PARAM1) ==
           UINT64_C(0x123456789abcdef0));
    assert(transport.descriptor->request_call_id == 42);

    const uint8_t expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    uint8_t actual[sizeof(expected)] = {0};
    cxl_gpu_transport_data_write(&transport, 3, expected, sizeof(expected));
    cxl_gpu_transport_data_read(&transport, 3, actual, sizeof(actual));
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    memset(actual, 0, sizeof(actual));
    assert(cxl_gpu_transport_batch_write(&transport, 7, expected,
                                         sizeof(expected)) == 0);
    assert(cxl_gpu_transport_batch_read(&transport, 7, actual,
                                        sizeof(actual)) == 0);
    assert(memcmp(expected, actual, sizeof(expected)) == 0);
    assert(cxl_gpu_transport_batch_write(&transport, CXL_GPU_BATCH_DATA_SIZE,
                                         expected, sizeof(expected)) == -1);

    DescriptorCompletion first = {
        .transport = &transport,
        .expected_submission = 1,
        .expected_command = CXL_GPU_CMD_CASE_BEGIN,
        .expected_epoch = 0,
        .completion_status = CXL_GPU_DESCRIPTOR_COMPLETION_COMPLETE,
        .result = CXL_GPU_SUCCESS,
    };
    uint32_t polls = 0;
    assert(execute_with_completion(&transport, &first, &polls) == CXL_GPU_SUCCESS);
    assert(polls >= 1);
    assert(cxl_gpu_transport_read64(&transport, CXL_GPU_REG_RESULT0) ==
           UINT64_C(0xfeedface));

    transport.descriptor->active_case_epoch = 17;
    DescriptorCompletion second = {
        .transport = &transport,
        .expected_submission = 2,
        .expected_command = CXL_GPU_CMD_CTX_SYNC,
        .expected_epoch = 17,
        .completion_status = CXL_GPU_DESCRIPTOR_COMPLETION_COMPLETE,
        .result = CXL_GPU_ERROR_INVALID_VALUE,
    };
    assert(execute_with_completion(&transport, &second, &polls) ==
           CXL_GPU_ERROR_INVALID_VALUE);
    assert(!transport.unusable);

    DescriptorCompletion protocol_error = {
        .transport = &transport,
        .expected_submission = 3,
        .expected_command = CXL_GPU_CMD_CASE_END,
        .expected_epoch = 17,
        .completion_status = CXL_GPU_DESCRIPTOR_COMPLETION_ERROR,
        .result = CXL_GPU_ERROR_UNKNOWN,
    };
    assert(execute_with_completion(&transport, &protocol_error, &polls) ==
           CXL_GPU_ERROR_UNKNOWN);
    assert(transport.unusable);
    uint64_t preserved_param = transport.descriptor->params[0];
    cxl_gpu_transport_write64(&transport, CXL_GPU_REG_PARAM0, 99);
    assert(transport.descriptor->params[0] == preserved_param);
    assert(cxl_gpu_transport_execute(&transport, CXL_GPU_CMD_NOP, &polls) ==
           CXL_GPU_ERROR_UNKNOWN);
    assert(polls == 0);

    free(bar);
    return 0;
}

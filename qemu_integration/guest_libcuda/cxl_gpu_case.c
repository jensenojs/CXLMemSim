#include "cxl_gpu_transport.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *stream) {
    fprintf(stream, "usage:\n"
                    "  cxl-gpu-case begin --case baseline|concordia --binding <u64>\n"
                    "  cxl-gpu-case end --epoch <u64> --application-exit <i32> --binding <u64>\n");
}

static int parse_u64(const char *text, uint64_t *value) {
    if (text[0] == '-')
        return -1;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0')
        return -1;
    *value = (uint64_t)parsed;
    return 0;
}

static int parse_i32(const char *text, int32_t *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX)
        return -1;
    *value = (int32_t)parsed;
    return 0;
}

static int run_begin(CxlGpuTransport *transport, int argc, char **argv) {
    if (argc != 6 || strcmp(argv[2], "--case") != 0 || strcmp(argv[4], "--binding") != 0)
        return 64;

    uint64_t case_id = 0;
    if (strcmp(argv[3], "baseline") == 0)
        case_id = CXL_GPU_CASE_BASELINE;
    else if (strcmp(argv[3], "concordia") == 0)
        case_id = CXL_GPU_CASE_CONCORDIA;
    else
        return 64;

    uint64_t binding = 0;
    if (parse_u64(argv[5], &binding) != 0)
        return 64;
    if (cxl_gpu_transport_open(transport, 1) != 0)
        return 69;
    if (cxl_gpu_transport_lock(transport) != 0)
        return 74;

    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM0, CXL_GPU_CASE_PROTOCOL_VERSION);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM1, case_id);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM2, binding);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM3, 0);
    uint32_t result = cxl_gpu_transport_execute(transport, CXL_GPU_CMD_CASE_BEGIN);
    uint64_t epoch = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT0);
    uint64_t acknowledged_case = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT1);
    uint64_t first_sequence = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT2);
    uint64_t config_binding = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT3);
    int unlock_result = cxl_gpu_transport_unlock(transport);
    /* `binding` authenticates this guest request against the paired run.  The
     * returned config binding is a separate, host-derived case identity: QEMU
     * hashes its AOF/manifest/limit configuration and guarantees it is nonzero.
     */
    if (result != CXL_GPU_SUCCESS || unlock_result != 0 || epoch == 0 || acknowledged_case != case_id ||
        config_binding == 0) {
        fprintf(stderr,
                "cxl_gpu_case=begin status=fail result=%" PRIu32 " unlock_result=%d epoch=%" PRIu64
                " acknowledged_case=%" PRIu64 " expected_case=%" PRIu64 " first_sequence=%" PRIu64
                " config_binding=%" PRIu64 " expected_binding=%" PRIu64 "\n",
                result, unlock_result, epoch, acknowledged_case, case_id, first_sequence, config_binding, binding);
        fflush(stderr);
        return 70;
    }

    printf("cxl_gpu_case=begin status=pass protocol=%u case=%s epoch=%" PRIu64 " first_sequence=%" PRIu64
           " binding=%" PRIu64 " config_binding=%" PRIu64 "\n",
           CXL_GPU_CASE_PROTOCOL_VERSION, argv[3], epoch, first_sequence, binding, config_binding);
    return 0;
}

static int run_end(CxlGpuTransport *transport, int argc, char **argv) {
    if (argc != 8 || strcmp(argv[2], "--epoch") != 0 || strcmp(argv[4], "--application-exit") != 0 ||
        strcmp(argv[6], "--binding") != 0)
        return 64;

    uint64_t epoch = 0;
    uint64_t binding = 0;
    int32_t application_exit = 0;
    if (parse_u64(argv[3], &epoch) != 0 || epoch == 0 || parse_i32(argv[5], &application_exit) != 0 ||
        parse_u64(argv[7], &binding) != 0)
        return 64;
    if (cxl_gpu_transport_open(transport, 1) != 0)
        return 69;
    if (cxl_gpu_transport_lock(transport) != 0)
        return 74;

    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM0, CXL_GPU_CASE_PROTOCOL_VERSION);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM1, epoch);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM2, (uint64_t)(int64_t)application_exit);
    cxl_gpu_transport_write64(transport, CXL_GPU_REG_PARAM3, binding);
    uint32_t result = cxl_gpu_transport_execute(transport, CXL_GPU_CMD_CASE_END);
    uint64_t acknowledged_epoch = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT0);
    uint64_t last_sequence = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT1);
    uint64_t concordia_status = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT2);
    uint64_t reset_status = cxl_gpu_transport_read64(transport, CXL_GPU_REG_RESULT3);
    int unlock_result = cxl_gpu_transport_unlock(transport);
    if (result != CXL_GPU_SUCCESS || unlock_result != 0 || acknowledged_epoch != epoch) {
        fprintf(stderr,
                "cxl_gpu_case=end status=fail result=%" PRIu32 " unlock_result=%d acknowledged_epoch=%" PRIu64
                " expected_epoch=%" PRIu64 " last_sequence=%" PRIu64 " concordia_status=%" PRIu64
                " reset_status=%" PRIu64 " application_exit=%" PRId32 " binding=%" PRIu64 "\n",
                result, unlock_result, acknowledged_epoch, epoch, last_sequence, concordia_status, reset_status,
                application_exit, binding);
        fflush(stderr);
        return 70;
    }

    printf("cxl_gpu_case=end status=pass protocol=%u epoch=%" PRIu64 " last_sequence=%" PRIu64
           " concordia_status=%" PRIu64 " reset_status=%" PRIu64 " application_exit=%" PRId32 " binding=%" PRIu64 "\n",
           CXL_GPU_CASE_PROTOCOL_VERSION, epoch, last_sequence, concordia_status, reset_status, application_exit,
           binding);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(stderr);
        return 64;
    }
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(stdout);
        return 0;
    }

    CxlGpuTransport transport = CXL_GPU_TRANSPORT_INITIALIZER;
    int result;
    if (strcmp(argv[1], "begin") == 0)
        result = run_begin(&transport, argc, argv);
    else if (strcmp(argv[1], "end") == 0)
        result = run_end(&transport, argc, argv);
    else
        result = 64;

    cxl_gpu_transport_close(&transport);
    if (result == 64)
        usage(stderr);
    return result;
}

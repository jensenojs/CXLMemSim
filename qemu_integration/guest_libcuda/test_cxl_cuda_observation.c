#include "cxl_cuda_observation.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1

CUresult cuCxlObservationDecodeBeginV1(uint64_t case_epoch);
CUresult cuCxlObservationSpanBeginV1(uint32_t category,
                                    uint64_t graph_ordinal,
                                    uint64_t operation_sequence,
                                    uint64_t *token);
CUresult cuCxlObservationSpanEndV1(uint64_t token,
                                  int32_t operation_status);
CUresult cuCxlObservationDecodeEndV1(uint64_t case_epoch);

static int run_complete(void) {
    uint64_t outer = 0;
    uint64_t inner = 0;
    uint64_t read = 0;

    if (cuCxlObservationDecodeBeginV1(17) != CUDA_SUCCESS ||
        cuCxlObservationSpanBeginV1(CXL_CUDA_OBS_SELECTED_RANGE_PLAN,
                                    3, 1, &outer) != CUDA_SUCCESS ||
        cuCxlObservationSpanBeginV1(CXL_CUDA_OBS_SELECTED_RANGE_PLAN,
                                    3, 2, &inner) != CUDA_SUCCESS ||
        cuCxlObservationSpanEndV1(inner, 0) != CUDA_SUCCESS ||
        cuCxlObservationSpanBeginV1(CXL_CUDA_OBS_HOST_RESULT_READ,
                                    3, 3, &read) != CUDA_SUCCESS ||
        cuCxlObservationSpanEndV1(read, 0) != CUDA_SUCCESS ||
        cuCxlObservationSpanEndV1(outer, 0) != CUDA_SUCCESS ||
        cuCxlObservationDecodeEndV1(17) != CUDA_SUCCESS)
        return 1;
    return 0;
}

static int run_incomplete(void) {
    if (cuCxlObservationDecodeBeginV1(23) != CUDA_SUCCESS ||
        cuCxlObservationSpanEndV1(UINT64_C(99), 0) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuCxlObservationDecodeEndV1(23) != CUDA_SUCCESS)
        return 1;
    return 0;
}

static int run_overflow(void) {
    uint64_t token = 0;

    if (cuCxlObservationDecodeBeginV1(29) != CUDA_SUCCESS)
        return 1;
    for (uint64_t operation = 1; operation <= 4096; operation++) {
        if (cuCxlObservationSpanBeginV1(CXL_CUDA_OBS_SOURCE_LEASE,
                                        1, operation, &token) != CUDA_SUCCESS)
            return 1;
    }
    if (cuCxlObservationSpanBeginV1(CXL_CUDA_OBS_SOURCE_LEASE,
                                    1, 4097, &token) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuCxlObservationDecodeEndV1(29) != CUDA_SUCCESS)
        return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s complete|incomplete|overflow\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "complete") == 0)
        return run_complete();
    if (strcmp(argv[1], "incomplete") == 0)
        return run_incomplete();
    if (strcmp(argv[1], "overflow") == 0)
        return run_overflow();
    return 2;
}

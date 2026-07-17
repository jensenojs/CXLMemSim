#ifndef CXL_GPU_CONTEXT_STATE_H
#define CXL_GPU_CONTEXT_STATE_H

#include <stdbool.h>
#include <stdint.h>

enum {
    CXL_CUDA_SUCCESS = 0,
    CXL_CUDA_ERROR_OUT_OF_MEMORY = 2,
    CXL_CUDA_ERROR_INVALID_CONTEXT = 201,
    CXL_CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CXL_CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CXL_CUDA_ERROR_NOT_SUPPORTED = 801,
};

typedef enum CxlCudaContextMode {
    CXL_CUDA_CONTEXT_EMPTY,
    CXL_CUDA_CONTEXT_ORDINARY_LIVE,
    CXL_CUDA_CONTEXT_ORDINARY_DESTROYED,
    CXL_CUDA_CONTEXT_PRIMARY_LIVE,
    CXL_CUDA_CONTEXT_PRIMARY_INACTIVE,
} CxlCudaContextMode;

/* One QEMU-issued token exists per guest process.  A thread stack only needs
 * a depth: every entry represents that same token. */
typedef struct CxlCudaContextStateView {
    uintptr_t token;
    CxlCudaContextMode mode;
    unsigned int primary_retain_count;
    unsigned int current_depth;
} CxlCudaContextStateView;

void cxl_cuda_context_state_reset(void);
CxlCudaContextStateView cxl_cuda_context_state_view(void);

int cxl_cuda_context_prepare_ordinary_create(void);
int cxl_cuda_context_commit_ordinary_create(uintptr_t token);
int cxl_cuda_context_prepare_primary_retain(bool *needs_create,
                                            uintptr_t *existing_token);
int cxl_cuda_context_commit_primary_retain(uintptr_t token);
int cxl_cuda_context_primary_release(void);
int cxl_cuda_context_primary_set_zero_flags(void);
int cxl_cuda_context_primary_get_state(unsigned int *flags, int *active);
int cxl_cuda_context_primary_reset(void);

int cxl_cuda_context_prepare_destroy(uintptr_t token);
int cxl_cuda_context_commit_destroy(uintptr_t token);
int cxl_cuda_context_get_current(uintptr_t *token);
int cxl_cuda_context_set_current(uintptr_t token);
int cxl_cuda_context_push_current(uintptr_t token);
int cxl_cuda_context_pop_current(uintptr_t *token);
int cxl_cuda_context_get_current_live(uintptr_t *token);

#endif

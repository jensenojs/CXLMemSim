#include "cxl_gpu_context_state.h"

#include <pthread.h>

typedef struct CxlCudaContextProcessState {
    pthread_mutex_t lock;
    uintptr_t token;
    CxlCudaContextMode mode;
    unsigned int primary_retain_count;
} CxlCudaContextProcessState;

static CxlCudaContextProcessState g_state = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .mode = CXL_CUDA_CONTEXT_EMPTY,
};

static _Thread_local unsigned int g_current_depth;

static bool token_is_live_locked(uintptr_t token)
{
    return token != 0 && token == g_state.token &&
           (g_state.mode == CXL_CUDA_CONTEXT_ORDINARY_LIVE ||
            g_state.mode == CXL_CUDA_CONTEXT_PRIMARY_LIVE);
}

void cxl_cuda_context_state_reset(void)
{
    pthread_mutex_lock(&g_state.lock);
    g_state.token = 0;
    g_state.mode = CXL_CUDA_CONTEXT_EMPTY;
    g_state.primary_retain_count = 0;
    pthread_mutex_unlock(&g_state.lock);
    g_current_depth = 0;
}

CxlCudaContextStateView cxl_cuda_context_state_view(void)
{
    CxlCudaContextStateView view;

    pthread_mutex_lock(&g_state.lock);
    view.token = g_state.token;
    view.mode = g_state.mode;
    view.primary_retain_count = g_state.primary_retain_count;
    pthread_mutex_unlock(&g_state.lock);
    view.current_depth = g_current_depth;
    return view;
}

int cxl_cuda_context_prepare_ordinary_create(void)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (g_state.mode != CXL_CUDA_CONTEXT_EMPTY) {
        result = CXL_CUDA_ERROR_NOT_SUPPORTED;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_commit_ordinary_create(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (token == 0 || g_state.mode != CXL_CUDA_CONTEXT_EMPTY) {
        result = CXL_CUDA_ERROR_NOT_SUPPORTED;
    } else {
        g_state.token = token;
        g_state.mode = CXL_CUDA_CONTEXT_ORDINARY_LIVE;
        g_current_depth = 1;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_prepare_primary_retain(bool *needs_create,
                                            uintptr_t *existing_token)
{
    int result = CXL_CUDA_SUCCESS;

    if (!needs_create || !existing_token) {
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_lock(&g_state.lock);
    switch (g_state.mode) {
    case CXL_CUDA_CONTEXT_EMPTY:
        *needs_create = true;
        *existing_token = 0;
        break;
    case CXL_CUDA_CONTEXT_PRIMARY_LIVE:
        g_state.primary_retain_count++;
        *needs_create = false;
        *existing_token = g_state.token;
        break;
    default:
        result = CXL_CUDA_ERROR_NOT_SUPPORTED;
        break;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_commit_primary_retain(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (token == 0 || g_state.mode != CXL_CUDA_CONTEXT_EMPTY) {
        result = CXL_CUDA_ERROR_NOT_SUPPORTED;
    } else {
        g_state.token = token;
        g_state.mode = CXL_CUDA_CONTEXT_PRIMARY_LIVE;
        g_state.primary_retain_count = 1;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_primary_release(void)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (g_state.mode != CXL_CUDA_CONTEXT_PRIMARY_LIVE ||
        g_state.primary_retain_count == 0) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else if (--g_state.primary_retain_count == 0) {
        g_state.mode = CXL_CUDA_CONTEXT_PRIMARY_INACTIVE;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_primary_set_zero_flags(void)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    switch (g_state.mode) {
    case CXL_CUDA_CONTEXT_EMPTY:
        break;
    case CXL_CUDA_CONTEXT_PRIMARY_LIVE:
        result = CXL_CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
        break;
    default:
        result = CXL_CUDA_ERROR_NOT_SUPPORTED;
        break;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_primary_get_state(unsigned int *flags, int *active)
{
    if (!flags || !active) {
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_lock(&g_state.lock);
    *flags = 0;
    *active = g_state.mode == CXL_CUDA_CONTEXT_PRIMARY_LIVE;
    pthread_mutex_unlock(&g_state.lock);
    return CXL_CUDA_SUCCESS;
}

int cxl_cuda_context_primary_reset(void)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (g_state.mode == CXL_CUDA_CONTEXT_PRIMARY_LIVE) {
        result = CXL_CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_prepare_destroy(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (!token_is_live_locked(token) ||
        g_state.mode != CXL_CUDA_CONTEXT_ORDINARY_LIVE) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_commit_destroy(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (!token_is_live_locked(token) ||
        g_state.mode != CXL_CUDA_CONTEXT_ORDINARY_LIVE) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else {
        g_state.mode = CXL_CUDA_CONTEXT_ORDINARY_DESTROYED;
        if (g_current_depth > 0) {
            g_current_depth--;
        }
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_get_current(uintptr_t *token)
{
    if (!token) {
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_lock(&g_state.lock);
    *token = g_current_depth == 0 ? 0 : g_state.token;
    pthread_mutex_unlock(&g_state.lock);
    return CXL_CUDA_SUCCESS;
}

int cxl_cuda_context_set_current(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (token == 0) {
        if (g_current_depth > 0) {
            g_current_depth--;
        }
    } else if (!token_is_live_locked(token)) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else if (g_current_depth == 0) {
        g_current_depth = 1;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_push_current(uintptr_t token)
{
    int result = CXL_CUDA_SUCCESS;

    pthread_mutex_lock(&g_state.lock);
    if (!token_is_live_locked(token)) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else if (g_current_depth == UINT32_MAX) {
        result = CXL_CUDA_ERROR_OUT_OF_MEMORY;
    } else {
        g_current_depth++;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

int cxl_cuda_context_pop_current(uintptr_t *token)
{
    if (!token) {
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_lock(&g_state.lock);
    if (g_current_depth == 0) {
        pthread_mutex_unlock(&g_state.lock);
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    *token = g_state.token;
    g_current_depth--;
    pthread_mutex_unlock(&g_state.lock);
    return CXL_CUDA_SUCCESS;
}

int cxl_cuda_context_get_current_live(uintptr_t *token)
{
    int result = CXL_CUDA_SUCCESS;

    if (!token) {
        return CXL_CUDA_ERROR_INVALID_CONTEXT;
    }
    pthread_mutex_lock(&g_state.lock);
    if (g_current_depth == 0) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else if (g_state.mode == CXL_CUDA_CONTEXT_ORDINARY_DESTROYED ||
               g_state.mode == CXL_CUDA_CONTEXT_PRIMARY_INACTIVE) {
        result = CXL_CUDA_ERROR_CONTEXT_IS_DESTROYED;
    } else if (!token_is_live_locked(g_state.token)) {
        result = CXL_CUDA_ERROR_INVALID_CONTEXT;
    } else {
        *token = g_state.token;
    }
    pthread_mutex_unlock(&g_state.lock);
    return result;
}

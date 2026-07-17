#include "cxl_gpu_context_state.h"

#include <pthread.h>
#include <stdio.h>

#define CHECK(expr)                                                           \
    do {                                                                      \
        if (!(expr)) {                                                        \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

typedef struct DestroyedContextThread {
    pthread_barrier_t ready;
    pthread_barrier_t destroyed;
    uintptr_t token;
    int result;
} DestroyedContextThread;

static void *thread_with_destroyed_current(void *opaque)
{
    DestroyedContextThread *thread = opaque;
    uintptr_t token = 0;

    thread->result = cxl_cuda_context_set_current(thread->token);
    if (thread->result == CXL_CUDA_SUCCESS) {
        pthread_barrier_wait(&thread->ready);
        pthread_barrier_wait(&thread->destroyed);
        thread->result = cxl_cuda_context_get_current(&token);
        if (thread->result == CXL_CUDA_SUCCESS && token != thread->token) {
            thread->result = CXL_CUDA_ERROR_INVALID_CONTEXT;
        }
        if (thread->result == CXL_CUDA_SUCCESS) {
            thread->result = cxl_cuda_context_get_current_live(&token);
        }
    }
    return NULL;
}

static int test_ordinary_context(void)
{
    uintptr_t token = 0;

    cxl_cuda_context_state_reset();
    CHECK(cxl_cuda_context_prepare_ordinary_create() == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_commit_ordinary_create(19) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_get_current(&token) == CXL_CUDA_SUCCESS);
    CHECK(token == 19);
    CHECK(cxl_cuda_context_set_current(0) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_get_current(&token) == CXL_CUDA_SUCCESS);
    CHECK(token == 0);
    CHECK(cxl_cuda_context_set_current(18) == CXL_CUDA_ERROR_INVALID_CONTEXT);
    CHECK(cxl_cuda_context_set_current(19) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_push_current(19) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_pop_current(&token) == CXL_CUDA_SUCCESS);
    CHECK(token == 19);
    CHECK(cxl_cuda_context_pop_current(&token) == CXL_CUDA_SUCCESS);
    CHECK(token == 19);
    CHECK(cxl_cuda_context_get_current_live(&token) == CXL_CUDA_ERROR_INVALID_CONTEXT);
    return 0;
}

static int test_destroyed_context_is_thread_local(void)
{
    DestroyedContextThread thread = { .token = 23 };
    pthread_t worker;

    cxl_cuda_context_state_reset();
    CHECK(cxl_cuda_context_commit_ordinary_create(thread.token) == CXL_CUDA_SUCCESS);
    CHECK(pthread_barrier_init(&thread.ready, NULL, 2) == 0);
    CHECK(pthread_barrier_init(&thread.destroyed, NULL, 2) == 0);
    CHECK(pthread_create(&worker, NULL, thread_with_destroyed_current, &thread) == 0);
    pthread_barrier_wait(&thread.ready);
    CHECK(cxl_cuda_context_prepare_destroy(thread.token) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_commit_destroy(thread.token) == CXL_CUDA_SUCCESS);
    pthread_barrier_wait(&thread.destroyed);
    CHECK(pthread_join(worker, NULL) == 0);
    CHECK(thread.result == CXL_CUDA_ERROR_CONTEXT_IS_DESTROYED);
    pthread_barrier_destroy(&thread.destroyed);
    pthread_barrier_destroy(&thread.ready);
    return 0;
}

static int test_primary_context(void)
{
    bool needs_create = false;
    uintptr_t token = 0;
    unsigned int flags = 1;
    int active = -1;

    cxl_cuda_context_state_reset();
    CHECK(cxl_cuda_context_prepare_primary_retain(&needs_create, &token) ==
          CXL_CUDA_SUCCESS);
    CHECK(needs_create && token == 0);
    CHECK(cxl_cuda_context_commit_primary_retain(31) == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_primary_get_state(&flags, &active) == CXL_CUDA_SUCCESS);
    CHECK(flags == 0 && active == 1);
    CHECK(cxl_cuda_context_prepare_primary_retain(&needs_create, &token) ==
          CXL_CUDA_SUCCESS);
    CHECK(!needs_create && token == 31);
    CHECK(cxl_cuda_context_primary_release() == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_primary_release() == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_primary_get_state(&flags, &active) == CXL_CUDA_SUCCESS);
    CHECK(flags == 0 && active == 0);
    CHECK(cxl_cuda_context_primary_reset() == CXL_CUDA_SUCCESS);
    CHECK(cxl_cuda_context_prepare_primary_retain(&needs_create, &token) ==
          CXL_CUDA_ERROR_NOT_SUPPORTED);
    return 0;
}

int main(void)
{
    return test_ordinary_context() || test_destroyed_context_is_thread_local() ||
           test_primary_context();
}

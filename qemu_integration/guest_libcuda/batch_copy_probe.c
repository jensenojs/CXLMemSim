#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef void *cudaStream_t;

typedef struct {
    int type;
    int id;
} cudaMemLocation;

typedef struct {
    int srcAccessOrder;
    cudaMemLocation srcLocHint;
    cudaMemLocation dstLocHint;
    unsigned int flags;
} cudaMemcpyAttributes;

typedef cudaError_t (*cuda_malloc_t)(void **pointer, size_t size);
typedef cudaError_t (*cuda_free_t)(void *pointer);
typedef cudaError_t (*cuda_stream_create_t)(cudaStream_t *stream);
typedef cudaError_t (*cuda_stream_synchronize_t)(cudaStream_t stream);
typedef cudaError_t (*cuda_stream_destroy_t)(cudaStream_t stream);
typedef cudaError_t (*cuda_memcpy_t)(void *destination, const void *source,
                                     size_t size, int kind);
typedef cudaError_t (*cuda_memcpy_batch_async_t)(
    void **destinations, void **sources, size_t *sizes, size_t count,
    cudaMemcpyAttributes *attributes, size_t *attribute_indices,
    size_t attribute_count, size_t *fail_index, cudaStream_t stream);

enum {
    CUDA_SUCCESS = 0,
    CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    CUDA_MEMCPY_SRC_ACCESS_ORDER_ANY = 3,
    RANGE_COUNT = 8,
    RANGE_CAPACITY = 32,
};

static void *required_symbol(void *library, const char *name) {
    dlerror();
    void *symbol = dlsym(library, name);
    const char *error = dlerror();
    if (!symbol || error) {
        printf("batch_copy_probe_symbol name=%s status=unavailable error=%s\n",
               name, error ? error : "null-symbol");
        return NULL;
    }
    return symbol;
}

int tiny_cuda_probe_run(void) {
    static const size_t sizes[RANGE_COUNT] = {3, 5, 7, 11, 13, 17, 19, 23};
    const char *runtime_path = getenv("BATCH_COPY_RUNTIME_LIBRARY");
    uint8_t sources[RANGE_COUNT][RANGE_CAPACITY] = {{0}};
    uint8_t outputs[RANGE_COUNT][RANGE_CAPACITY] = {{0}};
    void *destinations[RANGE_COUNT] = {0};
    void *source_pointers[RANGE_COUNT] = {0};
    size_t mutable_sizes[RANGE_COUNT] = {0};
    size_t attribute_index = 0;
    size_t fail_index = SIZE_MAX;
    cudaStream_t stream = NULL;
    int status = 1;

    if (!runtime_path || !*runtime_path) {
        runtime_path = "/usr/local/cuda/lib64/libcudart.so.12";
    }
    void *runtime = dlopen(runtime_path, RTLD_NOW | RTLD_GLOBAL);
    if (!runtime) {
        printf("batch_copy_probe_runtime path=%s status=unavailable error=%s\n",
               runtime_path, dlerror());
        return 20;
    }

    cuda_malloc_t cuda_malloc = required_symbol(runtime, "cudaMalloc");
    cuda_free_t cuda_free = required_symbol(runtime, "cudaFree");
    cuda_stream_create_t stream_create = required_symbol(runtime, "cudaStreamCreate");
    cuda_stream_synchronize_t stream_synchronize =
        required_symbol(runtime, "cudaStreamSynchronize");
    cuda_stream_destroy_t stream_destroy = required_symbol(runtime, "cudaStreamDestroy");
    cuda_memcpy_t cuda_memcpy = required_symbol(runtime, "cudaMemcpy");
    cuda_memcpy_batch_async_t batch_copy =
        required_symbol(runtime, "cudaMemcpyBatchAsync");
    if (!cuda_malloc || !cuda_free || !stream_create || !stream_synchronize ||
        !stream_destroy || !cuda_memcpy || !batch_copy) {
        status = 21;
        goto out;
    }

    cudaError_t result = stream_create(&stream);
    printf("batch_copy_probe_stream_create result=%d stream=%p\n", result, stream);
    if (result != CUDA_SUCCESS || !stream) {
        status = 22;
        goto out;
    }

    for (size_t range = 0; range < RANGE_COUNT; range++) {
        mutable_sizes[range] = sizes[range];
        source_pointers[range] = sources[range];
        for (size_t index = 0; index < sizes[range]; index++) {
            sources[range][index] = (uint8_t)(range * 31 + index + 1);
        }
        result = cuda_malloc(&destinations[range], sizes[range]);
        if (result != CUDA_SUCCESS || !destinations[range]) {
            printf("batch_copy_probe_malloc range=%zu bytes=%zu result=%d\n",
                   range, sizes[range], result);
            status = 23;
            goto out;
        }
    }

    cudaMemcpyAttributes attributes = {
        .srcAccessOrder = CUDA_MEMCPY_SRC_ACCESS_ORDER_ANY,
    };
    result = batch_copy(destinations, source_pointers, mutable_sizes,
                        RANGE_COUNT, &attributes, &attribute_index, 1,
                        &fail_index, stream);
    printf("batch_copy_probe_submit result=%d fail_index=%zu ranges=%d bytes=98\n",
           result, fail_index, RANGE_COUNT);
    if (result != CUDA_SUCCESS || fail_index != SIZE_MAX) {
        status = 24;
        goto out;
    }

    result = stream_synchronize(stream);
    printf("batch_copy_probe_synchronize result=%d\n", result);
    if (result != CUDA_SUCCESS) {
        status = 25;
        goto out;
    }

    for (size_t range = 0; range < RANGE_COUNT; range++) {
        result = cuda_memcpy(outputs[range], destinations[range], sizes[range],
                             CUDA_MEMCPY_DEVICE_TO_HOST);
        if (result != CUDA_SUCCESS ||
            memcmp(outputs[range], sources[range], sizes[range]) != 0) {
            printf("batch_copy_probe_oracle range=%zu bytes=%zu result=%d status=fail\n",
                   range, sizes[range], result);
            status = 26;
            goto out;
        }
    }

    printf("batch_copy_probe_oracle status=pass ranges=%d bytes=98\n",
           RANGE_COUNT);
    printf("=== BATCH_COPY_PROBE_PASS ===\n");
    status = 0;

out:
    for (size_t range = 0; range < RANGE_COUNT; range++) {
        if (destinations[range] && cuda_free) {
            cuda_free(destinations[range]);
        }
    }
    if (stream && stream_destroy) {
        stream_destroy(stream);
    }
    dlclose(runtime);
    return status;
}

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int initialized;

int cublasCreate_v2(void **handle) {
    initialized = 1;
    *handle = (void *)0x1;
    return 0;
}

int cublasDestroy_v2(void *handle) { return handle == (void *)0x1 ? 0 : 1; }

int cublasSgemm_v2(void *handle, int transa, int transb, int m, int n, int k, const float *alpha, const float *a,
                   int lda, const float *b, int ldb, const float *beta, float *c, int ldc) {
    if (handle != (void *)0x1 || transa != 0 || transb != 0 || !alpha || !a || !b || !beta || !c || m <= 0 || n <= 0 ||
        k <= 0 || lda < m || ldb < k || ldc < m) {
        return 1;
    }
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += a[row + inner * lda] * b[inner + column * ldb];
            }
            c[row + column * ldc] = *alpha * sum + *beta * c[row + column * ldc];
        }
    }
    return 0;
}

int cudaMalloc(void **pointer, size_t size) {
    *pointer = malloc(size);
    return *pointer ? 0 : 2;
}

int cudaMemcpy(void *destination, const void *source, size_t size, int kind) {
    if (!destination || !source || (kind != 1 && kind != 2)) {
        return 1;
    }
    memcpy(destination, source, size);
    return 0;
}

int cudaFree(void *pointer) {
    free(pointer);
    return 0;
}

int cudaDeviceSynchronize(void) { return 0; }

int cuModuleGetLoadingMode(int *mode) {
    if (!initialized) {
        return 3;
    }
    *mode = 2;
    return 0;
}

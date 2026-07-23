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

int cublasGemmEx(void *handle, int transa, int transb, int m, int n, int k, const void *alpha, const void *a, int atype,
                 int lda, const void *b, int btype, int ldb, const void *beta, void *c, int ctype, int ldc,
                 int compute_type, int algorithm) {
    const unsigned short half_zero = 0x0000;
    const unsigned short half_one = 0x3c00;
    const unsigned short half_expected = 0x5800;
    const unsigned short *alpha_half = alpha;
    const unsigned short *a_half = a;
    const unsigned short *b_half = b;
    const unsigned short *beta_half = beta;
    unsigned short *c_half = c;
    if (handle != (void *)0x1 || transa != 1 || transb != 0 || m != 128 || n != 128 || k != 128 || !alpha_half ||
        !a_half || !b_half || !beta_half || !c_half || atype != 2 || btype != 2 || ctype != 2 || lda != 128 ||
        ldb != 128 || ldc != 128 || compute_type != 64 || algorithm != 99 || *alpha_half != half_one ||
        *beta_half != half_zero) {
        return 1;
    }
    for (size_t index = 0; index < 128 * 128; ++index) {
        if (a_half[index] != half_one || b_half[index] != half_one) {
            return 1;
        }
        c_half[index] = half_expected;
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

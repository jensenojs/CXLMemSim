#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*cublas_create_t)(void **handle);
typedef int (*cublas_destroy_t)(void *handle);

static const char *configured_path(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && *value ? value : fallback;
}

static void *load_library(const char *label, const char *path) {
    dlerror();
    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    const char *error = dlerror();
    if (!handle) {
        printf("cublas_create_probe_load label=%s path=%s status=fail error=%s\n", label, path,
               error ? error : "unknown");
        return NULL;
    }
    printf("cublas_create_probe_load label=%s path=%s status=pass handle=%p\n", label, path, handle);
    return handle;
}

int tiny_cuda_probe_run(void) {
    const char *expected = getenv("CUBLAS_CREATE_EXPECTED_REGISTRATIONS");
    const char *ggml_path = configured_path("CUBLAS_CREATE_GGML_LIBRARY", "/opt/llama/bin/libggml-cuda.so.0");
    const char *cublas_path = configured_path("CUBLAS_CREATE_LIBRARY", "libcublas.so.12");
    void *ggml = NULL;
    void *cublas = NULL;
    void *cublas_handle = NULL;

    printf("=== CUBLAS_CREATE_PROBE_BEGIN ===\n");
    if (!expected || !*expected) {
        printf("cublas_create_probe_fail reason=missing-expected-registration-count\n");
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 30;
    }
    printf("cublas_create_probe_registration_expected=%s\n", expected);

    ggml = load_library("exact-libggml-cuda", ggml_path);
    if (!ggml) {
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 31;
    }
    cublas = load_library("libcublas", cublas_path);
    if (!cublas) {
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 32;
    }

    dlerror();
    cublas_create_t create = (cublas_create_t)dlsym(cublas, "cublasCreate_v2");
    const char *create_error = dlerror();
    dlerror();
    cublas_destroy_t destroy = (cublas_destroy_t)dlsym(cublas, "cublasDestroy_v2");
    const char *destroy_error = dlerror();
    if (!create || !destroy) {
        printf("cublas_create_probe_symbols status=fail create=%p destroy=%p create_error=%s destroy_error=%s\n",
               (void *)create, (void *)destroy, create_error ? create_error : "none",
               destroy_error ? destroy_error : "none");
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 33;
    }
    printf("cublas_create_probe_symbols status=pass create=%p destroy=%p\n", (void *)create, (void *)destroy);

    int create_result = create(&cublas_handle);
    printf("cublas_create_result=%d handle=%p\n", create_result, cublas_handle);
    if (create_result != 0 || !cublas_handle) {
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 34;
    }

    int destroy_result = destroy(cublas_handle);
    printf("cublas_destroy_result=%d\n", destroy_result);
    dlclose(cublas);
    dlclose(ggml);
    if (destroy_result != 0) {
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 35;
    }

    printf("=== CUBLAS_CREATE_PROBE_PASS ===\n");
    return 0;
}

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int (*cublas_create_t)(void **handle);
typedef int (*cublas_destroy_t)(void *handle);
typedef int (*cublas_sgemm_t)(void *handle, int transa, int transb, int m, int n, int k, const float *alpha,
                              const float *a, int lda, const float *b, int ldb, const float *beta, float *c, int ldc);
typedef int (*cuda_malloc_t)(void **pointer, size_t size);
typedef int (*cuda_memcpy_t)(void *destination, const void *source, size_t size, int kind);
typedef int (*cuda_free_t)(void *pointer);
typedef int (*cuda_device_synchronize_t)(void);
typedef int (*cu_module_get_loading_mode_t)(int *mode);

enum {
    CUDA_MEMCPY_HOST_TO_DEVICE = 1,
    CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    CUBLAS_OP_N = 0,
};

static const char *configured_path(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && *value ? value : fallback;
}

static void observe_module_loading_environment(const char *phase) {
    const char *module_loading = getenv("CUDA_MODULE_LOADING");
    const char *enable_lazy_loading = getenv("CUDA_ENABLE_MODULE_LAZY_LOADING");
    printf("cublas_module_loading_environment phase=%s CUDA_MODULE_LOADING=%s "
           "CUDA_ENABLE_MODULE_LAZY_LOADING=%s\n",
           phase, module_loading ? module_loading : "<unset>", enable_lazy_loading ? enable_lazy_loading : "<unset>");
}

/*
 * The cuBLAS initialization window is where libcudart invokes the guest
 * shim's cuLibraryLoadData registrations.  dladdr(code) records the
 * registration origin, while this inventory records the complete loader
 * image set before and after the two explicit dlopen calls below.  Keep the
 * output as one JSON value per line: the host verifier can reject malformed
 * records and bind a registration to path plus load base without interpreting
 * prose from a serial log.
 */
static void json_string(FILE *stream, const char *value) {
    fputc('"', stream);
    for (const unsigned char *cursor = (const unsigned char *)(value ? value : ""); *cursor; ++cursor) {
        switch (*cursor) {
        case '"':
            fputs("\\\"", stream);
            break;
        case '\\':
            fputs("\\\\", stream);
            break;
        case '\b':
            fputs("\\b", stream);
            break;
        case '\f':
            fputs("\\f", stream);
            break;
        case '\n':
            fputs("\\n", stream);
            break;
        case '\r':
            fputs("\\r", stream);
            break;
        case '\t':
            fputs("\\t", stream);
            break;
        default:
            if (*cursor < 0x20) {
                fprintf(stream, "\\u%04x", *cursor);
            } else {
                fputc(*cursor, stream);
            }
        }
    }
    fputc('"', stream);
}

struct loader_inventory_context {
    const char *phase;
    unsigned int count;
};

static int emit_loader_dso(struct dl_phdr_info *info, size_t size, void *opaque) {
    (void)size;
    struct loader_inventory_context *context = opaque;
    const char *path = info->dlpi_name ? info->dlpi_name : "";
    const char *kind = path[0] == '\0' ? "main-program"
                                       : (strncmp(path, "linux-vdso", strlen("linux-vdso")) == 0 ? "special" : "file");
    printf("cublas_loader_dso={\"schema_version\":1,\"phase\":");
    json_string(stdout, context->phase);
    printf(",\"kind\":");
    json_string(stdout, kind);
    printf(",\"path\":");
    json_string(stdout, path);
    printf(",\"load_base\":\"0x%llx\",\"phnum\":%u}\n", (unsigned long long)info->dlpi_addr,
           (unsigned int)info->dlpi_phnum);
    context->count++;
    return 0;
}

static void emit_loader_inventory(const char *phase) {
    struct loader_inventory_context context = {.phase = phase, .count = 0};
    printf("=== CUBLAS_LOADER_INVENTORY_BEGIN phase=%s ===\n", phase);
    dl_iterate_phdr(emit_loader_dso, &context);
    printf("=== CUBLAS_LOADER_INVENTORY_END phase=%s count=%u ===\n", phase, context.count);
    fflush(stdout);
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

static int observe_module_loading_mode(const char *phase, const char *driver_path, int *driver_result) {
    dlerror();
    void *driver = dlopen(driver_path, RTLD_NOW | RTLD_NOLOAD);
    const char *error = dlerror();
    if (!driver) {
        printf("cublas_module_loading_mode phase=%s status=driver-unavailable error=%s\n", phase,
               error ? error : "unknown");
        return 0;
    }

    dlerror();
    cu_module_get_loading_mode_t get_mode = (cu_module_get_loading_mode_t)dlsym(driver, "cuModuleGetLoadingMode");
    error = dlerror();
    if (!get_mode) {
        printf("cublas_module_loading_mode phase=%s status=unresolved error=%s\n", phase, error ? error : "unknown");
        dlclose(driver);
        return 0;
    }

    int mode = 0;
    int result = get_mode(&mode);
    printf("cublas_module_loading_mode phase=%s status=called result=%d mode=%d\n", phase, result, mode);
    *driver_result = result;
    dlclose(driver);
    return 1;
}

/*
 * The Type-2 GDB diagnostic preserves the RTLD_NOW load and cublasCreate
 * call. This optional rendezvous runs after both DSOs and cublasCreate_v2 are
 * resolved. The host uses the exact mapping identity to install guest-address
 * breakpoints, then creates the continuation file. It cannot change CUDA
 * arguments, loader mode, return values, BAR2, QEMU, or HetGPU behavior.
 */
static int wait_for_gdb_observer(cublas_create_t create) {
    const char *directory = getenv("CUBLAS_CREATE_GDB_SYNC_DIR");
    if (!directory || !*directory) {
        return 0;
    }

    Dl_info info = {0};
    if (dladdr((const void *)create, &info) == 0 || !info.dli_fname || !info.dli_fbase) {
        printf("cublas_gdb_sync status=fail reason=dladdr-create\n");
        return -1;
    }

    dlerror();
    void *ctx_init = dlsym(RTLD_DEFAULT, "cublasLtCtxInit");
    const char *ctx_init_error = dlerror();
    Dl_info ctx_init_info = {0};
    if (!ctx_init || ctx_init_error || dladdr(ctx_init, &ctx_init_info) == 0 || !ctx_init_info.dli_fname ||
        !ctx_init_info.dli_fbase) {
        printf("cublas_gdb_sync status=fail reason=dladdr-cublas-lt error=%s\n",
               ctx_init_error ? ctx_init_error : "unknown");
        return -1;
    }

    char ready[1024];
    char proceed[1024];
    if (snprintf(ready, sizeof(ready), "%s/ready.json", directory) >= (int)sizeof(ready) ||
        snprintf(proceed, sizeof(proceed), "%s/continue", directory) >= (int)sizeof(proceed)) {
        printf("cublas_gdb_sync status=fail reason=path-too-long\n");
        return -1;
    }

    int descriptor = open(ready, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0) {
        printf("cublas_gdb_sync status=fail reason=ready-create errno=%d\n", errno);
        return -1;
    }
    char record[1400];
    int length = snprintf(record, sizeof(record),
                          "{\"schema_version\":2,\"library\":\"%s\",\"load_base\":\"%p\","
                          "\"cublas_create\":\"%p\",\"cublas_lt_library\":\"%s\","
                          "\"cublas_lt_load_base\":\"%p\",\"cublas_lt_ctx_init\":\"%p\"}\n",
                          info.dli_fname, info.dli_fbase, (const void *)create, ctx_init_info.dli_fname,
                          ctx_init_info.dli_fbase, ctx_init);
    if (length < 0 || length >= (int)sizeof(record) || write(descriptor, record, (size_t)length) != length ||
        close(descriptor) != 0) {
        printf("cublas_gdb_sync status=fail reason=ready-write errno=%d\n", errno);
        return -1;
    }
    printf("cublas_gdb_sync status=ready library=%s load_base=%p cublas_lt_library=%s cublas_lt_load_base=%p\n",
           info.dli_fname, info.dli_fbase, ctx_init_info.dli_fname, ctx_init_info.dli_fbase);
    fflush(stdout);

    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000};
    for (unsigned int attempt = 0; attempt < 1200; ++attempt) {
        if (access(proceed, F_OK) == 0) {
            printf("cublas_gdb_sync status=continue\n");
            fflush(stdout);
            return 0;
        }
        nanosleep(&delay, NULL);
    }
    printf("cublas_gdb_sync status=fail reason=continue-timeout\n");
    return -1;
}

static int run_sgemm_oracle(void *cublas_handle, cublas_sgemm_t sgemm, cuda_malloc_t cuda_malloc,
                            cuda_memcpy_t cuda_memcpy, cuda_free_t cuda_free,
                            cuda_device_synchronize_t cuda_device_synchronize) {
    static const float host_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    static const float host_b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    static const float expected[4] = {23.0f, 34.0f, 31.0f, 46.0f};
    float host_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *device_a = NULL;
    float *device_b = NULL;
    float *device_c = NULL;
    int status = cuda_malloc((void **)&device_a, sizeof(host_a));
    if (status == 0) {
        status = cuda_malloc((void **)&device_b, sizeof(host_b));
    }
    if (status == 0) {
        status = cuda_malloc((void **)&device_c, sizeof(host_c));
    }
    printf("cublas_sgemm_allocation_result=%d a=%p b=%p c=%p\n", status, (void *)device_a, (void *)device_b,
           (void *)device_c);
    if (status != 0) {
        goto cleanup;
    }
    status = cuda_memcpy(device_a, host_a, sizeof(host_a), CUDA_MEMCPY_HOST_TO_DEVICE);
    if (status == 0) {
        status = cuda_memcpy(device_b, host_b, sizeof(host_b), CUDA_MEMCPY_HOST_TO_DEVICE);
    }
    printf("cublas_sgemm_input_copy_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }
    status =
        sgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, device_a, 2, device_b, 2, &beta, device_c, 2);
    printf("cublas_sgemm_result=%d m=2 n=2 k=2\n", status);
    if (status != 0) {
        goto cleanup;
    }
    status = cuda_device_synchronize();
    printf("cublas_sgemm_synchronize_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }
    status = cuda_memcpy(host_c, device_c, sizeof(host_c), CUDA_MEMCPY_DEVICE_TO_HOST);
    printf("cublas_sgemm_output_copy_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }
    float max_error = 0.0f;
    for (size_t index = 0; index < 4; ++index) {
        float error = host_c[index] - expected[index];
        if (error < 0.0f) {
            error = -error;
        }
        if (error > max_error) {
            max_error = error;
        }
        if (error > 0.000001f) {
            printf("cublas_sgemm_oracle_mismatch index=%zu actual=%.9g expected=%.9g abs_error=%.9g\n", index,
                   host_c[index], expected[index], error);
            status = 1;
            goto cleanup;
        }
    }
    printf("cublas_sgemm_numerical_oracle=pass elements=4 max_abs_error=%.9g\n", max_error);

cleanup:
    if (device_c && cuda_free(device_c) != 0 && status == 0) {
        status = 1;
    }
    if (device_b && cuda_free(device_b) != 0 && status == 0) {
        status = 1;
    }
    if (device_a && cuda_free(device_a) != 0 && status == 0) {
        status = 1;
    }
    printf("cublas_sgemm_cleanup_result=%d\n", status);
    return status;
}

int tiny_cuda_probe_run(void) {
    const char *expected = getenv("CUBLAS_CREATE_EXPECTED_REGISTRATIONS");
    const char *ggml_path = configured_path("CUBLAS_CREATE_GGML_LIBRARY", "/opt/llama/bin/libggml-cuda.so.0");
    const char *cublas_path = configured_path("CUBLAS_CREATE_LIBRARY", "libcublas.so.12");
    const char *runtime_path = configured_path("CUBLAS_CREATE_RUNTIME_LIBRARY", "libcudart.so.12");
    const char *driver_path = configured_path("CUBLAS_CREATE_DRIVER_LIBRARY", "libcuda.so.1");
    void *ggml = NULL;
    void *cublas = NULL;
    void *runtime = NULL;
    void *cublas_handle = NULL;
    int loading_mode_result = 0;

    printf("=== CUBLAS_CREATE_PROBE_BEGIN ===\n");
    if (!expected || !*expected) {
        printf("cublas_create_probe_fail reason=missing-expected-registration-count\n");
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 30;
    }
    printf("cublas_create_probe_registration_expected=%s\n", expected);

    emit_loader_inventory("before");

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
    emit_loader_inventory("after");

    dlerror();
    cublas_create_t create = (cublas_create_t)dlsym(cublas, "cublasCreate_v2");
    const char *create_error = dlerror();
    dlerror();
    cublas_destroy_t destroy = (cublas_destroy_t)dlsym(cublas, "cublasDestroy_v2");
    const char *destroy_error = dlerror();
    dlerror();
    cublas_sgemm_t sgemm = (cublas_sgemm_t)dlsym(cublas, "cublasSgemm_v2");
    const char *sgemm_error = dlerror();
    if (!create || !destroy || !sgemm) {
        printf("cublas_create_probe_symbols status=fail create=%p destroy=%p sgemm=%p create_error=%s "
               "destroy_error=%s sgemm_error=%s\n",
               (void *)create, (void *)destroy, (void *)sgemm, create_error ? create_error : "none",
               destroy_error ? destroy_error : "none", sgemm_error ? sgemm_error : "none");
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 33;
    }
    printf("cublas_create_probe_symbols status=pass create=%p destroy=%p sgemm=%p\n", (void *)create, (void *)destroy,
           (void *)sgemm);

    runtime = dlopen(runtime_path, RTLD_NOW | RTLD_NOLOAD);
    if (!runtime) {
        const char *runtime_error = dlerror();
        printf("cublas_create_probe_runtime status=fail path=%s error=%s\n", runtime_path,
               runtime_error ? runtime_error : "unknown");
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 39;
    }
    cuda_malloc_t cuda_malloc = (cuda_malloc_t)dlsym(runtime, "cudaMalloc");
    cuda_memcpy_t cuda_memcpy = (cuda_memcpy_t)dlsym(runtime, "cudaMemcpy");
    cuda_free_t cuda_free = (cuda_free_t)dlsym(runtime, "cudaFree");
    cuda_device_synchronize_t cuda_device_synchronize =
        (cuda_device_synchronize_t)dlsym(runtime, "cudaDeviceSynchronize");
    if (!cuda_malloc || !cuda_memcpy || !cuda_free || !cuda_device_synchronize) {
        printf("cublas_create_probe_runtime_symbols status=fail malloc=%p memcpy=%p free=%p synchronize=%p\n",
               (void *)cuda_malloc, (void *)cuda_memcpy, (void *)cuda_free, (void *)cuda_device_synchronize);
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 40;
    }
    printf("cublas_create_probe_runtime_symbols status=pass malloc=%p memcpy=%p free=%p synchronize=%p\n",
           (void *)cuda_malloc, (void *)cuda_memcpy, (void *)cuda_free, (void *)cuda_device_synchronize);

    if (wait_for_gdb_observer(create) != 0) {
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 38;
    }

    observe_module_loading_environment("before-create");
    if (!observe_module_loading_mode("before-create", driver_path, &loading_mode_result)) {
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 36;
    }

    int create_result = create(&cublas_handle);
    printf("cublas_create_result=%d handle=%p\n", create_result, cublas_handle);
    observe_module_loading_environment("after-create");
    if (create_result != 0 || !cublas_handle) {
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 34;
    }

    if (!observe_module_loading_mode("after-create", driver_path, &loading_mode_result) || loading_mode_result != 0) {
        destroy(cublas_handle);
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 37;
    }

    if (run_sgemm_oracle(cublas_handle, sgemm, cuda_malloc, cuda_memcpy, cuda_free, cuda_device_synchronize) != 0) {
        destroy(cublas_handle);
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 41;
    }

    int destroy_result = destroy(cublas_handle);
    printf("cublas_destroy_result=%d\n", destroy_result);
    dlclose(runtime);
    dlclose(cublas);
    dlclose(ggml);
    if (destroy_result != 0) {
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 35;
    }

    printf("=== CUBLAS_CREATE_PROBE_PASS ===\n");
    return 0;
}

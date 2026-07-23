#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef int (*cublas_create_t)(void **handle);
typedef int (*cublas_destroy_t)(void *handle);
typedef int (*cublas_sgemm_t)(void *handle, int transa, int transb, int m, int n, int k, const float *alpha,
                              const float *a, int lda, const float *b, int ldb, const float *beta, float *c, int ldc);
typedef int (*cublas_gemm_ex_t)(void *handle, int transa, int transb, int m, int n, int k, const void *alpha,
                                const void *a, int atype, int lda, const void *b, int btype, int ldb, const void *beta,
                                void *c, int ctype, int ldc, int compute_type, int algorithm);
typedef int (*cuda_malloc_t)(void **pointer, size_t size);
typedef int (*cuda_memcpy_t)(void *destination, const void *source, size_t size, int kind);
typedef int (*cuda_free_t)(void *pointer);
typedef int (*cuda_device_synchronize_t)(void);
typedef int (*cu_module_get_loading_mode_t)(int *mode);
typedef int (*cu_library_load_data_t)(void **library, const void *code, int *jit_options, void **jit_option_values,
                                      unsigned int num_jit_options, int *library_options,
                                      void **library_option_values, unsigned int num_library_options);
typedef int (*cu_library_get_kernel_t)(void **kernel, void *library, const char *name);
typedef int (*cu_kernel_get_function_t)(void **function, void *kernel);
typedef int (*cu_func_get_attribute_t)(int *value, int attribute, void *function);
typedef int (*cu_occupancy_t)(int *blocks, void *function, int block_size, size_t dynamic_shared_memory,
                              unsigned int flags);
typedef int (*cu_library_unload_t)(void *library);

enum {
    CUDA_MEMCPY_HOST_TO_DEVICE = 1,
    CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUDA_R_16F = 2,
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_LIBRARY_BINARY_IS_PRESERVED = 1,
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1,
};

static int is_lower_hex_string(const char *value, size_t length) {
    if (!value || strlen(value) != length) {
        return 0;
    }
    for (size_t index = 0; index < length; ++index) {
        if (!((value[index] >= '0' && value[index] <= '9') || (value[index] >= 'a' && value[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

static int parse_positive_size(const char *value, size_t *result) {
    if (!value || !*value || value[0] == '0') {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end || parsed > SIZE_MAX) {
        return 0;
    }
    *result = (size_t)parsed;
    return 1;
}

static int parse_unsigned(const char *value, unsigned int *result) {
    if (!value || !*value) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno || !end || *end || parsed > UINT32_MAX) {
        return 0;
    }
    *result = (unsigned int)parsed;
    return 1;
}

static int safe_corpus_file_name(const char *value) {
    if (!value || strncmp(value, "event-", 6) != 0 || strstr(value, "..") || strchr(value, '/')) {
        return 0;
    }
    const char *suffix = strstr(value, ".elf");
    return suffix && suffix[4] == '\0';
}

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

static int run_gemm_ex_oracle(void *cublas_handle, cublas_gemm_ex_t gemm_ex, cuda_malloc_t cuda_malloc,
                              cuda_memcpy_t cuda_memcpy, cuda_free_t cuda_free,
                              cuda_device_synchronize_t cuda_device_synchronize) {
    enum {
        m = 128,
        n = 128,
        k = 128,
        matrix_elements = 128 * 128,
    };
    const uint16_t half_zero = 0x0000;
    const uint16_t half_one = 0x3c00;
    const uint16_t half_expected = 0x5800;
    const size_t matrix_bytes = matrix_elements * sizeof(uint16_t);
    uint16_t *host_a = malloc(matrix_bytes);
    uint16_t *host_b = malloc(matrix_bytes);
    uint16_t *host_c = calloc(matrix_elements, sizeof(uint16_t));
    uint16_t *device_a = NULL;
    uint16_t *device_b = NULL;
    uint16_t *device_c = NULL;
    int status = 0;

    if (!host_a || !host_b || !host_c) {
        printf("cublas_gemm_ex_host_allocation_result=fail bytes=%zu\n", matrix_bytes);
        status = 1;
        goto cleanup;
    }
    for (size_t index = 0; index < matrix_elements; ++index) {
        host_a[index] = half_one;
        host_b[index] = half_one;
    }
    printf("cublas_gemm_ex_host_allocation_result=pass bytes=%zu\n", matrix_bytes);

    status = cuda_malloc((void **)&device_a, matrix_bytes);
    if (status == 0) {
        status = cuda_malloc((void **)&device_b, matrix_bytes);
    }
    if (status == 0) {
        status = cuda_malloc((void **)&device_c, matrix_bytes);
    }
    printf("cublas_gemm_ex_allocation_result=%d a=%p b=%p c=%p bytes=%zu\n", status, (void *)device_a, (void *)device_b,
           (void *)device_c, matrix_bytes);
    if (status != 0) {
        goto cleanup;
    }

    status = cuda_memcpy(device_a, host_a, matrix_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
    if (status == 0) {
        status = cuda_memcpy(device_b, host_b, matrix_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
    }
    if (status == 0) {
        status = cuda_memcpy(device_c, host_c, matrix_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
    }
    printf("cublas_gemm_ex_input_copy_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }

    status =
        gemm_ex(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &half_one, device_a, CUDA_R_16F, k, device_b,
                CUDA_R_16F, k, &half_zero, device_c, CUDA_R_16F, m, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    printf("cublas_gemm_ex_result=%d transa=%d transb=%d m=%d n=%d k=%d atype=%d btype=%d ctype=%d "
           "compute_type=%d algorithm=%d\n",
           status, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F,
           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != 0) {
        goto cleanup;
    }

    status = cuda_device_synchronize();
    printf("cublas_gemm_ex_synchronize_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }
    status = cuda_memcpy(host_c, device_c, matrix_bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
    printf("cublas_gemm_ex_output_copy_result=%d\n", status);
    if (status != 0) {
        goto cleanup;
    }

    for (size_t index = 0; index < matrix_elements; ++index) {
        if (host_c[index] != half_expected) {
            printf("cublas_gemm_ex_oracle_mismatch index=%zu actual_bits=0x%04x expected_bits=0x%04x\n", index,
                   host_c[index], half_expected);
            status = 1;
            goto cleanup;
        }
    }
    printf("cublas_gemm_ex_numerical_oracle=pass elements=%d expected_bits=0x%04x\n", matrix_elements, half_expected);

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
    free(host_c);
    free(host_b);
    free(host_a);
    printf("cublas_gemm_ex_cleanup_result=%d\n", status);
    return status;
}

struct library_corpus_api {
    cu_library_load_data_t load;
    cu_library_get_kernel_t get_kernel;
    cu_kernel_get_function_t get_function;
    cu_func_get_attribute_t get_attribute;
    cu_occupancy_t occupancy;
    cu_library_unload_t unload;
};

struct library_corpus_event_failures {
    const char *phases[10];
    unsigned int count;
    unsigned int blocked;
};

static void record_library_corpus_failure(struct library_corpus_event_failures *failures, const char *phase) {
    if (failures->count < sizeof(failures->phases) / sizeof(failures->phases[0])) {
        failures->phases[failures->count++] = phase;
    }
}

static void emit_library_corpus_blocked(struct library_corpus_event_failures *failures, unsigned int event,
                                        const char *phase, const char *blocked_by) {
    printf("library_corpus_event event=%u phase=%s result=blocked blocked-by=%s\n", event, phase, blocked_by);
    failures->blocked++;
}

static void emit_library_corpus_blocked_range(struct library_corpus_event_failures *failures, unsigned int event,
                                              const char *blocked_by, unsigned int first_phase) {
    static const char *const phases[] = {
        "open",          "identity", "mmap",         "load",         "get-kernel", "get-function",
        "get-attribute", "occupancy", "unload",      "stale-library", "stale-kernel",
    };

    for (unsigned int index = first_phase; index < sizeof(phases) / sizeof(phases[0]); ++index) {
        emit_library_corpus_blocked(failures, event, phases[index], blocked_by);
    }
}

static void emit_library_corpus_event_summary(unsigned int event,
                                              const struct library_corpus_event_failures *failures) {
    if (!failures->count) {
        printf("library_corpus_event_summary event=%u state=pass failed_phases=none blocked_phases=0\n", event);
        return;
    }

    printf("library_corpus_event_summary event=%u state=fail failed_phase=%s failed_phases=",
           event, failures->phases[0]);
    for (unsigned int index = 0; index < failures->count; ++index) {
        printf("%s%s", index ? "," : "", failures->phases[index]);
    }
    printf(" blocked_phases=%u\n", failures->blocked);
}

static void emit_library_corpus_contract_event_failure(unsigned int event, const char *reason) {
    struct library_corpus_event_failures failures = {0};

    printf("=== CUBLAS_LIBRARY_CORPUS_EVENT_BEGIN event=%u ===\n", event);
    printf("library_corpus_event event=%u phase=contract result=fail reason=%s\n", event, reason);
    record_library_corpus_failure(&failures, "contract");
    emit_library_corpus_blocked_range(&failures, event, "contract", 0);
    emit_library_corpus_event_summary(event, &failures);
    printf("=== CUBLAS_LIBRARY_CORPUS_EVENT_FAIL event=%u ===\n", event);
}

static int resolve_library_corpus_api(void *driver, struct library_corpus_api *api) {
    dlerror();
    api->load = (cu_library_load_data_t)dlsym(driver, "cuLibraryLoadData");
    api->get_kernel = (cu_library_get_kernel_t)dlsym(driver, "cuLibraryGetKernel");
    api->get_function = (cu_kernel_get_function_t)dlsym(driver, "cuKernelGetFunction");
    api->get_attribute = (cu_func_get_attribute_t)dlsym(driver, "cuFuncGetAttribute");
    api->occupancy =
        (cu_occupancy_t)dlsym(driver, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    api->unload = (cu_library_unload_t)dlsym(driver, "cuLibraryUnload");
    const char *error = dlerror();
    if (error || !api->load || !api->get_kernel || !api->get_function || !api->get_attribute || !api->occupancy ||
        !api->unload) {
        printf("library_corpus_symbols status=fail load=%p get_kernel=%p get_function=%p get_attribute=%p "
               "occupancy=%p unload=%p error=%s\n",
               (void *)api->load, (void *)api->get_kernel, (void *)api->get_function, (void *)api->get_attribute,
               (void *)api->occupancy, (void *)api->unload, error ? error : "none");
        return -1;
    }
    printf("library_corpus_symbols status=pass load=%p get_kernel=%p get_function=%p get_attribute=%p "
           "occupancy=%p unload=%p\n",
           (void *)api->load, (void *)api->get_kernel, (void *)api->get_function, (void *)api->get_attribute,
           (void *)api->occupancy, (void *)api->unload);
    return 0;
}

static int run_library_corpus_event(int root_fd, const struct library_corpus_api *api, unsigned int event,
                                    const char *file_name, size_t expected_size, const char *expected_sha256,
                                    const char *kernel_name) {
    int file = -1;
    void *code = MAP_FAILED;
    void *library = NULL;
    void *kernel = NULL;
    void *function = NULL;
    int library_loaded = 0;
    int library_unloaded = 0;
    struct library_corpus_event_failures failures = {0};

    printf("=== CUBLAS_LIBRARY_CORPUS_EVENT_BEGIN event=%u ===\n", event);
    file = openat(root_fd, file_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) {
        printf("library_corpus_event event=%u phase=open result=fail errno=%d file=%s\n", event, errno, file_name);
        record_library_corpus_failure(&failures, "open");
        emit_library_corpus_blocked_range(&failures, event, "open", 1);
        goto cleanup;
    }
    printf("library_corpus_event event=%u phase=open result=pass file=%s\n", event, file_name);
    struct stat facts = {0};
    if (fstat(file, &facts) != 0) {
        printf("library_corpus_event event=%u phase=identity result=fail reason=fstat errno=%d\n", event, errno);
        record_library_corpus_failure(&failures, "identity");
        emit_library_corpus_blocked_range(&failures, event, "identity", 2);
        goto cleanup;
    }
    if (!S_ISREG(facts.st_mode) || facts.st_size <= 0 ||
        (uintmax_t)facts.st_size != (uintmax_t)expected_size) {
        printf("library_corpus_event event=%u phase=identity result=fail expected_size=%zu actual_size=%jd "
               "regular=%d\n",
               event, expected_size, (intmax_t)facts.st_size, S_ISREG(facts.st_mode));
        record_library_corpus_failure(&failures, "identity");
        emit_library_corpus_blocked_range(&failures, event, "identity", 2);
        goto cleanup;
    }
    printf("library_corpus_event event=%u phase=identity result=pass file=%s raw_size=%zu sha256=%s kernel=%s\n",
           event, file_name, expected_size, expected_sha256, kernel_name);

    code = mmap(NULL, expected_size, PROT_READ, MAP_PRIVATE, file, 0);
    if (code == MAP_FAILED) {
        printf("library_corpus_event event=%u phase=mmap result=fail errno=%d\n", event, errno);
        record_library_corpus_failure(&failures, "mmap");
        emit_library_corpus_blocked_range(&failures, event, "mmap", 3);
        goto cleanup;
    }
    printf("library_corpus_event event=%u phase=mmap result=pass raw_size=%zu\n", event, expected_size);
    int option = CU_LIBRARY_BINARY_IS_PRESERVED;
    int result = api->load(&library, code, NULL, NULL, 0, &option, NULL, 1);
    printf("library_corpus_event event=%u phase=load result=%d library=%p option=%d option_values=null\n", event,
           result, library, option);
    if (result != 0 || !library) {
        record_library_corpus_failure(&failures, "load");
        emit_library_corpus_blocked_range(&failures, event, "load", 4);
        goto cleanup;
    }
    library_loaded = 1;

    result = api->get_kernel(&kernel, library, kernel_name);
    printf("library_corpus_event event=%u phase=get-kernel result=%d kernel=%p\n", event, result, kernel);
    if (result != 0 || !kernel) {
        record_library_corpus_failure(&failures, "get-kernel");
        emit_library_corpus_blocked(&failures, event, "get-function", "get-kernel");
        emit_library_corpus_blocked(&failures, event, "get-attribute", "get-kernel");
        emit_library_corpus_blocked(&failures, event, "occupancy", "get-kernel");
        goto unload;
    }
    result = api->get_function(&function, kernel);
    printf("library_corpus_event event=%u phase=get-function result=%d function=%p\n", event, result, function);
    if (result != 0 || !function) {
        record_library_corpus_failure(&failures, "get-function");
        emit_library_corpus_blocked(&failures, event, "get-attribute", "get-function");
        emit_library_corpus_blocked(&failures, event, "occupancy", "get-function");
        goto unload;
    }

    int attribute_value = 0;
    result = api->get_attribute(&attribute_value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function);
    printf("library_corpus_event event=%u phase=get-attribute result=%d attribute=%d value=%d\n", event, result,
           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, attribute_value);
    if (result != 0) {
        record_library_corpus_failure(&failures, "get-attribute");
        emit_library_corpus_blocked(&failures, event, "occupancy", "get-attribute");
        goto unload;
    }
    int blocks = 0;
    result = api->occupancy(&blocks, function, 128, 0, CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE);
    printf("library_corpus_event event=%u phase=occupancy result=%d block_size=128 dynamic_shared_memory=0 "
           "flags=%d blocks=%d\n",
           event, result, CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, blocks);
    if (result != 0 || blocks <= 0) {
        record_library_corpus_failure(&failures, "occupancy");
    }

unload:
    result = api->unload(library);
    printf("library_corpus_event event=%u phase=unload result=%d\n", event, result);
    if (result != 0) {
        record_library_corpus_failure(&failures, "unload");
        emit_library_corpus_blocked(&failures, event, "stale-library", "unload");
        emit_library_corpus_blocked(&failures, event, "stale-kernel", "unload");
        goto cleanup;
    }
    library_unloaded = 1;
    void *stale_kernel = NULL;
    result = api->get_kernel(&stale_kernel, library, kernel_name);
    printf("library_corpus_event event=%u phase=stale-library result=%d kernel=%p expected=%d\n", event, result,
           stale_kernel, CUDA_ERROR_INVALID_HANDLE);
    if (result != CUDA_ERROR_INVALID_HANDLE || stale_kernel) {
        record_library_corpus_failure(&failures, "stale-library");
    }
    if (kernel) {
        void *stale_function = NULL;
        result = api->get_function(&stale_function, kernel);
        printf("library_corpus_event event=%u phase=stale-kernel result=%d function=%p expected=%d\n", event, result,
               stale_function, CUDA_ERROR_INVALID_HANDLE);
        if (result != CUDA_ERROR_INVALID_HANDLE || stale_function) {
            record_library_corpus_failure(&failures, "stale-kernel");
        }
    } else {
        emit_library_corpus_blocked(&failures, event, "stale-kernel", "get-kernel");
    }

cleanup:
    if (library_loaded && !library_unloaded) {
        printf("library_corpus_event event=%u cleanup=library state=live-after-unload-failure\n", event);
    }
    if (code != MAP_FAILED) {
        munmap(code, expected_size);
    }
    if (file >= 0) {
        close(file);
    }
    emit_library_corpus_event_summary(event, &failures);
    printf("=== CUBLAS_LIBRARY_CORPUS_EVENT_%s event=%u ===\n", failures.count ? "FAIL" : "PASS", event);
    return failures.count ? 1 : 0;
}

static int run_library_corpus_replay(const char *root_path, unsigned int expected_events, const char *driver_path) {
    int root = open(root_path, O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (root < 0) {
        printf("library_corpus_replay status=fail phase=open-root root=%s errno=%d\n", root_path, errno);
        return -1;
    }
    int contract_fd = openat(root, "replay.tsv", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (contract_fd < 0) {
        printf("library_corpus_replay status=fail phase=open-contract errno=%d\n", errno);
        close(root);
        return -1;
    }
    FILE *contract = fdopen(contract_fd, "r");
    if (!contract) {
        printf("library_corpus_replay status=fail phase=fdopen errno=%d\n", errno);
        close(contract_fd);
        close(root);
        return -1;
    }

    void *driver = dlopen(driver_path, RTLD_NOW | RTLD_NOLOAD);
    if (!driver) {
        printf("library_corpus_replay status=fail phase=driver error=%s\n", dlerror());
        fclose(contract);
        close(root);
        return -1;
    }
    struct library_corpus_api api = {0};
    if (resolve_library_corpus_api(driver, &api) != 0) {
        dlclose(driver);
        fclose(contract);
        close(root);
        return -1;
    }

    char *line = NULL;
    size_t capacity = 0;
    ssize_t length = getline(&line, &capacity, contract);
    if (length <= 0 || line[length - 1] != '\n') {
        printf("library_corpus_replay status=fail phase=header reason=missing-or-unterminated\n");
        goto fail;
    }
    line[length - 1] = '\0';
    char *cursor = line;
    char *schema = strsep(&cursor, "\t");
    char *task = strsep(&cursor, "\t");
    char *digest = strsep(&cursor, "\t");
    char *manifest_sha256 = strsep(&cursor, "\t");
    char *count_text = strsep(&cursor, "\t");
    unsigned int declared_events = 0;
    if (!schema || strcmp(schema, "cxl-kimi-library-corpus-v1") != 0 || !task || !*task || !digest ||
        strncmp(digest, "sha256:", 7) != 0 || !is_lower_hex_string(digest + 7, 64) ||
        !is_lower_hex_string(manifest_sha256, 64) || !parse_unsigned(count_text, &declared_events) || cursor ||
        declared_events != expected_events) {
        printf("library_corpus_replay status=fail phase=header reason=invalid expected_events=%u\n",
               expected_events);
        goto fail;
    }
    printf("library_corpus_replay_identity schema=1 task=%s digest=%s manifest_sha256=%s events=%u\n", task,
           digest, manifest_sha256, declared_events);

    unsigned int failed_events = 0;
    unsigned int contract_failures = 0;
    for (unsigned int expected_event = 0; expected_event < expected_events; ++expected_event) {
        length = getline(&line, &capacity, contract);
        if (length <= 0 || line[length - 1] != '\n') {
            printf("library_corpus_replay status=fail phase=event reason=missing-or-unterminated event=%u\n",
                   expected_event);
            contract_failures++;
            for (unsigned int blocked_event = expected_event; blocked_event < expected_events; ++blocked_event) {
                emit_library_corpus_contract_event_failure(blocked_event, "missing-or-unterminated");
                failed_events++;
            }
            break;
        }
        line[length - 1] = '\0';
        cursor = line;
        char *event_text = strsep(&cursor, "\t");
        char *file_name = strsep(&cursor, "\t");
        char *size_text = strsep(&cursor, "\t");
        char *sha256 = strsep(&cursor, "\t");
        char *kernel_name = strsep(&cursor, "\t");
        unsigned int event = 0;
        size_t size = 0;
        if (!parse_unsigned(event_text, &event) || event != expected_event || !safe_corpus_file_name(file_name) ||
            !parse_positive_size(size_text, &size) || !is_lower_hex_string(sha256, 64) || !kernel_name ||
            !*kernel_name || cursor) {
            printf("library_corpus_replay status=fail phase=event reason=invalid event=%u\n", expected_event);
            emit_library_corpus_contract_event_failure(expected_event, "invalid-record");
            failed_events++;
            continue;
        }
        if (run_library_corpus_event(root, &api, event, file_name, size, sha256, kernel_name) != 0) {
            failed_events++;
        }
    }
    if (getline(&line, &capacity, contract) != -1) {
        printf("library_corpus_replay status=fail phase=trailer reason=extra-record\n");
        contract_failures++;
    }

    unsigned int total_failures = failed_events + contract_failures;
    printf("library_corpus_replay status=%s events=%u failed_events=%u contract_failures=%u total_failures=%u\n",
           total_failures ? "fail" : "pass", expected_events, failed_events, contract_failures, total_failures);
    free(line);
    dlclose(driver);
    fclose(contract);
    close(root);
    return total_failures ? -1 : 0;

fail:
    free(line);
    dlclose(driver);
    fclose(contract);
    close(root);
    return -1;
}

int tiny_cuda_probe_run(void) {
    const char *expected = getenv("CUBLAS_CREATE_EXPECTED_REGISTRATIONS");
    const char *ggml_path = configured_path("CUBLAS_CREATE_GGML_LIBRARY", "/opt/llama/bin/libggml-cuda.so.0");
    const char *cublas_path = configured_path("CUBLAS_CREATE_LIBRARY", "libcublas.so.12");
    const char *runtime_path = configured_path("CUBLAS_CREATE_RUNTIME_LIBRARY", "libcudart.so.12");
    const char *driver_path = configured_path("CUBLAS_CREATE_DRIVER_LIBRARY", "libcuda.so.1");
    const char *corpus_root = getenv("CUBLAS_CREATE_MODULE_CORPUS_ROOT");
    const char *corpus_expected_text = getenv("CUBLAS_CREATE_MODULE_CORPUS_EXPECTED");
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
    dlerror();
    cublas_gemm_ex_t gemm_ex = (cublas_gemm_ex_t)dlsym(cublas, "cublasGemmEx");
    const char *gemm_ex_error = dlerror();
    if (!create || !destroy || !sgemm || !gemm_ex) {
        printf("cublas_create_probe_symbols status=fail create=%p destroy=%p sgemm=%p gemm_ex=%p "
               "create_error=%s destroy_error=%s sgemm_error=%s gemm_ex_error=%s\n",
               (void *)create, (void *)destroy, (void *)sgemm, (void *)gemm_ex, create_error ? create_error : "none",
               destroy_error ? destroy_error : "none", sgemm_error ? sgemm_error : "none",
               gemm_ex_error ? gemm_ex_error : "none");
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 33;
    }
    printf("cublas_create_probe_symbols status=pass create=%p destroy=%p sgemm=%p gemm_ex=%p\n", (void *)create,
           (void *)destroy, (void *)sgemm, (void *)gemm_ex);

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

    if (run_gemm_ex_oracle(cublas_handle, gemm_ex, cuda_malloc, cuda_memcpy, cuda_free, cuda_device_synchronize) != 0) {
        destroy(cublas_handle);
        dlclose(runtime);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 42;
    }

    if ((corpus_root && *corpus_root) || (corpus_expected_text && *corpus_expected_text)) {
        unsigned int corpus_expected = 0;
        if (!corpus_root || !*corpus_root || !parse_unsigned(corpus_expected_text, &corpus_expected) ||
            !corpus_expected || run_library_corpus_replay(corpus_root, corpus_expected, driver_path) != 0) {
            destroy(cublas_handle);
            dlclose(runtime);
            dlclose(cublas);
            dlclose(ggml);
            printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
            return 43;
        }
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

#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*cublas_create_t)(void **handle);
typedef int (*cublas_destroy_t)(void *handle);
typedef int (*cu_module_get_loading_mode_t)(int *mode);

static const char *configured_path(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && *value ? value : fallback;
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
    cu_module_get_loading_mode_t get_mode =
        (cu_module_get_loading_mode_t)dlsym(driver, "cuModuleGetLoadingMode");
    error = dlerror();
    if (!get_mode) {
        printf("cublas_module_loading_mode phase=%s status=unresolved error=%s\n", phase,
               error ? error : "unknown");
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

int tiny_cuda_probe_run(void) {
    const char *expected = getenv("CUBLAS_CREATE_EXPECTED_REGISTRATIONS");
    const char *ggml_path = configured_path("CUBLAS_CREATE_GGML_LIBRARY", "/opt/llama/bin/libggml-cuda.so.0");
    const char *cublas_path = configured_path("CUBLAS_CREATE_LIBRARY", "libcublas.so.12");
    const char *driver_path = configured_path("CUBLAS_CREATE_DRIVER_LIBRARY", "libcuda.so.1");
    void *ggml = NULL;
    void *cublas = NULL;
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

    if (!observe_module_loading_mode("before-create", driver_path, &loading_mode_result)) {
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 36;
    }

    int create_result = create(&cublas_handle);
    printf("cublas_create_result=%d handle=%p\n", create_result, cublas_handle);
    if (create_result != 0 || !cublas_handle) {
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 34;
    }

    if (!observe_module_loading_mode("after-create", driver_path, &loading_mode_result) || loading_mode_result != 0) {
        destroy(cublas_handle);
        dlclose(cublas);
        dlclose(ggml);
        printf("=== CUBLAS_CREATE_PROBE_FAIL ===\n");
        return 37;
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

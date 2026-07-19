#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <link.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * This trigger deliberately knows only public CUDA Runtime ABI.  It loads an
 * exact libggml-cuda DSO so CUDA registers its real host stubs, validates one
 * explicitly supplied local-symbol offset against that DSO's executable ELF
 * segment, then calls cudaFuncSetAttribute on that registered stub.  It never
 * reads, writes, or invokes a CUDA private export-table entry.
 */

enum {
    CUDA_SUCCESS = 0,
    CUDA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE = 8,
};

typedef int cuda_error_t;
typedef cuda_error_t (*cuda_set_device_t)(int device);
typedef cuda_error_t (*cuda_func_set_attribute_t)(const void *function, int attribute, int value);

struct trigger_options {
    const char *library;
    const char *anchor_symbol;
    uintptr_t kernel_offset;
    int shared_memory_bytes;
};

struct executable_segment {
    uintptr_t base;
    uintptr_t address;
    bool found_object;
    bool executable;
};

static void usage(FILE *stream) {
    fprintf(stream,
            "用法：ggml-cuda-attribute-trigger --library PATH --anchor-symbol SYMBOL \\\n"
            "  --kernel-offset ELF_VADDR --shared-memory-bytes BYTES\n"
            "\n"
            "输入必须来自同一份 exact DSO 的静态与运行时证据：\n"
            "  --library              exact libggml-cuda 路径；调用前以 artifact manifest/sha256 校验。\n"
            "  --anchor-symbol        static collector 的 dynamic raw symbol；用于 dladdr 得到 DSO load base。\n"
            "  --kernel-offset        同一 DSO local registered host-stub 的 ELF virtual address，非 file offset。\n"
            "  --shared-memory-bytes  同一 Runtime/core 观察到的动态 shared-memory 字节数，不从反汇编猜测。\n"
            "\n"
            "程序 dlopen exact DSO 以完成真实 fatbin 注册，验证 base + ELF virtual address 落在该 DSO\n"
            "的 executable PT_LOAD，然后只调用公开 cudaSetDevice(0) 与 cudaFuncSetAttribute(stub, 8, BYTES)。\n"
            "成功只证明这次 public Runtime trigger 到达；它不证明 private ABI、guest shim、Type-2 或 Kimi correctness，\n"
            "也不会读取、写入或调用 CUDA private export-table entry。\n");
}

static int parse_uintptr(const char *text, uintptr_t *value) {
    char *end = NULL;
    unsigned long long parsed;

    if (!text || !*text) {
        return -1;
    }
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINTPTR_MAX) {
        return -1;
    }
    *value = (uintptr_t)parsed;
    return 0;
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    long parsed;

    if (!text || !*text) {
        return -1;
    }
    errno = 0;
    parsed = strtol(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_options(int argc, char **argv, struct trigger_options *options) {
    memset(options, 0, sizeof(*options));
    for (int index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
            usage(stdout);
            return 1;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "ggml_cuda_attribute_trigger_error=missing value for %s\n", argument);
            return -1;
        }
        const char *value = argv[++index];
        if (strcmp(argument, "--library") == 0) {
            options->library = value;
        } else if (strcmp(argument, "--anchor-symbol") == 0) {
            options->anchor_symbol = value;
        } else if (strcmp(argument, "--kernel-offset") == 0) {
            if (parse_uintptr(value, &options->kernel_offset) != 0 || options->kernel_offset == 0) {
                fprintf(stderr, "ggml_cuda_attribute_trigger_error=invalid --kernel-offset: %s\n", value);
                return -1;
            }
        } else if (strcmp(argument, "--shared-memory-bytes") == 0) {
            if (parse_positive_int(value, &options->shared_memory_bytes) != 0) {
                fprintf(stderr, "ggml_cuda_attribute_trigger_error=invalid --shared-memory-bytes: %s\n", value);
                return -1;
            }
        } else {
            fprintf(stderr, "ggml_cuda_attribute_trigger_error=unknown argument: %s\n", argument);
            return -1;
        }
    }
    if (!options->library || !options->anchor_symbol || options->kernel_offset == 0 || options->shared_memory_bytes == 0) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=all four required options are required\n");
        return -1;
    }
    return 0;
}

static int checked_add(uintptr_t left, uintptr_t right, uintptr_t *sum) {
    if (UINTPTR_MAX - left < right) {
        return -1;
    }
    *sum = left + right;
    return 0;
}

static int inspect_loaded_object(struct dl_phdr_info *info, size_t size, void *opaque) {
    (void)size;
    struct executable_segment *result = opaque;
    if ((uintptr_t)info->dlpi_addr != result->base) {
        return 0;
    }

    result->found_object = true;
    for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
        const ElfW(Phdr) *header = &info->dlpi_phdr[index];
        uintptr_t start;
        uintptr_t end;
        if (header->p_type != PT_LOAD || (header->p_flags & PF_X) == 0 ||
            checked_add(result->base, header->p_vaddr, &start) != 0 ||
            checked_add(start, header->p_memsz, &end) != 0) {
            continue;
        }
        if (result->address >= start && result->address < end) {
            result->executable = true;
            return 1;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    struct trigger_options options;
    const int parse_status = parse_options(argc, argv, &options);
    if (parse_status > 0) {
        return 0;
    }
    if (parse_status < 0) {
        usage(stderr);
        return 2;
    }

    printf("=== GGML_CUDA_ATTRIBUTE_TRIGGER_BEGIN ===\n");
    printf("ggml_cuda_attribute_library=%s\n", options.library);
    printf("ggml_cuda_attribute_anchor_symbol=%s\n", options.anchor_symbol);
    printf("ggml_cuda_attribute_kernel_offset=0x%" PRIxPTR "\n", options.kernel_offset);
    printf("ggml_cuda_attribute_public_attribute=%d\n", CUDA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE);
    printf("ggml_cuda_attribute_shared_memory_bytes=%d\n", options.shared_memory_bytes);
    fflush(stdout);

    void *handle = dlopen(options.library, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=dlopen_failed:%s\n", dlerror());
        return 20;
    }

    dlerror();
    void *anchor = dlsym(handle, options.anchor_symbol);
    const char *symbol_error = dlerror();
    if (symbol_error || !anchor) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=anchor_lookup_failed:%s\n",
                symbol_error ? symbol_error : "null anchor");
        dlclose(handle);
        return 21;
    }

    Dl_info anchor_info;
    if (dladdr(anchor, &anchor_info) == 0 || !anchor_info.dli_fbase || !anchor_info.dli_fname) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=anchor_dladdr_failed\n");
        dlclose(handle);
        return 22;
    }

    uintptr_t kernel_address;
    if (checked_add((uintptr_t)anchor_info.dli_fbase, options.kernel_offset, &kernel_address) != 0) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=kernel_address_overflow\n");
        dlclose(handle);
        return 23;
    }
    struct executable_segment segment = {
        .base = (uintptr_t)anchor_info.dli_fbase,
        .address = kernel_address,
        .found_object = false,
        .executable = false,
    };
    dl_iterate_phdr(inspect_loaded_object, &segment);
    if (!segment.found_object || !segment.executable) {
        fprintf(stderr,
                "ggml_cuda_attribute_trigger_error=kernel_offset_not_in_anchor_dso_executable_segment "
                "found_object=%d executable=%d\n",
                segment.found_object, segment.executable);
        dlclose(handle);
        return 24;
    }

    dlerror();
    cuda_set_device_t set_device = (cuda_set_device_t)dlsym(RTLD_DEFAULT, "cudaSetDevice");
    const char *set_device_error = dlerror();
    dlerror();
    cuda_func_set_attribute_t set_attribute =
        (cuda_func_set_attribute_t)dlsym(RTLD_DEFAULT, "cudaFuncSetAttribute");
    const char *set_attribute_error = dlerror();
    if (!set_device || !set_attribute) {
        fprintf(stderr, "ggml_cuda_attribute_trigger_error=public_runtime_symbol_missing set_device=%s set_attribute=%s\n",
                set_device_error ? set_device_error : "available",
                set_attribute_error ? set_attribute_error : "available");
        dlclose(handle);
        return 25;
    }

    printf("ggml_cuda_attribute_anchor_address=%p\n", anchor);
    printf("ggml_cuda_attribute_library_base=%p\n", anchor_info.dli_fbase);
    printf("ggml_cuda_attribute_library_loaded_path=%s\n", anchor_info.dli_fname);
    printf("ggml_cuda_attribute_kernel_address=0x%" PRIxPTR "\n", kernel_address);
    fflush(stdout);

    const cuda_error_t set_device_result = set_device(0);
    printf("ggml_cuda_attribute_set_device_result=%d\n", set_device_result);
    if (set_device_result != CUDA_SUCCESS) {
        printf("=== GGML_CUDA_ATTRIBUTE_TRIGGER_FAIL ===\n");
        dlclose(handle);
        return 26;
    }

    const cuda_error_t set_attribute_result =
        set_attribute((const void *)kernel_address, CUDA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                      options.shared_memory_bytes);
    printf("ggml_cuda_attribute_set_attribute_result=%d\n", set_attribute_result);
    if (set_attribute_result != CUDA_SUCCESS) {
        printf("=== GGML_CUDA_ATTRIBUTE_TRIGGER_FAIL ===\n");
        dlclose(handle);
        return 27;
    }

    printf("=== GGML_CUDA_ATTRIBUTE_TRIGGER_PASS ===\n");
    dlclose(handle);
    return 0;
}

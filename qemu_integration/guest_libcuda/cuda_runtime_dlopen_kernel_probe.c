#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

/*
 * 保持这个 launcher 不依赖 CUDA Runtime。这样即使目标 DSO 的 CUDA Runtime
 * 初始化失败，入口标记也已经写入 guest serial log。
 *
 * 实际 CUDA 调用、fatbin 注册和 kernel launch 都属于 libtiny_cuda.so；它在
 * launcher 的 main() 之后才被 dlopen。
 */

typedef int (*tiny_cuda_probe_run_t)(void);

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/mnt/payload/lib/libggml-cuda.so.0";

    printf("=== CUDA_RUNTIME_DLOPEN_KERNEL_PROBE_BEGIN ===\n");
    printf("dlopen_target=%s\n", path);
    fflush(stdout);

    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        printf("dlopen_failed=%s\n", dlerror());
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 20;
    }

    dlerror();
    tiny_cuda_probe_run_t run = (tiny_cuda_probe_run_t)dlsym(handle, "tiny_cuda_probe_run");
    const char *symbol_error = dlerror();
    if (symbol_error || !run) {
        printf("dlsym_failed=%s\n", symbol_error ? symbol_error : "null symbol");
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 21;
    }
    printf("dlopen_ok=%p probe=%p\n", handle, (void *)run);
    int result = run();
    dlclose(handle);
    if (result == 0) {
        printf("=== CUDA_RUNTIME_DLOPEN_KERNEL_PROBE_PASS ===\n");
    }
    return result;
}

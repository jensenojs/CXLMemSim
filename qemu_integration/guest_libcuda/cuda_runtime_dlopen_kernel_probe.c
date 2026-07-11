#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>

/*
 * 动态加载一个带 CUDA fatbin 的 DSO，再调用它导出的 host kernel stub。
 *
 * API survey 只能证明 libcudart 初始化没有立刻失败。这个 probe 继续要求
 * library、module、function、launch、device memory 和 copy-back 都产生真实结果。
 */

typedef int (*tiny_cuda_launch_t)(int *out);

static int print_cuda_result(const char *stage, cudaError_t error) {
    printf("kernel_probe stage=%s error=%d name=%s string=%s\n", stage, (int)error,
           cudaGetErrorName(error), cudaGetErrorString(error));
    return error == cudaSuccess ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/mnt/payload/lib/libggml-cuda.so.0";
    int failed = 0;
    int host_out = 0;
    int *device_out = NULL;

    printf("=== CUDA_RUNTIME_DLOPEN_KERNEL_PROBE_BEGIN ===\n");
    printf("dlopen_target=%s\n", path);

    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        printf("dlopen_failed=%s\n", dlerror());
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 20;
    }

    dlerror();
    tiny_cuda_launch_t launch = (tiny_cuda_launch_t)dlsym(handle, "tiny_cuda_launch");
    const char *symbol_error = dlerror();
    if (symbol_error || !launch) {
        printf("dlsym_failed=%s\n", symbol_error ? symbol_error : "null symbol");
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 21;
    }
    printf("dlopen_ok=%p launch=%p\n", handle, (void *)launch);

    failed |= print_cuda_result("set_device", cudaSetDevice(0));
    failed |= print_cuda_result("malloc", cudaMalloc((void **)&device_out, sizeof(*device_out)));
    if (!device_out) {
        printf("kernel_probe device_out=(nil)\n");
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 22;
    }

    failed |= print_cuda_result("copy_initial_htod",
                                cudaMemcpy(device_out, &host_out, sizeof(host_out), cudaMemcpyHostToDevice));
    (void)cudaGetLastError();

    failed |= print_cuda_result("launch", (cudaError_t)launch(device_out));
    failed |= print_cuda_result("synchronize", cudaDeviceSynchronize());
    failed |= print_cuda_result("copy_result_dtoh",
                                cudaMemcpy(&host_out, device_out, sizeof(host_out), cudaMemcpyDeviceToHost));
    failed |= print_cuda_result("free", cudaFree(device_out));

    printf("kernel_probe result=%d expected=1234\n", host_out);
    if (host_out != 1234) {
        failed = 1;
    }

    if (failed) {
        printf("=== CUDA_RUNTIME_DLOPEN_KERNEL_PROBE_FAIL ===\n");
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 30;
    }

    printf("=== CUDA_RUNTIME_DLOPEN_KERNEL_PROBE_PASS ===\n");
    printf("=== GGML_DLOPEN_PROBE_PASS ===\n");
    return 0;
}

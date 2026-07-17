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
typedef int CUresult;
typedef int CUdevice;
typedef CUresult (*cu_device_get_attribute_t)(int *value, int attribute, CUdevice device);

enum {
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
};

static int print_cuda_result(const char *stage, cudaError_t error) {
    printf("kernel_probe stage=%s error=%d name=%s string=%s\n", stage, (int)error, cudaGetErrorName(error),
           cudaGetErrorString(error));
    return error == cudaSuccess ? 0 : 1;
}

static int print_driver_attribute(cu_device_get_attribute_t get_attribute, int attribute) {
    int value = 0;
    CUresult result = get_attribute(&value, attribute, 0);

    printf("cuda_driver_attribute attribute=%d result=%d value=%d\n", attribute, result, value);
    return result == 0 ? 0 : 1;
}

static int print_device_properties(void) {
    struct cudaDeviceProp properties;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int failed = 0;

    printf("=== CUDA_DEVICE_PROPERTIES_BEGIN ===\n");
    cudaError_t properties_result = cudaGetDeviceProperties(&properties, 0);
    failed |= print_cuda_result("device_properties", properties_result);
    if (properties_result == cudaSuccess) {
        printf("cuda_device_properties total_global_mem=%zu shared_mem_per_block=%zu "
               "shared_mem_per_block_optin=%zu warp_size=%d multi_processor_count=%d "
               "max_threads_per_block=%d compute_capability=%d.%d\n",
               properties.totalGlobalMem, properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin,
               properties.warpSize, properties.multiProcessorCount, properties.maxThreadsPerBlock, properties.major,
               properties.minor);
    }

    cudaError_t meminfo_result = cudaMemGetInfo(&free_bytes, &total_bytes);
    failed |= print_cuda_result("mem_get_info", meminfo_result);
    if (meminfo_result == cudaSuccess) {
        printf("cuda_mem_info free=%zu total=%zu\n", free_bytes, total_bytes);
    }

    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    cu_device_get_attribute_t get_attribute =
        driver ? (cu_device_get_attribute_t)dlsym(driver, "cuDeviceGetAttribute") : NULL;
    if (!get_attribute) {
        printf("cuda_driver_attribute unavailable=%s\n", driver ? "symbol" : "library");
        failed = 1;
    } else {
        failed |= print_driver_attribute(get_attribute, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        failed |= print_driver_attribute(get_attribute, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    }
    if (driver) {
        dlclose(driver);
    }
    printf("=== CUDA_DEVICE_PROPERTIES_END ===\n");
    return failed;
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
    failed |= print_device_properties();
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

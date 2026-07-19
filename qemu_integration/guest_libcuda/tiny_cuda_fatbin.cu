#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>

typedef int CUresult;
typedef int CUdevice;
typedef CUresult (*cu_device_get_attribute_t)(int *value, int attribute, CUdevice device);

enum {
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
};

extern "C" __global__ void tiny_cuda_kernel(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 1234;
    }
}

/* 普通 C 导出负责建立 launch configuration，供 dlopen probe 调用。 */
extern "C" int tiny_cuda_launch(int *out) {
    tiny_cuda_kernel<<<1, 1>>>(out);
    return (int)cudaGetLastError();
}

static int print_cuda_result(const char *stage, cudaError_t error) {
    printf("kernel_probe stage=%s error=%d name=%s string=%s\n", stage, (int)error, cudaGetErrorName(error),
           cudaGetErrorString(error));
    return error == cudaSuccess ? 0 : 1;
}

/* CUDA Runtime, rather than this trigger, selects the private table and selector. */
static int set_tiny_kernel_attributes(void) {
    int failed = 0;
    failed |= print_cuda_result("set_attribute_dynamic_shared_memory",
                                cudaFuncSetAttribute(tiny_cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    failed |=
        print_cuda_result("set_attribute_shared_memory_carveout",
                          cudaFuncSetAttribute(tiny_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
    return failed;
}

static int report_tiny_kernel_occupancy(void) {
    int num_blocks = 0;
    cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, tiny_cuda_kernel, 1, 0);
    printf("kernel_probe occupancy error=%d name=%s string=%s num_blocks=%d\n", (int)error,
           cudaGetErrorName(error), cudaGetErrorString(error), num_blocks);
    return error == cudaSuccess && num_blocks > 0 ? 0 : 1;
}

static int print_driver_attribute(cu_device_get_attribute_t get_attribute, int attribute) {
    int value = 0;
    CUresult result = get_attribute(&value, attribute, 0);

    printf("cuda_driver_attribute attribute=%d result=%d value=%d\n", attribute, result, value);
    return result == 0 ? 0 : 1;
}

static int print_device_properties(void) {
    cudaDeviceProp properties;
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

extern "C" int tiny_cuda_probe_run(void) {
    int failed = 0;
    int host_out = 0;
    int *device_out = NULL;

    failed |= print_cuda_result("set_device", cudaSetDevice(0));
    failed |= print_device_properties();
    failed |= set_tiny_kernel_attributes();
    failed |= report_tiny_kernel_occupancy();
    failed |= print_cuda_result("malloc", cudaMalloc((void **)&device_out, sizeof(*device_out)));
    if (!device_out) {
        printf("kernel_probe device_out=(nil)\n");
        printf("=== GGML_DLOPEN_PROBE_FAIL ===\n");
        return 22;
    }

    failed |= print_cuda_result("copy_initial_htod",
                                cudaMemcpy(device_out, &host_out, sizeof(host_out), cudaMemcpyHostToDevice));
    (void)cudaGetLastError();

    failed |= print_cuda_result("launch", (cudaError_t)tiny_cuda_launch(device_out));
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

extern "C" int tiny_cuda_library_marker(void) { return 42; }

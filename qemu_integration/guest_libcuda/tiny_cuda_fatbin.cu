#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static int run_noncontiguous_2d_device_copy(void) {
    constexpr size_t source_pitch = 64;
    constexpr size_t destination_pitch = 80;
    constexpr size_t width = 17;
    constexpr size_t height = 3;
    constexpr size_t source_x = 7;
    constexpr size_t source_y = 1;
    constexpr size_t destination_x = 11;
    constexpr size_t destination_y = 2;
    constexpr unsigned char sentinel = 0xa5;
    constexpr size_t source_bytes = source_pitch * (source_y + height);
    constexpr size_t destination_bytes = destination_pitch * (destination_y + height);
    unsigned char source[source_bytes];
    unsigned char destination[destination_bytes];
    unsigned char *device_source = NULL;
    unsigned char *device_destination = NULL;
    int failed = 0;

    for (size_t index = 0; index < source_bytes; index++) {
        source[index] = (unsigned char)((index * 29U + 17U) & 0xffU);
    }
    memset(destination, sentinel, sizeof(destination));
    printf("kernel_probe copy_2d_contract width=%zu height=%zu src_pitch=%zu dst_pitch=%zu "
           "src_x=%zu src_y=%zu dst_x=%zu dst_y=%zu\n",
           width, height, source_pitch, destination_pitch, source_x, source_y, destination_x, destination_y);

    failed |= print_cuda_result("copy_2d_source_malloc", cudaMalloc((void **)&device_source, source_bytes));
    failed |= print_cuda_result("copy_2d_destination_malloc", cudaMalloc((void **)&device_destination, destination_bytes));
    if (!device_source || !device_destination) {
        failed = 1;
        goto cleanup;
    }
    failed |= print_cuda_result("copy_2d_source_htod",
                                cudaMemcpy(device_source, source, source_bytes, cudaMemcpyHostToDevice));
    failed |= print_cuda_result("copy_2d_destination_sentinel_htod",
                                cudaMemcpy(device_destination, destination, destination_bytes, cudaMemcpyHostToDevice));
    failed |= print_cuda_result("copy_2d_d2d_async",
                                cudaMemcpy2DAsync(device_destination + destination_y * destination_pitch + destination_x,
                                                  destination_pitch,
                                                  device_source + source_y * source_pitch + source_x,
                                                  source_pitch, width, height, cudaMemcpyDeviceToDevice, 0));
    failed |= print_cuda_result("copy_2d_synchronize", cudaDeviceSynchronize());
    failed |= print_cuda_result("copy_2d_destination_dtoh",
                                cudaMemcpy(destination, device_destination, destination_bytes, cudaMemcpyDeviceToHost));

    if (!failed) {
        size_t mismatch_count = 0;
        size_t sentinel_bytes = 0;
        for (size_t index = 0; index < destination_bytes; index++) {
            unsigned char expected = sentinel;
            for (size_t row = 0; row < height; row++) {
                size_t target = (destination_y + row) * destination_pitch + destination_x;
                if (index >= target && index < target + width) {
                    expected = source[(source_y + row) * source_pitch + source_x + index - target];
                    break;
                }
            }
            if (expected == sentinel) {
                sentinel_bytes++;
            }
            if (destination[index] != expected) {
                mismatch_count++;
            }
        }
        if (mismatch_count == 0) {
            printf("kernel_probe copy_2d_result=pass copied_bytes=%zu sentinel_bytes=%zu\n", width * height,
                   sentinel_bytes);
        } else {
            printf("kernel_probe copy_2d_result=fail mismatches=%zu\n", mismatch_count);
            failed = 1;
        }
    }

cleanup:
    if (device_destination) {
        failed |= print_cuda_result("copy_2d_destination_free", cudaFree(device_destination));
    }
    if (device_source) {
        failed |= print_cuda_result("copy_2d_source_free", cudaFree(device_source));
    }
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
    failed |= run_noncontiguous_2d_device_copy();

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

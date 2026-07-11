#include <cuda_runtime_api.h>

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

extern "C" int tiny_cuda_library_marker(void) {
    return 42;
}

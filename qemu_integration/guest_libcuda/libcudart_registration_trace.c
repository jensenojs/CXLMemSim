#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

/*
 * 只读跟踪 libcudart 的 fatbin/module 注册边界。
 * 真实函数始终通过 RTLD_NEXT 调用；tracer 只打印 runtime validation 状态，
 * 用来确认哪个注册步骤首次引入错误。
 */

enum {
    CUDA129_RUNTIME_GETTER_OFFSET = 0x29c50,
};

typedef void *(*runtime_getter_t)(void);

static uint32_t read_u32(const void *base, size_t offset) {
    return *(const uint32_t *)((const unsigned char *)base + offset);
}

static void trace_runtime(const char *stage) {
    void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    void *set_device = cudart ? dlsym(cudart, "cudaSetDevice") : NULL;
    Dl_info info;
    if (!set_device || !dladdr(set_device, &info) || !info.dli_fbase) {
        fprintf(stderr, "[CUDART-REG-TRACE] stage=%s runtime=unavailable\n", stage);
        return;
    }

    runtime_getter_t getter = (runtime_getter_t)((uintptr_t)info.dli_fbase + CUDA129_RUNTIME_GETTER_OFFSET);
    void *runtime = getter();
    if (!runtime) {
        fprintf(stderr, "[CUDART-REG-TRACE] stage=%s runtime=(nil)\n", stage);
        return;
    }

    fprintf(stderr, "[CUDART-REG-TRACE] stage=%s runtime=%p validation_state=%u validation_error=%u\n", stage,
            runtime, read_u32(runtime, 0x70), read_u32(runtime, 0x74));
}

void **__cudaRegisterFatBinary(void *fat_cubin) {
    typedef void **(*real_fn_t)(void *);
    static real_fn_t real_fn;
    if (!real_fn) {
        real_fn = (real_fn_t)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    }
    trace_runtime("before __cudaRegisterFatBinary");
    void **handle = real_fn(fat_cubin);
    fprintf(stderr, "[CUDART-REG-TRACE] __cudaRegisterFatBinary fat_cubin=%p handle=%p\n", fat_cubin, handle);
    trace_runtime("after __cudaRegisterFatBinary");
    return handle;
}

void __cudaRegisterFunction(void **fat_cubin_handle, const char *host_fun, char *device_fun,
                            const char *device_name, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *block_dim, dim3 *grid_dim, int *warp_size) {
    typedef void (*real_fn_t)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *,
                              int *);
    static real_fn_t real_fn;
    if (!real_fn) {
        real_fn = (real_fn_t)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    fprintf(stderr,
            "[CUDART-REG-TRACE] __cudaRegisterFunction handle=%p host_fun=%p device_fun=%p device_name=%s\n",
            fat_cubin_handle, host_fun, device_fun, device_name ? device_name : "(null)");
    trace_runtime("before __cudaRegisterFunction");
    real_fn(fat_cubin_handle, host_fun, device_fun, device_name, thread_limit, tid, bid, block_dim, grid_dim,
            warp_size);
    trace_runtime("after __cudaRegisterFunction");
}

char __cudaInitModule(void **fat_cubin_handle) {
    typedef char (*real_fn_t)(void **);
    static real_fn_t real_fn;
    if (!real_fn) {
        real_fn = (real_fn_t)dlsym(RTLD_NEXT, "__cudaInitModule");
    }
    trace_runtime("before __cudaInitModule");
    char result = real_fn(fat_cubin_handle);
    fprintf(stderr, "[CUDART-REG-TRACE] __cudaInitModule handle=%p result=%d\n", fat_cubin_handle, (int)result);
    trace_runtime("after __cudaInitModule");
    return result;
}

void __cudaRegisterFatBinaryEnd(void **fat_cubin_handle) {
    typedef void (*real_fn_t)(void **);
    static real_fn_t real_fn;
    if (!real_fn) {
        real_fn = (real_fn_t)dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
    }
    trace_runtime("before __cudaRegisterFatBinaryEnd");
    real_fn(fat_cubin_handle);
    trace_runtime("after __cudaRegisterFatBinaryEnd");
}

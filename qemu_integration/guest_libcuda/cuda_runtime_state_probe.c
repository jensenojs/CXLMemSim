#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * CUDA 12.9 专用诊断程序。
 *
 * libcudart 的四个失败 API 会先调用同一个内部函数取得 runtime 对象，
 * 再读取 runtime+0xa8 指向的状态表。这里仅观察该对象，不写入任何字段。
 * 偏移来自 payload 中固定的 libcudart.so.12；更换 libcudart 后必须重新取证。
 */

enum {
    CUDA129_RUNTIME_GETTER_OFFSET = 0x29c50,
    CUDA129_DRIVER_SLOT_CONTEXT_ENTER_OFFSET = 0x2b50c8,
    CUDA129_DRIVER_SLOT_FUNCTION_LOOKUP_OFFSET = 0x2b5058,
    CUDA129_DRIVER_SLOT_CONTEXT_LEAVE_OFFSET = 0x2b50d0,
    CUDA129_FUNCTION_RECORD_LIBRARY_RESOLVE_OFFSET = 0x265c0,
    CUDA129_FUNCTION_RECORD_PREPARE_OFFSET = 0x368e0,
    CUDA129_FUNCTION_RECORD_STATE_OFFSET = 0x3cfc0,
    CUDA129_FUNCTION_RECORD_ASSOCIATE_OFFSET = 0x3d170,
    CUDA129_LAZY_MODULE_VALIDATE_OFFSET = 0x41db0,
    CUDA129_LAZY_PRECHECK_OFFSET = 0x43420,
    CUDA129_LAZY_LOAD_OFFSET = 0x436f0,
    CUDA129_MODULE_FUNCTIONS_PREPARE_OFFSET = 0x44f10,
    CUDA129_MODULE_PREPARE_OFFSET = 0x45800,
};

typedef void *(*runtime_getter_t)(void);

static void dump_function_pointer_slot(const char *label, uintptr_t base, uintptr_t slot_offset) {
    void *target = *(void **)(base + slot_offset);
    Dl_info info;
    if (target && dladdr(target, &info)) {
        uintptr_t file_offset = info.dli_fbase ? (uintptr_t)target - (uintptr_t)info.dli_fbase : 0;
        printf("driver_slot label=%s slot_offset=0x%lx target=%p file=%s symbol=%s file_offset=0x%lx\n",
               label, (unsigned long)slot_offset, target, info.dli_fname ? info.dli_fname : "(unknown)",
               info.dli_sname ? info.dli_sname : "(unknown)", (unsigned long)file_offset);
        return;
    }
    printf("driver_slot label=%s slot_offset=0x%lx target=%p unresolved\n", label,
           (unsigned long)slot_offset, target);
}

static uint64_t read_u64(const void *base, size_t offset) {
    return *(const uint64_t *)((const unsigned char *)base + offset);
}

static uint32_t read_u32(const void *base, size_t offset) {
    return *(const uint32_t *)((const unsigned char *)base + offset);
}

static void dump_runtime_state(const char *stage) {
    Dl_info info;
    void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    void *set_device = cudart ? dlsym(cudart, "cudaSetDevice") : NULL;
    if (!set_device || !dladdr(set_device, &info) || !info.dli_fbase) {
        printf("runtime_state stage=%s unavailable=dladdr_failed\n", stage);
        return;
    }

    uintptr_t libcudart_base = (uintptr_t)info.dli_fbase;
    runtime_getter_t get_runtime = (runtime_getter_t)(libcudart_base + CUDA129_RUNTIME_GETTER_OFFSET);
    void *runtime = get_runtime();
    if (!runtime) {
        printf("runtime_state stage=%s libcudart_base=%p runtime=(nil)\n", stage, info.dli_fbase);
        return;
    }

    void *table98 = (void *)(uintptr_t)read_u64(runtime, 0x98);
    void *tablea0 = (void *)(uintptr_t)read_u64(runtime, 0xa0);
    void *capability = (void *)(uintptr_t)read_u64(runtime, 0xa8);
    uint32_t validation_state = read_u32(runtime, 0x70);
    uint32_t validation_error = read_u32(runtime, 0x74);
    printf("runtime_state stage=%s libcudart=%s base=%p runtime=%p validation_state=%u validation_error=%u table98=%p tablea0=%p capability=%p\n",
           stage, info.dli_fname ? info.dli_fname : "(unknown)", info.dli_fbase, runtime, validation_state,
           validation_error, table98, tablea0, capability);

    if (!capability) {
        return;
    }

    printf("runtime_capability stage=%s set_device_0x40=%u free_0x58=%u meminfo_0x78=%u sync_0x294=%u\n",
           stage, read_u32(capability, 0x40), read_u32(capability, 0x58), read_u32(capability, 0x78),
           read_u32(capability, 0x294));
}

static void probe_lazy_module_loader(void) {
    if (!getenv("CUDART_PROBE_DECOMPOSE_LAZY_LOAD")) {
        return;
    }

    void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    void *set_device = cudart ? dlsym(cudart, "cudaSetDevice") : NULL;
    Dl_info info;
    if (!set_device || !dladdr(set_device, &info) || !info.dli_fbase) {
        printf("lazy_module_probe unavailable=dladdr_failed\n");
        return;
    }

    uintptr_t base = (uintptr_t)info.dli_fbase;
    runtime_getter_t getter = (runtime_getter_t)(base + CUDA129_RUNTIME_GETTER_OFFSET);
    void *runtime = getter();
    void *registry = runtime ? (void *)(uintptr_t)read_u64(runtime, 0x88) : NULL;
    if (!registry) {
        printf("lazy_module_probe runtime=%p registry=(nil)\n", runtime);
        return;
    }

    typedef int (*precheck_fn_t)(void *registry);
    typedef int (*load_fn_t)(void *registry, void **module_out);
    typedef int (*module_validate_fn_t)(void *module);
    typedef int (*module_prepare_fn_t)(void *registry, void *module, void *module_records);
    typedef int (*module_functions_prepare_fn_t)(void *registry, void *module, void *function_records);
    typedef int (*function_record_prepare_fn_t)(void *record_payload, void *module);
    typedef int (*function_record_associate_fn_t)(void *module, void *record_payload);
    typedef int (*function_record_state_fn_t)(void *module, unsigned char *state, void *record_payload);
    typedef int (*function_record_library_resolve_fn_t)(void *record_payload, void **library_out);
    typedef int (*context_enter_fn_t)(void *context);
    typedef int (*function_lookup_fn_t)(void **module_out, void *library);
    typedef int (*context_leave_fn_t)(void **context_out);
    precheck_fn_t precheck = (precheck_fn_t)(base + CUDA129_LAZY_PRECHECK_OFFSET);
    load_fn_t load = (load_fn_t)(base + CUDA129_LAZY_LOAD_OFFSET);
    module_validate_fn_t module_validate =
        (module_validate_fn_t)(base + CUDA129_LAZY_MODULE_VALIDATE_OFFSET);
    module_prepare_fn_t module_prepare = (module_prepare_fn_t)(base + CUDA129_MODULE_PREPARE_OFFSET);
    module_functions_prepare_fn_t module_functions_prepare =
        (module_functions_prepare_fn_t)(base + CUDA129_MODULE_FUNCTIONS_PREPARE_OFFSET);
    function_record_prepare_fn_t function_record_prepare =
        (function_record_prepare_fn_t)(base + CUDA129_FUNCTION_RECORD_PREPARE_OFFSET);
    function_record_associate_fn_t function_record_associate =
        (function_record_associate_fn_t)(base + CUDA129_FUNCTION_RECORD_ASSOCIATE_OFFSET);
    function_record_state_fn_t function_record_state =
        (function_record_state_fn_t)(base + CUDA129_FUNCTION_RECORD_STATE_OFFSET);
    function_record_library_resolve_fn_t function_record_library_resolve =
        (function_record_library_resolve_fn_t)(base + CUDA129_FUNCTION_RECORD_LIBRARY_RESOLVE_OFFSET);

    int precheck_result = precheck(registry);
    void *module = NULL;
    int load_result = load(registry, &module);
    if (getenv("CUDART_PROBE_DECOMPOSE_FUNCTION_RECORDS")) {
        uint32_t bucket_count = module ? read_u32(module, 0x40) : 0;
        void **buckets = module ? (void **)(uintptr_t)read_u64(module, 0x50) : NULL;
        printf("function_record_probe module=%p bucket_count=%u buckets=%p\n", module, bucket_count,
               (void *)buckets);
        for (uint32_t bucket = 0; bucket < bucket_count; ++bucket) {
            void *node = buckets[bucket];
            uint32_t chain_index = 0;
            while (node) {
                void *next = (void *)(uintptr_t)read_u64(node, 0x0);
                void *record_payload = (void *)(uintptr_t)read_u64(node, 0x8);
                int result;
                  if (getenv("CUDART_PROBE_DECOMPOSE_FUNCTION_RECORD_CHILDREN")) {
                      unsigned char state = 0;
                      int associate_result = function_record_associate(module, record_payload);
                      dump_function_pointer_slot("context_enter", base, CUDA129_DRIVER_SLOT_CONTEXT_ENTER_OFFSET);
                      dump_function_pointer_slot("function_lookup", base,
                                                 CUDA129_DRIVER_SLOT_FUNCTION_LOOKUP_OFFSET);
                      dump_function_pointer_slot("context_leave", base, CUDA129_DRIVER_SLOT_CONTEXT_LEAVE_OFFSET);
                      if (getenv("CUDART_PROBE_DECOMPOSE_FUNCTION_RECORD_DIRECT_CALLS")) {
                          function_record_library_resolve_fn_t resolve = function_record_library_resolve;
                          context_enter_fn_t context_enter =
                              *(context_enter_fn_t *)(base + CUDA129_DRIVER_SLOT_CONTEXT_ENTER_OFFSET);
                          function_lookup_fn_t function_lookup =
                              *(function_lookup_fn_t *)(base + CUDA129_DRIVER_SLOT_FUNCTION_LOOKUP_OFFSET);
                          context_leave_fn_t context_leave =
                              *(context_leave_fn_t *)(base + CUDA129_DRIVER_SLOT_CONTEXT_LEAVE_OFFSET);
                          void *library = NULL;
                          void *driver_module = NULL;
                          void *popped_context = NULL;
                          void *context = module ? (void *)(uintptr_t)read_u64(module, 0x0) : NULL;
                          int resolve_result =
                              associate_result == 0 ? resolve(record_payload, &library) : -1;
                          int enter_result = resolve_result == 0 ? context_enter(context) : -1;
                          int lookup_result = enter_result == 0 ? function_lookup(&driver_module, library) : -1;
                          int leave_result = enter_result == 0 ? context_leave(&popped_context) : -1;
                          printf("function_record_direct_calls bucket=%u chain=%u payload=%p context=%p "
                                 "associate_result=%d resolve_result=%d library=%p enter_result=%d "
                                 "lookup_result=%d driver_module=%p leave_result=%d popped_context=%p\n",
                                 bucket, chain_index, record_payload, context, associate_result, resolve_result,
                                 library, enter_result, lookup_result, driver_module, leave_result, popped_context);
                          result = associate_result != 0   ? associate_result
                                   : resolve_result != 0   ? resolve_result
                                   : enter_result != 0     ? enter_result
                                   : lookup_result != 0    ? lookup_result
                                   : leave_result;
                          printf("function_record_probe bucket=%u chain=%u node=%p payload=%p result=%d\n",
                                 bucket, chain_index, node, record_payload, result);
                          if (result != 0) {
                              return;
                          }
                          node = next;
                          ++chain_index;
                          continue;
                      }
                      int state_result =
                          associate_result == 0 ? function_record_state(module, &state, record_payload) : -1;
                    printf("function_record_children bucket=%u chain=%u payload=%p associate_result=%d "
                           "state_result=%d state=%u\n",
                           bucket, chain_index, record_payload, associate_result, state_result, (unsigned)state);
                    result = associate_result != 0 ? associate_result : state_result;
                } else {
                    result = function_record_prepare(record_payload, module);
                }
                printf("function_record_probe bucket=%u chain=%u node=%p payload=%p result=%d\n", bucket,
                       chain_index, node, record_payload, result);
                if (result != 0) {
                    return;
                }
                node = next;
                ++chain_index;
            }
        }
        return;
    }
    if (getenv("CUDART_PROBE_DECOMPOSE_MODULE_FUNCTIONS")) {
        int module_functions_prepare_result =
            module ? module_functions_prepare(registry, module, (unsigned char *)module + 0x40) : -1;
        printf("lazy_module_probe runtime=%p registry=%p precheck_result=%d load_result=%d module=%p "
               "module_functions_prepare_result=%d\n",
               runtime, registry, precheck_result, load_result, module, module_functions_prepare_result);
        return;
    }
    if (getenv("CUDART_PROBE_DECOMPOSE_MODULE_VALIDATE")) {
        int module_prepare_result =
            module ? module_prepare(registry, module, (unsigned char *)module + 0x58) : -1;
        printf("lazy_module_probe runtime=%p registry=%p precheck_result=%d load_result=%d module=%p "
               "module_prepare_result=%d\n",
               runtime, registry, precheck_result, load_result, module, module_prepare_result);
        return;
    }
    int module_validate_result = module ? module_validate(module) : -1;
    printf("lazy_module_probe runtime=%p registry=%p precheck_result=%d load_result=%d module=%p "
           "module_validate_result=%d\n",
           runtime, registry, precheck_result, load_result, module, module_validate_result);
}

static void print_status(const char *label, cudaError_t err) {
    printf("%s err=%d name=%s string=%s\n", label, (int)err, cudaGetErrorName(err), cudaGetErrorString(err));
}

static void run_key_apis(const char *stage) {
    char label[96];
    size_t free_b = 0;
    size_t total_b = 0;

    snprintf(label, sizeof(label), "%s cudaSetDevice", stage);
    print_status(label, cudaSetDevice(0));
    dump_runtime_state(label);

    snprintf(label, sizeof(label), "%s cudaMemGetInfo", stage);
    cudaError_t meminfo = cudaMemGetInfo(&free_b, &total_b);
    print_status(label, meminfo);
    printf("%s values free=%zu total=%zu\n", label, free_b, total_b);

    snprintf(label, sizeof(label), "%s cudaFree0", stage);
    print_status(label, cudaFree(0));

    snprintf(label, sizeof(label), "%s cudaDeviceSynchronize", stage);
    print_status(label, cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/mnt/payload/lib/libggml-cuda.so.0";

    printf("=== CUDA_RUNTIME_STATE_PROBE_BEGIN ===\n");
    printf("dlopen_target=%s\n", path);

    print_status("startup cudaFree0", cudaFree(0));
    dump_runtime_state("before_dlopen");
    run_key_apis("before");

    void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        printf("dlopen_failed=%s\n", dlerror());
        printf("=== CUDA_RUNTIME_STATE_PROBE_FAIL ===\n");
        return 20;
    }

    printf("dlopen_ok=%p\n", handle);
    dump_runtime_state("immediately_after_dlopen");
    probe_lazy_module_loader();
    run_key_apis("after");
    dump_runtime_state("after_survey");

    printf("=== CUDA_RUNTIME_STATE_PROBE_DONE ===\n");
    printf("=== GGML_DLOPEN_PROBE_PASS ===\n");
    return 0;
}

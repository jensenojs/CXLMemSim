#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CXL_CUDA_PROVENANCE_FUNCTION_CAPACITY 4096
#define CXL_CUDA_PROVENANCE_CALLER_CAPACITY 4096
#define NO_INSTRUMENT __attribute__((no_instrument_function))

typedef struct {
    const void *address;
    const char *name;
    int is_public_cuda_api;
} FunctionRecord;

typedef struct {
    const void *function;
    const void *caller;
} CallerRecord;

static FunctionRecord g_functions[CXL_CUDA_PROVENANCE_FUNCTION_CAPACITY];
static CallerRecord g_callers[CXL_CUDA_PROVENANCE_CALLER_CAPACITY];
static size_t g_function_count;
static size_t g_caller_count;
static volatile int g_lock;
static int g_enabled = -1;
static int g_capacity_reported;

static void provenance_lock(void) NO_INSTRUMENT;
static void provenance_unlock(void) NO_INSTRUMENT;
static int provenance_enabled(void) NO_INSTRUMENT;
static FunctionRecord *function_record(const void *address) NO_INSTRUMENT;
static int first_function_caller(const void *function, const void *caller) NO_INSTRUMENT;
static void emit_natural_event(const char *symbol, const void *caller) NO_INSTRUMENT;

static void provenance_lock(void) {
    while (__sync_lock_test_and_set(&g_lock, 1)) {
    }
}

static void provenance_unlock(void) { __sync_lock_release(&g_lock); }

static int provenance_enabled(void) {
    if (g_enabled < 0) {
        g_enabled = getenv("CXL_CUDA_API_PROVENANCE") ? 1 : 0;
    }
    return g_enabled;
}

static FunctionRecord *function_record(const void *address) {
    for (size_t index = 0; index < g_function_count; index++) {
        if (g_functions[index].address == address) {
            return &g_functions[index];
        }
    }
    if (g_function_count == CXL_CUDA_PROVENANCE_FUNCTION_CAPACITY) {
        return NULL;
    }
    FunctionRecord *record = &g_functions[g_function_count++];
    record->address = address;
    Dl_info info = {0};
    record->name = dladdr(address, &info) && info.dli_sname ? info.dli_sname : NULL;
    record->is_public_cuda_api = record->name && record->name[0] == 'c' && record->name[1] == 'u' &&
                                 record->name[2] >= 'A' && record->name[2] <= 'Z';
    return record;
}

static int first_function_caller(const void *function, const void *caller) {
    for (size_t index = 0; index < g_caller_count; index++) {
        if (g_callers[index].function == function && g_callers[index].caller == caller) {
            return 0;
        }
    }
    if (g_caller_count == CXL_CUDA_PROVENANCE_CALLER_CAPACITY) {
        return -1;
    }
    g_callers[g_caller_count++] = (CallerRecord){.function = function, .caller = caller};
    return 1;
}

static void emit_natural_event(const char *symbol, const void *caller) {
    Dl_info info = {0};
    const int resolved = dladdr(caller, &info) != 0 && info.dli_fbase && info.dli_fname;
    if (resolved) {
        uintptr_t base = (uintptr_t)info.dli_fbase;
        fprintf(stderr,
                "[CXL-CUDA] api_provenance event=natural pid=%ld symbol=%s caller_status=resolved "
                "caller_file=%s caller_base=0x%llx caller_offset=0x%llx\n",
                (long)getpid(), symbol, info.dli_fname, (unsigned long long)base,
                (unsigned long long)((uintptr_t)caller - base));
    } else {
        fprintf(stderr,
                "[CXL-CUDA] api_provenance event=natural pid=%ld symbol=%s caller_status=unresolved "
                "caller_address=%p\n",
                (long)getpid(), symbol, caller);
    }
}

void __cyg_profile_func_enter(void *this_function, void *call_site) NO_INSTRUMENT;
void __cyg_profile_func_exit(void *this_function, void *call_site) NO_INSTRUMENT;

void __cyg_profile_func_enter(void *this_function, void *call_site) {
    if (!provenance_enabled()) {
        return;
    }
    provenance_lock();
    FunctionRecord *function = function_record(this_function);
    if (!function || !function->is_public_cuda_api) {
        if (!function && !g_capacity_reported) {
            g_capacity_reported = 1;
            fprintf(stderr, "[CXL-CUDA] api_provenance event=capacity-exhausted kind=function\n");
        }
        provenance_unlock();
        return;
    }
    const int first = first_function_caller(this_function, call_site);
    if (first < 0) {
        if (!g_capacity_reported) {
            g_capacity_reported = 1;
            fprintf(stderr, "[CXL-CUDA] api_provenance event=capacity-exhausted kind=caller\n");
        }
        provenance_unlock();
        return;
    }
    if (first > 0) {
        emit_natural_event(function->name, call_site);
    }
    provenance_unlock();
}

void __cyg_profile_func_exit(void *this_function, void *call_site) {
    (void)this_function;
    (void)call_site;
}

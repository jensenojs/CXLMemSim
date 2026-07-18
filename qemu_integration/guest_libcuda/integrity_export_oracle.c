#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/*
 * CUDA 12.9 calls the private INTEGRITY_CHECK export while it initializes its
 * Runtime-side Driver API table.  The callback ABI is observed from the real
 * NVIDIA export table; this probe deliberately does not invoke table slot 2,
 * whose signature has not yet been recovered.
 *
 * The host driver and Type-2 shim necessarily have different table and code
 * addresses.  Equality of their version 12092 output is therefore not an
 * oracle.  The useful comparison is narrower: fixed protocol revisions must
 * agree, and holding the supplied second constant while varying process or
 * thread identity reveals whether the real callback includes either identity
 * in its dynamic branch.
 */

typedef int CUresult;

typedef struct {
    unsigned char bytes[16];
} CUuuid;

typedef CUresult (*cu_get_export_table_t)(const void **table, const CUuuid *uuid);
typedef CUresult (*integrity_check_t)(uint32_t version, uint64_t unix_seconds, uint64_t result[2]);
typedef CUresult (*tools_tls_get_t)(void **out);

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_SUPPORTED = 801,
};

static const CUuuid integrity_check_uuid = {
    .bytes = {0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2},
};

static const CUuuid tools_tls_uuid = {
    .bytes = {0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc},
};

typedef struct {
    const char *label;
    void *handle;
    const void *table;
    uintptr_t size_word;
    const void *slot1;
    const void *slot2;
    integrity_check_t callback;
} ExportTable;

typedef struct {
    pid_t pid;
    uintptr_t thread;
    CUresult result;
    uint64_t words[2];
} CallbackResult;

typedef struct {
    integrity_check_t callback;
    uint32_t version;
    uint64_t seconds;
    CallbackResult result;
} ThreadCall;

typedef struct {
    const char *label;
    void *handle;
    const void *table;
    uintptr_t size_word;
    const void *slot1;
    const void *slot2;
    tools_tls_get_t get;
} ToolsTlsTable;

typedef struct {
    pid_t pid;
    uintptr_t thread;
    CUresult result;
    void *out;
} ToolsTlsResult;

typedef struct {
    tools_tls_get_t get;
    ToolsTlsResult result;
} ToolsTlsThreadCall;

static void usage(FILE *stream) {
    fprintf(stream, "usage: cuda-integrity-export-oracle --shim PATH [--driver PATH] [--seconds UNIX_SECONDS] "
                    "[--table integrity|tools-tls]\n"
                    "\n"
                    "Loads the real NVIDIA Driver and one Type-2 shim with RTLD_LOCAL. integrity\n"
                    "inspects INTEGRITY_CHECK and calls slot 1; tools-tls inspects TOOLS_TLS and\n"
                    "calls host slot 2 before and after cuInit.\n");
}

static int parse_seconds(const char *text, uint64_t *seconds) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *seconds = (uint64_t)value;
    return 0;
}

static void print_symbol_location(const char *label, const void *address) {
    Dl_info info;
    if (address && dladdr(address, &info) != 0) {
        uintptr_t base = (uintptr_t)info.dli_fbase;
        uintptr_t offset = (uintptr_t)address - base;
        printf("integrity_oracle_symbol label=%s address=%p file=%s base=%p offset=0x%" PRIxPTR " symbol=%s\n", label,
               address, info.dli_fname ? info.dli_fname : "<unknown>", info.dli_fbase, offset,
               info.dli_sname ? info.dli_sname : "<unknown>");
        return;
    }
    printf("integrity_oracle_symbol label=%s address=%p unresolved=1\n", label, address);
}

static int load_export_table(ExportTable *export_table, const char *label, const char *path) {
    cu_get_export_table_t get_export_table;

    memset(export_table, 0, sizeof(*export_table));
    export_table->label = label;
    export_table->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!export_table->handle) {
        fprintf(stderr, "integrity_oracle_error label=%s stage=dlopen path=%s error=%s\n", label, path, dlerror());
        return -1;
    }

    dlerror();
    get_export_table = (cu_get_export_table_t)dlsym(export_table->handle, "cuGetExportTable");
    const char *symbol_error = dlerror();
    if (symbol_error || !get_export_table) {
        fprintf(stderr, "integrity_oracle_error label=%s stage=dlsym symbol=cuGetExportTable error=%s\n", label,
                symbol_error ? symbol_error : "<null>");
        return -1;
    }

    CUresult result = get_export_table(&export_table->table, &integrity_check_uuid);
    if (result != CUDA_SUCCESS || !export_table->table) {
        fprintf(stderr, "integrity_oracle_error label=%s stage=get_export_table result=%d table=%p\n", label, result,
                export_table->table);
        return -1;
    }

    const void *const *slots = export_table->table;
    export_table->size_word = (uintptr_t)slots[0];
    export_table->slot1 = slots[1];
    export_table->slot2 = slots[2];
    export_table->callback = (integrity_check_t)export_table->slot1;
    if (!export_table->callback) {
        fprintf(stderr, "integrity_oracle_error label=%s stage=slot1-null\n", label);
        return -1;
    }

    printf("integrity_oracle_table label=%s path=%s table=%p size_word=%" PRIuPTR " slot1=%p slot2=%p\n", label, path,
           export_table->table, export_table->size_word, export_table->slot1, export_table->slot2);
    print_symbol_location("slot1", export_table->slot1);
    print_symbol_location("slot2", export_table->slot2);
    return 0;
}

static int load_tools_tls_table(ToolsTlsTable *table, const char *label, const char *path) {
    cu_get_export_table_t get_export_table;

    memset(table, 0, sizeof(*table));
    table->label = label;
    table->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!table->handle) {
        fprintf(stderr, "tools_tls_oracle_error label=%s stage=dlopen path=%s error=%s\n", label, path, dlerror());
        return -1;
    }

    dlerror();
    get_export_table = (cu_get_export_table_t)dlsym(table->handle, "cuGetExportTable");
    const char *symbol_error = dlerror();
    if (symbol_error || !get_export_table) {
        fprintf(stderr, "tools_tls_oracle_error label=%s stage=dlsym symbol=cuGetExportTable error=%s\n", label,
                symbol_error ? symbol_error : "<null>");
        return -1;
    }

    CUresult result = get_export_table(&table->table, &tools_tls_uuid);
    if (result != CUDA_SUCCESS || !table->table) {
        fprintf(stderr, "tools_tls_oracle_error label=%s stage=get_export_table result=%d table=%p\n", label, result,
                table->table);
        return -1;
    }

    const void *const *slots = table->table;
    table->size_word = (uintptr_t)slots[0];
    if (table->size_word < 3 * sizeof(void *) || table->size_word % sizeof(void *) != 0 ||
        table->size_word > 64 * sizeof(void *)) {
        fprintf(stderr, "tools_tls_oracle_error label=%s stage=table-size size_word=%" PRIuPTR "\n", label,
                table->size_word);
        return -1;
    }
    table->slot1 = slots[1];
    table->slot2 = slots[2];
    table->get = (tools_tls_get_t)table->slot2;

    printf("tools_tls_oracle_table label=%s path=%s table=%p size_word=%" PRIuPTR " slots=%" PRIuPTR "\n", label, path,
           table->table, table->size_word, table->size_word / sizeof(void *));
    for (uintptr_t index = 0; index < table->size_word / sizeof(void *); index++) {
        printf("tools_tls_oracle_slot label=%s index=%" PRIuPTR " address=%p present=%d\n", label, index, slots[index],
               slots[index] != NULL);
    }
    print_symbol_location("tools_tls_slot1", table->slot1);
    print_symbol_location("tools_tls_slot2", table->slot2);
    return 0;
}

static int initialize_driver_handle(void *handle, const char *label) {
    typedef CUresult (*cu_init_t)(unsigned int flags);

    dlerror();
    cu_init_t init = (cu_init_t)dlsym(handle, "cuInit");
    const char *symbol_error = dlerror();
    if (symbol_error || !init) {
        fprintf(stderr, "export_oracle_error label=%s stage=dlsym symbol=cuInit error=%s\n", label,
                symbol_error ? symbol_error : "<null>");
        return -1;
    }

    CUresult result = init(0);
    printf("export_oracle_driver_init label=%s pid=%ld result=%d\n", label, (long)getpid(), result);
    return result == CUDA_SUCCESS ? 0 : -1;
}

static int print_driver_identity_handle(void *handle, const char *label) {
    typedef CUresult (*cu_driver_get_version_t)(int *version);
    typedef CUresult (*cu_device_get_count_t)(int *count);
    typedef CUresult (*cu_device_get_uuid_t)(CUuuid *uuid, int device);
    typedef CUresult (*cu_device_get_attribute_t)(int *value, int attribute, int device);

    cu_driver_get_version_t driver_get_version = dlsym(handle, "cuDriverGetVersion");
    cu_device_get_count_t device_get_count = dlsym(handle, "cuDeviceGetCount");
    cu_device_get_uuid_t device_get_uuid = dlsym(handle, "cuDeviceGetUuid_v2");
    cu_device_get_attribute_t device_get_attribute = dlsym(handle, "cuDeviceGetAttribute");
    if (!driver_get_version || !device_get_count || !device_get_uuid || !device_get_attribute) {
        fprintf(stderr, "export_oracle_error label=%s stage=identity-symbol\n", label);
        return -1;
    }

    int version = 0;
    int count = 0;
    CUuuid uuid = {0};
    int domain = 0;
    int bus = 0;
    int device = 0;
    CUresult version_result = driver_get_version(&version);
    CUresult count_result = device_get_count(&count);
    CUresult uuid_result = device_get_uuid(&uuid, 0);
    CUresult domain_result = device_get_attribute(&domain, 50, 0);
    CUresult bus_result = device_get_attribute(&bus, 33, 0);
    CUresult device_result = device_get_attribute(&device, 34, 0);
    printf("integrity_oracle_identity label=%s driver_result=%d driver_version=%d count_result=%d count=%d "
           "uuid_result=%d uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x "
           "pci_domain_result=%d pci_domain=%d pci_bus_result=%d pci_bus=%d pci_device_result=%d pci_device=%d\n",
           label, version_result, version, count_result, count, uuid_result, uuid.bytes[0], uuid.bytes[1],
           uuid.bytes[2], uuid.bytes[3], uuid.bytes[4], uuid.bytes[5], uuid.bytes[6], uuid.bytes[7], uuid.bytes[8],
           uuid.bytes[9], uuid.bytes[10], uuid.bytes[11], uuid.bytes[12], uuid.bytes[13], uuid.bytes[14],
           uuid.bytes[15], domain_result, domain, bus_result, bus, device_result, device);
    return version_result == CUDA_SUCCESS && count_result == CUDA_SUCCESS && count > 0 && uuid_result == CUDA_SUCCESS &&
                   domain_result == CUDA_SUCCESS && bus_result == CUDA_SUCCESS && device_result == CUDA_SUCCESS
               ? 0
               : -1;
}

static CallbackResult call_callback(integrity_check_t callback, uint32_t version, uint64_t seconds) {
    CallbackResult callback_result = {
        .pid = getpid(),
        .thread = (uintptr_t)pthread_self(),
        .result = CUDA_ERROR_NOT_SUPPORTED,
        .words = {0, 0},
    };
    callback_result.result = callback(version, seconds, callback_result.words);
    return callback_result;
}

static void print_result(const char *label, const char *scope, uint32_t version, uint64_t seconds,
                         CallbackResult callback_result) {
    printf("integrity_oracle_result label=%s scope=%s pid=%ld tid=%" PRIuPTR " version=%u seconds=%" PRIu64
           " result=%d out0=%016" PRIx64 " out1=%016" PRIx64 "\n",
           label, scope, (long)callback_result.pid, callback_result.thread, version, seconds, callback_result.result,
           callback_result.words[0], callback_result.words[1]);
}

static int same_result(CallbackResult left, CallbackResult right) {
    return left.result == right.result && left.words[0] == right.words[0] && left.words[1] == right.words[1];
}

static void *thread_call(void *argument) {
    ThreadCall *call = argument;
    call->result = call_callback(call->callback, call->version, call->seconds);
    return NULL;
}

static ToolsTlsResult call_tools_tls(tools_tls_get_t get) {
    ToolsTlsResult result = {
        .pid = getpid(),
        .thread = (uintptr_t)pthread_self(),
        .result = CUDA_ERROR_NOT_SUPPORTED,
        .out = (void *)(uintptr_t)UINT64_C(0xfeedfacefeedface),
    };
    if (get) {
        result.result = get(&result.out);
    }
    return result;
}

static void print_tools_tls_result(const char *label, const char *scope, ToolsTlsResult result) {
    printf("tools_tls_oracle_result label=%s scope=%s pid=%ld tid=%" PRIuPTR " result=%d out=%p\n", label, scope,
           (long)result.pid, result.thread, result.result, result.out);
}

static void *tools_tls_thread_call(void *argument) {
    ToolsTlsThreadCall *call = argument;
    call->result = call_tools_tls(call->get);
    return NULL;
}

static int run_tools_tls_oracle(const char *driver_path, const char *shim_path) {
    ToolsTlsTable host;
    ToolsTlsTable shim;
    if (load_tools_tls_table(&host, "host", driver_path) != 0 || load_tools_tls_table(&shim, "shim", shim_path) != 0) {
        return 3;
    }

    ToolsTlsResult host_preinit = call_tools_tls(host.get);
    print_tools_tls_result("host", "preinit", host_preinit);
    if (initialize_driver_handle(host.handle, "host") != 0 || print_driver_identity_handle(host.handle, "host") != 0) {
        return 4;
    }

    ToolsTlsResult host_first = call_tools_tls(host.get);
    ToolsTlsResult host_second = call_tools_tls(host.get);
    print_tools_tls_result("host", "postinit-first", host_first);
    print_tools_tls_result("host", "postinit-second", host_second);

    ToolsTlsThreadCall thread_call_data = {
        .get = host.get,
        .result = {.result = CUDA_ERROR_NOT_SUPPORTED, .out = NULL},
    };
    pthread_t thread;
    if (pthread_create(&thread, NULL, tools_tls_thread_call, &thread_call_data) != 0 ||
        pthread_join(thread, NULL) != 0) {
        fprintf(stderr, "tools_tls_oracle_error label=host stage=thread\n");
        return 4;
    }
    print_tools_tls_result("host", "postinit-thread", thread_call_data.result);

    ToolsTlsResult shim_result = call_tools_tls(shim.get);
    print_tools_tls_result("shim", "slot2", shim_result);
    printf("tools_tls_oracle_relation label=shim slot2_present=%d\n", shim.get != NULL);
    printf("tools_tls_oracle=complete\n");
    dlclose(shim.handle);
    dlclose(host.handle);
    return 0;
}

int main(int argc, char **argv) {
    const char *shim_path = NULL;
    const char *driver_path = "libcuda.so.1";
    uint64_t seconds = (uint64_t)time(NULL);
    enum { TABLE_INTEGRITY, TABLE_TOOLS_TLS } table = TABLE_INTEGRITY;

    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--help") == 0) {
            usage(stdout);
            return 0;
        }
        if (strcmp(argv[index], "--shim") == 0 && index + 1 < argc) {
            shim_path = argv[++index];
            continue;
        }
        if (strcmp(argv[index], "--driver") == 0 && index + 1 < argc) {
            driver_path = argv[++index];
            continue;
        }
        if (strcmp(argv[index], "--seconds") == 0 && index + 1 < argc) {
            if (parse_seconds(argv[++index], &seconds) != 0) {
                fprintf(stderr, "integrity_oracle_error stage=parse-seconds value=%s\n", argv[index]);
                return 2;
            }
            continue;
        }
        if (strcmp(argv[index], "--table") == 0 && index + 1 < argc) {
            const char *value = argv[++index];
            if (strcmp(value, "integrity") == 0) {
                table = TABLE_INTEGRITY;
                continue;
            }
            if (strcmp(value, "tools-tls") == 0) {
                table = TABLE_TOOLS_TLS;
                continue;
            }
            fprintf(stderr, "integrity_oracle_error stage=parse-table value=%s\n", value);
            return 2;
        }
        usage(stderr);
        return 2;
    }
    if (!shim_path) {
        usage(stderr);
        return 2;
    }

    if (table == TABLE_TOOLS_TLS) {
        return run_tools_tls_oracle(driver_path, shim_path);
    }

    ExportTable host;
    ExportTable shim;
    if (load_export_table(&host, "host", driver_path) != 0 || load_export_table(&shim, "shim", shim_path) != 0) {
        return 3;
    }

    int fork_pipe[2];
    if (pipe(fork_pipe) != 0) {
        perror("integrity_oracle_error stage=pipe-preinit");
        return 4;
    }
    pid_t preinit_child = fork();
    if (preinit_child < 0) {
        perror("integrity_oracle_error stage=fork-preinit");
        close(fork_pipe[0]);
        close(fork_pipe[1]);
        return 4;
    }
    if (preinit_child == 0) {
        close(fork_pipe[0]);
        if (initialize_driver_handle(host.handle, host.label) != 0) {
            close(fork_pipe[1]);
            _exit(2);
        }
        CallbackResult child_result = call_callback(host.callback, 12092, seconds);
        ssize_t written = write(fork_pipe[1], &child_result, sizeof(child_result));
        close(fork_pipe[1]);
        _exit(written == (ssize_t)sizeof(child_result) ? 0 : 1);
    }
    close(fork_pipe[1]);

    if (initialize_driver_handle(host.handle, host.label) != 0) {
        close(fork_pipe[0]);
        waitpid(preinit_child, NULL, 0);
        return 4;
    }
    if (print_driver_identity_handle(host.handle, host.label) != 0) {
        close(fork_pipe[0]);
        waitpid(preinit_child, NULL, 0);
        return 4;
    }

    const uint32_t versions[] = {12090, 12091, 12092};
    CallbackResult host_12092 = {
        .pid = 0,
        .thread = 0,
        .result = CUDA_ERROR_NOT_SUPPORTED,
        .words = {0, 0},
    };
    for (size_t index = 0; index < sizeof(versions) / sizeof(versions[0]); index++) {
        uint32_t version = versions[index];
        CallbackResult host_result = call_callback(host.callback, version, seconds);
        CallbackResult shim_result = call_callback(shim.callback, version, seconds);
        print_result("host", "parent", version, seconds, host_result);
        print_result("shim", "parent", version, seconds, shim_result);
        if (version == 12092) {
            host_12092 = host_result;
        }
        if (version == 12090 || version == 12091) {
            printf("integrity_oracle_relation label=fixed version=%u equal=%d\n", version,
                   same_result(host_result, shim_result));
        }
    }

    ThreadCall thread_call_data = {
        .callback = host.callback,
        .version = 12092,
        .seconds = seconds,
        .result = {.result = CUDA_ERROR_NOT_SUPPORTED, .words = {0, 0}},
    };
    pthread_t thread;
    if (pthread_create(&thread, NULL, thread_call, &thread_call_data) != 0 || pthread_join(thread, NULL) != 0) {
        fprintf(stderr, "integrity_oracle_error label=host stage=thread\n");
        return 4;
    }
    print_result("host", "thread", 12092, seconds, thread_call_data.result);
    printf("integrity_oracle_relation label=host relation=thread_same_second equal=%d\n",
           same_result(host_12092, thread_call_data.result));

    CallbackResult preinit_child_result = {
        .pid = 0,
        .thread = 0,
        .result = CUDA_ERROR_NOT_SUPPORTED,
        .words = {0, 0},
    };
    ssize_t preinit_received = read(fork_pipe[0], &preinit_child_result, sizeof(preinit_child_result));
    close(fork_pipe[0]);
    int preinit_wait_status = 0;
    if (waitpid(preinit_child, &preinit_wait_status, 0) != preinit_child ||
        preinit_received != (ssize_t)sizeof(preinit_child_result) || !WIFEXITED(preinit_wait_status) ||
        WEXITSTATUS(preinit_wait_status) != 0) {
        fprintf(stderr, "integrity_oracle_error label=host stage=preinit-forked-callback received=%zd wait_status=%d\n",
                preinit_received, preinit_wait_status);
        return 5;
    }
    print_result("host", "fork-child-preinit", 12092, seconds, preinit_child_result);
    printf("integrity_oracle_relation label=host relation=fork_preinit_same_second equal=%d\n",
           same_result(host_12092, preinit_child_result));

    printf("integrity_oracle=pass\n");
    dlclose(shim.handle);
    dlclose(host.handle);
    return 0;
}

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

/* This is an audit-only observer.  It never changes a selected symbol value.
 * The fixed records avoid allocation while glibc holds loader state. */
#define CXL_LOADER_AUDIT_SCHEMA_VERSION 1
#define CXL_LOADER_AUDIT_MAX_OBJECTS 1024
#define CXL_LOADER_AUDIT_PATH_LIMIT 3072
#define CXL_LOADER_AUDIT_LINE_LIMIT 8192

typedef struct {
    uintptr_t cookie;
    uintptr_t load_base;
    Lmid_t namespace_id;
    char path[CXL_LOADER_AUDIT_PATH_LIMIT];
    const char *kind;
} LoaderObject;

static int g_log_fd = -1;
static pid_t g_pid;
static pid_t g_ppid;
static uint64_t g_sequence;
static LoaderObject g_objects[CXL_LOADER_AUDIT_MAX_OBJECTS];
static size_t g_object_count;

static LoaderObject *object_for_cookie(uintptr_t cookie);

static size_t json_string(char *destination, size_t capacity, const char *source) {
    size_t written = 0;
    if (capacity == 0) {
        return 0;
    }
    for (const unsigned char *cursor = (const unsigned char *)(source ? source : ""); *cursor; cursor++) {
        const unsigned char byte = *cursor;
        if (byte == '"' || byte == '\\') {
            if (written + 2 >= capacity) {
                break;
            }
            destination[written++] = '\\';
            destination[written++] = (char)byte;
        } else if (byte >= 0x20 && byte <= 0x7e) {
            if (written + 1 >= capacity) {
                break;
            }
            destination[written++] = (char)byte;
        } else {
            static const char hex[] = "0123456789abcdef";
            if (written + 6 >= capacity) {
                break;
            }
            destination[written++] = '\\';
            destination[written++] = 'u';
            destination[written++] = '0';
            destination[written++] = '0';
            destination[written++] = hex[byte >> 4];
            destination[written++] = hex[byte & 0x0f];
        }
    }
    destination[written] = '\0';
    return written;
}

static void write_line(const char *line, size_t length) {
    if (g_log_fd < 0 || length == 0 || length >= CXL_LOADER_AUDIT_LINE_LIMIT) {
        return;
    }
    size_t written = 0;
    while (written < length) {
        ssize_t result = write(g_log_fd, line + written, length - written);
        if (result <= 0) {
            return;
        }
        written += (size_t)result;
    }
}

static void emit_process_start(const char *executable) {
    char escaped[CXL_LOADER_AUDIT_PATH_LIMIT * 2 + 1];
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    json_string(escaped, sizeof(escaped), executable);
    int length = snprintf(line, sizeof(line),
                          "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"process-start\","
                          "\"sequence\":%llu,\"pid\":%ld,\"ppid\":%ld,\"executable\":\"%s\"}\n",
                          CXL_LOADER_AUDIT_SCHEMA_VERSION, (unsigned long long)++g_sequence, (long)g_pid,
                          (long)g_ppid, escaped);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static void emit_object_event(const char *event, const LoaderObject *object) {
    char escaped[CXL_LOADER_AUDIT_PATH_LIMIT * 2 + 1];
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    json_string(escaped, sizeof(escaped), object->path);
    int length = snprintf(
        line, sizeof(line),
        "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"%s\",\"sequence\":%llu,"
        "\"pid\":%ld,\"ppid\":%ld,\"object_kind\":\"%s\",\"path\":\"%s\","
        "\"load_base\":\"0x%llx\",\"namespace\":%ld,\"cookie\":\"0x%llx\"}\n",
        CXL_LOADER_AUDIT_SCHEMA_VERSION, event, (unsigned long long)++g_sequence, (long)g_pid, (long)g_ppid,
        object->kind, escaped, (unsigned long long)object->load_base, (long)object->namespace_id,
        (unsigned long long)object->cookie);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static void emit_activity_event(const char *activity) {
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    int length = snprintf(line, sizeof(line),
                          "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"activity-%s\","
                          "\"sequence\":%llu,\"pid\":%ld,\"ppid\":%ld}\n",
                          CXL_LOADER_AUDIT_SCHEMA_VERSION, activity, (unsigned long long)++g_sequence, (long)g_pid,
                          (long)g_ppid);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static void emit_symbol_bind(const char *symbol, uintptr_t reference_cookie, uintptr_t definition_cookie) {
    char escaped[512];
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    json_string(escaped, sizeof(escaped), symbol);
    int length = snprintf(
        line, sizeof(line),
        "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"symbol-bind\","
        "\"sequence\":%llu,\"pid\":%ld,\"ppid\":%ld,\"symbol\":\"%s\","
        "\"reference_cookie\":\"0x%llx\",\"definition_cookie\":\"0x%llx\"}\n",
        CXL_LOADER_AUDIT_SCHEMA_VERSION, (unsigned long long)++g_sequence, (long)g_pid, (long)g_ppid, escaped,
        (unsigned long long)reference_cookie, (unsigned long long)definition_cookie);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static int is_cublas_create_boundary(const char *symbol, uintptr_t reference_cookie) {
    LoaderObject *reference = object_for_cookie(reference_cookie);
    if (!reference || !symbol || !strstr(reference->path, "/libcublas.so.12")) {
        return 0;
    }
    return strcmp(symbol, "cublasLtCtxInit") == 0 || strcmp(symbol, "pthread_once") == 0;
}

static void emit_plt_event(const char *event, const char *symbol, uintptr_t reference_cookie, uintptr_t definition_cookie,
                           uint64_t result) {
    char escaped[512];
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    json_string(escaped, sizeof(escaped), symbol);
    int length = snprintf(
        line, sizeof(line),
        "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"%s\","
        "\"sequence\":%llu,\"pid\":%ld,\"ppid\":%ld,\"symbol\":\"%s\","
        "\"reference_cookie\":\"0x%llx\",\"definition_cookie\":\"0x%llx\",\"rax\":\"0x%llx\"}\n",
        CXL_LOADER_AUDIT_SCHEMA_VERSION, event, (unsigned long long)++g_sequence, (long)g_pid, (long)g_ppid, escaped,
        (unsigned long long)reference_cookie, (unsigned long long)definition_cookie, (unsigned long long)result);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static void emit_capacity_exhausted(const char *kind) {
    char line[CXL_LOADER_AUDIT_LINE_LIMIT];
    int length = snprintf(line, sizeof(line),
                          "{\"schema_version\":%d,\"kind\":\"loader-audit\",\"event\":\"capacity-exhausted\","
                          "\"sequence\":%llu,\"pid\":%ld,\"ppid\":%ld,\"capacity_kind\":\"%s\"}\n",
                          CXL_LOADER_AUDIT_SCHEMA_VERSION, (unsigned long long)++g_sequence, (long)g_pid,
                          (long)g_ppid, kind);
    if (length > 0) {
        write_line(line, (size_t)length);
    }
}

static LoaderObject *object_for_cookie(uintptr_t cookie) {
    if (cookie == 0 || cookie > g_object_count) {
        return NULL;
    }
    return &g_objects[cookie - 1];
}

static const char *object_kind_for(const char *path) {
    if (!path || !*path) {
        return "main-program";
    }
    if (path[0] == '[' || strcmp(path, "linux-vdso.so.1") == 0) {
        return "special";
    }
    return "file";
}

static int initialize_logger(void) {
    const char *directory = getenv("CXL_LOADER_AUDIT_DIR");
    char destination[PATH_MAX];
    char executable[PATH_MAX];
    if (!directory || !*directory) {
        return -1;
    }
    g_pid = getpid();
    g_ppid = getppid();
    int path_length = snprintf(destination, sizeof(destination), "%s/%ld.jsonl", directory, (long)g_pid);
    if (path_length <= 0 || (size_t)path_length >= sizeof(destination)) {
        return -1;
    }
    g_log_fd = open(destination, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (g_log_fd < 0) {
        return -1;
    }
    ssize_t executable_length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (executable_length < 0) {
        close(g_log_fd);
        g_log_fd = -1;
        return -1;
    }
    executable[executable_length] = '\0';
    emit_process_start(executable);
    return 0;
}

unsigned int la_version(unsigned int version) {
    if (version < LAV_CURRENT || initialize_logger() != 0) {
        return 0;
    }
    return LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t namespace_id, uintptr_t *cookie) {
    if (!map || !cookie) {
        return 0;
    }
    if (g_object_count == CXL_LOADER_AUDIT_MAX_OBJECTS) {
        emit_capacity_exhausted("object");
        return 0;
    }
    LoaderObject *object = &g_objects[g_object_count];
    object->cookie = (uintptr_t)(g_object_count + 1);
    object->load_base = (uintptr_t)map->l_addr;
    object->namespace_id = namespace_id;
    object->kind = object_kind_for(map->l_name);
    const char *path = map->l_name && *map->l_name ? map->l_name : "";
    size_t copied = strnlen(path, sizeof(object->path) - 1);
    memcpy(object->path, path, copied);
    object->path[copied] = '\0';
    g_object_count++;
    *cookie = object->cookie;
    emit_object_event("open", object);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

void la_activity(uintptr_t *cookie, unsigned int flag) {
    (void)cookie;
    switch (flag) {
    case LA_ACT_CONSISTENT:
        emit_activity_event("consistent");
        break;
    case LA_ACT_ADD:
        emit_activity_event("add");
        break;
    case LA_ACT_DELETE:
        emit_activity_event("delete");
        break;
    default:
        emit_activity_event("unknown");
        break;
    }
}

unsigned int la_objclose(uintptr_t *cookie) {
    LoaderObject *object = cookie ? object_for_cookie(*cookie) : NULL;
    if (object) {
        emit_object_event("close", object);
    }
    return 0;
}

uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index, uintptr_t *reference_cookie, uintptr_t *definition_cookie,
                       unsigned int *flags, const char *symbol_name) {
    (void)index;
    /* Keep the loader's chosen address but request PLT callbacks for the
     * selected cuBLAS boundary after a lazy or eager symbol bind. */
    if (flags) {
        *flags &= ~(LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT);
    }
    if (symbol_name && symbol_name[0] == 'c' && symbol_name[1] == 'u') {
        emit_symbol_bind(symbol_name, reference_cookie ? *reference_cookie : 0,
                         definition_cookie ? *definition_cookie : 0);
    }
    return symbol->st_value;
}

Elf64_Addr la_x86_64_gnu_pltenter(Elf64_Sym *symbol, unsigned int index, uintptr_t *reference_cookie,
                                  uintptr_t *definition_cookie, La_x86_64_regs *registers, unsigned int *flags,
                                  const char *symbol_name, long int *framesize) {
    (void)symbol;
    (void)index;
    (void)flags;
    uintptr_t reference = reference_cookie ? *reference_cookie : 0;
    uintptr_t definition = definition_cookie ? *definition_cookie : 0;
    if (is_cublas_create_boundary(symbol_name, reference)) {
        /* glibc dispatches la_pltexit only after the observer requests a return frame. */
        if (framesize) {
            *framesize = (long int)sizeof(*registers);
        }
        emit_plt_event("plt-enter", symbol_name, reference, definition, 0);
    }
    return symbol->st_value;
}

unsigned int la_x86_64_gnu_pltexit(Elf64_Sym *symbol, unsigned int index, uintptr_t *reference_cookie,
                                   uintptr_t *definition_cookie, const La_x86_64_regs *registers,
                                   La_x86_64_retval *result, const char *symbol_name) {
    (void)symbol;
    (void)index;
    (void)registers;
    uintptr_t reference = reference_cookie ? *reference_cookie : 0;
    uintptr_t definition = definition_cookie ? *definition_cookie : 0;
    if (is_cublas_create_boundary(symbol_name, reference)) {
        emit_plt_event("plt-exit", symbol_name, reference, definition, result ? result->lrv_rax : 0);
    }
    return 0;
}

#define _GNU_SOURCE

#include "cxl_gpu_transport.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>

static void transport_log(const CxlGpuTransport *transport, const char *format, ...) {
    if (!transport->debug)
        return;

    va_list args;
    va_start(args, format);
    fprintf(stderr, "[CXL-GPU-TRANSPORT] ");
    vfprintf(stderr, format, args);
    va_end(args);
}

static uint64_t monotonic_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int read_hex_u16(const char *path, uint16_t *value) {
    char text[32];
    char *end = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;

    ssize_t count = read(fd, text, sizeof(text) - 1);
    int saved_errno = errno;
    close(fd);
    errno = saved_errno;
    if (count <= 0)
        return -1;
    text[count] = '\0';

    errno = 0;
    unsigned long parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || parsed > UINT16_MAX)
        return -1;
    while (isspace((unsigned char)*end))
        end++;
    if (*end != '\0')
        return -1;
    *value = (uint16_t)parsed;
    return 0;
}

static int device_path(char *path, size_t capacity, const char *device, const char *leaf) {
    static const char prefix[] = "/sys/bus/pci/devices/";
    size_t device_length = strnlen(device, sizeof(((CxlGpuTransport *)0)->pci_bdf));
    size_t leaf_length = strlen(leaf);
    size_t prefix_length = sizeof(prefix) - 1;
    if (device_length == sizeof(((CxlGpuTransport *)0)->pci_bdf) ||
        prefix_length + device_length + 1 + leaf_length + 1 > capacity) {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(path, prefix, prefix_length);
    memcpy(path + prefix_length, device, device_length);
    path[prefix_length + device_length] = '/';
    memcpy(path + prefix_length + device_length + 1, leaf, leaf_length + 1);
    return 0;
}

uint32_t cxl_gpu_transport_read32(const CxlGpuTransport *transport, uint32_t offset) {
    if (offset >= CXL_GPU_REG_PARAM0 && offset <= CXL_GPU_REG_PARAM7 &&
        (offset - CXL_GPU_REG_PARAM0) % 8 == 0)
        return (uint32_t)transport->descriptor->params[(offset - CXL_GPU_REG_PARAM0) / 8];
    return *(volatile uint32_t *)((volatile uint8_t *)transport->regs + offset);
}

uint64_t cxl_gpu_transport_read64(const CxlGpuTransport *transport, uint32_t offset) {
    if (offset >= CXL_GPU_REG_PARAM0 && offset <= CXL_GPU_REG_PARAM7 &&
        (offset - CXL_GPU_REG_PARAM0) % 8 == 0)
        return transport->descriptor->params[(offset - CXL_GPU_REG_PARAM0) / 8];
    if (offset == CXL_GPU_REG_CALL_ID)
        return transport->descriptor->request_call_id;
    if (offset >= CXL_GPU_REG_RESULT0 && offset <= CXL_GPU_REG_RESULT3 &&
        (offset - CXL_GPU_REG_RESULT0) % 8 == 0)
        return transport->descriptor->results[(offset - CXL_GPU_REG_RESULT0) / 8];
    return *(volatile uint64_t *)((volatile uint8_t *)transport->regs + offset);
}

void cxl_gpu_transport_write32(CxlGpuTransport *transport, uint32_t offset, uint32_t value) {
    if (transport->unusable)
        return;
    if (offset >= CXL_GPU_REG_PARAM0 && offset <= CXL_GPU_REG_PARAM7 &&
        (offset - CXL_GPU_REG_PARAM0) % 8 == 0) {
        transport->descriptor->params[(offset - CXL_GPU_REG_PARAM0) / 8] = value;
        return;
    }
    *(volatile uint32_t *)((volatile uint8_t *)transport->regs + offset) = value;
    __sync_synchronize();
}

void cxl_gpu_transport_write64(CxlGpuTransport *transport, uint32_t offset, uint64_t value) {
    if (transport->unusable)
        return;
    if (offset >= CXL_GPU_REG_PARAM0 && offset <= CXL_GPU_REG_PARAM7 &&
        (offset - CXL_GPU_REG_PARAM0) % 8 == 0) {
        transport->descriptor->params[(offset - CXL_GPU_REG_PARAM0) / 8] = value;
        return;
    }
    if (offset == CXL_GPU_REG_CALL_ID) {
        transport->descriptor->request_call_id = value;
        return;
    }
    if (offset >= CXL_GPU_REG_RESULT0 && offset <= CXL_GPU_REG_RESULT3 &&
        (offset - CXL_GPU_REG_RESULT0) % 8 == 0) {
        transport->descriptor->results[(offset - CXL_GPU_REG_RESULT0) / 8] = value;
        return;
    }
    *(volatile uint64_t *)((volatile uint8_t *)transport->regs + offset) = value;
    __sync_synchronize();
}

void cxl_gpu_transport_data_write(CxlGpuTransport *transport, size_t offset, const void *src, size_t len) {
    if (offset > CXL_GPU_DATA_SIZE || len > CXL_GPU_DATA_SIZE - offset)
        return;

    /* Protocol v1.10 maps only the data window as QEMU RAM. Unlike command
     * registers, it has no per-access device side effect, so libc may use
     * wide stores. The full barrier below publishes every payload byte before
     * the caller writes params and the command doorbell through volatile MMIO. */
    memcpy((void *)(transport->data + offset), src, len);
#if 0
    /* Historical I/O-window implementation. It limited every store to the
     * former QEMU MemoryRegionOps maximum access width. The v1.10 data window
     * is RAM-backed; enabling this loop again would restore the TCG narrow-
     * store cost without preserving any additional device semantics.
     * knockout: restore only with an older I/O-backed data-window protocol,
     * and only together with a distinct protocol version and QEMU mapping. */
    const uint8_t *source = src;
    volatile uint8_t *destination = transport->data + offset;
    size_t index = 0;
    for (; index + sizeof(uint64_t) <= len; index += sizeof(uint64_t)) {
        uint64_t value;
        memcpy(&value, source + index, sizeof(value));
        *(volatile uint64_t *)(void *)(destination + index) = value;
    }
    for (; index < len; index++)
        destination[index] = source[index];
#endif
    /* QEMU must not observe the command doorbell before the RAM payload. */
    __sync_synchronize();
}

void cxl_gpu_transport_data_read(const CxlGpuTransport *transport, size_t offset, void *dst, size_t len) {
    if (offset > CXL_GPU_DATA_SIZE || len > CXL_GPU_DATA_SIZE - offset)
        return;

    /* Observe QEMU's completed command and payload writes before bulk read. */
    __sync_synchronize();
    /* The same RAM-window argument permits wide loads on the return path. */
    memcpy(dst, (const void *)(transport->data + offset), len);
#if 0
    /* Historical peer of the write loop above. Keep it only as evidence of
     * the old I/O-window access-width contract. */
    uint8_t *destination = dst;
    volatile uint8_t *source = transport->data + offset;
    size_t index = 0;
    for (; index + sizeof(uint64_t) <= len; index += sizeof(uint64_t)) {
        uint64_t value = *(volatile uint64_t *)(const void *)(source + index);
        memcpy(destination + index, &value, sizeof(value));
    }
    for (; index < len; index++)
        destination[index] = source[index];
#endif
}

int cxl_gpu_transport_batch_write(CxlGpuTransport *transport, size_t offset,
                                  const void *src, size_t len) {
    if (!transport || !transport->batch_data || !src ||
        offset > CXL_GPU_BATCH_DATA_SIZE ||
        len > CXL_GPU_BATCH_DATA_SIZE - offset)
        return -1;
    memcpy((void *)(transport->batch_data + offset), src, len);
    __sync_synchronize();
    return 0;
}

int cxl_gpu_transport_batch_read(const CxlGpuTransport *transport, size_t offset,
                                 void *dst, size_t len) {
    if (!transport || !transport->batch_data || !dst ||
        offset > CXL_GPU_BATCH_DATA_SIZE ||
        len > CXL_GPU_BATCH_DATA_SIZE - offset)
        return -1;
    __sync_synchronize();
    memcpy(dst, (const void *)(transport->batch_data + offset), len);
    return 0;
}

int cxl_gpu_transport_lock(CxlGpuTransport *transport) {
    if (transport->pci_fd < 0)
        return 0;
    return flock(transport->pci_fd, LOCK_EX);
}

int cxl_gpu_transport_unlock(CxlGpuTransport *transport) {
    if (transport->pci_fd < 0)
        return 0;
    return flock(transport->pci_fd, LOCK_UN);
}

uint32_t cxl_gpu_transport_execute(CxlGpuTransport *transport, uint32_t command, uint32_t *poll_count) {
    static const uint64_t command_timeout_ns = UINT64_C(60) * UINT64_C(1000000000);
    static const uint32_t spin_poll_limit = 64;
    static const struct timespec poll_pause = {.tv_nsec = 1000};

    volatile CXLGPURAMCommandDescriptor *descriptor = transport->descriptor;
    if (transport->unusable || !descriptor) {
        if (poll_count)
            *poll_count = 0;
        return CXL_GPU_ERROR_UNKNOWN;
    }

    uint64_t generation = __atomic_load_n(&descriptor->device_generation, __ATOMIC_ACQUIRE);
    uint64_t previous = __atomic_load_n(&descriptor->completion_submission, __ATOMIC_ACQUIRE);
    if (generation == 0 || previous == UINT64_MAX) {
        transport->unusable = 1;
        if (poll_count)
            *poll_count = 0;
        return CXL_GPU_ERROR_UNKNOWN;
    }
    uint64_t submission = previous + 1;
    descriptor->protocol_version = CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION;
    descriptor->descriptor_size = CXL_GPU_DESCRIPTOR_WIRE_SIZE;
    descriptor->request_submission = submission;
    descriptor->request_device_generation = generation;
    descriptor->request_command = command;
    descriptor->request_case_epoch = command == CXL_GPU_CMD_CASE_BEGIN
                                         ? 0
                                         : __atomic_load_n(&descriptor->active_case_epoch, __ATOMIC_ACQUIRE);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *(volatile uint32_t *)((volatile uint8_t *)transport->regs +
                           CXL_GPU_DESCRIPTOR_DOORBELL_OFFSET) =
        CXL_GPU_DESCRIPTOR_DOORBELL_VALUE;

    uint32_t polls = 0;
    uint64_t deadline_ns = 0;
    for (;;) {
        uint64_t status = __atomic_load_n(&descriptor->completion_status, __ATOMIC_ACQUIRE);
        polls++;
        if ((status == CXL_GPU_DESCRIPTOR_COMPLETION_COMPLETE ||
             status == CXL_GPU_DESCRIPTOR_COMPLETION_ERROR) &&
            descriptor->completion_submission == submission &&
            descriptor->completion_device_generation == generation) {
            if (poll_count)
                *poll_count = polls;
            if (status == CXL_GPU_DESCRIPTOR_COMPLETION_ERROR)
                transport->unusable = 1;
            return (uint32_t)descriptor->result;
        }

        if (deadline_ns == 0) {
            uint64_t now_ns = monotonic_ns();
            if (now_ns == 0)
                break;
            deadline_ns = now_ns + command_timeout_ns;
        }
        if (polls >= spin_poll_limit) {
            uint64_t now_ns = monotonic_ns();
            if (now_ns == 0 || now_ns >= deadline_ns)
                break;
            nanosleep(&poll_pause, NULL);
        }
    }
    if (poll_count)
        *poll_count = polls;
    transport->unusable = 1;
    return CXL_GPU_ERROR_UNKNOWN;
}

void cxl_gpu_transport_close(CxlGpuTransport *transport) {
    if (!transport)
        return;
    if (transport->regs) {
        munmap((void *)transport->regs, transport->bar_size);
        transport->regs = NULL;
        transport->data = NULL;
        transport->descriptor = NULL;
        transport->batch_data = NULL;
    }
    if (transport->pci_fd >= 0) {
        close(transport->pci_fd);
        transport->pci_fd = -1;
    }
    transport->bar_size = 0;
    transport->unusable = 0;
    transport->pci_bdf[0] = '\0';
}

int cxl_gpu_transport_open(CxlGpuTransport *transport, int debug) {
    if (!transport || transport->regs || transport->pci_fd >= 0)
        return -1;

    transport->debug = debug;
    DIR *directory = opendir("/sys/bus/pci/devices");
    if (!directory) {
        transport_log(transport, "cannot open /sys/bus/pci/devices: %s\n", strerror(errno));
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_name[0] == '.')
            continue;

        char path[256];
        uint16_t vendor = 0;
        uint16_t device = 0;
        if (device_path(path, sizeof(path), entry->d_name, "vendor") != 0)
            continue;
        if (read_hex_u16(path, &vendor) != 0)
            continue;
        if (device_path(path, sizeof(path), entry->d_name, "device") != 0)
            continue;
        if (read_hex_u16(path, &device) != 0)
            continue;
        if (vendor != CXL_GPU_PCI_VENDOR_ID || device != CXL_GPU_PCI_DEVICE_ID)
            continue;

        transport_log(transport, "found CXL Type-2 device at %s\n", entry->d_name);
        size_t bdf_length = strlen(entry->d_name);
        memcpy(transport->pci_bdf, entry->d_name, bdf_length + 1);

        if (device_path(path, sizeof(path), entry->d_name, "enable") != 0) {
            closedir(directory);
            return -1;
        }
        int enable_fd = open(path, O_WRONLY);
        if (enable_fd >= 0) {
            ssize_t ignored = write(enable_fd, "1", 1);
            (void)ignored;
            close(enable_fd);
        }

        if (device_path(path, sizeof(path), entry->d_name, "resource") != 0) {
            closedir(directory);
            return -1;
        }
        FILE *resources = fopen(path, "r");
        if (resources) {
            uint64_t start = 0;
            uint64_t end = 0;
            uint64_t flags = 0;
            for (int index = 0; index < 2; index++) {
                if (fscanf(resources, "0x%" SCNx64 " 0x%" SCNx64 " 0x%" SCNx64 "\n", &start, &end, &flags) != 3)
                    break;
            }
            if (fscanf(resources, "0x%" SCNx64 " 0x%" SCNx64 " 0x%" SCNx64, &start, &end, &flags) == 3 && end >= start)
                transport->bar_size = (size_t)(end - start + 1);
            fclose(resources);
        }

        if (device_path(path, sizeof(path), entry->d_name, "resource2") != 0) {
            closedir(directory);
            return -1;
        }
        transport->pci_fd = open(path, O_RDWR | O_SYNC);
        if (transport->pci_fd < 0) {
            transport_log(transport, "cannot open %s: %s\n", path, strerror(errno));
            closedir(directory);
            return -1;
        }

        if (transport->bar_size == 0)
            transport->bar_size = CXL_GPU_CMD_REG_SIZE;
        if (transport->bar_size < CXL_GPU_CMD_REG_SIZE) {
            transport_log(transport, "BAR2 size %zu is smaller than protocol size %u\n", transport->bar_size,
                          CXL_GPU_CMD_REG_SIZE);
            cxl_gpu_transport_close(transport);
            closedir(directory);
            return -1;
        }
        void *mapping = mmap(NULL, transport->bar_size, PROT_READ | PROT_WRITE, MAP_SHARED, transport->pci_fd, 0);
        if (mapping == MAP_FAILED) {
            transport_log(transport, "cannot mmap BAR2: %s\n", strerror(errno));
            cxl_gpu_transport_close(transport);
            closedir(directory);
            return -1;
        }
        transport->regs = mapping;
        transport->data = (volatile uint8_t *)mapping + CXL_GPU_DATA_OFFSET;
        transport->descriptor = (volatile CXLGPURAMCommandDescriptor *)
            ((volatile uint8_t *)mapping + CXL_GPU_DESCRIPTOR_OFFSET);
        transport->batch_data =
            (volatile uint8_t *)mapping + CXL_GPU_BATCH_DATA_OFFSET;

        uint32_t magic = cxl_gpu_transport_read32(transport, CXL_GPU_REG_MAGIC);
        if (magic != CXL_GPU_MAGIC) {
            transport_log(transport, "invalid magic 0x%x, expected 0x%x\n", magic, CXL_GPU_MAGIC);
            cxl_gpu_transport_close(transport);
            closedir(directory);
            return -1;
        }
        uint32_t version = cxl_gpu_transport_read32(transport, CXL_GPU_REG_VERSION);
        if (version != CXL_GPU_VERSION) {
            transport_log(transport, "protocol version mismatch device=0x%x guest=0x%x\n", version,
                          CXL_GPU_VERSION);
            cxl_gpu_transport_close(transport);
            closedir(directory);
            return -1;
        }
        uint32_t status = cxl_gpu_transport_read32(transport, CXL_GPU_REG_STATUS);
        if (!(status & CXL_GPU_STATUS_READY)) {
            transport_log(transport, "device not ready, status=0x%x\n", status);
            cxl_gpu_transport_close(transport);
            continue;
        }

        transport_log(transport, "mapped BAR2 size=%zu magic=0x%x version=0x%x\n", transport->bar_size, magic,
                      version);
        closedir(directory);
        return 0;
    }

    closedir(directory);
    transport_log(transport, "CXL Type-2 device not found\n");
    return -1;
}

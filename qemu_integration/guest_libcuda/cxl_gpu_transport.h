#ifndef CXL_GPU_TRANSPORT_H
#define CXL_GPU_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

#include "cxl_gpu_cmd.h"

#define CXL_GPU_PCI_VENDOR_ID 0x8086
#define CXL_GPU_PCI_DEVICE_ID 0x0d92

typedef struct {
    volatile uint32_t *regs;
    volatile uint8_t *data;
    volatile CXLGPURAMCommandDescriptor *descriptor;
    volatile uint8_t *batch_data;
    int pci_fd;
    int source_fd;
    size_t bar_size;
    char pci_bdf[64];
    int debug;
    int unusable;
} CxlGpuTransport;

#define CXL_GPU_TRANSPORT_INITIALIZER                                                                                  \
    {.regs = NULL, .data = NULL, .descriptor = NULL, .batch_data = NULL, .pci_fd = -1, .source_fd = -1, .bar_size = 0, .pci_bdf = "", .debug = 0, .unusable = 0}

int cxl_gpu_transport_open(CxlGpuTransport *transport, int debug);
int cxl_gpu_transport_open_source(CxlGpuTransport *transport);
void cxl_gpu_transport_close(CxlGpuTransport *transport);

uint32_t cxl_gpu_transport_read32(const CxlGpuTransport *transport, uint32_t offset);
uint64_t cxl_gpu_transport_read64(const CxlGpuTransport *transport, uint32_t offset);
void cxl_gpu_transport_write32(CxlGpuTransport *transport, uint32_t offset, uint32_t value);
void cxl_gpu_transport_write64(CxlGpuTransport *transport, uint32_t offset, uint64_t value);
void cxl_gpu_transport_data_write(CxlGpuTransport *transport, size_t offset, const void *src, size_t len);
void cxl_gpu_transport_data_read(const CxlGpuTransport *transport, size_t offset, void *dst, size_t len);
int cxl_gpu_transport_batch_write(CxlGpuTransport *transport, size_t offset,
                                  const void *src, size_t len);
int cxl_gpu_transport_batch_read(const CxlGpuTransport *transport, size_t offset,
                                 void *dst, size_t len);

int cxl_gpu_transport_lock(CxlGpuTransport *transport);
int cxl_gpu_transport_unlock(CxlGpuTransport *transport);
int cxl_gpu_transport_try_elide_stream_sync(CxlGpuTransport *transport,
                                            uint64_t stream_wire);
uint32_t cxl_gpu_transport_execute(CxlGpuTransport *transport, uint32_t command, uint32_t *poll_count);

#endif

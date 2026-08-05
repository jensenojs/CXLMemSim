/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _UAPI_LINUX_CXL_TYPE2_ACCEL_H
#define _UAPI_LINUX_CXL_TYPE2_ACCEL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define CXL_TYPE2_TMATMUL_UAPI_VERSION	2

#define CXL_TYPE2_SOURCE_UAPI_VERSION	1
#define CXL_TYPE2_SOURCE_IOC_MAGIC	0xCF
#define CXL_TYPE2_SOURCE_MAX_RANGES	256
#define CXL_TYPE2_SOURCE_MAX_RUNS	4096

/*
 * Result flags returned by CXL_TYPE2_TMATMUL_RUN_CSR_ONLY.
 *
 * STALLED is the success indicator: the device reached the trailing 'stall'
 * instruction of the user-provided program.  Userspace verifies the output
 * region of its own mmap to confirm the result.
 */
#define CXL_TYPE2_TMATMUL_RESULT_STALLED	(1U << 0)
#define CXL_TYPE2_TMATMUL_RESULT_DMA_ERROR	(1U << 2)

#define CXL_TYPE2_MEM_REQ_READ			0
#define CXL_TYPE2_MEM_REQ_WRITE			1
#define CXL_TYPE2_MEM_REQ_MAX_BYTES		(1U << 20)

struct cxl_type2_tmatmul_info {
	__u32 version;
	__u32 dev_id;
	__u32 num_instances;
	__u32 dim_d;
	__u32 ddr_data_width;
	__u32 mc_status;
	__u32 reserved0;
	__u64 reserved1[2];	/* was default_hpa_base/_size in v1 */
	__u64 reserved2[4];
};

struct cxl_type2_tmatmul_csr_run {
	/* in */
	__u32 timeout_ms;
	__u32 flags;		/* reserved, must be 0 */

	/* out (updated on both success and error) */
	__u32 dma_status;
	__u32 stall_status;
	__u32 instr_count;
	__u32 dim_d;
	__u32 result_flags;
	__u32 reserved0;
	__u64 reserved1[4];
};

struct cxl_type2_mem_req {
	/* Inputs. hpa_base/hpa_size may be zero to use module defaults. */
	__u64 hpa_base;
	__u64 hpa_size;
	__u64 offset;
	__u64 user_ptr;
	__u32 size;
	__u32 op;
	__u32 flags;
	__u32 reserved0;
	__u64 reserved1[4];
};

struct cxl_type2_source_info_v1 {
	__u32 version;
	__u32 flags;
	__u32 max_ranges;
	__u32 max_runs;
	__aligned_u64 dmap_bytes;
	__aligned_u64 reserved[5];
};

struct cxl_type2_source_range_v1 {
	__aligned_u64 user_address;
	__aligned_u64 length;
	__u32 first_run;
	__u32 run_count;
	__aligned_u64 first_run_byte_offset;
	__aligned_u64 reserved;
};

struct cxl_type2_source_run_v1 {
	__aligned_u64 guest_phys_addr;
	__aligned_u64 length;
};

struct cxl_type2_source_acquire_v1 {
	__u32 version;
	__u32 flags;
	__aligned_u64 ranges_ptr;
	__aligned_u64 runs_ptr;
	__u32 range_count;
	__u32 run_capacity;
	__u32 run_count;
	__u32 reserved0;
	__aligned_u64 lease_handle;
	__aligned_u64 logical_bytes;
	__aligned_u64 unique_dmap_bytes;
};

struct cxl_type2_source_release_v1 {
	__u32 version;
	__u32 flags;
	__aligned_u64 lease_handle;
	__aligned_u64 reserved[2];
};

#define CXL_TYPE2_TMATMUL_IOC_MAGIC		0xCE
#define CXL_TYPE2_TMATMUL_GET_INFO		\
	_IOR(CXL_TYPE2_TMATMUL_IOC_MAGIC, 0x00, struct cxl_type2_tmatmul_info)
#define CXL_TYPE2_TMATMUL_RUN			\
	_IOWR(CXL_TYPE2_TMATMUL_IOC_MAGIC, 0x01, struct cxl_type2_tmatmul_run)
#define CXL_TYPE2_MEM_IO			\
	_IOWR(CXL_TYPE2_TMATMUL_IOC_MAGIC, 0x02, struct cxl_type2_mem_req)
/* ioctl 0x01 was CXL_TYPE2_TMATMUL_RUN in v1; retired with no replacement. */
#define CXL_TYPE2_TMATMUL_RUN_CSR_ONLY		\
	_IOWR(CXL_TYPE2_TMATMUL_IOC_MAGIC, 0x03, struct cxl_type2_tmatmul_csr_run)

#define CXL_TYPE2_SOURCE_GET_INFO		\
	_IOR(CXL_TYPE2_SOURCE_IOC_MAGIC, 0x00, struct cxl_type2_source_info_v1)
#define CXL_TYPE2_SOURCE_ACQUIRE			\
	_IOWR(CXL_TYPE2_SOURCE_IOC_MAGIC, 0x01, struct cxl_type2_source_acquire_v1)
#define CXL_TYPE2_SOURCE_RELEASE			\
	_IOW(CXL_TYPE2_SOURCE_IOC_MAGIC, 0x02, struct cxl_type2_source_release_v1)

#endif /* _UAPI_LINUX_CXL_TYPE2_ACCEL_H */

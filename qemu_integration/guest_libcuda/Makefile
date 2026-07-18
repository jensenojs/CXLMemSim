CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=
NVCC ?= nvcc
NVCC_HOST_CXX ?= g++
CUDA_ARCH ?= sm_89
CUDA_HOME ?= /usr/local/cuda

CUDA_SHIM := libcuda.so.1
CUDA_LINK := libcuda.so
CASE_CONTROL := cxl-gpu-case
TRANSPORT_TEST := test_cxl_gpu_transport
CONTEXT_STATE_TEST := test_cxl_gpu_context_state
CONTEXT_SHIM_TEST := test_cxl_gpu_context_shim
TINY_CUDA_LIBRARY := libtiny_cuda.so
TINY_CUDA_PROBE := cuda-runtime-dlopen-kernel-probe
INTEGRITY_EXPORT_ORACLE := cuda-integrity-export-oracle

CUDA_TESTS := cuda_test cuda_advanced_test
BAR_TESTS := cxl_bar_benchmark
BAR_STATIC := cxl_bar_benchmark.static
TARGETS := $(CUDA_SHIM) $(CUDA_LINK) $(CASE_CONTROL) $(CUDA_TESTS) $(BAR_TESTS)

.PHONY: all clean static test test-binaries tiny-probe integrity-export-oracle test-context-state test-context-shim

all: $(TARGETS)

test-binaries: $(CUDA_TESTS) $(BAR_TESTS) $(TRANSPORT_TEST)

tiny-probe: $(TINY_CUDA_LIBRARY) $(TINY_CUDA_PROBE)

integrity-export-oracle: $(INTEGRITY_EXPORT_ORACLE)

$(CUDA_SHIM): libcuda.c cxl_gpu_context_state.c cxl_gpu_context_state.h cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ libcuda.c cxl_gpu_context_state.c cxl_gpu_transport.c -ldl \
		-Wl,--no-as-needed -l:liblz4.so.1 -l:libzstd.so.1 -Wl,--as-needed \
		$(LDFLAGS) $(LDLIBS)

$(CUDA_LINK): $(CUDA_SHIM)
	ln -sf $(CUDA_SHIM) $@

$(CASE_CONTROL): cxl_gpu_case.c cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -o $@ cxl_gpu_case.c cxl_gpu_transport.c $(LDFLAGS) $(LDLIBS)

$(TRANSPORT_TEST): test_cxl_gpu_transport.c cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -o $@ test_cxl_gpu_transport.c cxl_gpu_transport.c $(LDFLAGS) $(LDLIBS)

$(CONTEXT_STATE_TEST): test_cxl_gpu_context_state.c cxl_gpu_context_state.c cxl_gpu_context_state.h
	$(CC) $(CFLAGS) -pthread -o $@ test_cxl_gpu_context_state.c cxl_gpu_context_state.c $(LDFLAGS) $(LDLIBS)

test-context-state: $(CONTEXT_STATE_TEST)
	./$(CONTEXT_STATE_TEST)

$(CONTEXT_SHIM_TEST): test_cxl_gpu_context_shim.c libcuda.c cxl_gpu_context_state.c cxl_gpu_context_state.h cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -DCXL_GPU_CONTEXT_SHIM_TEST -o $@ test_cxl_gpu_context_shim.c libcuda.c cxl_gpu_context_state.c cxl_gpu_transport.c -ldl \
		-Wl,--no-as-needed -l:liblz4.so.1 -l:libzstd.so.1 -Wl,--as-needed \
		$(LDFLAGS) $(LDLIBS)

test-context-shim: $(CONTEXT_SHIM_TEST)
	./$(CONTEXT_SHIM_TEST)

$(TINY_CUDA_LIBRARY): tiny_cuda_fatbin.cu
	$(NVCC) -ccbin $(NVCC_HOST_CXX) -O2 -arch=$(CUDA_ARCH) -cudart=shared -shared -Xcompiler=-fPIC -o $@ $<

$(TINY_CUDA_PROBE): cuda_runtime_dlopen_kernel_probe.c
	# This executable must reach main() before anything loads CUDA Runtime.  The
	# DSO below owns both the tiny fatbin and all CUDA Runtime calls.
	$(CC) $(CFLAGS) -o $@ $< -ldl $(LDFLAGS) $(LDLIBS)

$(INTEGRITY_EXPORT_ORACLE): integrity_export_oracle.c
	$(CC) $(CFLAGS) -pthread -o $@ $< -ldl $(LDFLAGS) $(LDLIBS)

test: $(CASE_CONTROL) $(TRANSPORT_TEST)
	./$(TRANSPORT_TEST)
	./$(CASE_CONTROL) --help
	! ./$(CASE_CONTROL)
	! ./$(CASE_CONTROL) begin --case invalid --binding 1
	! ./$(CASE_CONTROL) begin --case baseline --binding invalid
	! ./$(CASE_CONTROL) begin --case baseline --binding -1
	! ./$(CASE_CONTROL) end --epoch invalid --application-exit 0 --binding 1
	! ./$(CASE_CONTROL) end --epoch 0 --application-exit 0 --binding 1
	! ./$(CASE_CONTROL) end --epoch 1 --application-exit invalid --binding 1

cuda_test: cuda_test.c $(CUDA_LINK)
	$(CC) $(CFLAGS) -o $@ cuda_test.c -L. -lcuda -ldl -lrt -Wl,-rpath,. $(LDFLAGS) $(LDLIBS)

cuda_advanced_test: cuda_advanced_test.c $(CUDA_LINK)
	$(CC) $(CFLAGS) -o $@ cuda_advanced_test.c -L. -lcuda -ldl -lrt -Wl,-rpath,. $(LDFLAGS) $(LDLIBS)

cxl_bar_benchmark: cxl_bar_benchmark.c cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -o $@ cxl_bar_benchmark.c -lrt -lpthread $(LDFLAGS) $(LDLIBS)

static: $(BAR_STATIC)

$(BAR_STATIC): cxl_bar_benchmark.c cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -static -o $@ cxl_bar_benchmark.c -lrt -lpthread $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS) $(BAR_STATIC) $(TRANSPORT_TEST) $(CONTEXT_STATE_TEST) $(CONTEXT_SHIM_TEST) $(TINY_CUDA_LIBRARY) $(TINY_CUDA_PROBE) $(INTEGRITY_EXPORT_ORACLE)

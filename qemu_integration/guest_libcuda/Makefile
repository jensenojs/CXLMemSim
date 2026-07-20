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
GGML_CUDA_ATTRIBUTE_TRIGGER := ggml-cuda-attribute-trigger
GGML_FLASH_ATTN_EXT_TRIGGER := ggml-flash-attn-ext-trigger
CUDA_ELF_STATIC_EVIDENCE := collect_cuda_elf_static_evidence.py
CUDA_DEBUG_CAPTURE := run_cuda_debug_capture.sh
CUDA_FATBIN_CUBIN_LOAD_PROBE := probe_cuda_fatbin_cubin_load.py
CUDA_FATBIN_LIBRARY_LOAD_PROBE := probe_cuda_fatbin_library_load.py
PRIVATE_EXPORT_PROBE := private_export_probe.py
PRIVATE_EXPORT_RUNNER := run_private_export_probe.sh

CUDA_TESTS := cuda_test cuda_advanced_test
BAR_TESTS := cxl_bar_benchmark
BAR_STATIC := cxl_bar_benchmark.static
TARGETS := $(CUDA_SHIM) $(CUDA_LINK) $(CASE_CONTROL) $(CUDA_TESTS) $(BAR_TESTS)

.PHONY: all clean static test test-binaries tiny-probe private-export-table-probe integrity-export-oracle \
	fatbin-cubin-load-probe fatbin-library-load-probe \
	ggml-cuda-attribute-probe-tools ggml-flash-attn-ext-probe cuda-elf-static-evidence-tools cuda-debug-capture-tools test-context-state test-context-shim

all: $(TARGETS)

test-binaries: $(CUDA_TESTS) $(BAR_TESTS) $(TRANSPORT_TEST)

tiny-probe: $(TINY_CUDA_LIBRARY) $(TINY_CUDA_PROBE)

private-export-table-probe: tiny-probe $(PRIVATE_EXPORT_PROBE) $(PRIVATE_EXPORT_RUNNER)
	python3 -m py_compile $(PRIVATE_EXPORT_PROBE)
	bash -n $(PRIVATE_EXPORT_RUNNER)

integrity-export-oracle: $(INTEGRITY_EXPORT_ORACLE)

ggml-cuda-attribute-probe-tools: $(GGML_CUDA_ATTRIBUTE_TRIGGER) $(CUDA_ELF_STATIC_EVIDENCE)
	python3 -m py_compile $(CUDA_ELF_STATIC_EVIDENCE)

ggml-flash-attn-ext-probe: $(GGML_FLASH_ATTN_EXT_TRIGGER)

$(GGML_FLASH_ATTN_EXT_TRIGGER): ggml_flash_attn_ext_trigger.cpp
	@test -n "$(GGML_INCLUDE_DIR)" || { echo 'GGML_INCLUDE_DIR is required and must name the frozen llama source headers' >&2; exit 2; }
	@test -n "$(GGML_LIBRARY_DIR)" || { echo 'GGML_LIBRARY_DIR is required and must name the frozen guest lib directory' >&2; exit 2; }
	@test -n "$(GGML_LINK_LIBRARY_DIR)" || { echo 'GGML_LINK_LIBRARY_DIR is required and must name the frozen guest NCCL directory used only for link-time transitive closure' >&2; exit 2; }
	@test -f "$(GGML_INCLUDE_DIR)/ggml.h" || { echo 'GGML_INCLUDE_DIR does not contain ggml.h' >&2; exit 2; }
	@test -f "$(GGML_INCLUDE_DIR)/ggml-backend.h" || { echo 'GGML_INCLUDE_DIR does not contain ggml-backend.h' >&2; exit 2; }
	@test -f "$(GGML_INCLUDE_DIR)/ggml-cuda.h" || { echo 'GGML_INCLUDE_DIR does not contain ggml-cuda.h' >&2; exit 2; }
	@test -f "$(GGML_LIBRARY_DIR)/libggml-cuda.so.0" || { echo 'GGML_LIBRARY_DIR does not contain libggml-cuda.so.0' >&2; exit 2; }
	@test -f "$(GGML_LIBRARY_DIR)/libggml-base.so.0" || { echo 'GGML_LIBRARY_DIR does not contain libggml-base.so.0' >&2; exit 2; }
	@test -f "$(GGML_LIBRARY_DIR)/libggml.so.0" || { echo 'GGML_LIBRARY_DIR does not contain libggml.so.0' >&2; exit 2; }
	@test -f "$(GGML_LINK_LIBRARY_DIR)/libnccl.so.2" || { echo 'GGML_LINK_LIBRARY_DIR does not contain libnccl.so.2' >&2; exit 2; }
	$(CXX) $(CXXFLAGS) -std=c++17 -I"$(GGML_INCLUDE_DIR)" -o $@ $< -L"$(GGML_LIBRARY_DIR)" \
		-Wl,-rpath,"$(GGML_LIBRARY_DIR)" -Wl,-rpath-link,"$(GGML_LINK_LIBRARY_DIR)" \
		-l:libggml-cuda.so.0 -l:libggml-base.so.0 -l:libggml.so.0 $(LDFLAGS) $(LDLIBS)

cuda-elf-static-evidence-tools: $(CUDA_ELF_STATIC_EVIDENCE)
	python3 -m py_compile $(CUDA_ELF_STATIC_EVIDENCE)

cuda-debug-capture-tools: $(CUDA_DEBUG_CAPTURE)
	bash -n $(CUDA_DEBUG_CAPTURE)

fatbin-cubin-load-probe: $(CUDA_FATBIN_CUBIN_LOAD_PROBE)
	python3 -m py_compile $(CUDA_FATBIN_CUBIN_LOAD_PROBE)

fatbin-library-load-probe: $(CUDA_FATBIN_LIBRARY_LOAD_PROBE)
	python3 -m py_compile $(CUDA_FATBIN_LIBRARY_LOAD_PROBE)

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
	$(CC) $(CFLAGS) -DCXL_GPU_CONTEXT_SHIM_TEST -Wl,--export-dynamic -o $@ test_cxl_gpu_context_shim.c libcuda.c cxl_gpu_context_state.c cxl_gpu_transport.c -ldl \
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

$(GGML_CUDA_ATTRIBUTE_TRIGGER): ggml_cuda_attribute_trigger.c
	$(CC) $(CFLAGS) -o $@ $< -ldl $(LDFLAGS) $(LDLIBS)

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
	rm -f $(TARGETS) $(BAR_STATIC) $(TRANSPORT_TEST) $(CONTEXT_STATE_TEST) $(CONTEXT_SHIM_TEST) $(TINY_CUDA_LIBRARY) $(TINY_CUDA_PROBE) $(INTEGRITY_EXPORT_ORACLE) $(GGML_CUDA_ATTRIBUTE_TRIGGER) $(GGML_FLASH_ATTN_EXT_TRIGGER)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

CUDA_SHIM := libcuda.so.1
CUDA_LINK := libcuda.so
CASE_CONTROL := cxl-gpu-case
TRANSPORT_TEST := test_cxl_gpu_transport

CUDA_TESTS := cuda_test cuda_advanced_test
BAR_TESTS := cxl_bar_benchmark
BAR_STATIC := cxl_bar_benchmark.static
TARGETS := $(CUDA_SHIM) $(CUDA_LINK) $(CASE_CONTROL) $(CUDA_TESTS) $(BAR_TESTS)

.PHONY: all clean static test test-binaries

all: $(TARGETS)

test-binaries: $(CUDA_TESTS) $(BAR_TESTS) $(TRANSPORT_TEST)

$(CUDA_SHIM): libcuda.c cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ libcuda.c cxl_gpu_transport.c -ldl $(LDFLAGS) $(LDLIBS)

$(CUDA_LINK): $(CUDA_SHIM)
	ln -sf $(CUDA_SHIM) $@

$(CASE_CONTROL): cxl_gpu_case.c cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -o $@ cxl_gpu_case.c cxl_gpu_transport.c $(LDFLAGS) $(LDLIBS)

$(TRANSPORT_TEST): test_cxl_gpu_transport.c cxl_gpu_transport.c cxl_gpu_transport.h cxl_gpu_cmd.h
	$(CC) $(CFLAGS) -o $@ test_cxl_gpu_transport.c cxl_gpu_transport.c $(LDFLAGS) $(LDLIBS)

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
	rm -f $(TARGETS) $(BAR_STATIC) $(TRANSPORT_TEST)

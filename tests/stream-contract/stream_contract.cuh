// Shared machinery for the stream-contract consistency micro-tests.
//
// Contract under test: work enqueued on one CUDA stream executes in issue
// order. Concretely: producer kernel writes a buffer on stream S, then an
// async copy on the SAME stream S must observe the producer's writes (DtoD /
// 2D variants) or must be ordered after the producer's writes to the same
// buffer (HtoD variant). If the forwarding chain drops the stream and runs
// the copy on the host legacy default stream, the copy can overtake the
// producer running on a non-blocking stream, and the destination keeps the
// stale pattern. The producer's clock64 delay loop makes that race window
// deterministic: the copy has `delay_cycles` of real time to overtake.
//
// Output contract: key=value lines only, plus `verdict=PASS|FAIL`.
// Exit code: 0 = PASS, 1 = FAIL (contract violated), 2 = usage/env error.

#ifndef STREAM_CONTRACT_CUH
#define STREAM_CONTRACT_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Distinct 32-bit fill patterns. Every word of the validated region must end
// up EXPECTED; anything else is attributable:
//   STALE    -> the copy read the source before the producer finished
//   DST_INIT -> the copy never landed in the destination
//   other    -> corruption outside the two modeled outcomes
static const unsigned int PAT_STALE = 0x5A1E5A1Eu;
static const unsigned int PAT_FRESH = 0xFE55FE55u;  // producer writes this
static const unsigned int PAT_DST_INIT = 0x11111111u;
static const unsigned int PAT_HOST = 0xACEDACEDu;   // HtoD source content

struct CaseArgs {
    bool nonblocking = false;   // false = legacy default stream (stream 0)
    long long delay_cycles = 3000000000LL;  // ~1.7 s on a 1.78 GHz SM clock
    size_t bytes = 1u << 20;    // 1 MiB per copy
};

static void usage(const char *prog) {
    std::fprintf(stderr,
                 "usage: %s --stream=legacy|nonblocking [--delay-cycles=N] [--bytes=N]\n",
                 prog);
}

static bool parse_args(int argc, char **argv, CaseArgs *out) {
    for (int i = 1; i < argc; i++) {
        if (std::strncmp(argv[i], "--stream=", 9) == 0) {
            const char *v = argv[i] + 9;
            if (std::strcmp(v, "legacy") == 0) {
                out->nonblocking = false;
            } else if (std::strcmp(v, "nonblocking") == 0) {
                out->nonblocking = true;
            } else {
                return false;
            }
        } else if (std::strncmp(argv[i], "--delay-cycles=", 15) == 0) {
            out->delay_cycles = std::atoll(argv[i] + 15);
            if (out->delay_cycles < 0) return false;
        } else if (std::strncmp(argv[i], "--bytes=", 8) == 0) {
            out->bytes = (size_t)std::strtoull(argv[i] + 8, nullptr, 0);
            if (out->bytes == 0 || out->bytes % sizeof(unsigned int) != 0) return false;
        } else {
            return false;
        }
    }
    return true;
}

// Spin for delay_cycles of SM clock, then fill words with pattern. Single
// block: the delay gates every write, so an unordered copy sees only stale
// data no matter which word it reads first.
__global__ void producer_kernel(unsigned int *buf, size_t words,
                                unsigned int pattern, long long delay_cycles) {
    long long start = clock64();
    while (clock64() - start < delay_cycles) {
    }
    for (size_t i = threadIdx.x; i < words; i += blockDim.x) {
        buf[i] = pattern;
    }
}

#define CUDA_RT_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::printf("error=runtime api=%s code=%d detail=%s\n", #call,      \
                        (int)err_, cudaGetErrorString(err_));                   \
            return 2;                                                           \
        }                                                                       \
    } while (0)

#define CUDA_DRV_CHECK(call)                                                    \
    do {                                                                        \
        CUresult err_ = (call);                                                 \
        if (err_ != CUDA_SUCCESS) {                                             \
            const char *name_ = nullptr;                                        \
            cuGetErrorName(err_, &name_);                                       \
            std::printf("error=driver api=%s code=%d detail=%s\n", #call,       \
                        (int)err_, name_ ? name_ : "unknown");                  \
            return 2;                                                           \
        }                                                                       \
    } while (0)

// Common setup: pick device 0, allocate src/dst, fill src=STALE dst=DST_INIT
// synchronously, then make the fills visible to every later stream.
struct Buffers {
    CUdeviceptr src = 0;
    CUdeviceptr dst = 0;
    size_t words = 0;
};

static int setup_buffers(const CaseArgs &args, Buffers *out) {
    CUDA_RT_CHECK(cudaFree(0));  // establish the primary context
    CUDA_DRV_CHECK(cuMemAlloc_v2(&out->src, args.bytes));
    CUDA_DRV_CHECK(cuMemAlloc_v2(&out->dst, args.bytes));
    out->words = args.bytes / sizeof(unsigned int);
    CUDA_DRV_CHECK(cuMemsetD32_v2(out->src, PAT_STALE, out->words));
    CUDA_DRV_CHECK(cuMemsetD32_v2(out->dst, PAT_DST_INIT, out->words));
    CUDA_DRV_CHECK(cuCtxSynchronize());
    return 0;
}

static int make_stream(bool nonblocking, cudaStream_t *out) {
    if (nonblocking) {
        CUDA_RT_CHECK(cudaStreamCreateWithFlags(out, cudaStreamNonBlocking));
    } else {
        *out = 0;  // legacy default stream
    }
    return 0;
}

// Read dst back and classify every word against `expected` (contract-honoring
// outcome), `bad_marker` (the recognizable contract-violating outcome),
// DST_INIT (operation never landed) and other (unmodeled corruption).
static int validate(const char *api, const CaseArgs &args, CUdeviceptr dst,
                    unsigned int expected, unsigned int bad_marker,
                    const char *bad_diagnosis, size_t words, bool *pass_out) {
    unsigned int *host = (unsigned int *)std::malloc(args.bytes);
    if (!host) {
        std::printf("error=host_alloc bytes=%zu\n", args.bytes);
        return 2;
    }
    CUresult rc = cuMemcpyDtoH_v2(host, dst, args.bytes);
    if (rc != CUDA_SUCCESS) {
        std::printf("error=driver api=cuMemcpyDtoH_v2 code=%d\n", (int)rc);
        std::free(host);
        return 2;
    }
    size_t good = 0, bad = 0, dst_init = 0, other = 0;
    size_t first_bad = (size_t)-1;
    unsigned int first_bad_value = 0;
    for (size_t i = 0; i < words; i++) {
        if (host[i] == expected) {
            good++;
        } else {
            if (host[i] == bad_marker) bad++;
            else if (host[i] == PAT_DST_INIT) dst_init++;
            else other++;
            if (first_bad == (size_t)-1) {
                first_bad = i;
                first_bad_value = host[i];
            }
        }
    }
    bool pass = (good == words);
    std::printf("test=%s stream=%s delay_cycles=%lld bytes=%zu\n", api,
                args.nonblocking ? "nonblocking" : "legacy",
                args.delay_cycles, args.bytes);
    std::printf("total_words=%zu expected_words=%zu bad_marker_words=%zu "
                "dst_init_words=%zu other_words=%zu\n",
                words, good, bad, dst_init, other);
    if (!pass) {
        std::printf("first_mismatch_index=%zu first_mismatch_value=0x%08x "
                    "expected_value=0x%08x\n",
                    first_bad, first_bad_value, expected);
        std::printf("diagnosis=%s\n",
                    bad ? bad_diagnosis
                        : dst_init ? "operation_never_landed"
                                   : "unexpected_content");
    }
    std::printf("verdict=%s\n", pass ? "PASS" : "FAIL");
    std::free(host);
    *pass_out = pass;
    return 0;
}

#endif  // STREAM_CONTRACT_CUH

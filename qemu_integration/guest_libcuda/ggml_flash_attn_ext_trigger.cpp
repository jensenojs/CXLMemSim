#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

/*
 * This trigger executes the exact public GGML graph shape that selects the
 * Kimi failing flash-attention wrapper on an Ada L40. It deliberately has no
 * knowledge of CUDA Runtime private export tables: Python-GDB observes only
 * calls the Runtime makes naturally while this public graph runs.
 */

namespace {

constexpr int64_t kHeadQ = 576;
constexpr int64_t kHeadV = 512;
constexpr int64_t kQueryTokens = 2;
constexpr int64_t kQueryHeads = 16;
// The frozen CUDA GQA dispatcher requires K.ne[1] to be divisible by
// FATTN_KQ_STRIDE (256).  Keeping this equal to the smallest supported
// length makes the public graph reach the same <576,512,2,16> wrapper
// selected in the Kimi core rather than being rejected at supports_op.
constexpr int64_t kKeyValueTokens = 256;
constexpr int64_t kKeyValueHeads = 1;
constexpr size_t kContextBytes = 1024 * 1024;
constexpr float kExpectedValue = 0.25f;

void usage(FILE *stream) {
    std::fprintf(
        stream, "用法：ggml-flash-attn-ext-trigger [--help|--hint]\n"
                "\n"
                "运行一次公开 GGML flash-attention-ext 图。构建时必须显式指定与 guest artifact 匹配的\n"
                "GGML_INCLUDE_DIR、GGML_LIBRARY_DIR 和 GGML_LINK_LIBRARY_DIR；入口不会从系统路径搜索或回退到其他 "
                "llama/ggml 库。\n"
                "最后一项只让链接器解析 frozen libggml-cuda 的 libnccl.so.2 传递依赖；实际运行仍使用固定 L40 工具链。\n"
                "固定形状 Q=[576,2,16,1]、K=[576,256,1,1]、V=K 的 [512,...] view、F16 mask，\n"
                "在 L40 上选择 exact libggml-cuda 的 <576,512,2,16> MMA wrapper。\n"
                "它只调用公开 GGML API；不读取、写入、调用或推断 CUDA private export-table slot。\n");
}

void hint(FILE *stream) {
    std::fprintf(
        stream,
        "ggml_flash_attn_ext_trigger_hint=self=qemu_integration/guest_libcuda/ggml_flash_attn_ext_trigger.cpp\n"
        "ggml_flash_attn_ext_trigger_hint=problem=the exact public cudaFuncSetAttribute replay and backend "
        "initialization reached callback-hooks slots 2 and 6 but never slot 1; the missing precondition is the exact "
        "GGML flash-attention dispatcher rather than an invented private ABI call\n"
        "ggml_flash_attn_ext_trigger_hint=mental_model=build one small public GGML graph whose Q/K/V/mask shapes "
        "select the exact <576,512,2,16> CUDA wrapper from the Kimi core; the CUDA Runtime then performs its own "
        "natural table calls, observed separately by run_private_export_probe.sh\n"
        "ggml_flash_attn_ext_trigger_hint=role=bridge static exact-wrapper evidence and a real CUDA "
        "graph/module/function/kernel path without loading the Kimi model or starting QEMU\n"
        "ggml_flash_attn_ext_trigger_hint=inputs=compile with exact source headers through GGML_INCLUDE_DIR, exact "
        "extracted guest GGML libraries through GGML_LIBRARY_DIR and the guest libnccl.so.2 directory through "
        "GGML_LINK_LIBRARY_DIR; the link directory is not added to runtime lookup\n"
        "ggml_flash_attn_ext_trigger_hint=outputs=graph shape, backend support, allocation, compute and "
        "synchronization markers; private-table events remain in the paired Python-GDB capture directory\n"
        "ggml_flash_attn_ext_trigger_hint=interpret=pass proves this public graph executed through the frozen GGML "
        "CUDA artifact; a slot 1 natural capture is still required before any guest shim implementation\n"
        "ggml_flash_attn_ext_trigger_hint=does_not_prove=private slot signature or semantics, guest shim correctness, "
        "BAR2/Type-2 execution, Kimi output correctness or TPS\n"
        "ggml_flash_attn_ext_trigger_hint=next=run through cxl-lab run_kimi_private_abi_probe.sh with the matching "
        "private-table capture; use a reached entry/return only to constrain the smallest guest shim oracle\n");
}

int fail(const char *stage) {
    std::fprintf(stderr, "ggml_flash_attn_ext_trigger_error=%s\n", stage);
    std::printf("=== GGML_FLASH_ATTN_EXT_TRIGGER_FAIL ===\n");
    return 2;
}

void fill_tensor(struct ggml_tensor *tensor, unsigned char value) {
    std::vector<unsigned char> bytes(ggml_nbytes(tensor), value);
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        usage(stdout);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--hint") {
        hint(stdout);
        return 0;
    }
    if (argc != 1) {
        usage(stderr);
        return 2;
    }

    std::printf("=== GGML_FLASH_ATTN_EXT_TRIGGER_BEGIN ===\n");
    std::printf(
        "ggml_flash_attn_ext_trigger_shape=q=[%lld,%lld,%lld,1] k=[%lld,%lld,%lld,1] v=view-[%lld,%lld,%lld,1] "
        "mask=[%lld,%lld,1,1]\n",
        static_cast<long long>(kHeadQ), static_cast<long long>(kQueryTokens), static_cast<long long>(kQueryHeads),
        static_cast<long long>(kHeadQ), static_cast<long long>(kKeyValueTokens), static_cast<long long>(kKeyValueHeads),
        static_cast<long long>(kHeadV), static_cast<long long>(kKeyValueTokens), static_cast<long long>(kKeyValueHeads),
        static_cast<long long>(kKeyValueTokens), static_cast<long long>(kQueryTokens));
    std::fflush(stdout);

    struct ggml_init_params params = {
        /*.mem_size   =*/kContextBytes,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        return fail("ggml_init");
    }

    struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadQ, kQueryTokens, kQueryHeads, 1);
    struct ggml_tensor *k_storage =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kHeadQ, 2 * kKeyValueTokens, kKeyValueHeads, 1);
    struct ggml_tensor *k = ggml_view_4d(ctx, k_storage, kHeadQ, kKeyValueTokens, kKeyValueHeads, 1, k_storage->nb[1],
                                         k_storage->nb[2], k_storage->nb[3], 0);
    struct ggml_tensor *v =
        ggml_view_4d(ctx, k, kHeadV, kKeyValueTokens, kKeyValueHeads, 1, k->nb[1], k->nb[2], k->nb[3], 0);
    struct ggml_tensor *mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kKeyValueTokens, kQueryTokens, 1, 1);
    if (!q || !k_storage || !k || !v || !mask) {
        ggml_free(ctx);
        return fail("tensor_construction");
    }

    struct ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / 24.0f, 0.0f, 0.0f);
    if (!out) {
        ggml_free(ctx);
        return fail("flash_attn_graph_construction");
    }
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8, false);
    if (!graph) {
        ggml_free(ctx);
        return fail("graph_construction");
    }
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        ggml_free(ctx);
        return fail("cuda_backend_init");
    }
    const bool supported = ggml_backend_supports_op(backend, out);
    std::printf("ggml_flash_attn_ext_trigger_backend_support=%d\n", supported ? 1 : 0);
    if (!supported) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return fail("cuda_backend_does_not_support_flash_attn_ext");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return fail("backend_tensor_allocation");
    }
    fill_tensor(q, 0);
    fill_tensor(mask, 0);

    // Q is zero, so every attention score is zero. A constant, exactly
    // representable V makes every output element equal to kExpectedValue.
    std::vector<ggml_fp16_t> key_values(ggml_nelements(k_storage), ggml_fp32_to_fp16(kExpectedValue));
    ggml_backend_tensor_set(k_storage, key_values.data(), 0, key_values.size() * sizeof(key_values[0]));

    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> poisoned_output(ggml_nelements(out), poison);
    ggml_backend_tensor_set(out, poisoned_output.data(), 0, poisoned_output.size() * sizeof(poisoned_output[0]));
    const enum ggml_status compute = ggml_backend_graph_compute(backend, graph);
    std::printf("ggml_flash_attn_ext_trigger_compute_status=%d\n", static_cast<int>(compute));
    if (compute != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return fail("graph_compute");
    }
    ggml_backend_synchronize(backend);
    std::printf("ggml_flash_attn_ext_trigger_synchronize=pass\n");
    std::vector<float> output(ggml_nelements(out));
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    double max_abs_error = 0.0;
    for (size_t index = 0; index < output.size(); ++index) {
        const double error = std::abs(static_cast<double>(output[index]) - kExpectedValue);
        max_abs_error = std::max(max_abs_error, error);
        if (!std::isfinite(output[index]) || error > 1.0e-5) {
            std::fprintf(stderr,
                         "ggml_flash_attn_ext_trigger_oracle_mismatch index=%zu value=%g expected=%g error=%g\n",
                         index, static_cast<double>(output[index]), static_cast<double>(kExpectedValue), error);
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return fail("numerical_oracle");
        }
    }
    std::printf("ggml_flash_attn_ext_trigger_numerical_oracle=pass elements=%zu expected=%g max_abs_error=%g\n",
                output.size(), static_cast<double>(kExpectedValue), max_abs_error);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("=== GGML_FLASH_ATTN_EXT_TRIGGER_PASS ===\n");
    return 0;
}

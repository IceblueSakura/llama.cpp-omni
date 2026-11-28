// Opt-in MiniCPM-V 4.7 35B Vulkan oracle; one model/context per process.
#include "llama.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>

struct observed {
    ggml_backend_dev_t device;
    int single = 0;
    int unexpected = 0;
    int stream = 0;
    uint64_t layers = 0;
    int consumed = 0;
};

static bool observe(ggml_tensor * t, bool ask, void * opaque) {
    auto & counts = *static_cast<observed *>(opaque);
    int layer = -1;
    if (t->op == GGML_OP_MAP_CUSTOM3 && std::sscanf(t->name, "ffn_moe_out-%d", &layer) == 1) {
        if (!ask) {
            if (layer < 0 || layer >= 40) { throw std::runtime_error("unexpected streamed layer"); }
            ++counts.stream;
            counts.layers |= uint64_t(1) << layer;
        }
        return true;
    }
    if (std::strcmp(t->name, "attn_post_norm-1") == 0) {
        if (!ask) { ++counts.consumed; }
        return true;
    }
    if (t->op != GGML_OP_MUL_MAT_ID || std::strncmp(t->name, "ffn_moe_", 8) != 0) {
        return false;
    }
    if (!ask) {
        const auto * ids = t->src[2];
        if (!t->buffer || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) != counts.device || ids->ne[0] != 8) {
            ++counts.unexpected;
        } else if (ids->ne[1] == 1) {
            ++counts.single;
        } else {
            ++counts.unexpected;
        }
    }
    return true;
}

static uint64_t read_bytes() {
    std::ifstream in("/proc/self/io");
    std::string key;
    uint64_t value;
    while (in >> key >> value) {
        if (key == "read_bytes:") { return value; }
    }
    throw std::runtime_error("missing read_bytes counter");
}

int main(int argc, char ** argv) try {
    const uint64_t io_start = read_bytes();
    if (argc != 6) { throw std::runtime_error("usage: probe MODEL BANK LOGITS off|on|fail|abort|prefill-fail eval|plain"); }
    const std::string_view observation = argv[5];
    if (observation != "eval" && observation != "plain") { throw std::runtime_error("invalid observation mode"); }
    const std::string mode = argv[4];
    const bool prefill_fail = mode == "prefill-fail";
    const int fault = mode == "fail" || prefill_fail ? 1 : mode == "abort" ? 2 : 0;
    if (!fault && mode != "off" && mode != "on") { throw std::runtime_error("invalid probe mode"); }
    const bool enabled = mode != "off";
    if (enabled) { setenv("LLAMA_MOE_VK_SINGLE_SLOT", "1", 1); }
    else { unsetenv("LLAMA_MOE_VK_SINGLE_SLOT"); }
    using arm_t = void (*)(size_t, size_t, size_t, int);
    using seen_t = int (*)();
    auto arm = reinterpret_cast<arm_t>(dlsym(RTLD_DEFAULT, "les_probe_arm"));
    auto seen = reinterpret_cast<seen_t>(dlsym(RTLD_DEFAULT, "les_probe_seen"));
    if (fault && (!arm || !seen)) { throw std::runtime_error("fault shim missing"); }
    size_t base = 0, size = 0;
    if (fault) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(argv[1], {true, nullptr}), gguf_free);
        if (!meta) { throw std::runtime_error("missing metadata"); }
        int64_t tid = gguf_find_tensor(meta.get(), "blk.0.moe_exps_packed.weight");
        if (tid < 0) { throw std::runtime_error("missing carrier"); }
        base = gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), tid);
        size = gguf_get_tensor_size(meta.get(), tid);
    }
    ggml_backend_load_all();
    observed counts{ggml_backend_dev_by_name("Vulkan0")};
    if (!counts.device || ggml_backend_dev_type(counts.device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::runtime_error("Vulkan0 not available");
    }
    ggml_backend_dev_t devices[] = {counts.device, nullptr};
    const llama_model_tensor_buft_override host_experts[] = {
        {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr},
    };
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    mp.moe_slot_bank = std::stoi(argv[2]);
    mp.tensor_buft_overrides = mp.moe_slot_bank ? nullptr : host_experts;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) { throw std::runtime_error("model load failed"); }

    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 4;
    cp.n_ubatch = enabled ? 2 : 1; // exercise the opt-in cap
    cp.n_threads = cp.n_threads_batch = 4;
    if (fault == 2) {
        cp.abort_callback = [](void * ud) { return (*static_cast<seen_t *>(ud))() == 2; };
        cp.abort_callback_data = &seen;
    }
    const bool eval = fault || observation == "eval";
    cp.cb_eval = eval ? observe : nullptr;
    cp.cb_eval_user_data = &counts;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx || llama_n_ubatch(ctx.get()) != 1) { throw std::runtime_error("unexpected context geometry"); }

    if (enabled) {
        llama_set_n_expert_used(ctx.get(), 4);
        llama_set_moe_lookahead_prefetch_depth(ctx.get(), 1);
        if (llama_n_expert_used(ctx.get()) != 8 || llama_get_moe_lookahead_prefetch_depth(ctx.get()) != 0) {
            throw std::runtime_error("single-slot context admitted an unsupported configuration change");
        }
    }
    const auto io_infer = read_bytes();
    std::cout << "read_bytes_setup=" << io_infer - io_start << std::endl;
    auto io_prev = io_infer;

    constexpr char text[] = "The first benchmark compares computation and memory traffic on a discrete GPU.";
    std::array<llama_token, sizeof(text)> tokens{};
    const auto * vocab = llama_model_get_vocab(model.get());
    constexpr int prefill = 3, decode = 5;
    if (llama_tokenize(vocab, text, sizeof(text) - 1, tokens.data(), tokens.size(), true, false) < prefill + decode) {
        throw std::runtime_error("not enough oracle tokens");
    }
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    std::cout << "tokens=";
    for (int i = 0; i < prefill + decode; ++i) { std::cout << (i ? "," : "") << tokens[i]; }
    std::cout << " n_vocab=" << n_vocab << std::endl;
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    for (int step = 0; step <= decode; ++step) {
        counts.single = counts.unexpected = counts.stream = counts.consumed = 0;
        counts.layers = 0;
        const bool inject = fault && step == (prefill_fail ? 0 : 1);
        if (inject) { arm(base, size, size / 256, fault); }
        auto batch = llama_batch_get_one(tokens.data() + (step ? prefill + step - 1 : 0), step ? 1 : prefill);
        const auto rc = llama_decode(ctx.get(), batch);
        if (inject) {
            std::cout << "fault=" << mode << " seen=" << seen() << " rc=" << rc << " layer1=" << counts.consumed << std::endl;
            if (seen() != fault || rc != (fault == 1 ? -3 : 2) || counts.consumed != 0) {
                throw std::runtime_error("fault did not stop downstream execution");
            }
            const int retry = llama_decode(ctx.get(), batch);
            std::cout << "retry_rc=" << retry << std::endl;
            if (retry != -3) { throw std::runtime_error("a poisoned recurrent context was accepted"); }
            return 0;
        }
        const float * logits = rc == 0 ? llama_get_logits_ith(ctx.get(), -1) : nullptr;
        if (!logits) { throw std::runtime_error("decode/logits failed: " + std::to_string(rc)); }
        out.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
        std::cout << "step=" << step << " one_row=" << counts.single
                  << " unexpected=" << counts.unexpected << " stream=" << counts.stream << " mask=" << std::hex << counts.layers << std::dec << " layer1=" << counts.consumed << std::endl;
        const auto io_now = read_bytes();
        std::cout << "read_bytes_step=" << step << ':' << io_now - io_prev << std::endl;
        io_prev = io_now;
        const int groups = step ? 1 : prefill;
        const int stream = enabled ? 40 * groups : 0;
        if (eval && (counts.single != (enabled ? 0 : 120 * groups) ||
                counts.unexpected || counts.stream != stream || counts.consumed != groups ||
                counts.layers != (enabled ? (uint64_t(1) << 40) - 1 : 0))) {
            throw std::runtime_error("unexpected device or MoE matmul shape/count");
        }
    }
    out.close();
    if (!out) { throw std::runtime_error("write logits failed"); }
    std::cout << "read_bytes_inference=" << io_prev - io_infer << std::endl;
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) { std::cout << "peak_rss_kib=" << usage.ru_maxrss << '\n'; }
    return 0;
} catch (const std::exception & e) {
    std::cerr << "probe: " << e.what() << '\n';
    return 1;
}

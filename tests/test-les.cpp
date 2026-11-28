#include "llama-moe-experts.h"
#include "llama-moe-storage.h"
#include "llama-moe-slot.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "gguf.h"
#include "llama.h"
#include "llama-ext.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <unordered_map>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

namespace {
// Observe and forward real backend work. No fake kernels, completion or data.
struct backend_probe {
    ggml_backend_t backend;
    ggml_backend_i original;
    int synchronizations = 0;
    std::function<void(ggml_cgraph *)> before_compute, after_compute;
    static inline thread_local std::unordered_map<ggml_backend_t, backend_probe *> probes;

    explicit backend_probe(ggml_backend_t backend) : backend(backend), original(backend->iface) {
        check(probes.emplace(backend, this).second, "backend already observed");
        backend->iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) {
            auto & self = *probes.at(backend);
            if (self.before_compute) { self.before_compute(graph); }
            const auto status = self.original.graph_compute(backend, graph);
            if (status == GGML_STATUS_SUCCESS && self.after_compute) { self.after_compute(graph); }
            return status;
        };
        backend->iface.synchronize = [](ggml_backend_t backend) {
            auto & self = *probes.at(backend);
            ++self.synchronizations;
            if (self.original.synchronize) { self.original.synchronize(backend); }
        };
    }
    ~backend_probe() {
        ggml_backend_synchronize(backend);
        backend->iface = original;
        probes.erase(backend);
    }
    backend_probe(const backend_probe &) = delete;
    backend_probe & operator=(const backend_probe &) = delete;
};

constexpr int dim = 256;
constexpr int experts = 4;
constexpr int slots = 2;

static void require_async_vulkan(ggml_backend_t backend) {
    auto * dev = ggml_backend_get_device(backend);
    check(std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Vulkan") == 0,
        "this boundary test requires a real Vulkan backend");
    check(backend->iface.get_tensor_async && !std::getenv("GGML_VK_DISABLE_ASYNC") &&
        !std::getenv("GGML_VK_PERF_LOGGER"),
        "boundary tests require native async execution without disable-async/perf logger");
#ifdef TEST_LES_VULKAN_SYNCHRONOUS
    throw std::runtime_error("boundary tests require a Vulkan build without internal result checking");
#endif
}

struct family {
    std::string name;
    ggml_tensor * full = nullptr;
    ggml_tensor * view = nullptr;
    std::vector<uint8_t> bytes;
    size_t offset = 0;
};

struct fixture {
    ggml_context_ptr meta{ggml_init({64 * ggml_tensor_overhead(), nullptr, true})};
    std::unique_ptr<FILE, int (*)(FILE *)> file{tmpfile(), fclose};
    std::vector<family> families;
    std::unique_ptr<llama_moe_experts> source;
    nlohmann::json layout;
    size_t stride = 0;
    size_t data_offset = 0;
    int n_experts;

    fixture(bool fused, ggml_type type, int n_experts = experts) : n_experts(n_experts) {
        check(meta && file, "cannot create fixture");
        const std::vector<std::string> names = fused ?
            std::vector<std::string>{"ffn_gate_up_exps", "ffn_down_exps"} :
            std::vector<std::string>{"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        nlohmann::json tensors = nlohmann::json::array();
        for (size_t i = 0; i < names.size(); ++i) {
            family f;
            f.name = "blk.0." + names[i] + ".weight";
            const int rows = names[i] == "ffn_gate_up_exps" ? 2 * dim : dim;
            const auto ft = type == GGML_TYPE_Q4_K && names[i] == "ffn_down_exps" ? GGML_TYPE_Q6_K : type;
            f.full = ggml_new_tensor_3d(meta.get(), ft, dim, rows, n_experts);
            ggml_set_name(f.full, f.name.c_str());
            std::vector<float> values(ggml_nelements(f.full));
            for (size_t j = 0; j < values.size(); ++j) {
                values[j] = 0.025f * std::sin(float(j * 7 + i * 19)) + 0.001f * float(j / (dim * rows));
            }
            f.bytes.resize(ggml_nbytes(f.full));
            if (ft == GGML_TYPE_F32) {
                std::memcpy(f.bytes.data(), values.data(), f.bytes.size());
            } else {
                check(ggml_quantize_chunk(ft, values.data(), f.bytes.data(), 0,
                    rows * n_experts, dim, nullptr) == f.bytes.size(), "quantization failed");
            }
            f.offset = stride;
            stride += f.full->nb[2];
            auto * base = ggml_new_tensor_3d(meta.get(), ft, dim, rows, 2 * slots + 1);
            // Nonzero offset exercises cloning and selective uploads through views.
            f.view = ggml_view_3d(meta.get(), base, dim, rows, 2 * slots, base->nb[1], base->nb[2], base->nb[2]);
            ggml_set_name(f.view, f.name.c_str());
            tensors.push_back({{"tensor_family", names[i]}, {"type", ggml_type_name(ft)},
                {"offset_in_expert", f.offset}, {"bytes_per_expert", f.full->nb[2]}});
            families.push_back(std::move(f));
        }
        layout = {{"layout", "expert_major_packed"}, {"layers", nlohmann::json::array({{
            {"layer", 0}, {"n_expert", n_experts}, {"n_expert_used", slots},
            {"groups", nlohmann::json::array({{{"expert_begin", 0}, {"expert_count", n_experts},
                {"expert_stride", stride}, {"tensor_name", "blk.0.moe_exps_packed.weight"}, {"tensors", tensors}}})}
        }})}};
        std::vector<uint8_t> packed(stride * n_experts);
        for (const auto & f : families) {
            for (int e = 0; e < n_experts; ++e) {
                std::memcpy(packed.data() + e * stride + f.offset,
                    f.bytes.data() + e * f.full->nb[2], f.full->nb[2]);
            }
        }
        gguf_context_ptr gguf(gguf_init_empty());
        gguf_set_alignment(gguf.get(), 4096);
        gguf_set_val_str(gguf.get(), "les.moe.layout", layout.dump().c_str());
        auto * carrier = ggml_new_tensor_4d(meta.get(), GGML_TYPE_I8, stride, 1, 1, n_experts);
        ggml_set_name(carrier, "blk.0.moe_exps_packed.weight");
        carrier->data = packed.data();
        gguf_add_tensor(gguf.get(), carrier);
        check(gguf_write_to_file_ptr(gguf.get(), file.get(), false), "cannot write GGUF fixture");
        std::rewind(file.get());
        gguf_context_ptr read(gguf_init_from_file_ptr(file.get(), {true, nullptr}));
        check(bool(read), "cannot read GGUF fixture");
        const auto key = gguf_find_key(read.get(), "les.moe.layout");
        source = llama_moe_experts::from_layout(gguf_get_val_str(read.get(), key), "fixture");
        data_offset = gguf_get_data_offset(read.get()) + gguf_get_tensor_offset(read.get(), 0);
        source->bind_region("blk.0.moe_exps_packed.weight", data_offset, packed.size(), "fixture");
        carrier->data = nullptr;
    }

    void read(int expert, const family & f, void * dst) {
        const auto * entry = source->find(f.name.c_str());
        check(entry != nullptr, "missing family");
        check(fseek(file.get(), long(entry->offset + expert * entry->expert_stride), SEEK_SET) == 0, "seek failed");
        check(fread(dst, 1, entry->bytes_per_expert, file.get()) == entry->bytes_per_expert, "short packed read");
    }

    void verify() {
        for (const auto & f : families) {
            std::vector<uint8_t> actual(f.full->nb[2]);
            for (int e = 0; e < n_experts; ++e) {
                read(e, f, actual.data());
                check(std::memcmp(actual.data(), f.bytes.data() + e * actual.size(), actual.size()) == 0,
                    "packed source differs from family-major oracle");
            }
        }
    }
};

// A real, small model/manifest exercises the runtime rather than a second cache
// implementation. Its named file can be truncated after loading to force preadv
// to partially overwrite a slot and then fail.
struct runtime_fixture {
    fixture source;
    std::filesystem::path directory, path;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    ggml_context_ptr tensors{ggml_init({32 * ggml_tensor_overhead(), nullptr, false})};
    ggml_tensor * ids = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_I32, slots, 1);
    ggml_tensor * mapped = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_I32, slots, 1);
    ggml_tensor * prediction = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_I32, 1, 1);
    std::vector<char> original;

    explicit runtime_fixture(int bank, bool fused = false,
            std::function<void(gguf_context *, ggml_tensor *)> edit_carrier = {}) :
            source(fused, GGML_TYPE_F32, 16) {
        directory = std::filesystem::temp_directory_path() /
            ("llama-test-les-" + std::to_string(std::random_device{}()));
        check(std::filesystem::create_directory(directory), "cannot create runtime fixture directory");
        path = directory / "model.gguf";
        gguf_context_ptr gguf(gguf_init_empty());
        gguf_set_alignment(gguf.get(), 4096);
        gguf_set_val_str(gguf.get(), "general.architecture", "minicpm5_moe");
        gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "no_vocab");
        auto key = [&](const char * name, uint32_t value) {
            gguf_set_val_u32(gguf.get(), (std::string("minicpm5_moe.") + name).c_str(), value);
        };
        key("context_length", 32);
        key("embedding_length", dim);
        key("block_count", 2);
        key("vocab_size", 128);
        key("attention.head_count", 2);
        key("attention.head_count_kv", 2);
        key("feed_forward_length", dim);
        key("expert_count", source.n_experts);
        key("expert_used_count", slots);
        key("expert_feed_forward_length", dim);
        key("leading_dense_block_count", 0);
        key("expert_shared_count", 0);
        gguf_set_val_f32(gguf.get(), "minicpm5_moe.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_f32(gguf.get(), "minicpm5_moe.expert_weights_scale", 1.0f);
        gguf_set_val_bool(gguf.get(), "minicpm5_moe.expert_weights_norm", false);
        ggml_context_ptr meta(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        std::vector<std::vector<float>> weights;
        auto weight = [&](const std::string & name, int64_t n0, int64_t n1 = 1) {
            auto * t = ggml_new_tensor_2d(meta.get(), GGML_TYPE_F32, n0, n1);
            ggml_set_name(t, name.c_str());
            weights.emplace_back(ggml_nelements(t));
            uint32_t state = 0x9e3779b9u + uint32_t(weights.size());
            for (auto & value : weights.back()) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                value = name.find("norm.weight") != std::string::npos ? 1.0f :
                    float(int32_t(state & 0xffff) - 32768) * (0.02f / 32768);
            }
            t->data = weights.back().data();
            gguf_add_tensor(gguf.get(), t);
        };
        weight("token_embd.weight", dim, 128);
        weight("output_norm.weight", dim);
        for (int layer = 0; layer < 2; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            weight(prefix + "attn_norm.weight", dim);
            for (const auto * name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight"}) {
                weight(prefix + name, dim, dim);
            }
            weight(prefix + "ffn_norm.weight", dim);
            weight(prefix + "ffn_gate_inp.weight", dim, source.n_experts);
        }
        std::vector<uint8_t> packed(source.stride * source.n_experts);
        nlohmann::json families = nlohmann::json::array();
        for (const auto & f : source.families) {
            families.push_back({{"tensor_family", f.name.substr(6, f.name.size() - 13)},
                {"type", "F32"}, {"offset_in_expert", f.offset},
                {"bytes_per_expert", f.full->nb[2]}, {"ne", {dim, f.full->ne[1]}}});
            for (int e = 0; e < source.n_experts; ++e) {
                std::memcpy(packed.data() + e * source.stride + f.offset,
                    f.bytes.data() + e * f.full->nb[2], f.full->nb[2]);
            }
        }
        nlohmann::json layers = nlohmann::json::array();
        for (int layer = 0; layer < 2; ++layer) {
            const std::string name = "blk." + std::to_string(layer) + ".moe_exps_packed.weight";
            auto * t = ggml_new_tensor_4d(meta.get(), GGML_TYPE_I8, source.stride, 1, 1, source.n_experts);
            ggml_set_name(t, name.c_str());
            t->data = packed.data();
            if (edit_carrier) { edit_carrier(gguf.get(), t); }
            gguf_add_tensor(gguf.get(), t);
            layers.push_back({{"layer", layer}, {"n_expert", source.n_experts}, {"n_expert_used", slots},
                {"groups", nlohmann::json::array({{{"expert_begin", 0}, {"expert_count", source.n_experts},
                    {"expert_stride", source.stride}, {"tensor_name", name}, {"tensors", families}}})}});
        }
        const nlohmann::json layout = {{"layout", "expert_major_packed"}, {"layers", layers}};
        gguf_set_val_str(gguf.get(), "les.moe.layout", layout.dump().c_str());
        check(gguf_write_to_file(gguf.get(), path.string().c_str(), false), "cannot write runtime model");
        std::ifstream in(path, std::ios::binary);
        original.assign(std::istreambuf_iterator<char>(in), {});
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.use_mmap = false;
        params.moe_slot_bank = bank;
        model.reset(llama_model_load_from_file(path.string().c_str(), params));
        if (!model) {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
        check(bool(model), "cannot load runtime fixture");
    }

    ~runtime_fixture() {
        model.reset();
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    std::filesystem::path write_unpacked(
            std::function<void(gguf_context *, ggml_tensor *)> edit_family = {}) {
        ggml_context * raw = nullptr;
        gguf_context_ptr input(gguf_init_from_file(path.string().c_str(), {false, &raw}));
        ggml_context_ptr data(raw);
        check(input && data, "cannot read tiny model for unpacking");
        gguf_context_ptr output(gguf_init_empty());
        gguf_set_kv(output.get(), input.get());
        gguf_remove_key(output.get(), "les.moe.layout");
        gguf_set_alignment(output.get(), 4096);
        for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
            const char * name = gguf_get_tensor_name(input.get(), i);
            if (!std::strstr(name, ".moe_exps_packed.")) {
                gguf_add_tensor(output.get(), ggml_get_tensor(data.get(), name));
            }
        }
        ggml_context_ptr meta(ggml_init({8 * ggml_tensor_overhead(), nullptr, true}));
        std::vector<std::vector<uint8_t>> layer_bytes;
        for (int layer = 0; layer < 2; ++layer) {
            for (const auto & family : source.families) {
                auto * tensor = ggml_dup_tensor(meta.get(), family.full);
                ggml_set_name(tensor, ("blk." + std::to_string(layer) + "." + family.name.substr(6)).c_str());
                tensor->data = const_cast<uint8_t *>(family.bytes.data());
                if (layer == 1) {
                    auto & bytes = layer_bytes.emplace_back(family.bytes.size());
                    const size_t stride = family.full->nb[2];
                    for (int e = 0; e < source.n_experts; ++e) {
                        std::memcpy(bytes.data() + e * stride,
                            family.bytes.data() + ((e + 1) % source.n_experts) * stride, stride);
                    }
                    tensor->data = bytes.data();
                }
                if (edit_family) { edit_family(output.get(), tensor); }
                gguf_add_tensor(output.get(), tensor);
            }
        }
        const auto result = directory / "unpacked.gguf";
        check(gguf_write_to_file(output.get(), result.string().c_str(), false), "cannot write unpacked tiny model");
        return result;
    }

    void request(llama_moe_slot_runtime & runtime, int layer, std::array<int32_t, slots> values,
                 int predicted = -1) {
        request_rows(runtime, layer, {values.begin(), values.end()}, predicted);
    }

    void request_rows(llama_moe_slot_runtime & runtime, int layer, const std::vector<int32_t> & values,
                      int predicted = -1) {
        if (ids->ne[1] != int64_t(values.size() / slots)) {
            ids = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_I32, slots, values.size() / slots);
            mapped = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_I32, slots, values.size() / slots);
        }
        std::memcpy(ids->data, values.data(), values.size() * sizeof(int32_t));
        *static_cast<int32_t *>(prediction->data) = predicted;
        runtime.process(layer, ids, mapped, predicted >= 0 ? prediction : nullptr,
            runtime.matmul_split() > 0 && !model->layers[layer].ffn_gate_up_exps);
    }

    void verify(llama_moe_slot_runtime & runtime, int layer, std::array<int32_t, slots> values) {
        verify_rows(runtime, layer, {values.begin(), values.end()}, true);
    }

    void verify_rows(llama_moe_slot_runtime & runtime, int layer, const std::vector<int32_t> & values,
                     bool decode) {
        const auto & l = model->layers[layer];
        const std::vector<const ggml_tensor *> templates = l.ffn_gate_up_exps ?
            std::vector<const ggml_tensor *>{l.ffn_gate_up_exps, l.ffn_down_exps} :
            std::vector<const ggml_tensor *>{l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps};
        for (size_t family = 0; family < source.families.size(); ++family) {
            auto * bank = runtime.bank_tensor(templates[family], decode);
            for (size_t k = 0; k < values.size(); ++k) {
                const int slot = static_cast<int32_t *>(mapped->data)[k];
                check(slot >= 0 && slot < bank->ne[2], "runtime returned invalid slot");
                check(std::memcmp(static_cast<uint8_t *>(bank->data) + slot * bank->nb[2],
                    source.families[family].bytes.data() + values[k] * bank->nb[2], bank->nb[2]) == 0,
                    "runtime published a partial or overwritten expert");
            }
        }
    }

    void restore() {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(original.data(), original.size());
        check(bool(out), "cannot restore runtime model");
    }

    void prepare_prewarm() {
        const auto table = directory / "prewarm.txt";
        { std::ofstream out(table); out << "3\n3\n"; }
#ifdef _WIN32
        _putenv_s("LLAMA_MOE_PREWARM", table.string().c_str());
#else
        setenv("LLAMA_MOE_PREWARM", table.string().c_str(), 1);
#endif
    }
};

static void quantizer_product() {
    for (bool fused : {false, true}) {
        runtime_fixture f(4, fused);
        const auto input = f.write_unpacked();
        gguf_context_ptr original(gguf_init_from_file(input.string().c_str(), {true, nullptr}));
        check(bool(original), "cannot read original quantizer fixture");
        for (int mode = 0; mode < 4; ++mode) {
            std::printf("quantizer: fused=%d mode=%d\n", fused, mode);
            const auto plain = f.directory / "quantized.gguf";
            const auto packed = f.directory / "quantized-les.gguf";
            auto qp = llama_model_quantize_default_params();
            qp.nthread = 2;
            qp.only_copy = mode == 0 || mode == 3;
            qp.ftype = mode == 1 ? LLAMA_FTYPE_MOSTLY_F16 : LLAMA_FTYPE_MOSTLY_Q4_K_M;
            const llama_model_tensor_override types[] = {
                {"ffn_down_exps\\.weight", GGML_TYPE_F16},
                {"ffn_.*_exps\\.weight", GGML_TYPE_Q4_K},
                {nullptr, GGML_TYPE_COUNT},
            };
            if (mode == 2) { qp.tt_overrides = types; }
            llama_model_kv_override overrides_topk[2]{};
            overrides_topk[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            std::strcpy(overrides_topk[0].key, "minicpm5_moe.expert_used_count");
            overrides_topk[0].val_i64 = 1;
            if (mode == 3) { qp.kv_overrides = overrides_topk; }
            check(llama_model_quantize(input.string().c_str(), plain.string().c_str(), &qp) == 0,
                "plain tiny quantization failed");
            qp.les_experts = true;
            check(llama_model_quantize(input.string().c_str(), packed.string().c_str(), &qp) == 0,
                "LES tiny quantization failed");
            gguf_context_ptr reference(gguf_init_from_file(plain.string().c_str(), {true, nullptr}));
            gguf_context_ptr actual(gguf_init_from_file(packed.string().c_str(), {true, nullptr}));
            check(reference && actual, "cannot read quantizer output table");
            auto layout = llama_moe_experts::from_gguf(actual.get(), packed.string(), 2);
            check(bool(layout), "quantizer omitted LES layout");
            for (const auto & name : layout->region_names()) {
                const int64_t id = gguf_find_tensor(actual.get(), name.c_str());
                check(id >= 0 && gguf_get_tensor_type(actual.get(), id) == GGML_TYPE_I8, "missing I8 carrier");
                layout->bind_region(name, gguf_get_data_offset(actual.get()) + gguf_get_tensor_offset(actual.get(), id),
                    gguf_get_tensor_size(actual.get(), id), packed.string());
            }
            auto read_bytes = [](const std::filesystem::path & path, size_t offset, size_t size) {
                std::vector<uint8_t> bytes(size);
                std::ifstream in(path, std::ios::binary);
                in.seekg(offset);
                in.read(reinterpret_cast<char *>(bytes.data()), size);
                check(bool(in), "short quantizer payload");
                return bytes;
            };
            int family_count = 0, resident_count = 0;
            for (int64_t i = 0; i < gguf_get_n_tensors(reference.get()); ++i) {
                const auto * name = gguf_get_tensor_name(reference.get(), i);
                const auto expected = read_bytes(plain,
                    gguf_get_data_offset(reference.get()) + gguf_get_tensor_offset(reference.get(), i),
                    gguf_get_tensor_size(reference.get(), i));
                if (qp.only_copy) {
                    const auto id = gguf_find_tensor(original.get(), name);
                    check(id >= 0 && read_bytes(input,
                        gguf_get_data_offset(original.get()) + gguf_get_tensor_offset(original.get(), id),
                        gguf_get_tensor_size(original.get(), id)) == expected, "only-copy changed source bytes");
                }
                if (const auto * entry = layout->find(name)) {
                    ++family_count;
                    const auto expected_type = qp.only_copy ? GGML_TYPE_F32 : mode == 1 ? GGML_TYPE_F16 :
                        entry->tensor_family == "ffn_down_exps" ? GGML_TYPE_F16 : GGML_TYPE_Q4_K;
                    check(entry->type == expected_type, "requested expert quantization was not exercised");
                    check(entry->type == gguf_get_tensor_type(reference.get(), i) &&
                        entry->bytes_per_expert * f.source.n_experts == expected.size(), "quantized family geometry changed");
                    check(gguf_find_tensor(actual.get(), name) < 0, "packed output kept duplicate family tensor");
                    for (int e = 0; e < f.source.n_experts; ++e) {
                        const auto bytes = read_bytes(packed, entry->offset + e * entry->expert_stride, entry->bytes_per_expert);
                        check(std::memcmp(bytes.data(), expected.data() + e * bytes.size(), bytes.size()) == 0,
                            "LES expert payload differs from ordinary quantization");
                    }
                } else {
                    ++resident_count;
                    const int64_t id = gguf_find_tensor(actual.get(), name);
                    check(id >= 0 && gguf_get_tensor_type(actual.get(), id) == gguf_get_tensor_type(reference.get(), i) &&
                        gguf_get_tensor_size(actual.get(), id) == expected.size(), "resident tensor metadata changed");
                    check(read_bytes(packed, gguf_get_data_offset(actual.get()) + gguf_get_tensor_offset(actual.get(), id),
                        expected.size()) == expected, "LES resident payload differs from ordinary quantization");
                }
            }
            check(family_count == int(2 * f.source.families.size()) &&
                gguf_get_n_tensors(actual.get()) == resident_count + 2, "quantizer lost or added expert families");
            std::printf("quantizer: %d resident tensors and %d expert families byte-identical\n", resident_count, family_count);
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 0;
            mp.use_mmap = false;
            const llama_model_tensor_buft_override overrides[] = {llm_ffn_exps_cpu_override(), {nullptr, nullptr}};
            mp.tensor_buft_overrides = overrides;
            using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
            model_ptr plain_model(llama_model_load_from_file(plain.string().c_str(), mp), llama_model_free);
            mp.moe_slot_bank = 4;
            model_ptr packed_model(llama_model_load_from_file(packed.string().c_str(), mp), llama_model_free);
            check(plain_model && packed_model, "quantizer product cannot be loaded as a model");
            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
            context_ptr oracle(llama_init_from_model(plain_model.get(), cp), llama_free);
            context_ptr streamed(llama_init_from_model(packed_model.get(), cp), llama_free);
            check(oracle && streamed, "cannot construct quantizer product contexts");
            llama_token tokens[] = {1, 2};
            for (int count : {1, 2, 1}) {
                check(llama_decode(oracle.get(), llama_batch_get_one(tokens, count)) == 0 &&
                    llama_decode(streamed.get(), llama_batch_get_one(tokens, count)) == 0,
                    "quantizer product decode failed");
                const auto * expected = llama_get_logits(oracle.get());
                const auto * logits = llama_get_logits(streamed.get());
                check(expected && logits, "missing quantizer product logits");
                bool varied = false;
                double error = 0, norm = 0, max_abs = 0;
                for (int i = 0; i < 128; ++i) {
                    check(std::isfinite(logits[i]) && std::isfinite(expected[i]), "nonfinite quantizer product logits");
                    const double delta = double(logits[i]) - expected[i];
                    error += delta * delta;
                    norm += double(expected[i]) * expected[i];
                    max_abs = std::max(max_abs, std::abs(delta));
                    varied |= expected[i] != expected[0];
                }
                const double nmse = error / std::max(norm, 1e-30);
                std::printf("quantizer: fused=%d mode=%d count=%d NMSE=%.9g max_abs=%.9g\n",
                    fused, mode, count, nmse, max_abs);
                std::fflush(stdout);
                // Bytes above are exact; logits follow the existing model-case
                // contract rather than requiring identical floating-point kernels.
                check(nmse < 1e-4, "quantizer product logits differ from ordinary same-quantization model");
                check(varied, "quantizer product oracle has degenerate logits");
            }
            std::printf("quantizer: fused=%d mode=%d decode/prefill/continuation matched\n", fused, mode);
        }
    }
}

static void quantizer_inputs() {
    struct input_case {
        const char * name;
        const char * error;
        bool fused;
        std::function<void(gguf_context *, ggml_tensor *)> edit;
    };
    const std::vector<input_case> cases = {
        {"missing down", "incomplete or mixed expert families", true, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.ffn_down_exps.weight") == 0) { ggml_set_name(t, "unused.weight"); }
        }},
        {"missing up", "incomplete or mixed expert families", false, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.ffn_up_exps.weight") == 0) { ggml_set_name(t, "unused.weight"); }
        }},
        {"wrong expert count", "invalid expert shape", true, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.ffn_gate_up_exps.weight") == 0) { --t->ne[2]; }
        }},
        {"same bytes wrong shape", "invalid expert shape", true, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.ffn_gate_up_exps.weight") != 0) { return; }
            t->ne[0] /= 2;
            t->ne[1] *= 2;
            t->nb[1] /= 2;
        }},
        {"mixed fused and split", "incomplete or mixed expert families", true, [](gguf_context * g, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.ffn_gate_up_exps.weight") != 0) { return; }
            auto extra = *t;
            extra.ne[1] /= 2;
            extra.nb[2] /= 2;
            extra.nb[3] /= 2;
            for (const auto * name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
                ggml_set_name(&extra, name);
                gguf_add_tensor(g, &extra);
            }
        }},
    };
    for (const auto & item : cases) {
        runtime_fixture f(4, item.fused);
        const auto input = f.write_unpacked(item.edit);
        const auto output = f.directory / "rejected.gguf";
        auto qp = llama_model_quantize_default_params();
        qp.nthread = 2;
        qp.only_copy = qp.les_experts = true;
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * ud) {
            static_cast<std::string *>(ud)->append(text);
        }, &log);
        const auto result = llama_model_quantize(input.string().c_str(), output.string().c_str(), &qp);
        llama_log_set(nullptr, nullptr);
        if (result == 0 || log.find(item.error) == std::string::npos || std::filesystem::exists(output)) {
            std::fprintf(stderr, "%s\n", log.c_str());
            throw std::runtime_error(item.name);
        }
        std::printf("quantizer rejected before write: %s\n", item.name);
    }
    runtime_fixture f(4);
    const auto input = f.write_unpacked();
    const auto output = f.directory / "alignment.gguf";
    auto qp = llama_model_quantize_default_params();
    qp.only_copy = qp.les_experts = true;
    llama_model_kv_override overrides[2]{};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::strcpy(overrides[0].key, "general.alignment");
    overrides[0].val_i64 = 32;
    qp.kv_overrides = overrides;
    check(llama_model_quantize(input.string().c_str(), output.string().c_str(), &qp) != 0 &&
        !std::filesystem::exists(output), "conflicting LES alignment override was accepted");
    qp.kv_overrides = nullptr;
    check(llama_model_quantize(input.string().c_str(), output.string().c_str(), &qp) == 0,
        "quantizer did not recover after rejected input");
}

static void invalid_layout() {
    using json = nlohmann::json;
    fixture f(false, GGML_TYPE_F32);
    const std::vector<std::pair<const char *, std::function<void(json &)>>> cases = {
        {"overlapping families", [](json & j) {
            j["layers"][0]["groups"][0]["tensors"][1]["offset_in_expert"] = 0;
        }},
        {"negative offset", [](json & j) {
            j["layers"][0]["groups"][0]["tensors"][0]["offset_in_expert"] = -32;
        }},
        {"offset addition overflow", [](json & j) {
            j["layers"][0]["groups"][0]["tensors"][0]["offset_in_expert"] =
                std::numeric_limits<size_t>::max() & ~size_t(31);
        }},
        {"fractional expert count", [](json & j) { j["layers"][0]["n_expert"] = 4.5; }},
        {"expert count narrowing", [](json & j) {
            j["layers"][0]["n_expert"] = uint64_t(1) << 32 | experts;
        }},
        {"layer outside supported range", [](json & j) { j["layers"][0]["layer"] = 512; }},
        {"duplicate carrier", [](json & j) {
            auto layer = j["layers"][0];
            layer["layer"] = 1;
            j["layers"].push_back(layer);
        }},
        {"inconsistent top-k", [](json & j) {
            auto layer = j["layers"][0];
            layer["layer"] = 1;
            layer["n_expert_used"] = 1;
            layer["groups"][0]["tensor_name"] = "blk.1.moe_exps_packed.weight";
            j["layers"].push_back(layer);
        }},
        {"region size overflow", [](json & j) {
            auto & group = j["layers"][0]["groups"][0];
            const size_t stride = size_t(1) << (sizeof(size_t) * 8 - 1);
            const size_t first = group["tensors"][0]["bytes_per_expert"];
            group["expert_stride"] = stride;
            group["tensors"][2]["bytes_per_expert"] = stride - 2 * first;
        }},
        {"zero stride", [](json & j) {
            auto & group = j["layers"][0]["groups"][0];
            group["expert_stride"] = 0;
            for (auto & t : group["tensors"]) {
                t["offset_in_expert"] = 0;
                t["bytes_per_expert"] = 0;
            }
        }},
        {"unaligned family", [](json & j) {
            j["layers"][0]["groups"][0]["tensors"][0]["offset_in_expert"] = 1;
        }},
        {"unsupported JSON version", [](json & j) { j["les_version"] = 2; }},
        {"empty families", [](json & j) { j["layers"][0]["groups"][0]["tensors"] = json::array(); }},
    };
    int accepted = 0;
    for (const auto & item : cases) {
        auto layout = f.layout;
        item.second(layout);
        bool rejected = false;
        try { llama_moe_experts::from_layout(layout.dump(), "invalid fixture"); }
        catch (const std::exception &) { rejected = true; }
        std::printf("layout: %s %s\n", item.first, rejected ? "rejected" : "ACCEPTED");
        accepted += !rejected;
    }
    check(accepted == 0, "invalid expert layouts were accepted");
    bool rejected = false;
    try {
        f.source->bind_region("blk.0.moe_exps_packed.weight",
            std::numeric_limits<size_t>::max() - 15, f.stride * f.n_experts, "overflow");
    } catch (const std::exception &) { rejected = true; }
    check(rejected, "absolute carrier extent overflow was accepted");
}

static void carrier_bounds() {
    struct input_case {
        const char * name;
        const char * error;
        std::function<void(gguf_context *, ggml_tensor *)> edit;
    };
    const std::vector<input_case> cases = {
        // The next valid carrier supplies bytes beyond this shortened extent.
        {"undersized carrier", "bytes, expected", [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") == 0) { --t->ne[3]; }
        }},
        {"wrong carrier type", "must have type I8", [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") != 0) { return; }
            t->type = GGML_TYPE_I16;
            t->ne[0] /= 2;
            t->nb[0] = 2; // Same byte extent and valid higher strides.
        }},
        {"missing carrier", "model has no such tensor", [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") == 0) {
                ggml_set_name(t, "missing.weight");
            }
        }},
        {"resident after carrier", "must precede expert regions", [](gguf_context * gguf, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.1.moe_exps_packed.weight") == 0) {
                ggml_tensor sentinel = *t;
                ggml_set_name(&sentinel, "sentinel.weight");
                sentinel.ne[3] = 1;
                gguf_add_tensor(gguf, &sentinel);
            }
        }},
    };
    for (const auto & item : cases) {
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * ud) {
            static_cast<std::string *>(ud)->append(text);
        }, &log);
        bool rejected = false;
        try { runtime_fixture f(4, false, item.edit); }
        catch (const std::exception & error) {
            rejected = std::strcmp(error.what(), "cannot load runtime fixture") == 0;
        }
        llama_log_set(nullptr, nullptr);
        std::printf("loader: %s %s\n", item.name, rejected ? "rejected" : "ACCEPTED");
        if (!rejected || log.find(item.error) == std::string::npos) {
            std::fprintf(stderr, "%s\n", log.c_str());
            throw std::runtime_error(item.name);
        }
    }
}

static void layout_metadata() {
    fixture f(false, GGML_TYPE_F32);
    const std::vector<std::pair<const char *, std::function<void(gguf_context *)>>> cases = {
        {"unknown version", [](gguf_context * g) { gguf_set_val_u32(g, "les.moe.version", 2); }},
        {"wrong version type", [](gguf_context * g) { gguf_set_val_str(g, "les.moe.version", "1"); }},
        {"wrong layout type", [](gguf_context * g) { gguf_set_val_bool(g, "les.moe.layout", true); }},
        {"wrong requires type", [](gguf_context * g) { gguf_set_val_str(g, "les.moe.requires", "x"); }},
        {"wrong capability element type", [](gguf_context * g) {
            const uint32_t value = 1;
            gguf_set_arr_data(g, "les.moe.requires", GGUF_TYPE_UINT32, &value, 1);
        }},
        {"unknown capability", [](gguf_context * g) {
            const char * value = "unsupported.expert-layout";
            gguf_set_arr_str(g, "les.moe.requires", &value, 1);
        }},
        {"unaligned GGUF", [](gguf_context * g) { gguf_set_alignment(g, 32); }},
        {"alignment mismatch", [&](gguf_context * g) {
            auto layout = f.layout;
            layout["alignment"] = 2048;
            gguf_set_val_str(g, "les.moe.layout", layout.dump().c_str());
        }},
        {"layer outside model", [&](gguf_context * g) {
            auto layout = f.layout;
            layout["layers"][0]["layer"] = 1;
            gguf_set_val_str(g, "les.moe.layout", layout.dump().c_str());
        }},
    };
    for (const auto & item : cases) {
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_alignment(meta.get(), 4096);
        gguf_set_val_u32(meta.get(), "les.moe.version", 1);
        gguf_set_val_str(meta.get(), "les.moe.layout", f.layout.dump().c_str());
        item.second(meta.get());
        bool rejected = false;
        try { llama_moe_experts::from_gguf(meta.get(), "metadata fixture", 1); }
        catch (const std::exception &) { rejected = true; }
        std::printf("metadata: %s %s\n", item.first, rejected ? "rejected" : "ACCEPTED");
        check(rejected, item.first);
    }
    // Both deployed legacy-v1 metadata and explicit v1 remain loadable.
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_alignment(meta.get(), 4096);
    gguf_set_val_str(meta.get(), "les.moe.layout", f.layout.dump().c_str());
    check(bool(llama_moe_experts::from_gguf(meta.get(), "legacy", 1)), "legacy v1 rejected");
    gguf_set_val_u32(meta.get(), "les.moe.version", 1);
    const char * capability = "moe.expert_major_packed_v1";
    gguf_set_arr_str(meta.get(), "les.moe.requires", &capability, 1);
    check(bool(llama_moe_experts::from_gguf(meta.get(), "v1", 1)), "explicit v1 rejected");
    auto defaults = f.layout;
    defaults["defaults"] = {{"n_expert_used", slots}};
    defaults["layers"][0].erase("n_expert_used");
    gguf_set_val_str(meta.get(), "les.moe.layout", defaults.dump().c_str());
    check(llama_moe_experts::from_gguf(meta.get(), "defaults", 1)->expert_used_count() == slots,
        "default top-k was not applied");
}

static void runtime_rollback() {
    runtime_fixture f(4);
    llama_moe_slot_runtime runtime(*f.model);
    check(runtime.decode_window() == 2, "rollback test requires decode window 2");
    const bool deferred = runtime.matmul_split() > 0;
    for (int round = 0; round < 3; ++round) {
        f.request(runtime, 1, {0, 1});
        runtime.process_deferred(1);
        f.verify(runtime, 1, {0, 1});
        const auto * region = f.model->moe_experts_entry("blk.1.ffn_gate_exps.weight");
        std::filesystem::resize_file(f.path, region->offset + 3 * region->expert_stride + region->bytes_per_expert / 2);
        bool failed = false;
        try {
            f.request(runtime, 1, {2, 3});
            runtime.process_deferred(1);
        } catch (const std::runtime_error &) { failed = true; }
        check(failed, deferred ? "truncated deferred expert unexpectedly loaded" : "truncated expert unexpectedly loaded");
        f.restore();
        f.request(runtime, 1, {2, 3});
        runtime.process_deferred(1);
        f.verify(runtime, 1, {2, 3});
    }
}

static void prewarm_rollback() {
    // Failed prewarm entries must not be re-published at the prefill/decode
    // transition, even if another layer's prewarm succeeded.
    runtime_fixture f(4);
    f.prepare_prewarm();
    llama_moe_slot_runtime runtime(*f.model);
    const auto * region = f.model->moe_experts_entry("blk.1.ffn_gate_exps.weight");
    std::filesystem::resize_file(f.path, region->offset + 3 * region->expert_stride + region->bytes_per_expert / 2);
    runtime.prewarm_experts();
    f.restore();
    f.request(runtime, 1, {3, 4});
    f.verify(runtime, 1, {3, 4});
}

static void prewarm_churn() {
    for (bool idle : {false, true}) {
        runtime_fixture f(4);
        f.prepare_prewarm();
        llama_moe_slot_runtime runtime(*f.model);
        for (int layer : {0, 1}) { f.request(runtime, layer, {3, 0}); }
        if (idle) { runtime.prewarm_experts(); }
        for (int layer : {0, 1}) { f.request(runtime, layer, {1, 2}); }
        for (int layer : {0, 1}) { f.request_rows(runtime, layer, {0, 1, 2, 3}); }
        for (int layer : {0, 1}) {
            f.request(runtime, layer, {3, 0});
            f.verify(runtime, layer, {3, 0});
        }
    }
}

static void runtime_decode_rollback(bool abort) {
    runtime_fixture f(4);
    for (int initial : {1, 2}) {
        bool cancel = false;
        auto params = llama_context_default_params();
        params.n_ctx = 32;
        params.n_batch = params.n_ubatch = 2;
        params.n_threads = params.n_threads_batch = 4;
        if (abort) {
            params.abort_callback = [](void * data) { return *static_cast<bool *>(data); };
            params.abort_callback_data = &cancel;
            params.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
                if (std::strcmp(t->name, "ffn_moe_down-0") != 0) { return false; }
                if (!ask) { *static_cast<bool *>(data) = true; }
                return true;
            };
            params.cb_eval_user_data = &cancel;
        }
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(f.model.get(), params), llama_free);
        check(bool(ctx), "cannot create tiny LES context");
        if (!abort) {
            const auto * region = f.model->moe_experts_entry("blk.1.ffn_gate_exps.weight");
            std::filesystem::resize_file(f.path, region->offset + region->bytes_per_expert / 2);
        }
        llama_token tokens[] = {1, 2};
        const int result = llama_decode(ctx.get(), llama_batch_get_one(tokens, initial));
        check(abort ? result == 2 && cancel : result < 0, "expected decode failure/abort was not exercised");
        if (!abort) { f.restore(); }
        llama_set_abort_callback(ctx.get(), nullptr, nullptr);
        const int next = 3 - initial;
        check(llama_decode(ctx.get(), llama_batch_get_one(tokens, next)) == 0,
            "decode could not recover across phases");
        params.cb_eval = nullptr;
        params.abort_callback = nullptr;
        context_ptr oracle(llama_init_from_model(f.model.get(), params), llama_free);
        check(oracle && llama_decode(oracle.get(), llama_batch_get_one(tokens, next)) == 0,
            "cannot compute fresh context oracle");
        const float * logits = llama_get_logits(ctx.get());
        const float * expected = llama_get_logits(oracle.get());
        check(logits && expected, "missing recovered logits");
        for (int i = 0; i < 128; ++i) {
            check(std::isfinite(logits[i]) && logits[i] == expected[i],
                "recovered context used the wrong phase bank");
        }
    }
}

static void gpu_rollback(ggml_backend_t device, bool abort) {
    require_async_vulkan(device);
    auto * dev = ggml_backend_get_device(device);
    runtime_fixture f(4);
    f.model.reset();
    ggml_backend_dev_t devices[] = {dev, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    mp.use_mmap = false;
    mp.moe_slot_bank = 4;
    f.model.reset(llama_model_load_from_file(f.path.string().c_str(), mp));
    check(bool(f.model), "cannot load GPU runtime model");
    for (int initial : {1, 2}) {
        bool cancel = false;
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_batch = cp.n_ubatch = 2;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.abort_callback = [](void * data) { return *static_cast<bool *>(data); };
        cp.abort_callback_data = &cancel;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(f.model.get(), cp), llama_free);
        check(bool(ctx), "cannot construct GPU recovery context");
        auto * sched = ctx->get_sched();
        ggml_backend_t gpu_backend = nullptr, cpu_backend = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(backend) == dev) { gpu_backend = backend; }
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_backend = backend;
            }
        }
        check(gpu_backend && cpu_backend, "recovery test did not get both real backends");
        backend_probe gpu(gpu_backend), cpu(cpu_backend);
        bool submitted = false, triggered = false;
        int sync_at_failure = 0, matmuls = 0;
        gpu.after_compute = [&](ggml_cgraph * graph) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                matmuls += node->op == GGML_OP_MUL_MAT_ID;
                submitted |= std::strcmp(node->name, "ffn_moe_down-0") == 0;
            }
        };
        cpu.before_compute = [&](ggml_cgraph * graph) {
            for (int i = 0; i < ggml_graph_n_nodes(graph) && !triggered; ++i) {
                if (std::strcmp(ggml_graph_node(graph, i)->name, "ffn_moe_slots-1") != 0) { continue; }
                check(submitted, "failure occurred before the first layer GPU submission");
                triggered = true;
                sync_at_failure = gpu.synchronizations;
                if (abort) {
                    cancel = true;
                } else {
                    const auto * region = f.model->moe_experts_entry("blk.1.ffn_gate_exps.weight");
                    std::filesystem::resize_file(f.path, region->offset + region->bytes_per_expert / 2);
                }
            }
        };
        llama_token tokens[] = {1, 2};
        const int result = llama_decode(ctx.get(), llama_batch_get_one(tokens, initial));
        check(triggered && matmuls >= 3, "failure test did not execute the first GPU expert layer");
        check(abort ? result == 2 : result < 0, "GPU failure/abort did not propagate");
        check(gpu.synchronizations > sync_at_failure, "failed context returned before draining GPU backend");
        cpu.before_compute = {};
        gpu.after_compute = {};
        cancel = false;
        llama_set_abort_callback(ctx.get(), nullptr, nullptr);
        if (!abort) { f.restore(); }
        cp.abort_callback = nullptr;
        context_ptr oracle(llama_init_from_model(f.model.get(), cp), llama_free);
        check(bool(oracle), "cannot create fresh GPU oracle");
        for (int batch_size : {3 - initial, 1}) {
            check(llama_decode(ctx.get(), llama_batch_get_one(tokens, batch_size)) == 0 &&
                llama_decode(oracle.get(), llama_batch_get_one(tokens, batch_size)) == 0,
                "GPU retry or continuation failed");
            const auto * actual = llama_get_logits(ctx.get());
            const auto * expected = llama_get_logits(oracle.get());
            check(actual && expected, "missing GPU retry logits");
            bool varied = false;
            for (int i = 0; i < 128; ++i) {
                check(std::isfinite(actual[i]) && actual[i] == expected[i], "GPU retry differs from fresh context");
                varied |= expected[i] != expected[0];
            }
            check(varied, "GPU recovery oracle has degenerate vocabulary logits");
        }
        std::printf("GPU recovery: abort=%d initial=%d retry=%d continuation matched fresh context\n",
            abort, initial, 3 - initial);
    }
}

static void shared_a2_abort(ggml_backend_t device) {
    auto * dev = ggml_backend_get_device(device);
    check(std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") == 0 &&
        ggml_backend_buft_is_host_addressable(ggml_backend_get_default_buffer_type(device)),
        "shared A2 recovery requires a real shared Metal backend");
    runtime_fixture f(4);
    f.model.reset();
    ggml_backend_dev_t devices[] = {dev, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    mp.use_mmap = false;
    mp.moe_slot_bank = 4;
    f.model.reset(llama_model_load_from_file(f.path.string().c_str(), mp));
    check(bool(f.model), "cannot load shared A2 model");
    for (int retry_size : {1, 2}) {
        bool cancel = false;
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_batch = cp.n_ubatch = 2;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.abort_callback = [](void * data) { return *static_cast<bool *>(data); };
        cp.abort_callback_data = &cancel;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(f.model.get(), cp), llama_free);
        check(bool(ctx), "cannot construct shared A2 context");
        auto * sched = ctx->get_sched();
        ggml_backend_t gpu_backend = nullptr, cpu_backend = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(backend) == dev) { gpu_backend = backend; }
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_backend = backend;
            }
        }
        check(gpu_backend && cpu_backend, "shared A2 test did not get both real backends");
        backend_probe gpu(gpu_backend), cpu(cpu_backend);
        const ggml_tensor * bank = nullptr;
        bool submitted_a = false, triggered = false;
        int submitted_b = 0, sync_at_abort = 0;
        gpu.after_compute = [&](ggml_cgraph * graph) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (std::strcmp(node->name, "ffn_moe_split_a-0") == 0) {
                    submitted_a = true;
                    bank = node->src[0];
                }
                submitted_b += std::strcmp(node->name, "ffn_moe_split_b-0") == 0;
            }
        };
        cpu.before_compute = [&](ggml_cgraph * graph) {
            for (int i = 0; i < ggml_graph_n_nodes(graph) && !triggered; ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (std::strcmp(node->name, "ffn_moe_deferred_read-0") != 0) { continue; }
                check(submitted_a && bank && submitted_b == 0, "abort did not fall between A and B");
                check(bank->data && bank->buffer &&
                    ggml_backend_buft_is_host_addressable(ggml_backend_buffer_get_type(bank->buffer)) &&
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(bank->buffer)) == dev,
                    "A2 weights did not use target shared storage");
                int32_t reserved;
                std::memcpy(&reserved, static_cast<const uint8_t *>(node->src[0]->data) + sizeof(int32_t), sizeof(reserved));
                check(reserved >= 0 && reserved < bank->ne[2], "invalid B reservation");
                const auto * bytes = static_cast<const uint8_t *>(bank->data) + reserved * bank->nb[2];
                for (size_t j = 0; j < bank->nb[2]; ++j) {
                    check(bytes[j] == 0, "B expert was already installed before abort");
                }
                sync_at_abort = gpu.synchronizations;
                triggered = cancel = true;
            }
        };
        llama_token tokens[] = {1, 2};
        check(llama_decode(ctx.get(), llama_batch_get_one(tokens, 1)) == 2, "shared A2 abort did not propagate");
        check(triggered && submitted_b == 0 && gpu.synchronizations > sync_at_abort,
            "shared A2 abort did not drain before returning");
        gpu.after_compute = cpu.before_compute = nullptr;
        llama_set_abort_callback(ctx.get(), nullptr, nullptr);
        cp.abort_callback = nullptr;
        context_ptr oracle(llama_init_from_model(f.model.get(), cp), llama_free);
        check(bool(oracle), "cannot construct shared A2 oracle");
        for (int count : {retry_size, 1}) {
            check(llama_decode(ctx.get(), llama_batch_get_one(tokens, count)) == 0 &&
                llama_decode(oracle.get(), llama_batch_get_one(tokens, count)) == 0,
                "shared A2 retry or continuation failed");
            const auto * actual = llama_get_logits(ctx.get());
            const auto * expected = llama_get_logits(oracle.get());
            check(actual && expected, "missing shared A2 recovery logits");
            bool varied = false;
            for (int i = 0; i < 128; ++i) {
                check(std::isfinite(actual[i]) && actual[i] == expected[i], "shared A2 retry differs from fresh context");
                varied |= expected[i] != expected[0];
            }
            check(varied, "shared A2 recovery oracle has degenerate logits");
        }
        std::printf("shared A2 recovery: abort before B, retry=%d and continuation matched fresh context\n", retry_size);
    }
}

static void a2_contract() {
    for (bool fused : {false, true}) {
        runtime_fixture f(4, fused);
        llama_moe_slot_runtime runtime(*f.model);
        check(runtime.matmul_split() == 1, "A2 contract requires split 1");
        // Expert 1 first occurs in B, but A on the second row needs it too.
        f.request_rows(runtime, 0, {0, 1, 1, 0});
        f.verify_rows(runtime, 0, {0, 1, 1, 0}, true);
        runtime.process_deferred(0);
        f.verify_rows(runtime, 0, {0, 1, 1, 0}, true);
    }
}

static void a2_prefetch() {
    runtime_fixture f(12);
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, proceed = false, timed_out = false;
    llama_moe_slot_runtime runtime(*f.model, [&](int, int) {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return proceed; })) {
            timed_out = true;
            throw std::runtime_error("test did not release A2 prefetch barrier");
        }
    });
    check(runtime.decode_window() == 3 && runtime.window_max_rows() == 2 &&
        runtime.matmul_split() == 1 && runtime.lookahead_prefetch_depth() == 1,
        "A2 prefetch test requires a four-slot unified window and split 1");
    f.request(runtime, 0, {0, 1}, 3);
    runtime.process_deferred(0);
    {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }),
            "A2 prefetch worker did not reach barrier");
    }
    f.request(runtime, 1, {0, 3}); // B reserves a slot for the predicted expert.
    const auto before = runtime.counters_snapshot();
    {
        std::lock_guard<std::mutex> lock(mutex);
        proceed = true;
    }
    cv.notify_all();
    runtime.set_lookahead_prefetch_depth(0); // Worker publishes first; B must keep its own slot.
    check(!timed_out, "A2 prefetch barrier timed out");
    check(runtime.counters_snapshot().bytes - before.bytes == int64_t(f.source.stride),
        "A2 prefetch did not finish reading the predicted expert");
    runtime.process_deferred(1);
    f.verify(runtime, 1, {0, 3});

    const std::vector<int32_t> full_window = {0, 1, 2, 3};
    f.request_rows(runtime, 0, full_window);
    runtime.process_deferred(0);
    f.request_rows(runtime, 1, full_window);
    runtime.process_deferred(1);
    f.verify_rows(runtime, 1, full_window, true);
}

static void prefetch_rollback() {
    runtime_fixture f(12);
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    llama_moe_slot_runtime runtime(*f.model, [&](int, int) {
        std::lock_guard<std::mutex> lock(mutex);
        started = true;
        cv.notify_all();
    });
    check(runtime.lookahead_prefetch_depth() == 1, "prefetch rollback requires depth 1");
    const auto * region = f.model->moe_experts_entry("blk.1.ffn_gate_exps.weight");
    std::filesystem::resize_file(f.path, region->offset + 15 * region->expert_stride + region->bytes_per_expert / 2);
    f.request(runtime, 0, {0, 1}, 15);
    {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return started; }),
            "failed prefetch was never attempted");
    }
    runtime.set_lookahead_prefetch_depth(0); // Wait until the failed read has unwound.
    f.restore();
    const std::vector<int32_t> values = {0, 1, 2, 3};
    f.request_rows(runtime, 0, values);
    f.request_rows(runtime, 1, values); // Requires every slot; a leaked Pending fails.
    f.verify_rows(runtime, 1, values, true);
}

static void late_prefetch(bool window, bool partitioned = false) {
    runtime_fixture f(12);
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, proceed = false, waiting = false, done = false, timed_out = false, primed = false;
    llama_moe_slot_runtime runtime(*f.model, [&](int, int expert) {
        std::unique_lock<std::mutex> lock(mutex);
        if (partitioned && expert == 4) {
            primed = true;
            cv.notify_all();
            return;
        }
        entered = true;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return proceed; })) {
            timed_out = true;
            throw std::runtime_error("test did not release prefetch barrier");
        }
    }, [&] {
        std::lock_guard<std::mutex> lock(mutex);
        waiting = entered;
        cv.notify_all();
    });
    check(runtime.lookahead_prefetch_depth() == 1 && runtime.decode_window() == (window ? 3 : 0),
        "late-prefetch requires depth 1 and the configured window");
    if (partitioned) {
        f.request(runtime, 0, {0, 1}, 4);
        {
            std::unique_lock<std::mutex> lock(mutex);
            check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return primed; }), "partition was not primed");
        }
        runtime.set_lookahead_prefetch_depth(0);
        runtime.set_lookahead_prefetch_depth(1);
        f.request(runtime, 1, {4, 1}); // Expert 4 occupies the only prefetch slot.
    }
    f.request(runtime, 0, {0, 1}, 15);
    if (!partitioned) { f.request(runtime, 1, {0, 1}); }
    {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }),
            "prefetch worker did not reach barrier");
    }
    const std::vector<int32_t> values = partitioned ? std::vector<int32_t>{4, 1} :
        (window ? std::vector<int32_t>{0, 1, 2, 3} :
            std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    std::exception_ptr error;
    std::thread consumer([&] {
        try {
            if (!partitioned) { f.request_rows(runtime, 0, values); }
            f.request_rows(runtime, 1, values);
        } catch (...) { error = std::current_exception(); }
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cv.notify_all();
    });
    bool reached;
    {
        std::unique_lock<std::mutex> lock(mutex);
        // Old code returns while the writer is blocked; corrected code first
        // reaches the explicit I/O completion boundary. Neither needs a sleep.
        reached = cv.wait_for(lock, std::chrono::seconds(5), [&] { return done || waiting; });
        proceed = true;
    }
    cv.notify_all();
    consumer.join();
    runtime.set_lookahead_prefetch_depth(0); // Join this I/O batch, not a timed sleep.
    check(reached && !timed_out, "prefetch barrier timed out");
    if (error) { std::rethrow_exception(error); }
    f.verify_rows(runtime, 1, values, window);
}

struct install {
    fixture & source;
    llama_moe_storage & storage;
    int calls = 0;

    static void map(ggml_tensor * dst, const ggml_tensor * ids, int ith, int, void * userdata) {
        if (ith != 0) {
            return;
        }
        auto & self = *static_cast<install *>(userdata);
        const auto * selected = static_cast<const int32_t *>(ids->data);
        for (const auto & f : self.source.families) {
            auto * bank = self.storage.get(f.view);
            for (int slot = 0; slot < slots; ++slot) {
                self.source.read(selected[slot], f, static_cast<uint8_t *>(bank->data) + (2 * slot + 1) * bank->nb[2]);
            }
        }
        for (int32_t slot = 0; slot < slots; ++slot) {
            static_cast<int32_t *>(dst->data)[slot] = 2 * slot + 1;
        }
        ++self.calls;
    }
};

static ggml_tensor * ffn(ggml_context * ctx, const std::vector<ggml_tensor *> & weights, ggml_tensor * input, ggml_tensor * ids) {
    ggml_tensor * gate;
    ggml_tensor * up;
    if (weights.size() == 2) {
        auto * gu = ggml_mul_mat_id(ctx, weights[0], input, ids);
        gate = ggml_view_3d(ctx, gu, dim, slots, 1, gu->nb[1], gu->nb[2], 0);
        up = ggml_view_3d(ctx, gu, dim, slots, 1, gu->nb[1], gu->nb[2], dim * sizeof(float));
    } else {
        gate = ggml_mul_mat_id(ctx, weights[0], input, ids);
        up = ggml_mul_mat_id(ctx, weights[1], input, ids);
    }
    auto * activated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat_id(ctx, weights.back(), activated, ids);
}

// Match runtime banks, including Metal's alloc-only mapped/shared buffer type.
static ggml_backend_buffer_type_t bank_buffer_type(ggml_backend_t backend) {
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    using alloc_only_fn = ggml_backend_buffer_type_t (*)(size_t);
    auto alloc_only = (alloc_only_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_buffer_type_alloc_only");
    if (alloc_only) {
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            if (ggml_backend_reg_dev_get(reg, i) == dev) {
                if (auto * buft = alloc_only(i)) {
                    return buft;
                }
                break;
            }
        }
    }
    return ggml_backend_get_default_buffer_type(backend);
}

static void run(fixture & f, ggml_backend_t device, bool parallel) {
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    check(bool(cpu), "CPU init failed");
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu.get()));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) {
        set_threads(cpu.get(), 4);
    }
    const bool gpu = ggml_backend_get_device(device) != ggml_backend_get_device(cpu.get());
    llama_moe_storage storage(64), other(64), estimate(64, true);
    auto * buft = bank_buffer_type(device);
    std::vector<ggml_tensor *> banks;
    for (const auto & family : f.families) {
        auto * bank = storage.clone(family.view, buft);
        banks.push_back(bank);
        auto * private_bank = other.clone(family.view, buft);
        check(bank->buffer != private_bank->buffer && bank->view_offs == family.view->view_offs,
            "bank ownership or view offset lost");
        auto * dummy = estimate.clone(family.view, buft);
        check(dummy->data == nullptr && dummy->view_src && dummy->view_src->data == nullptr &&
            ggml_backend_buffer_get_size(dummy->buffer) == 0 &&
            dummy->buffer == dummy->view_src->buffer &&
            dummy->view_offs == bank->view_offs && std::memcmp(dummy->nb, bank->nb, sizeof(bank->nb)) == 0,
            "no-alloc bank allocated data or lost view geometry");
        if (gpu && !storage.requires_upload()) {
            check(ggml_backend_buft_is_host_addressable(buft) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(bank->buffer)) == ggml_backend_get_device(device) &&
                ggml_backend_dev_supports_buft(ggml_backend_get_device(device), ggml_backend_buffer_get_type(bank->buffer)),
                "shared bank is not directly usable by the requested device");
        }
    }
    check(storage.size_bytes() > 0 && estimate.size_bytes() == storage.size_bytes() &&
        estimate.uploaded_bytes() == 0, "bank allocation or no-alloc estimate not accounted");
    install installation{f, storage};
    ggml_context_ptr graph_ctx(ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false), nullptr, true}));
    auto * graph = ggml_new_graph_custom(graph_ctx.get(), 128, false);
    auto * ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, slots, 1);
    auto * input = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, dim, 1, 1);
    ggml_set_input(ids);
    ggml_set_input(input);
    auto * mapped = ggml_map_custom1(graph_ctx.get(), ids, install::map, 1, &installation);
    auto * output = ffn(graph_ctx.get(), banks, input, mapped);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    std::vector<ggml_backend_t> backends = gpu ? std::vector<ggml_backend_t>{device, cpu.get()} : std::vector<ggml_backend_t>{cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 128, parallel, true));
    ggml_backend_sched_set_tensor_backend(sched.get(), mapped, cpu.get());
    int device_mats = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID) {
            check(ggml_backend_supports_op(device, node), "target backend does not support fixture matmul");
            ggml_backend_sched_set_tensor_backend(sched.get(), node, gpu ? device : cpu.get());
            ++device_mats;
        }
    }
    check(device_mats >= 2, "no device matmuls");
    ggml_backend_sched_set_copy_callback(sched.get(),
        [](ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * split, void * ud) {
            return static_cast<llama_moe_storage *>(ud)->copy(backend, src, dst, split);
        }, &storage);
    check(ggml_backend_sched_alloc_graph(sched.get(), graph), "scheduler allocation failed");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID) {
            check(ggml_backend_sched_get_tensor_backend(sched.get(), node) == (gpu ? device : cpu.get()),
                "matmul silently fell back to another backend");
        }
    }

    ggml_context_ptr oracle_ctx(ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false), nullptr, true}));
    std::vector<ggml_tensor *> full;
    for (const auto & family : f.families) {
        auto * tensor = ggml_dup_tensor(oracle_ctx.get(), family.full);
        full.push_back(tensor);
    }
    auto * oracle_ids = ggml_new_tensor_2d(oracle_ctx.get(), GGML_TYPE_I32, slots, 1);
    auto * oracle_input = ggml_new_tensor_3d(oracle_ctx.get(), GGML_TYPE_F32, dim, 1, 1);
    auto * oracle_output = ffn(oracle_ctx.get(), full, oracle_input, oracle_ids);
    auto * oracle_graph = ggml_new_graph_custom(oracle_ctx.get(), 128, false);
    ggml_build_forward_expand(oracle_graph, oracle_output);
    ggml_backend_buffer_ptr oracle_buffer(ggml_backend_alloc_ctx_tensors(oracle_ctx.get(), cpu.get()));
    check(bool(oracle_buffer), "oracle allocation failed");
    for (size_t i = 0; i < full.size(); ++i) {
        ggml_backend_tensor_set(full[i], f.families[i].bytes.data(), 0, f.families[i].bytes.size());
    }
    std::vector<float> activation(dim);
    for (int i = 0; i < dim; ++i) {
        activation[i] = std::cos(float(i)) * 0.2f;
    }
    // Reuse one allocated graph while changing the experts installed in its bank.
    const std::array<std::array<int32_t, slots>, 6> requests = {{{3, 1}, {2, 0}, {1, 3}, {0, 2}, {3, 0}, {2, 1}}};
    for (const auto & request : requests) {
        ggml_backend_tensor_set(ids, request.data(), 0, sizeof(request));
        ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
        check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "LES graph compute failed");
        ggml_backend_tensor_set(oracle_ids, request.data(), 0, sizeof(request));
        ggml_backend_tensor_set(oracle_input, activation.data(), 0, activation.size() * sizeof(float));
        check(ggml_backend_graph_compute(cpu.get(), oracle_graph) == GGML_STATUS_SUCCESS, "oracle compute failed");
        std::vector<float> actual(ggml_nelements(output)), expected(actual.size());
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_tensor_get(oracle_output, expected.data(), 0, expected.size() * sizeof(float));
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite output");
            error += double(actual[i] - expected[i]) * (actual[i] - expected[i]);
            norm += double(expected[i]) * expected[i];
        }
        const double nmse = error / std::max(norm, 1e-30);
        std::printf("  request=(%d,%d) NMSE=%.8g\n", request[0], request[1], nmse);
        check(nmse < 5e-4, "LES output differs from unpacked CPU oracle");
    }
    check(installation.calls == int(requests.size()), "slot installer was not executed for every forward");
    if (gpu && storage.requires_upload()) {
        check(storage.uploaded_bytes() > 0, "GPU test never uploaded LES weights");
    } else {
        check(storage.uploaded_bytes() == 0, "direct bank unexpectedly uploaded LES weights");
    }
    for (const auto & family : f.families) {
        auto * untouched = other.get(family.view);
        const auto * bytes = static_cast<const uint8_t *>(untouched->data);
        for (size_t i = 0; i < ggml_nbytes(untouched); ++i) {
            check(bytes[i] == 0, "one context modified another context's bank");
        }
    }
    std::printf("  bank=%zu bytes uploaded=%llu bytes target=%s matmuls=%d\n", storage.size_bytes(),
        (unsigned long long) storage.uploaded_bytes(), ggml_backend_name(device), device_mats);
}

// Optional integration test: fixed token IDs avoid chat-template/tokenizer changes
// during packing. Two LES contexts alternate against unpacked contexts on the
// same compute backend. Shared GPUs use native device weights; staged GPUs use
// host experts with offload enabled, isolating packing from backend differences.
static void run_model(const std::string & packed, const std::string & reference, ggml_backend_t device) {
    check(!packed.empty() && !reference.empty(), "model case requires --model and --reference");
    const bool gpu = ggml_backend_dev_type(ggml_backend_get_device(device)) != GGML_BACKEND_DEVICE_TYPE_CPU;
    const bool shared_gpu = gpu &&
        ggml_backend_buft_is_host_addressable(ggml_backend_get_default_buffer_type(device));
    if (gpu && !shared_gpu) {
        const char * min_batch = std::getenv("GGML_OP_OFFLOAD_MIN_BATCH");
        check(min_batch && std::string(min_batch) == "1",
            "set GGML_OP_OFFLOAD_MIN_BATCH=1 so the unpacked oracle also computes experts on the GPU");
    }
    using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    ggml_backend_dev_t target_devices[] = {ggml_backend_get_device(device), nullptr};
    auto params = llama_model_default_params();
    params.n_gpu_layers = gpu ? 99 : 0;
    params.devices = target_devices;
    const llama_model_tensor_buft_override overrides[] = {llm_ffn_exps_cpu_override(), {nullptr, nullptr}};
    // CPU_REPACK expert buffers are not Metal-readable and can force a CPU
    // matmul even with min_batch=1. Use Metal's native mapped weights instead.
    params.tensor_buft_overrides = shared_gpu ? nullptr : overrides;
    model_ptr oracle(llama_model_load_from_file(reference.c_str(), params), llama_model_free);
    check(bool(oracle), "cannot load unpacked oracle");
    params.tensor_buft_overrides = nullptr;
    params.moe_slot_bank = 8;
    model_ptr streamed(llama_model_load_from_file(packed.c_str(), params), llama_model_free);
    check(bool(streamed), "cannot load packed LES model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 12;
    struct placement {
        ggml_backend_dev_t device;
        int matmuls = 0;
    };
    std::array<placement, 4> placements = {{{target_devices[0], 0}, {target_devices[0], 0},
        {target_devices[0], 0}, {target_devices[0], 0}}};
    std::vector<context_ptr> expected, actual;
    for (int i = 0; i < 2; ++i) {
        if (gpu) {
            cp.cb_eval = [](ggml_tensor * t, bool ask, void * ud) {
                if (t->op != GGML_OP_MUL_MAT_ID || std::strncmp(t->name, "ffn_moe_", 8) != 0) {
                    return false;
                }
                if (!ask) {
                    auto & p = *static_cast<placement *>(ud);
                    if (!t->buffer || ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) != p.device) {
                        std::fprintf(stderr, "misplaced matmul=%s buffer=%s target=%s\n", t->name,
                            t->buffer ? ggml_backend_buffer_name(t->buffer) : "(null)",
                            ggml_backend_dev_name(p.device));
                    }
                    check(t->buffer && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == p.device,
                        "real model matmul fell back from requested device");
                    ++p.matmuls;
                }
                return true;
            };
            cp.cb_eval_user_data = &placements[2 * i];
        }
        expected.emplace_back(llama_init_from_model(oracle.get(), cp), llama_free);
        cp.cb_eval_user_data = &placements[2 * i + 1];
        actual.emplace_back(llama_init_from_model(streamed.get(), cp), llama_free);
        check(expected.back() && actual.back(), "cannot create model contexts");
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(oracle.get()));
    check(n_vocab == llama_vocab_n_tokens(llama_model_get_vocab(streamed.get())), "vocabulary mismatch");
    const std::array<std::array<llama_token, 4>, 2> requests = {{{1, 1000, 12345, 1000}, {2, 1111, 23456, 1111}}};
    for (size_t step = 0; step < requests[0].size(); ++step) {
        for (size_t stream = 0; stream < requests.size(); ++stream) {
            llama_token token = requests[stream][step];
            check(token < n_vocab, "test token outside vocabulary");
            auto batch = llama_batch_get_one(&token, 1);
            check(llama_decode(expected[stream].get(), batch) == 0, "unpacked oracle decode failed");
            check(llama_decode(actual[stream].get(), batch) == 0, "LES decode failed");
            auto * a = llama_get_logits(actual[stream].get());
            auto * b = llama_get_logits(expected[stream].get());
            double error = 0, norm = 0;
            float max_error = 0;
            int top_a = 0, top_b = 0;
            for (int i = 0; i < n_vocab; ++i) {
                check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite model logits");
                error += double(a[i] - b[i]) * (a[i] - b[i]);
                norm += double(b[i]) * b[i];
                max_error = std::max(max_error, std::abs(a[i] - b[i]));
                if (a[i] > a[top_a]) { top_a = i; }
                if (b[i] > b[top_b]) { top_b = i; }
            }
            const double nmse = error / std::max(norm, 1e-30);
            std::printf("model stream=%zu step=%zu NMSE=%.8g max_abs=%.6g top=%d/%d\n",
                stream, step, nmse, max_error, top_a, top_b);
            check(nmse < 1e-4, "LES logits differ from unpacked same-backend oracle");
        }
    }
    for (const auto & p : placements) {
        check(!gpu || p.matmuls > 0, "real model did not execute any checked device matmuls");
        std::printf("verified device matmuls=%d\n", p.matmuls);
    }
}

static void scheduler_boundaries(ggml_backend_t device) {
    require_async_vulkan(device);
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    check(bool(cpu), "cannot initialize scheduler CPU backend");
    auto * dev = ggml_backend_get_device(device);
    ggml_backend_t backends[] = {device, cpu.get()};
    {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 64, true, true));
        const int copies = ggml_backend_sched_get_n_copies(sched.get());
        check(copies > 1, "parallel scheduler did not allocate multiple copies");
        std::set<void *> destinations;
        for (int round = 0; round < 2 * copies + 1; ++round) {
            ggml_context_ptr ctx(ggml_init({32 * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(64, false), nullptr, true}));
            auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
            auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
            ggml_set_input(input);
            auto * mapped = ggml_map_custom1(ctx.get(), input,
                [](ggml_tensor * dst, const ggml_tensor * src, int ith, int, void *) {
                    if (ith == 0) { std::memcpy(dst->data, src->data, ggml_nbytes(src)); }
                }, 1, nullptr);
            auto * output = ggml_scale(ctx.get(), mapped, 2);
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
            ggml_backend_sched_set_tensor_backend(sched.get(), input, cpu.get());
            ggml_backend_sched_set_tensor_backend(sched.get(), mapped, cpu.get());
            ggml_backend_sched_set_tensor_backend(sched.get(), output, device);
            check(ggml_backend_sched_alloc_graph(sched.get(), graph), "rotation graph allocation failed");
            check(ggml_backend_sched_get_n_splits(sched.get()) == 2, "rotation graph did not cross backends");
            destinations.insert(output->src[0]->data);
            std::array<float, 16> values, actual;
            for (int i = 0; i < 16; ++i) { values[i] = float(round * 16 + i + 1); }
            ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
            check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS,
                "rotation graph compute failed");
            ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
            for (int i = 0; i < 16; ++i) { check(actual[i] == 2 * values[i], "stale scheduler copy"); }
            // Reset while this graph's tensors still exist, then discard them.
            // No synchronize between reset and the next alloc: that resets copy 0.
            ggml_backend_sched_reset(sched.get());
        }
        check(destinations.size() == size_t(copies), "scheduler did not rotate every physical copy");
        std::printf("scheduler rotation: %d copies, %d marked allocations passed\n", copies, 2 * copies + 1);
    }
    for (bool parallel : {false, true}) {
        ggml_context_ptr ctx(ggml_init({32 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(64, false), nullptr, true}));
        auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        auto * seed = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_input(input);
        ggml_set_input(seed);
        auto * marker = ggml_scale(ctx.get(), input, 2);
        ggml_set_output(marker);
        auto * host_type = ggml_backend_dev_host_buffer_type(dev);
        check(host_type, "Vulkan pinned readback type absent");
        ggml_backend_buffer_ptr readback(ggml_backend_buft_alloc_buffer(host_type, 16 * sizeof(float)));
        check(readback && ggml_backend_buffer_get_type(readback.get()) == host_type,
            "pinned readback fell back to synchronous pageable memory");
        auto * values = static_cast<float *>(ggml_backend_buffer_get_base(readback.get()));
        backend_probe probe(device);
        struct observation {
            backend_probe & probe;
            const float * values;
            int sync_after_submit = 0;
            bool submitted = false, waited = false, visible = false;
        } seen{probe, values};
        auto * consumer = ggml_map_custom1(ctx.get(), seed,
            [](ggml_tensor * dst, const ggml_tensor * src, int ith, int, void * ud) {
                if (ith != 0) { return; }
                auto & seen = *static_cast<observation *>(ud);
                seen.waited = seen.submitted && seen.probe.synchronizations > seen.sync_after_submit;
                seen.visible = true;
                for (int i = 0; i < 16; ++i) { seen.visible &= seen.values[i] == 2 * (i + 1); }
                std::memcpy(dst->data, src->data, sizeof(float));
            }, 1, &seen);
        ggml_set_output(consumer);
        ggml_build_forward_expand(graph, marker);
        ggml_build_forward_expand(graph, consumer); // No graph input crosses GPU -> CPU.
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 64, parallel, true));
        ggml_backend_sched_set_tensor_backend(sched.get(), input, device);
        ggml_backend_sched_set_tensor_backend(sched.get(), marker, device);
        ggml_backend_sched_set_tensor_backend(sched.get(), seed, cpu.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), consumer, cpu.get());
        check(ggml_backend_sched_alloc_graph(sched.get(), graph), "no-input graph allocation failed");
        check(ggml_backend_sched_get_n_splits(sched.get()) == 2, "no-input graph did not cross backends");
        std::array<float, 16> input_values;
        for (int i = 0; i < 16; ++i) { input_values[i] = float(i + 1); }
        ggml_backend_tensor_set(input, input_values.data(), 0, sizeof(input_values));
        const float zero = 0;
        ggml_backend_tensor_set(seed, &zero, 0, sizeof(zero));
        std::memset(values, 0, sizeof(input_values));
        probe.after_compute = [&](ggml_cgraph *) {
            ggml_backend_tensor_get_async(device, marker, values, 0, sizeof(input_values));
            seen.sync_after_submit = probe.synchronizations;
            seen.submitted = true;
        };
        check(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS,
            "no-input graph compute failed");
        // Assert before any test-side synchronization can hide the boundary.
        check(seen.waited && seen.visible, "no-input CPU split observed GPU data before completion");
        ggml_backend_sched_synchronize(sched.get());
        std::printf("scheduler no-input: parallel=%d native readback visible\n", parallel);
    }
}

static size_t no_alloc_model(const std::string & packed, ggml_backend_t device, int bank = 8) {
    check(!packed.empty(), "no-alloc requires --model");
    auto params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {ggml_backend_get_device(device), nullptr};
    params.devices = devices;
    params.n_gpu_layers = ggml_backend_dev_type(devices[0]) == GGML_BACKEND_DEVICE_TYPE_CPU ? 0 : 99;
    params.moe_slot_bank = bank;
    params.no_alloc = true;
    // Match the fit/estimation caller: no_alloc cannot wrap model mmap buffers.
    params.use_mmap = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(packed.c_str(), params), llama_model_free);
    check(bool(model), "cannot load metadata-only LES model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 1;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    check(bool(ctx), "cannot construct metadata-only LES context");
    size_t context_bytes = 0;
    for (const auto & item : llama_get_memory_breakdown(ctx.get())) {
        context_bytes += item.second.context;
    }
    check(context_bytes > 0, "metadata-only context omitted its memory estimate");
    std::printf("no-alloc context estimate=%zu bytes\n", context_bytes);
    return context_bytes;
}

static void tiny_no_alloc(ggml_backend_t device) {
    runtime_fixture f(4);
    const size_t small = no_alloc_model(f.path.string(), device, 4);
    const size_t large = no_alloc_model(f.path.string(), device, 8);
    // The prefill bank is shared across layers; the fixed per-layer decode
    // windows and KV geometry must not contribute to this four-slot increment.
    check(large > small && large - small == 4 * f.source.stride,
        "context estimate omitted or double-counted LES bank capacity");
    std::printf("no-alloc four-slot increment=%zu bytes\n", large - small);
}

struct env_guard {
    std::string key;
    std::string prev;
    bool had = false;
    env_guard(const char * key, const char * value) : key(key) {
        if (const char * old = std::getenv(key)) {
            had = true;
            prev = old;
        }
        set(value);
    }
    ~env_guard() {
        if (had) { set(prev.c_str()); }
        else { unset(); }
    }
    env_guard(const env_guard &) = delete;
    env_guard & operator=(const env_guard &) = delete;
    void set(const char * v) const {
#ifdef _WIN32
        _putenv_s(key.c_str(), v ? v : "");
#else
        if (v) { setenv(key.c_str(), v, 1); } else { unsetenv(key.c_str()); }
#endif
    }
    void unset() const {
#ifdef _WIN32
        _putenv_s(key.c_str(), "");
#else
        unsetenv(key.c_str());
#endif
    }
};

static void check_logits_close(const float * actual, const float * expected, int n_vocab, const char * label) {
    check(actual && expected, "missing logits");
    bool varied = false;
    double error = 0, norm = 0, max_abs = 0;
    for (int i = 0; i < n_vocab; ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite logits");
        const double delta = double(actual[i]) - expected[i];
        error += delta * delta;
        norm += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
        varied |= expected[i] != expected[0];
    }
    const double nmse = error / std::max(norm, 1e-30);
    std::printf("%s: NMSE=%.9g max_abs=%.9g\n", label, nmse, max_abs);
    check(nmse < 1e-4, "logits differ from oracle");
    check(varied, "oracle produced degenerate logits");
}

static void write_mutated_les_gguf(
        const std::filesystem::path & src_path,
        const std::filesystem::path & dst_path,
        const std::function<void(gguf_context *)> & edit_kv,
        const std::function<void(gguf_context *, ggml_tensor *)> & edit_tensor,
        const std::function<void(gguf_context *, ggml_context *)> & after_tensors) {
    ggml_context * raw = nullptr;
    gguf_context_ptr input(gguf_init_from_file(src_path.string().c_str(), {false, &raw}));
    ggml_context_ptr data(raw);
    check(input && data, "cannot read source LES GGUF");
    gguf_context_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), input.get());
    gguf_set_alignment(output.get(), gguf_get_alignment(input.get()));
    if (edit_kv) {
        edit_kv(output.get());
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(input.get()); ++i) {
        const char * name = gguf_get_tensor_name(input.get(), i);
        ggml_tensor * t = ggml_get_tensor(data.get(), name);
        ggml_tensor copy = *t;
        if (edit_tensor) {
            edit_tensor(output.get(), &copy);
        }
        gguf_add_tensor(output.get(), &copy);
    }
    if (after_tensors) {
        after_tensors(output.get(), data.get());
    }
    check(gguf_write_to_file(output.get(), dst_path.string().c_str(), false), "cannot write mutated LES GGUF");
}

static void packed_inplace(ggml_backend_t device) {
    runtime_fixture f(4);
    const auto unpacked = f.write_unpacked();

    struct bad_case {
        const char * name;
        const char * error;
        std::function<void(gguf_context *)> edit_kv;
        std::function<void(gguf_context *, ggml_tensor *)> edit_tensor;
        std::function<void(gguf_context *, ggml_context *)> after_tensors;
    };
    const std::vector<bad_case> bad_cases = {
        {"version 2", "les.moe.version", [](gguf_context * g) {
            gguf_set_val_u32(g, "les.moe.version", 2);
        }, {}, {}},
        {"version string", "les.moe.version", [](gguf_context * g) {
            gguf_set_val_str(g, "les.moe.version", "1");
        }, {}, {}},
        {"layout bool", "les.moe.layout", [](gguf_context * g) {
            gguf_set_val_bool(g, "les.moe.layout", true);
        }, {}, {}},
        {"requires string", "les.moe.requires", [](gguf_context * g) {
            gguf_set_val_str(g, "les.moe.requires", "x");
        }, {}, {}},
        {"requires uint32 element", "les.moe.requires", [](gguf_context * g) {
            const uint32_t v = 1;
            gguf_set_arr_data(g, "les.moe.requires", GGUF_TYPE_UINT32, &v, 1);
        }, {}, {}},
        {"requires unknown cap", "unsupported required capability", [](gguf_context * g) {
            const char * v = "unsupported.cap";
            gguf_set_arr_str(g, "les.moe.requires", &v, 1);
        }, {}, {}},
        {"unaligned gguf", "page-aligned GGUF layout", [](gguf_context * g) {
            gguf_set_alignment(g, 32);
        }, {}, {}},
        {"alignment mismatch", "page-aligned GGUF layout", [](gguf_context * g) {
            const int k = gguf_find_key(g, "les.moe.layout");
            auto j = nlohmann::json::parse(gguf_get_val_str(g, k));
            j["alignment"] = 2048;
            gguf_set_val_str(g, "les.moe.layout", j.dump().c_str());
        }, {}, {}},
        {"layer index >= block_count", "exceeds model layer count", [](gguf_context * g) {
            const int k = gguf_find_key(g, "les.moe.layout");
            auto j = nlohmann::json::parse(gguf_get_val_str(g, k));
            j["layers"][0]["layer"] = 2;
            gguf_set_val_str(g, "les.moe.layout", j.dump().c_str());
        }, {}, {}},
        {"undersized carrier", "bytes, expected", {}, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") == 0) { --t->ne[3]; }
        }, {}},
        {"non-I8 carrier", "must have type I8", {}, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") == 0) {
                t->type = GGML_TYPE_I16;
                t->ne[0] /= 2;
                t->nb[0] = 2;
            }
        }, {}},
        {"missing carrier", "no such tensor", {}, [](gguf_context *, ggml_tensor * t) {
            if (std::strcmp(t->name, "blk.0.moe_exps_packed.weight") == 0) {
                ggml_set_name(t, "missing.weight");
            }
        }, {}},
        {"resident after carrier", "must precede expert regions", {}, {}, [](gguf_context * g, ggml_context * data) {
            ggml_tensor * base = ggml_get_tensor(data, "blk.1.moe_exps_packed.weight");
            ggml_tensor sentinel = *base;
            ggml_set_name(&sentinel, "sentinel.weight");
            sentinel.ne[3] = 1;
            gguf_add_tensor(g, &sentinel);
        }},
        {"family size mismatch", "byte size mismatch against layout", [](gguf_context * g) {
            const int k = gguf_find_key(g, "les.moe.layout");
            auto j = nlohmann::json::parse(gguf_get_val_str(g, k));
            auto & tensors = j["layers"][0]["groups"][0]["tensors"];
            tensors[0]["bytes_per_expert"] = size_t(tensors[0]["bytes_per_expert"]) - 32;
            tensors[1]["offset_in_expert"] = size_t(tensors[1]["offset_in_expert"]) - 32;
            tensors[1]["bytes_per_expert"] = size_t(tensors[1]["bytes_per_expert"]) + 32;
            gguf_set_val_str(g, "les.moe.layout", j.dump().c_str());
        }, {}, {}},
    };

    for (const auto & item : bad_cases) {
        const auto bad_path = f.directory / "bad-inplace.gguf";
        write_mutated_les_gguf(f.path, bad_path, item.edit_kv, item.edit_tensor, item.after_tensors);
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * ud) {
            static_cast<std::string *>(ud)->append(text);
        }, &log);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_mmap = false;
        mp.moe_slot_bank = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> bad(
            llama_model_load_from_file(bad_path.string().c_str(), mp), llama_model_free);
        llama_log_set(nullptr, nullptr);
        std::printf("packed-inplace negative [%s]: rejected=%d\n", item.name, bad == nullptr ? 1 : 0);
        if (bad != nullptr || log.find(item.error) == std::string::npos) {
            std::fprintf(stderr, "missing expected error '%s' in log:\n%s\n", item.error, log.c_str());
            throw std::runtime_error(item.name);
        }
    }

    const auto positive_les = f.directory / "positive-inplace.gguf";
    {
        auto qp = llama_model_quantize_default_params();
        qp.nthread = 2;
        qp.only_copy = true;
        qp.les_experts = true;
        check(llama_model_quantize(unpacked.string().c_str(), positive_les.string().c_str(), &qp) == 0,
            "cannot pack unpacked fixture for packed-inplace");
    }

    const bool gpu = ggml_backend_dev_type(ggml_backend_get_device(device)) != GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_dev_t devs[] = {ggml_backend_get_device(device), nullptr};
    for (bool use_mmap : {true, false}) {
        auto mp = llama_model_default_params();
        mp.devices = devs;
        mp.n_gpu_layers = gpu ? 99 : 0;
        mp.use_mmap = use_mmap;
        mp.moe_slot_bank = 0;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        model_ptr oracle_model(llama_model_load_from_file(unpacked.string().c_str(), mp), llama_model_free);
        model_ptr inplace_model(llama_model_load_from_file(positive_les.string().c_str(), mp), llama_model_free);
        check(oracle_model && inplace_model, "cannot load packed-inplace positive models");
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_batch = cp.n_ubatch = 2;
        cp.n_threads = cp.n_threads_batch = 2;
        context_ptr oracle(llama_init_from_model(oracle_model.get(), cp), llama_free);
        context_ptr inplace(llama_init_from_model(inplace_model.get(), cp), llama_free);
        check(oracle && inplace, "cannot create packed-inplace contexts");
        llama_token toks_prefill[] = {1, 2};
        llama_token tok_decode = 3;
        check(llama_decode(oracle.get(), llama_batch_get_one(toks_prefill, 2)) == 0 &&
            llama_decode(inplace.get(), llama_batch_get_one(toks_prefill, 2)) == 0,
            "packed-inplace prefill failed");
        const std::string tag_p = std::string("packed-inplace prefill mmap=") + (use_mmap ? "1" : "0");
        check_logits_close(llama_get_logits(inplace.get()), llama_get_logits(oracle.get()), 128, tag_p.c_str());
        check(llama_decode(oracle.get(), llama_batch_get_one(&tok_decode, 1)) == 0 &&
            llama_decode(inplace.get(), llama_batch_get_one(&tok_decode, 1)) == 0,
            "packed-inplace decode failed");
        const std::string tag_d = std::string("packed-inplace decode mmap=") + (use_mmap ? "1" : "0");
        check_logits_close(llama_get_logits(inplace.get()), llama_get_logits(oracle.get()), 128, tag_d.c_str());
    }

    {
        auto mp = llama_model_default_params();
        mp.devices = devs;
        mp.n_gpu_layers = gpu ? 99 : 0;
        mp.use_mmap = false;
        mp.no_alloc = true;
        mp.moe_slot_bank = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> m(
            llama_model_load_from_file(positive_les.string().c_str(), mp), llama_model_free);
        check(bool(m), "packed-inplace no_alloc model failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_batch = cp.n_ubatch = 2;
        std::unique_ptr<llama_context, decltype(&llama_free)> c(llama_init_from_model(m.get(), cp), llama_free);
        check(bool(c), "packed-inplace no_alloc context failed");
        size_t total_bytes = 0;
        for (const auto & item : llama_get_memory_breakdown(c.get())) {
            total_bytes += item.second.model + item.second.context + item.second.compute;
        }
        check(total_bytes > 0, "packed-inplace no_alloc omitted memory estimate");
    }
}

static std::filesystem::path write_per_layer_unpacked_model(
        const std::filesystem::path & dir,
        bool fused,
        bool wrong_shape = false,
        bool dense_has_expert = false) {
    const int n_layers = 3;
    const int n_exp = 4;
    const int n_used = 2;
    const std::array<uint32_t, 3> n_ff_arr = {0, 256, 512};
    const auto path = dir / (std::string("per-layer-") + (fused ? "fused" : "split") +
        (wrong_shape ? "-badshape" : "") + (dense_has_expert ? "-denseexp" : "") + ".gguf");
    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_alignment(gguf.get(), 4096);
    gguf_set_val_str(gguf.get(), "general.architecture", "minicpm5_moe");
    gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "no_vocab");
    auto key = [&](const char * name, uint32_t value) {
        gguf_set_val_u32(gguf.get(), (std::string("minicpm5_moe.") + name).c_str(), value);
    };
    key("context_length", 32);
    key("embedding_length", dim);
    key("block_count", n_layers);
    key("vocab_size", 128);
    key("attention.head_count", 2);
    key("attention.head_count_kv", 2);
    key("feed_forward_length", dim);
    key("expert_count", n_exp);
    key("expert_used_count", n_used);
    key("leading_dense_block_count", 1);
    key("expert_shared_count", 0);
    gguf_set_arr_data(gguf.get(), "minicpm5_moe.expert_feed_forward_length", GGUF_TYPE_UINT32, n_ff_arr.data(), n_ff_arr.size());
    gguf_set_val_f32(gguf.get(), "minicpm5_moe.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_f32(gguf.get(), "minicpm5_moe.expert_weights_scale", 1.0f);
    gguf_set_val_bool(gguf.get(), "minicpm5_moe.expert_weights_norm", false);

    ggml_context_ptr meta(ggml_init({128 * ggml_tensor_overhead(), nullptr, true}));
    std::vector<std::vector<float>> weights;
    auto add_tensor = [&](const std::string & name, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        auto * t = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        weights.emplace_back(ggml_nelements(t));
        uint32_t state = 0x9e3779b9u + uint32_t(weights.size() * 131u);
        for (size_t i = 0; i < weights.back().size(); ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            weights.back()[i] = name.find("norm.weight") != std::string::npos ? 1.0f :
                float(int32_t(state & 0xffff) - 32768) * (0.02f / 32768);
        }
        t->data = weights.back().data();
        gguf_add_tensor(gguf.get(), t);
    };

    add_tensor("token_embd.weight", dim, 128);
    add_tensor("output_norm.weight", dim);
    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        add_tensor(p + "attn_norm.weight", dim);
        for (const auto * name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight"}) {
            add_tensor(p + name, dim, dim);
        }
        add_tensor(p + "ffn_norm.weight", dim);
        if (layer == 0) {
            add_tensor(p + "ffn_gate.weight", dim, dim);
            add_tensor(p + "ffn_up.weight", dim, dim);
            add_tensor(p + "ffn_down.weight", dim, dim);
            if (dense_has_expert) {
                add_tensor(p + "ffn_gate_exps.weight", dim, 256, n_exp);
            }
        } else {
            const int64_t n_ff = (layer == 1 && wrong_shape) ? 512 : int64_t(n_ff_arr[layer]);
            add_tensor(p + "ffn_gate_inp.weight", dim, n_exp);
            if (fused) {
                add_tensor(p + "ffn_gate_up_exps.weight", dim, 2 * n_ff, n_exp);
                add_tensor(p + "ffn_down_exps.weight", n_ff, dim, n_exp);
            } else {
                add_tensor(p + "ffn_gate_exps.weight", dim, n_ff, n_exp);
                add_tensor(p + "ffn_up_exps.weight", dim, n_ff, n_exp);
                add_tensor(p + "ffn_down_exps.weight", n_ff, dim, n_exp);
            }
        }
    }
    check(gguf_write_to_file(gguf.get(), path.string().c_str(), false), "cannot write per-layer model");
    return path;
}

static void quantizer_per_layer_ffn() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("llama-test-les-perlayer-" + std::to_string(std::random_device{}()));
    check(std::filesystem::create_directory(dir), "cannot create per-layer temp dir");
    struct cleanup {
        std::filesystem::path d;
        ~cleanup() { std::error_code ec; std::filesystem::remove_all(d, ec); }
    } guard{dir};

    auto read_bytes = [](const std::filesystem::path & path, size_t offset, size_t size) {
        std::vector<uint8_t> bytes(size);
        std::ifstream in(path, std::ios::binary);
        in.seekg(offset);
        in.read(reinterpret_cast<char *>(bytes.data()), size);
        check(bool(in), "short quantizer payload");
        return bytes;
    };

    for (bool fused : {false, true}) {
        const auto input = write_per_layer_unpacked_model(dir, fused);
        for (int mode = 0; mode < 3; ++mode) {
            std::printf("quantizer-per-layer-ffn: fused=%d mode=%d\n", fused, mode);
            const auto plain = dir / "plain.gguf";
            const auto packed = dir / "packed.gguf";
            auto qp = llama_model_quantize_default_params();
            qp.nthread = 2;
            qp.only_copy = (mode == 0);
            qp.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_S;
            const llama_model_tensor_override mixed_types[] = {
                {"blk\\.1\\.ffn_.*_exps\\.weight", GGML_TYPE_F32},
                {"blk\\.2\\.ffn_.*_exps\\.weight", GGML_TYPE_F16},
                {nullptr, GGML_TYPE_COUNT},
            };
            if (mode == 2) {
                qp.tt_overrides = mixed_types;
            }
            check(llama_model_quantize(input.string().c_str(), plain.string().c_str(), &qp) == 0,
                "plain per-layer quantization failed");
            qp.les_experts = true;
            check(llama_model_quantize(input.string().c_str(), packed.string().c_str(), &qp) == 0,
                "LES per-layer quantization failed");

            gguf_context_ptr ref(gguf_init_from_file(plain.string().c_str(), {true, nullptr}));
            gguf_context_ptr act(gguf_init_from_file(packed.string().c_str(), {true, nullptr}));
            check(ref && act, "cannot read per-layer quantize outputs");
            auto layout = llama_moe_experts::from_gguf(act.get(), packed.string(), 3);
            for (const auto & name : layout->region_names()) {
                const int64_t id = gguf_find_tensor(act.get(), name.c_str());
                check(id >= 0 && gguf_get_tensor_type(act.get(), id) == GGML_TYPE_I8, "missing I8 carrier");
                layout->bind_region(name,
                    gguf_get_data_offset(act.get()) + gguf_get_tensor_offset(act.get(), id),
                    gguf_get_tensor_size(act.get(), id), packed.string());
            }
            for (int64_t i = 0; i < gguf_get_n_tensors(ref.get()); ++i) {
                const auto * name = gguf_get_tensor_name(ref.get(), i);
                const auto expected = read_bytes(plain,
                    gguf_get_data_offset(ref.get()) + gguf_get_tensor_offset(ref.get(), i),
                    gguf_get_tensor_size(ref.get(), i));
                if (const auto * entry = layout->find(name)) {
                    check(entry->type == gguf_get_tensor_type(ref.get(), i), "expert type mismatch");
                    check(entry->bytes_per_expert * 4 == expected.size(), "expert size mismatch");
                    for (int e = 0; e < 4; ++e) {
                        const auto bytes = read_bytes(packed, entry->offset + e * entry->expert_stride, entry->bytes_per_expert);
                        check(bytes.size() == entry->bytes_per_expert &&
                            std::memcmp(bytes.data(), expected.data() + e * bytes.size(), bytes.size()) == 0,
                            "per-layer LES expert payload differs");
                    }
                } else {
                    const int64_t id = gguf_find_tensor(act.get(), name);
                    check(id >= 0 && gguf_get_tensor_type(act.get(), id) == gguf_get_tensor_type(ref.get(), i) &&
                        gguf_get_tensor_size(act.get(), id) == expected.size(), "resident metadata differs");
                    check(read_bytes(packed, gguf_get_data_offset(act.get()) + gguf_get_tensor_offset(act.get(), id),
                        expected.size()) == expected, "resident bytes differ");
                }
            }

            using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
            using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 0;
            mp.use_mmap = false;
            mp.use_extra_bufts = false;
            model_ptr plain_m(llama_model_load_from_file(plain.string().c_str(), mp), llama_model_free);
            mp.moe_slot_bank = 4;
            model_ptr slot_m(llama_model_load_from_file(packed.string().c_str(), mp), llama_model_free);
            mp.moe_slot_bank = 0;
            model_ptr inplace_m(llama_model_load_from_file(packed.string().c_str(), mp), llama_model_free);
            check(plain_m && slot_m && inplace_m, "cannot load quantized per-layer models");

            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            context_ptr oracle(llama_init_from_model(plain_m.get(), cp), llama_free);
            context_ptr slotted(llama_init_from_model(slot_m.get(), cp), llama_free);
            context_ptr inplace(llama_init_from_model(inplace_m.get(), cp), llama_free);
            check(oracle && slotted && inplace, "cannot create quantized per-layer contexts");
            llama_token toks_p[] = {1, 2};
            llama_token tok_d = 3;
            for (int n : {2, 1}) {
                const llama_token * ptr = n == 2 ? toks_p : &tok_d;
                check(llama_decode(oracle.get(), llama_batch_get_one(const_cast<llama_token *>(ptr), n)) == 0 &&
                    llama_decode(slotted.get(), llama_batch_get_one(const_cast<llama_token *>(ptr), n)) == 0 &&
                    llama_decode(inplace.get(), llama_batch_get_one(const_cast<llama_token *>(ptr), n)) == 0,
                    "per-layer quantized decode failed");
                check_logits_close(llama_get_logits(slotted.get()), llama_get_logits(oracle.get()), 128, "per-layer slot");
                check_logits_close(llama_get_logits(inplace.get()), llama_get_logits(oracle.get()), 128, "per-layer inplace");
            }
        }
    }

    for (bool dense_err : {false, true}) {
        const auto bad_in = write_per_layer_unpacked_model(dir, false, !dense_err, dense_err);
        const auto bad_out = dir / "rejected-perlayer.gguf";
        auto qp = llama_model_quantize_default_params();
        qp.only_copy = true;
        qp.les_experts = true;
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * ud) {
            static_cast<std::string *>(ud)->append(text);
        }, &log);
        const int rc = llama_model_quantize(bad_in.string().c_str(), bad_out.string().c_str(), &qp);
        llama_log_set(nullptr, nullptr);
        check(rc != 0 && !std::filesystem::exists(bad_out),
            dense_err ? "dense layer expert tensor was accepted" : "wrong per-layer n_ff shape was accepted");
    }
}

static void write_predictor_models(
        const std::filesystem::path & dir,
        bool with_prev,
        std::filesystem::path & out_les,
        std::filesystem::path & out_unpacked) {
    const int n_layers = 3;
    const int n_exp = 16;
    const int n_used = 2;
    const int64_t stride = 3 * dim * dim * int64_t(sizeof(float));
    out_les = dir / (std::string("pred-les-") + (with_prev ? "prev" : "noprev") + ".gguf");
    out_unpacked = dir / (std::string("pred-unpacked-") + (with_prev ? "prev" : "noprev") + ".gguf");

    ggml_context_ptr meta(ggml_init({256 * ggml_tensor_overhead(), nullptr, true}));
    std::vector<std::vector<float>> f32_bufs;
    std::vector<std::vector<uint8_t>> packed_bufs;

    auto make_f32 = [&](const std::string & name, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        auto * t = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        f32_bufs.emplace_back(ggml_nelements(t));
        uint32_t state = 0x517cc1b7u + uint32_t(f32_bufs.size() * 97u);
        for (size_t i = 0; i < f32_bufs.back().size(); ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if (name.find("norm.weight") != std::string::npos) {
                f32_bufs.back()[i] = 1.0f;
            } else if (name.find("exp_probs_b") != std::string::npos) {
                f32_bufs.back()[i] = 0.001f * float(i);
            } else {
                f32_bufs.back()[i] = float(int32_t(state & 0xffff) - 32768) * (0.02f / 32768);
            }
        }
        t->data = f32_bufs.back().data();
        return t;
    };

    auto init_gguf = [&]() {
        gguf_context_ptr g(gguf_init_empty());
        gguf_set_alignment(g.get(), 4096);
        gguf_set_val_str(g.get(), "general.architecture", "minicpm5_moe");
        gguf_set_val_str(g.get(), "tokenizer.ggml.model", "no_vocab");
        auto key = [&](const char * name, uint32_t value) {
            gguf_set_val_u32(g.get(), (std::string("minicpm5_moe.") + name).c_str(), value);
        };
        key("context_length", 32);
        key("embedding_length", dim);
        key("block_count", n_layers);
        key("vocab_size", 128);
        key("attention.head_count", 2);
        key("attention.head_count_kv", 2);
        key("feed_forward_length", dim);
        key("expert_count", n_exp);
        key("expert_used_count", n_used);
        key("expert_feed_forward_length", dim);
        key("leading_dense_block_count", 1);
        key("expert_shared_count", 0);
        gguf_set_val_f32(g.get(), "minicpm5_moe.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_f32(g.get(), "minicpm5_moe.expert_weights_scale", 1.0f);
        gguf_set_val_bool(g.get(), "minicpm5_moe.expert_weights_norm", false);
        return g;
    };

    gguf_context_ptr g_les = init_gguf();
    gguf_context_ptr g_unp = init_gguf();

    auto add_resident = [&](ggml_tensor * t) {
        gguf_add_tensor(g_les.get(), t);
        gguf_add_tensor(g_unp.get(), t);
    };

    add_resident(make_f32("token_embd.weight", dim, 128));
    add_resident(make_f32("output_norm.weight", dim));
    std::vector<std::array<ggml_tensor *, 3>> layer_exps(n_layers);
    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        add_resident(make_f32(p + "attn_norm.weight", dim));
        for (const auto * name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight"}) {
            add_resident(make_f32(p + name, dim, dim));
        }
        add_resident(make_f32(p + "ffn_norm.weight", dim));
        if (layer == 0) {
            add_resident(make_f32(p + "ffn_gate.weight", dim, dim));
            add_resident(make_f32(p + "ffn_up.weight", dim, dim));
            add_resident(make_f32(p + "ffn_down.weight", dim, dim));
        } else {
            add_resident(make_f32(p + "ffn_gate_inp.weight", dim, n_exp));
            add_resident(make_f32(p + "exp_probs_b.bias", n_exp));
            add_resident(make_f32(p + "ffn_gate_inp_pred.weight", dim, n_exp));
            if (with_prev) {
                add_resident(make_f32(p + "ffn_gate_inp_pred_prev.weight", dim, n_exp));
            }
            layer_exps[layer][0] = make_f32(p + "ffn_gate_exps.weight", dim, dim, n_exp);
            layer_exps[layer][1] = make_f32(p + "ffn_up_exps.weight", dim, dim, n_exp);
            layer_exps[layer][2] = make_f32(p + "ffn_down_exps.weight", dim, dim, n_exp);
            for (int f = 0; f < 3; ++f) {
                gguf_add_tensor(g_unp.get(), layer_exps[layer][f]);
            }
        }
    }

    const size_t fam_bytes = size_t(dim) * dim * sizeof(float);
    nlohmann::json layers_json = nlohmann::json::array();
    for (int layer = 1; layer < n_layers; ++layer) {
        const std::string cname = "blk." + std::to_string(layer) + ".moe_exps_packed.weight";
        packed_bufs.emplace_back(size_t(stride) * n_exp);
        auto & buf = packed_bufs.back();
        const char * fnames[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        nlohmann::json fams_json = nlohmann::json::array();
        for (int f = 0; f < 3; ++f) {
            const uint8_t * src = static_cast<const uint8_t *>(layer_exps[layer][f]->data);
            for (int e = 0; e < n_exp; ++e) {
                std::memcpy(buf.data() + size_t(e) * stride + size_t(f) * fam_bytes, src + size_t(e) * fam_bytes, fam_bytes);
            }
            fams_json.push_back({
                {"tensor_family", fnames[f]},
                {"type", "F32"},
                {"offset_in_expert", size_t(f) * fam_bytes},
                {"bytes_per_expert", fam_bytes},
                {"ne", {dim, dim}},
            });
        }
        auto * carrier = ggml_new_tensor_4d(meta.get(), GGML_TYPE_I8, stride, 1, 1, n_exp);
        ggml_set_name(carrier, cname.c_str());
        carrier->data = buf.data();
        gguf_add_tensor(g_les.get(), carrier);
        layers_json.push_back({
            {"layer", layer},
            {"n_expert", n_exp},
            {"n_expert_used", n_used},
            {"groups", nlohmann::json::array({{
                {"expert_begin", 0},
                {"expert_count", n_exp},
                {"expert_stride", stride},
                {"tensor_name", cname},
                {"tensors", fams_json},
            }})},
        });
    }
    const nlohmann::json layout = {{"layout", "expert_major_packed"}, {"layers", layers_json}};
    gguf_set_val_u32(g_les.get(), "les.moe.version", 1);
    const char * cap = "moe.expert_major_packed_v1";
    gguf_set_arr_str(g_les.get(), "les.moe.requires", &cap, 1);
    gguf_set_val_str(g_les.get(), "les.moe.layout", layout.dump().c_str());

    check(gguf_write_to_file(g_les.get(), out_les.string().c_str(), false), "cannot write predictor LES GGUF");
    check(gguf_write_to_file(g_unp.get(), out_unpacked.string().c_str(), false), "cannot write predictor unpacked GGUF");
}

static void predictor_tests(ggml_backend_t device) {
    const auto dir = std::filesystem::temp_directory_path() /
        ("llama-test-les-pred-" + std::to_string(std::random_device{}()));
    check(std::filesystem::create_directory(dir), "cannot create predictor temp dir");
    struct cleanup {
        std::filesystem::path d;
        ~cleanup() { std::error_code ec; std::filesystem::remove_all(d, ec); }
    } guard{dir};

    std::filesystem::path les_prev, unp_prev, les_noprev, unp_noprev;
    write_predictor_models(dir, true, les_prev, unp_prev);
    write_predictor_models(dir, false, les_noprev, unp_noprev);

    using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    auto * dev = ggml_backend_get_device(device);
    const bool gpu = ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;

    if (!gpu) {
        for (bool with_prev : {false, true}) {
            const auto & les_path = with_prev ? les_prev : les_noprev;
            const auto & unp_path = with_prev ? unp_prev : unp_noprev;
            for (const char * dec_win : {"0", "3"}) {
                env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
                env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2");
                env_guard e_w("LLAMA_MOE_DECODE_WINDOW", dec_win);
                env_guard e_r("LLAMA_MOE_WINDOW_MAX_ROWS", "1");
                env_guard e_s("LLAMA_MOE_MATMUL_SPLIT", "0");

                auto mp = llama_model_default_params();
                mp.n_gpu_layers = 0;
                mp.use_mmap = false;
                model_ptr om(llama_model_load_from_file(unp_path.string().c_str(), mp), llama_model_free);
                mp.moe_slot_bank = 12;
                mp.moe_expert_predictor = true;
                model_ptr sm(llama_model_load_from_file(les_path.string().c_str(), mp), llama_model_free);
                check(om && sm && sm->moe_expert_predictor_enabled(), "CPU predictor model load failed");

                auto cp = llama_context_default_params();
                cp.n_ctx = 32;
                cp.n_batch = cp.n_ubatch = 2;
                cp.n_threads = cp.n_threads_batch = 2;
                context_ptr oracle(llama_init_from_model(om.get(), cp), llama_free);
                context_ptr ctx_a(llama_init_from_model(sm.get(), cp), llama_free);
                context_ptr ctx_b(llama_init_from_model(sm.get(), cp), llama_free);
                check(oracle && ctx_a && ctx_b, "CPU predictor context init failed");

                ggml_backend_t cpu_be = nullptr;
                auto * sched = ctx_a->get_sched();
                for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                    auto * b = ggml_backend_sched_get_backend(sched, i);
                    if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                        cpu_be = b;
                    }
                }
                check(cpu_be != nullptr, "CPU backend missing from sched");
                backend_probe probe(cpu_be);
                int n_mc1 = 0, n_mc2 = 0, n_set = 0, n_get = 0;
                probe.after_compute = [&](ggml_cgraph * g) {
                    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                        const auto * node = ggml_graph_node(g, i);
                        const auto op = node->op;
                        n_mc1 += op == GGML_OP_MAP_CUSTOM1;
                        n_mc2 += op == GGML_OP_MAP_CUSTOM2;
                        n_set += op == GGML_OP_SET_ROWS && node->src[2] &&
                            std::strcmp(node->src[2]->name, "moe_pred_prev_state") == 0;
                        n_get += op == GGML_OP_GET_ROWS && node->src[0] &&
                            std::strcmp(node->src[0]->name, "moe_pred_prev_state") == 0;
                    }
                };

                llama_token prefill[] = {1, 2};
                const std::array<llama_token, 4> dec = {3, 4, 5, 6};
                check(llama_decode(oracle.get(), llama_batch_get_one(prefill, 2)) == 0 &&
                    llama_decode(ctx_a.get(), llama_batch_get_one(prefill, 2)) == 0 &&
                    llama_decode(ctx_b.get(), llama_batch_get_one(prefill, 2)) == 0,
                    "CPU predictor prefill failed");
                check_logits_close(llama_get_logits(ctx_a.get()), llama_get_logits(oracle.get()), 128, "CPU pred prefill A");
                check_logits_close(llama_get_logits(ctx_b.get()), llama_get_logits(oracle.get()), 128, "CPU pred prefill B");
                for (size_t s = 0; s < dec.size(); ++s) {
                    llama_token t = dec[s];
                    check(llama_decode(oracle.get(), llama_batch_get_one(&t, 1)) == 0 &&
                        llama_decode(ctx_a.get(), llama_batch_get_one(&t, 1)) == 0 &&
                        llama_decode(ctx_b.get(), llama_batch_get_one(&t, 1)) == 0,
                        "CPU predictor decode failed");
                    check_logits_close(llama_get_logits(ctx_a.get()), llama_get_logits(oracle.get()), 128, "CPU pred dec A");
                    check_logits_close(llama_get_logits(ctx_b.get()), llama_get_logits(oracle.get()), 128, "CPU pred dec B");
                }
                check(n_mc1 > 0 && n_mc2 > 0, "CPU predictor did not execute lookahead custom ops");
                if (with_prev) {
                    check(n_set > 0 && n_get > 0, "CPU predictor _prev did not execute SET_ROWS/GET_ROWS");
                } else {
                    check(n_set == 0 && n_get == 0, "CPU predictor without _prev unexpectedly touched state rows");
                }
            }
        }

        // no_alloc memory accounting for predictor_state
        {
            env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
            env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2");
            env_guard e_w("LLAMA_MOE_DECODE_WINDOW", "0");
            auto measure = [&](const std::filesystem::path & path, bool pred_on) -> size_t {
                auto mp = llama_model_default_params();
                mp.n_gpu_layers = 0;
                mp.use_mmap = false;
                mp.no_alloc = true;
                mp.moe_slot_bank = 12;
                mp.moe_expert_predictor = pred_on;
                model_ptr m(llama_model_load_from_file(path.string().c_str(), mp), llama_model_free);
                check(bool(m), "CPU predictor no_alloc model failed");
                auto cp = llama_context_default_params();
                cp.n_ctx = 32;
                cp.n_batch = cp.n_ubatch = 2;
                context_ptr c(llama_init_from_model(m.get(), cp), llama_free);
                check(bool(c), "CPU predictor no_alloc context failed");
                size_t total = 0;
                for (const auto & item : llama_get_memory_breakdown(c.get())) {
                    total += item.second.context;
                }
                return total;
            };
            const size_t expected_state = size_t(dim) * (2 * 3) * sizeof(float);
            check(measure(les_prev, true) - measure(les_prev, false) == expected_state,
                "no_alloc omitted predictor_state bytes");
            check(measure(les_noprev, true) == measure(les_noprev, false),
                "no_alloc without _prev unexpectedly allocated predictor_state");
        }

        // Exception capture in map_custom_lookahead_only
        {
            env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
            env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "1");
            auto mp = llama_model_default_params();
            mp.n_gpu_layers = 0;
            mp.use_mmap = false;
            mp.moe_slot_bank = 12;
            mp.moe_expert_predictor = true;
            model_ptr m(llama_model_load_from_file(les_prev.string().c_str(), mp), llama_model_free);
            check(bool(m), "exception capture model load failed");
            llama_moe_slot_runtime runtime(*m);
            ggml_context_ptr gctx(ggml_init({16 * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(16, false), nullptr, true}));
            auto * bad_pred = ggml_new_tensor_1d(gctx.get(), GGML_TYPE_F32, 2);
            auto * out = runtime.build_lookahead_only(gctx.get(), bad_pred, 1);
            check(out != nullptr, "build_lookahead_only returned null");
            auto * gf = ggml_new_graph_custom(gctx.get(), 16, false);
            ggml_build_forward_expand(gf, out);
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(gctx.get(), device));
            check(bool(buf), "exception test alloc failed");
            check(ggml_backend_graph_compute(device, gf) == GGML_STATUS_SUCCESS, "graph compute failed");
            check(runtime.failed(), "map_custom_lookahead_only did not capture type error");
            std::printf("predictor exception capture: runtime.failed()=1\n");
        }
        return;
    }

    // Non-UMA GPU path
    auto * default_buft = ggml_backend_dev_buffer_type(dev);
    check(!ggml_backend_buft_is_host_addressable(default_buft),
        "predictor GPU test requires a non-UMA GPU backend");
    ggml_backend_dev_t devs[] = {dev, nullptr};

    // Subtest 1: staged full GPU offload rejects lookahead=1, succeeds with lookahead=0, rejects runtime toggle
    {
        env_guard e_w("LLAMA_MOE_DECODE_WINDOW", "0");
        env_guard e_s("LLAMA_MOE_MATMUL_SPLIT", "0");
        {
            env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
            env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2");
            auto mp = llama_model_default_params();
            mp.devices = devs;
            mp.n_gpu_layers = 99;
            mp.use_mmap = false;
            mp.moe_slot_bank = 12;
            mp.moe_expert_predictor = true;
            model_ptr sm(llama_model_load_from_file(les_prev.string().c_str(), mp), llama_model_free);
            check(bool(sm), "staged GPU predictor model load failed");
            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            std::string log;
            llama_log_set([](ggml_log_level, const char * text, void * ud) {
                static_cast<std::string *>(ud)->append(text);
            }, &log);
            context_ptr rejected(llama_init_from_model(sm.get(), cp), llama_free);
            llama_log_set(nullptr, nullptr);
            check(rejected == nullptr, "staged GPU unexpectedly allowed lookahead prefetch at init");
        }
        {
            env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "0");
            auto mp = llama_model_default_params();
            mp.devices = devs;
            mp.n_gpu_layers = 99;
            mp.use_mmap = false;
            const llama_model_tensor_buft_override cpu_exps[] = {llm_ffn_exps_cpu_override(), {nullptr, nullptr}};
            mp.tensor_buft_overrides = cpu_exps;
            model_ptr om(llama_model_load_from_file(unp_prev.string().c_str(), mp), llama_model_free);
            mp.tensor_buft_overrides = nullptr;
            mp.moe_slot_bank = 12;
            mp.moe_expert_predictor = true;
            model_ptr sm(llama_model_load_from_file(les_prev.string().c_str(), mp), llama_model_free);
            check(om && sm, "staged GPU predictor (lookahead=0) model load failed");

            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            context_ptr oracle(llama_init_from_model(om.get(), cp), llama_free);
            context_ptr ctx(llama_init_from_model(sm.get(), cp), llama_free);
            check(oracle && ctx, "staged GPU predictor context init failed");
            llama_token prefill[] = {1, 2};
            check(llama_decode(oracle.get(), llama_batch_get_one(prefill, 2)) == 0 &&
                llama_decode(ctx.get(), llama_batch_get_one(prefill, 2)) == 0,
                "staged GPU prefill failed");
            check_logits_close(llama_get_logits(ctx.get()), llama_get_logits(oracle.get()), 128, "staged GPU prefill");
            for (llama_token tok : {3, 4, 5}) {
                check(llama_decode(oracle.get(), llama_batch_get_one(&tok, 1)) == 0 &&
                    llama_decode(ctx.get(), llama_batch_get_one(&tok, 1)) == 0,
                    "staged GPU decode failed");
                check_logits_close(llama_get_logits(ctx.get()), llama_get_logits(oracle.get()), 128, "staged GPU dec");
            }
            bool threw = false;
            try { llama_set_moe_lookahead_prefetch_depth(ctx.get(), 1); }
            catch (const std::runtime_error &) { threw = true; }
            check(threw, "staged GPU unexpectedly allowed runtime lookahead toggle");
            llama_token extra = 6;
            check(llama_decode(oracle.get(), llama_batch_get_one(&extra, 1)) == 0 &&
                llama_decode(ctx.get(), llama_batch_get_one(&extra, 1)) == 0,
                "staged GPU decode after rejected toggle failed");
            check_logits_close(llama_get_logits(ctx.get()), llama_get_logits(oracle.get()), 128, "staged GPU post-toggle");
        }
    }

    // Subtest 2: hybrid non-UMA GPU predictor + CPU experts with lookahead ON
    {
        const llama_model_tensor_buft_override pred_gpu[] = {
            {"ffn_gate_inp_pred", default_buft},
            {nullptr, nullptr},
        };
        for (const char * dec_win : {"0", "3"}) {
            env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
            env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2");
            env_guard e_w("LLAMA_MOE_DECODE_WINDOW", dec_win);
            env_guard e_r("LLAMA_MOE_WINDOW_MAX_ROWS", "1");
            env_guard e_s("LLAMA_MOE_MATMUL_SPLIT", "0");

            auto mp = llama_model_default_params();
            mp.devices = devs;
            mp.n_gpu_layers = 0;
            mp.use_mmap = false;
            model_ptr om(llama_model_load_from_file(unp_prev.string().c_str(), mp), llama_model_free);
            mp.moe_slot_bank = 12;
            mp.moe_expert_predictor = true;
            mp.tensor_buft_overrides = pred_gpu;
            model_ptr sm(llama_model_load_from_file(les_prev.string().c_str(), mp), llama_model_free);
            check(om && sm && sm->moe_expert_predictor_enabled(), "hybrid GPU predictor model load failed");
            check(sm->moe_expert_predictor_buft() == default_buft &&
                !ggml_backend_buft_is_host_addressable(sm->moe_expert_predictor_buft()),
                "hybrid predictor buft is not non-UMA GPU");

            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            context_ptr oracle(llama_init_from_model(om.get(), cp), llama_free);
            context_ptr ctx(llama_init_from_model(sm.get(), cp), llama_free);
            check(oracle && ctx, "hybrid GPU predictor context init failed");

            ggml_backend_t gpu_be = nullptr, cpu_be = nullptr;
            auto * sched = ctx->get_sched();
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                auto * b = ggml_backend_sched_get_backend(sched, i);
                if (ggml_backend_get_device(b) == dev) { gpu_be = b; }
                if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    cpu_be = b;
                }
            }
            check(gpu_be && cpu_be, "hybrid GPU predictor missing backend in sched");
            backend_probe gp(gpu_be), cp_probe(cpu_be);
            int gpu_set = 0, gpu_get = 0, gpu_mm = 0, cpu_mc1 = 0, cpu_mc2 = 0;
            gp.after_compute = [&](ggml_cgraph * g) {
                for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                    const auto * node = ggml_graph_node(g, i);
                    const auto op = node->op;
                    gpu_set += op == GGML_OP_SET_ROWS && node->src[2] &&
                        std::strcmp(node->src[2]->name, "moe_pred_prev_state") == 0;
                    gpu_get += op == GGML_OP_GET_ROWS && node->src[0] &&
                        std::strcmp(node->src[0]->name, "moe_pred_prev_state") == 0;
                    gpu_mm  += op == GGML_OP_MUL_MAT;
                }
            };
            cp_probe.after_compute = [&](ggml_cgraph * g) {
                for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                    const auto op = ggml_graph_node(g, i)->op;
                    cpu_mc1 += op == GGML_OP_MAP_CUSTOM1;
                    cpu_mc2 += op == GGML_OP_MAP_CUSTOM2;
                }
            };

            llama_token prefill[] = {1, 2};
            check(llama_decode(oracle.get(), llama_batch_get_one(prefill, 2)) == 0 &&
                llama_decode(ctx.get(), llama_batch_get_one(prefill, 2)) == 0,
                "hybrid GPU predictor prefill failed");
            check_logits_close(llama_get_logits(ctx.get()), llama_get_logits(oracle.get()), 128, "hybrid GPU pred prefill");
            for (llama_token tok : {3, 4, 5, 6}) {
                check(llama_decode(oracle.get(), llama_batch_get_one(&tok, 1)) == 0 &&
                    llama_decode(ctx.get(), llama_batch_get_one(&tok, 1)) == 0,
                    "hybrid GPU predictor decode failed");
                check_logits_close(llama_get_logits(ctx.get()), llama_get_logits(oracle.get()), 128, "hybrid GPU pred dec");
            }
            check(gpu_set > 0 && gpu_get > 0 && gpu_mm > 0,
                "hybrid GPU predictor did not execute SET_ROWS/GET_ROWS/MUL_MAT on GPU");
            check(cpu_mc1 > 0 && cpu_mc2 > 0,
                "hybrid GPU predictor did not execute lookahead custom ops on CPU");
        }

        // Hybrid no_alloc memory breakdown check on non-UMA GPU buft
        env_guard e_pf("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1");
        env_guard e_d("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2");
        env_guard e_w("LLAMA_MOE_DECODE_WINDOW", "0");
        auto measure_dev = [&](bool pred_on) -> size_t {
            auto mp = llama_model_default_params();
            mp.devices = devs;
            mp.n_gpu_layers = 0;
            mp.use_mmap = false;
            mp.no_alloc = true;
            mp.moe_slot_bank = 12;
            mp.moe_expert_predictor = pred_on;
            mp.tensor_buft_overrides = pred_gpu;
            model_ptr m(llama_model_load_from_file(les_prev.string().c_str(), mp), llama_model_free);
            check(bool(m), "hybrid no_alloc model failed");
            auto cp = llama_context_default_params();
            cp.n_ctx = 32;
            cp.n_batch = cp.n_ubatch = 2;
            context_ptr c(llama_init_from_model(m.get(), cp), llama_free);
            check(bool(c), "hybrid no_alloc context failed");
            for (const auto & item : llama_get_memory_breakdown(c.get())) {
                if (item.first == default_buft) {
                    return item.second.context;
                }
            }
            return 0;
        };
        const size_t expected_state = size_t(dim) * (2 * 3) * sizeof(float);
        check(measure_dev(true) - measure_dev(false) == expected_state,
            "hybrid GPU no_alloc omitted predictor_state context bytes");
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        std::string test_case = "source", backend = "CPU", packed, reference;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            check(i + 1 < argc, "expected --case NAME or --backend DEVICE");
            if (arg == "--case") {
                test_case = argv[++i];
            } else if (arg == "--backend") {
                backend = argv[++i];
            } else if (arg == "--model") {
                packed = argv[++i];
            } else if (arg == "--reference") {
                reference = argv[++i];
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        check(test_case == "source" || test_case == "cpu-packed" || test_case == "device-packed" ||
            test_case == "scheduler" || test_case == "scheduler-boundaries" || test_case == "model" ||
            test_case == "quantizer" || test_case == "quantizer-inputs" ||
            test_case == "gpu-decode-rollback" || test_case == "gpu-abort-rollback" || test_case == "shared-a2-abort" ||
            test_case == "no-alloc" || test_case == "tiny-no-alloc" || test_case == "layout-metadata" ||
            test_case == "invalid-layout" || test_case == "carrier-bounds" ||
            test_case == "runtime-rollback" ||
            test_case == "deferred-rollback" || test_case == "prefetch-rollback" || test_case == "prewarm-rollback" ||
            test_case == "runtime-decode-rollback" || test_case == "runtime-abort-rollback" ||
            test_case == "prewarm-churn" || test_case == "a2-contract" || test_case == "a2-prefetch" ||
            test_case == "late-prefetch" || test_case == "late-prefetch-prefill" || test_case == "late-prefetch-partition" ||
            test_case == "packed-inplace" || test_case == "quantizer-per-layer-ffn" || test_case == "predictor",
            "unknown test case");
        ggml_backend_load_all();
        if (test_case == "quantizer") { quantizer_product(); return 0; }
        if (test_case == "quantizer-inputs") { quantizer_inputs(); return 0; }
        if (test_case == "quantizer-per-layer-ffn") { quantizer_per_layer_ffn(); return 0; }
        ggml_backend_ptr device(ggml_backend_init_by_name(backend.c_str(), nullptr));
        check(bool(device), "requested backend is absent (not a passing hardware test)");
        if (test_case == "packed-inplace") { packed_inplace(device.get()); return 0; }
        if (test_case == "predictor") { predictor_tests(device.get()); return 0; }
        if (test_case == "invalid-layout") { invalid_layout(); return 0; }
        if (test_case == "carrier-bounds") { carrier_bounds(); return 0; }
        if (test_case == "layout-metadata") { layout_metadata(); return 0; }
        if (test_case == "tiny-no-alloc") { tiny_no_alloc(device.get()); return 0; }
        if (test_case == "scheduler-boundaries") { scheduler_boundaries(device.get()); return 0; }
        if (test_case == "shared-a2-abort") { shared_a2_abort(device.get()); return 0; }
        if (test_case == "gpu-decode-rollback" || test_case == "gpu-abort-rollback") {
            gpu_rollback(device.get(), test_case == "gpu-abort-rollback");
            return 0;
        }
        if (test_case == "device-packed" || test_case == "scheduler") {
            check(ggml_backend_dev_type(ggml_backend_get_device(device.get())) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                ggml_backend_dev_type(ggml_backend_get_device(device.get())) == GGML_BACKEND_DEVICE_TYPE_IGPU,
                "device-packed requires a real GPU backend");
        }
        if (test_case == "model") {
            run_model(packed, reference, device.get());
            return 0;
        }
        if (test_case == "runtime-rollback" || test_case == "deferred-rollback" ||
            test_case == "prefetch-rollback" || test_case == "prewarm-rollback" ||
            test_case == "runtime-decode-rollback" || test_case == "runtime-abort-rollback" ||
            test_case == "prewarm-churn" || test_case == "a2-contract" || test_case == "a2-prefetch" ||
            test_case == "late-prefetch" || test_case == "late-prefetch-prefill" || test_case == "late-prefetch-partition") {
            if (test_case == "runtime-rollback" || test_case == "deferred-rollback") { runtime_rollback(); }
            else if (test_case == "prewarm-rollback") { prewarm_rollback(); }
            else if (test_case == "prefetch-rollback") { prefetch_rollback(); }
            else if (test_case == "runtime-decode-rollback" || test_case == "runtime-abort-rollback") {
                runtime_decode_rollback(test_case == "runtime-abort-rollback");
            }
            else if (test_case == "prewarm-churn") { prewarm_churn(); }
            else if (test_case == "a2-contract") { a2_contract(); }
            else if (test_case == "a2-prefetch") { a2_prefetch(); }
            else { late_prefetch(test_case != "late-prefetch-prefill", test_case == "late-prefetch-partition"); }
            return 0;
        }
        if (test_case == "no-alloc") {
            no_alloc_model(packed, device.get());
            return 0;
        }
        for (const bool fused : {false, true}) {
            for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_K}) {
                std::printf("case=%s fused=%d type=%s\n", test_case.c_str(), fused, ggml_type_name(type));
                fixture f(fused, type);
                f.verify();
                if (test_case != "source") {
                    run(f, device.get(), test_case == "scheduler");
                }
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-les: %s\n", error.what());
        return 1;
    }
}

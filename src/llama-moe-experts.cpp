#include "llama-moe-experts.h"

#include "llama-impl.h"
#include "llama-hparams.h"

#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

ggml_type parse_type(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return (char) std::tolower(value);
    });

    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = (ggml_type) value;
        const char * type_name = ggml_type_name(type);
        if (!type_name) {
            continue;
        }
        std::string candidate(type_name);
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char item) {
            return (char) std::tolower(item);
        });
        if (candidate == name) {
            return type;
        }
    }

    throw std::runtime_error(format("unknown MoE experts quant type '%s'", name.c_str()));
}

// ---------------------------------------------------------------------------------------------
// Source: the layout declared by the model file itself
// ---------------------------------------------------------------------------------------------

constexpr const char * LES_LAYOUT_KIND   = "expert_major_packed";
// 一个 family 在一个专家内的对齐。K 量化的 block 是 32 元素，所以字节数天然是 32 的
// 倍数；引擎侧的 slot 读取（一条 preadv 读一个专家）依赖这个前提。
constexpr size_t      LES_TENSOR_ALIGN   = 32;

std::string layer_tensor_name(int32_t layer, const std::string & tensor_family) {
    return format("blk.%d.%s.weight", layer, tensor_family.c_str());
}

uint64_t layout_uint(const nlohmann::json & value, const char * field,
                     uint64_t maximum = std::numeric_limits<size_t>::max()) {
    if (!value.is_number_integer() ||
            (!value.is_number_unsigned() && value.get<int64_t>() < 0) ||
            value.get<uint64_t>() > maximum) {
        throw std::runtime_error(format("expert layout: '%s' must be a nonnegative integer <= %llu",
            field, (unsigned long long) maximum));
    }
    return value.get<uint64_t>();
}

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("expert layout: byte offset/size overflow");
    }
    return a + b;
}

size_t checked_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error("expert layout: byte size overflow");
    }
    return a * b;
}

} // namespace

std::unique_ptr<llama_moe_experts> llama_moe_experts::from_gguf(
        const gguf_context * metadata, const std::string & path_model, int32_t n_layers) {
    const int64_t key = gguf_find_key(metadata, "les.moe.layout");
    if (key < 0 || gguf_get_kv_type(metadata, key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("expert layout: 'les.moe.layout' must be a string");
    }
    // Older v1 writers omitted version/requires. Absence retains that v1
    // interpretation; an explicit unknown version or capability never does.
    const int64_t version = gguf_find_key(metadata, "les.moe.version");
    if (version >= 0 && (gguf_get_kv_type(metadata, version) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(metadata, version) != 1)) {
        throw std::runtime_error("expert layout: unsupported 'les.moe.version' (expected uint32 1)");
    }
    const int64_t key_requires = gguf_find_key(metadata, "les.moe.requires");
    if (key_requires >= 0) {
        if (gguf_get_kv_type(metadata, key_requires) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(metadata, key_requires) != GGUF_TYPE_STRING) {
            throw std::runtime_error("expert layout: 'les.moe.requires' must be a string array");
        }
        for (size_t i = 0; i < gguf_get_arr_n(metadata, key_requires); ++i) {
            const std::string capability = gguf_get_arr_str(metadata, key_requires, i);
            if (capability != "moe.expert_major_packed_v1") {
                throw std::runtime_error(format("expert layout: unsupported required capability '%s'",
                    capability.c_str()));
            }
        }
    }
    auto out = from_layout(gguf_get_val_str(metadata, key), path_model);
    if (n_layers <= 0 || out->bytes_per_slot_layer.size() > (size_t) n_layers) {
        throw std::runtime_error("expert layout: layer index exceeds model layer count");
    }
    const size_t alignment = gguf_get_alignment(metadata);
    if (alignment < 2048 || (alignment & (alignment - 1)) ||
            (out->alignment && out->alignment != alignment)) {
        throw std::runtime_error("expert layout: alignment must match the page-aligned GGUF layout");
    }
    out->alignment = alignment;
    return out;
}

std::unique_ptr<llama_moe_experts> llama_moe_experts::from_layout(
        const std::string & layout_json, const std::string & path_model) {
    nlohmann::json layout;
    try {
        layout = nlohmann::json::parse(layout_json);
    } catch (const std::exception & e) {
        throw std::runtime_error(format("failed to parse the expert layout: %s", e.what()));
    }

    if (layout.value("layout", std::string()) != LES_LAYOUT_KIND) {
        throw std::runtime_error(format(
            "expert layout '%s' is not '%s'",
            layout.value("layout", std::string()).c_str(), LES_LAYOUT_KIND));
    }
    if (!layout.contains("layers") || !layout.at("layers").is_array() || layout.at("layers").empty()) {
        throw std::runtime_error("expert layout declares no layers");
    }

    auto out = std::unique_ptr<llama_moe_experts>(new llama_moe_experts());
    if (layout.contains("les_version") && layout_uint(layout.at("les_version"), "les_version") != 1) {
        throw std::runtime_error("expert layout: unsupported JSON les_version");
    }
    if (layout.contains("alignment")) {
        out->alignment = layout_uint(layout.at("alignment"), "alignment");
        if (out->alignment < 2048 || (out->alignment & (out->alignment - 1))) {
            throw std::runtime_error("expert layout: alignment must be a power of two >= 2048");
        }
    }

    // 每一层的声明都是权威；这里只保留一个全局副本，用来给 slot 记账提供单一几何。
    int32_t n_expert      = -1;
    int32_t n_expert_used = -1;
    std::unordered_set<int32_t> layers;
    std::unordered_set<std::string> carriers;

    for (const auto & item : layout.at("layers")) {
        const int32_t layer = layout_uint(item.at("layer"), "layer", LLAMA_MAX_LAYERS - 1);
        const int32_t layer_experts = layout_uint(item.at("n_expert"), "n_expert", INT32_MAX);
        const auto & defaults = layout.contains("defaults") ? layout.at("defaults") : nlohmann::json::object();
        const auto used = item.contains("n_expert_used") ? item.at("n_expert_used") :
            defaults.value("n_expert_used", nlohmann::json(layer_experts));
        const int32_t layer_used = layout_uint(used, "n_expert_used", INT32_MAX);

        if (layer < 0 || layer_experts <= 0 || layer_used <= 0 || layer_used > layer_experts) {
            throw std::runtime_error(format(
                "expert layout: invalid expert counts for layer %d", layer));
        }
        if (!layers.insert(layer).second) {
            throw std::runtime_error(format("expert layout: duplicate layer %d", layer));
        }
        if (n_expert < 0) {
            n_expert      = layer_experts;
            n_expert_used = layer_used;
        } else if (layer_experts != n_expert || layer_used != n_expert_used) {
            // 这个构建要求所有层用同一个专家数：计算图的 expert 轴、slot 记账和预读
            // 都是按单值分配的。不一致时明确拒绝，而不是静默地算出错的结果。
            throw std::runtime_error(format(
                "expert layout: layer %d has expert counts/top-k %d/%d, expected %d/%d; "
                "this build requires uniform expert counts and top-k across layers",
                layer, layer_experts, layer_used, n_expert, n_expert_used));
        }

        if (!item.contains("groups") || !item.at("groups").is_array() ||
                item.at("groups").empty()) {
            throw std::runtime_error(format("expert layout: layer %d has no groups", layer));
        }
        if (item.at("groups").size() > 1) {
            // 一个 group 对应一族固定的 slot 张量几何。这个构建只支持每层一个 group。
            throw std::runtime_error(format(
                "expert layout: layer %d has %zu groups; this build supports one group per layer",
                layer, item.at("groups").size()));
        }

        size_t  layer_slot_bytes = 0;
        int32_t covered          = 0;
        for (const auto & group : item.at("groups")) {
            const int32_t expert_begin = layout_uint(group.at("expert_begin"), "expert_begin", INT32_MAX);
            const int32_t group_count = layout_uint(group.at("expert_count"), "expert_count", INT32_MAX);
            const size_t expert_stride = layout_uint(group.at("expert_stride"), "expert_stride");
            const std::string tensor_name   = group.at("tensor_name").get<std::string>();

            if (expert_begin != covered || group_count <= 0 ||
                    expert_begin > layer_experts || group_count > layer_experts - expert_begin) {
                throw std::runtime_error(format(
                    "expert layout: layer %d group [%d,+%d) does not cover experts contiguously "
                    "(covered up to %d of %d)",
                    layer, expert_begin, group_count, covered, layer_experts));
            }
            if (expert_stride == 0 || expert_stride % LES_TENSOR_ALIGN) {
                throw std::runtime_error(format(
                    "expert layout: layer %d expert_stride %zu is not a multiple of %zu",
                    layer, expert_stride, LES_TENSOR_ALIGN));
            }

            if (tensor_name.empty() || !carriers.insert(tensor_name).second) {
                throw std::runtime_error("expert layout: carrier tensor must be unique to one group");
            }
            checked_mul(expert_stride, (size_t) group_count);
            if (!group.at("tensors").is_array() || group.at("tensors").empty()) {
                throw std::runtime_error("expert layout: group declares no tensor families");
            }
            std::vector<std::pair<size_t, size_t>> intervals;
            for (const auto & tensor : group.at("tensors")) {
                const std::string tensor_family    = tensor.at("tensor_family").get<std::string>();
                if (tensor_family.empty()) {
                    throw std::runtime_error("expert layout: tensor family name must not be empty");
                }
                const size_t offset_in_expert = layout_uint(tensor.at("offset_in_expert"), "offset_in_expert");
                const size_t bytes_per_expert = layout_uint(tensor.at("bytes_per_expert"), "bytes_per_expert");

                if (bytes_per_expert == 0 || offset_in_expert % LES_TENSOR_ALIGN ||
                        bytes_per_expert % LES_TENSOR_ALIGN) {
                    throw std::runtime_error(format(
                        "expert layout: layer %d tensor '%s' is not %zu-byte aligned",
                        layer, tensor_family.c_str(), LES_TENSOR_ALIGN));
                }
                if (offset_in_expert > expert_stride || bytes_per_expert > expert_stride - offset_in_expert) {
                    throw std::runtime_error(format(
                        "expert layout: layer %d tensor '%s' runs past expert_stride "
                        "(%zu + %zu > %zu)",
                        layer, tensor_family.c_str(), offset_in_expert, bytes_per_expert,
                        expert_stride));
                }

                llama_moe_experts_entry entry;
                entry.layer               = layer;
                entry.tensor_family       = tensor_family;
                entry.tensor_name         = layer_tensor_name(layer, tensor_family);
                entry.packed_tensor_name  = tensor_name;
                entry.offset_in_expert    = offset_in_expert;
                entry.path                = path_model;
                entry.type                = parse_type(tensor.at("type").get<std::string>());
                entry.offset              = offset_in_expert; // bind_region() 之后重定基
                entry.bytes_per_expert    = bytes_per_expert;
                entry.expert_stride       = expert_stride;
                entry.expert_count        = layer_experts;
                entry.tensor_bytes        = checked_mul(bytes_per_expert, (size_t) layer_experts);

                if (!out->entries.emplace(entry.tensor_name, std::move(entry)).second) {
                    throw std::runtime_error(format(
                        "expert layout: duplicate tensor '%s'",
                        layer_tensor_name(layer, tensor_family).c_str()));
                }
                intervals.emplace_back(offset_in_expert, offset_in_expert + bytes_per_expert);
            }
            std::sort(intervals.begin(), intervals.end());
            size_t end = 0;
            for (const auto & interval : intervals) {
                if (interval.first != end) {
                    throw std::runtime_error("expert layout: tensor families overlap or leave a gap");
                }
                end = interval.second;
            }
            if (end != expert_stride) {
                throw std::runtime_error("expert layout: tensor families do not fill expert_stride");
            }
            layer_slot_bytes = checked_add(layer_slot_bytes, expert_stride);
            covered = expert_begin + group_count;
        }
        if (covered != layer_experts) {
            throw std::runtime_error(format(
                "expert layout: layer %d groups cover %d of %d experts",
                layer, covered, layer_experts));
        }

        if ((int32_t) out->bytes_per_slot_layer.size() <= layer) {
            out->bytes_per_slot_layer.resize((size_t) layer + 1, 0);
        }
        out->bytes_per_slot_layer[(size_t) layer] = layer_slot_bytes;
    }

    out->n_expert      = n_expert;
    out->n_expert_used = n_expert_used;
    out->n_layer       = (int32_t) layers.size();
    for (size_t bytes : out->bytes_per_slot_layer) {
        out->bytes_per_slot = checked_add(out->bytes_per_slot, bytes);
    }

    if (out->entries.empty()) {
        throw std::runtime_error("expert layout declares no expert tensors");
    }
    return out;
}

void llama_moe_experts::bind_region(const std::string & packed_tensor_name,
                                    size_t tensor_offset,
                                    size_t tensor_bytes,
                                    const std::string & path) {
    // 载体张量在文件里的位置一旦解析出来，就可以给引用它的每条 entry 定绝对偏移。
    // 同一层三个 family 共用一个载体，所以 base 相同、只有 offset_in_expert 不同 ——
    // `finalize_pack` 要求的「同文件、同 stride、族在 stride 内」由此自动成立。
    int bound = 0;
    checked_add(tensor_offset, tensor_bytes);
    if (alignment && tensor_offset % alignment) {
        throw std::runtime_error("expert layout: carrier offset is not aligned");
    }
    for (const auto & kv : entries) {
        const auto & entry = kv.second;
        if (entry.packed_tensor_name == packed_tensor_name) {
            const size_t expected = checked_mul(entry.expert_stride, (size_t) entry.expert_count);
            if (tensor_bytes != expected) {
                throw std::runtime_error(format("expert layout: carrier '%s' has %zu bytes, expected %zu",
                    packed_tensor_name.c_str(), tensor_bytes, expected));
            }
            ++bound;
        }
    }
    if (bound == 0) {
        throw std::runtime_error(format(
            "expert layout references tensor '%s' but no entry uses it", packed_tensor_name.c_str()));
    }
    for (auto & kv : entries) {
        auto & entry = kv.second;
        if (entry.packed_tensor_name != packed_tensor_name) {
            continue;
        }
        entry.offset = tensor_offset + entry.offset_in_expert;
        entry.path   = path;
    }
}

std::vector<std::string> llama_moe_experts::region_names() const {
    std::vector<std::string> names;
    names.reserve(entries.size());
    for (const auto & kv : entries) {
        const auto & name = kv.second.packed_tensor_name;
        if (name.empty()) {
            continue;
        }
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

const llama_moe_experts_entry * llama_moe_experts::find(const char * tensor_name) const {    const auto found = entries.find(tensor_name);
    return found == entries.end() ? nullptr : &found->second;
}

int32_t llama_moe_experts::expert_count() const {
    return n_expert;
}

int32_t llama_moe_experts::expert_used_count() const {
    return n_expert_used;
}

int32_t llama_moe_experts::layer_count() const {
    return n_layer;
}

size_t llama_moe_experts::bytes_per_slot_all_layers() const {
    return bytes_per_slot;
}

const std::vector<size_t> & llama_moe_experts::bytes_per_slot_per_layer() const {
    return bytes_per_slot_layer;
}

// ---------------------------------------------------------------------------------------------
// Reading the geometry without loading the model
// ---------------------------------------------------------------------------------------------

bool llama_moe_experts_geometry_from_file(const char * path_model,
                                          struct llama_moe_experts_geometry * out) {
    if (!path_model || !out) {
        return false;
    }
    // 只要 metadata：no_alloc 让 gguf 不去碰张量数据，所以对十几 GiB 的文件也只是
    // 一次 header 读。
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path_model, params);
    if (!ctx) {
        return false;
    }

    bool ok = false;
    const int key_layout = gguf_find_key(ctx, "les.moe.layout");
    if (key_layout >= 0) {
        try {
            const auto source = llama_moe_experts::from_gguf(ctx, path_model, LLAMA_MAX_LAYERS);
            out->n_layer        = source->layer_count();
            out->n_expert       = source->expert_count();
            out->n_expert_used  = source->expert_used_count();
            out->bytes_per_slot = source->bytes_per_slot_all_layers();
            ok = true;
        } catch (const std::exception &) {
            ok = false;
        }
    }

    gguf_free(ctx);
    return ok;
}

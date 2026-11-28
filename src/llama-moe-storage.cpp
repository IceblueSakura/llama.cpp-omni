#include "llama-moe-storage.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

llama_moe_storage::llama_moe_storage(size_t n_tensors, bool no_alloc) : no_alloc(no_alloc) {
    ctx.reset(ggml_init({ggml_tensor_overhead() * n_tensors, nullptr, true}));
    if (!ctx) {
        throw std::runtime_error("failed to create LES storage metadata");
    }
}

ggml_tensor * llama_moe_storage::clone(const ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
    if (!tensor) {
        return nullptr;
    }
    // Resolve the storage type before deduplicating shared roots: CPU layers and
    // Metal shared layers must not accidentally alias a CPU-only allocation.
    if (!ggml_backend_buft_is_host_addressable(buft)) {
        staged = true;
        buft = ggml_backend_cpu_buffer_type();
    }
    auto & by_type = clones[buft];
    const auto it = by_type.find(tensor);
    if (it != by_type.end()) {
        return it->second;
    }
    ggml_tensor * base = tensor->view_src ? clone(tensor->view_src, buft) : nullptr;
    auto * result = ggml_dup_tensor(ctx.get(), tensor);
    std::memcpy(result->nb, tensor->nb, sizeof(result->nb));
    ggml_set_name(result, ggml_get_name(tensor));
    if (base) {
        result->view_src = base;
        result->view_offs = tensor->view_offs;
        if (no_alloc) {
            result->buffer = base->buffer;
        } else if (ggml_backend_view_init(result) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize LES bank view");
        }
    } else {
        // A device address is not a preadv destination. The selected host/shared
        // allocation is bounded by the template geometry, never by model size.
        const size_t size = ggml_backend_buft_get_alloc_size(buft, result);
        ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, no_alloc ? 0 : size));
        if (!buffer) {
            throw std::runtime_error("failed to allocate LES bank");
        }
        if (no_alloc) {
            result->buffer = buffer.get();
        } else if (ggml_backend_tensor_alloc(buffer.get(), result,
                ggml_backend_buffer_get_base(buffer.get())) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize LES bank");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Selective uploads can include backend padding after a requested expert.
        if (!no_alloc) {
            ggml_backend_buffer_clear(buffer.get(), 0);
        }
        sizes[buft] += size;
        buffers.push_back(std::move(buffer));
    }
    tensors.emplace(tensor, result);
    by_type.emplace(tensor, result);
    owned.insert(result);
    return result;
}

ggml_tensor * llama_moe_storage::get(const ggml_tensor * tensor) const {
    const auto it = tensors.find(tensor);
    return it == tensors.end() ? nullptr : it->second;
}

bool llama_moe_storage::owns(const ggml_tensor * tensor) const {
    return owned.count(tensor) != 0;
}

size_t llama_moe_storage::size_bytes() const {
    size_t size = 0;
    for (const auto & item : sizes) {
        size += item.second;
    }
    return size;
}

bool llama_moe_storage::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    if (!owns(src)) {
        return false;
    }
    // G1 deliberately uses a synchronous upload boundary. No I/O worker touches
    // backend state, and neither the source nor destination is reused in flight.
    ggml_backend_synchronize(backend);
    std::vector<bool> used(src->ne[2], false);
    bool found = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
            continue;
        }
        found = true;
        const auto * ids = node->src[2];
        std::vector<uint8_t> data(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, data.data(), 0, data.size());
        for (int64_t row = 0; row < ids->ne[1]; ++row) {
            for (int64_t k = 0; k < ids->ne[0]; ++k) {
                int32_t id;
                std::memcpy(&id, data.data() + row * ids->nb[1] + k * ids->nb[0], sizeof(id));
                GGML_ASSERT(id >= 0 && id < src->ne[2]);
                used[id] = true;
            }
        }
    }
    // LES weights must be consumed by an indirect matmul in this split. A full
    // copy here would hide a graph binding error and copy uninstalled slots.
    GGML_ASSERT(found);
    for (int64_t first = 0; first < src->ne[2];) {
        if (!used[first]) {
            ++first;
            continue;
        }
        int64_t end = first + 1;
        while (end < src->ne[2] && used[end]) {
            ++end;
        }
        const size_t offset = first * src->nb[2];
        const size_t padding = end < src->ne[2] ? std::min<size_t>(src->nb[2], 512) : 0;
        const size_t bytes = (end - first) * src->nb[2] + padding;
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, bytes);
        n_uploaded += bytes;
        first = end;
    }
    ggml_backend_synchronize(backend);
    return true;
}

#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Context-owned LES storage. Model tensors are metadata templates only. Shared
// backends read directly into these banks; other backends consume scheduler copies.
class llama_moe_storage {
public:
    explicit llama_moe_storage(size_t n_tensors, bool no_alloc = false,
                               ggml_backend_buffer_type_t pinned_staging = nullptr);
    ggml_tensor * clone(const ggml_tensor * tensor, ggml_backend_buffer_type_t buft);
    ggml_tensor * get(const ggml_tensor * tensor) const;
    bool owns(const ggml_tensor * tensor) const;

    // Runs on the scheduler submission thread, after non-weight inputs are ready.
    // Returns false only for tensors that do not belong to this LES context.
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);
    uint64_t uploaded_bytes() const { return n_uploaded; }
    bool requires_upload() const { return staged; }
    bool pinned_staging_enabled() const { return pinned_staging != nullptr; }
    size_t pinned_bytes() const;
    size_t size_bytes() const;
    const std::map<ggml_backend_buffer_type_t, size_t> & memory_breakdown() const { return sizes; }

private:
    ggml_context_ptr ctx;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> tensors;
    std::unordered_map<ggml_backend_buffer_type_t,
        std::unordered_map<const ggml_tensor *, ggml_tensor *>> clones;
    std::unordered_set<const ggml_tensor *> owned;
    uint64_t n_uploaded = 0;
    bool no_alloc;
    bool staged = false;
    ggml_backend_buffer_type_t pinned_staging;
    std::map<ggml_backend_buffer_type_t, size_t> sizes;
};

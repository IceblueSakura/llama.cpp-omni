#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-metal-device.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static void balanced(const ggml_metal_test_stats & before) {
    const auto after = ggml_metal_test_get_stats();
    if (before.metadata != after.metadata || before.host_bytes != after.host_bytes ||
        before.buffers != after.buffers || before.rsets != after.rsets || before.registered != after.registered) {
        std::fprintf(stderr, "live delta: metadata=%lld host=%lld buffers=%lld rsets=%lld registered=%lld\n",
            (long long)(after.metadata-before.metadata), (long long)(after.host_bytes-before.host_bytes),
            (long long)(after.buffers-before.buffers), (long long)(after.rsets-before.rsets),
            (long long)(after.registered-before.registered));
        throw std::runtime_error("Metal allocation rollback leaked resources");
    }
}

int main() {
    try {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("MTL0");
        if (!dev) { std::puts("SKIP: Metal device unavailable"); return 77; }
        auto * native = ggml_metal_device_get(0);
        const auto * props = ggml_metal_device_get_props(native);
        check(props->use_shared_buffers, "allocator test requires shared buffers enabled");
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        ggml_metal_test_configure({});
        const auto baseline = ggml_metal_test_get_stats();
        int cases = 0;
        auto recovered = [&] {
            ggml_metal_test_configure({});
            {
                ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer(buft, page));
                check(bool(b), "allocation after injected failure did not recover");
                ggml_backend_buffer_clear(b.get(), 0x5a);
                check(*(const unsigned char *)ggml_backend_buffer_get_base(b.get()) == 0x5a,
                    "recovered shared allocation is unusable");
            }
            balanced(baseline);
            ++cases;
        };
        for (auto failure : {GGML_METAL_TEST_METADATA, GGML_METAL_TEST_HOST, GGML_METAL_TEST_BUFFER,
                             GGML_METAL_TEST_RSET, GGML_METAL_TEST_RSET_CREATED}) {
            if (!props->use_residency_sets &&
                (failure == GGML_METAL_TEST_RSET || failure == GGML_METAL_TEST_RSET_CREATED)) {
                std::puts("SKIP: residency fault case (residency sets unavailable)");
                continue;
            }
            ggml_metal_test_config config{};
            config.failure = failure;
            config.nth = 1;
            ggml_metal_test_configure(config);
            ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer(buft, page));
            check(!b, "failed shared allocation was published as success");
            check(ggml_metal_test_get_stats().injected == 1, "shared fault was not exercised");
            balanced(baseline);
            recovered();
        }
        for (auto failure : {GGML_METAL_TEST_BUFFER, GGML_METAL_TEST_RSET, GGML_METAL_TEST_RSET_CREATED}) {
            if (!props->use_residency_sets && failure != GGML_METAL_TEST_BUFFER) { continue; }
            ggml_metal_test_config config{};
            config.failure = failure;
            config.nth = 1;
            ggml_metal_test_configure(config);
            check(ggml_metal_buffer_init(native, page, false) == nullptr,
                "failed private allocation was published as success");
            check(ggml_metal_test_get_stats().injected == 1, "private fault was not exercised");
            balanced(baseline);
            recovered();
            auto * retry = ggml_metal_buffer_init(native, page, false);
            check(retry != nullptr, "private allocation after injected failure did not recover");
            ggml_metal_buffer_free(retry);
            balanced(baseline);
        }
        void * data = mmap(nullptr, 6 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        check(data != MAP_FAILED, "cannot allocate small external mapping");
        struct mapping_owner {
            void * data;
            size_t size;
            ~mapping_owner() { munmap(data, size); }
        } owner{data, 6 * page};
        std::memset(data, 0xa5, 6 * page);
        auto external_intact = [&] {
            const auto * bytes = static_cast<const unsigned char *>(data);
            for (size_t i = 0; i < 6 * page; ++i) {
                check(bytes[i] == 0xa5, "Metal changed or released loader-owned memory");
            }
        };
        for (int view = 1; view <= 3; ++view) {
            ggml_metal_test_config config{};
            config.failure = GGML_METAL_TEST_BUFFER;
            config.nth = view;
            config.map_max_size = 4 * page; // Three overlapping views, not a huge model.
            ggml_metal_test_configure(config);
            ggml_backend_buffer_ptr b(ggml_backend_dev_buffer_from_host_ptr(dev, data, 6 * page, page));
            check(!b, "failed multi-view mapping was published as success");
            check(ggml_metal_test_get_stats().injected == 1, "multi-view fault was not exercised");
            balanced(baseline);
            external_intact();
            recovered();
        }
        for (auto failure : {GGML_METAL_TEST_METADATA, GGML_METAL_TEST_RSET, GGML_METAL_TEST_RSET_CREATED,
                             GGML_METAL_TEST_NONE}) {
            if (!props->use_residency_sets &&
                (failure == GGML_METAL_TEST_RSET || failure == GGML_METAL_TEST_RSET_CREATED)) {
                std::puts("SKIP: mapped residency fault (residency sets unavailable)");
                continue;
            }
            ggml_metal_test_config config{};
            config.failure = failure;
            config.nth = 1;
            config.map_max_size = 4 * page;
            ggml_metal_test_configure(config);
            {
                ggml_backend_buffer_ptr b(ggml_backend_dev_buffer_from_host_ptr(dev,
                    static_cast<char *>(data) + 1, 6 * page - 2, page));
                check(bool(b) == (failure == GGML_METAL_TEST_NONE), "unexpected mapped allocation result");
                check(ggml_metal_test_get_stats().injected == (failure == GGML_METAL_TEST_NONE ? 0 : 1),
                    "mapping fault was not exercised");
            }
            balanced(baseline);
            external_intact();
            recovered();
        }
        for (auto failure : {GGML_METAL_TEST_BUFFER, GGML_METAL_TEST_NONE}) {
            ggml_metal_test_config config{};
            config.failure = failure;
            config.nth = 1;
            ggml_metal_test_configure(config);
            {
                ggml_backend_buffer_ptr b(ggml_backend_dev_buffer_from_host_ptr(dev, data, page, page));
                check(bool(b) == (failure == GGML_METAL_TEST_NONE), "single-view mapping result is wrong");
                check(ggml_metal_test_get_stats().injected == (failure == GGML_METAL_TEST_NONE ? 0 : 1),
                    "single-view fault was not exercised");
            }
            balanced(baseline);
            external_intact();
            recovered();
        }
        if (props->use_residency_sets) {
            for (auto failure : {GGML_METAL_TEST_RSET, GGML_METAL_TEST_RSET_CREATED}) {
                ggml_metal_test_configure({});
                auto * b = ggml_metal_buffer_init(native, page, true);
                check(b != nullptr, "cannot allocate park/unpark fixture");
                std::memset(ggml_metal_buffer_get_base(b), 0x5a, page);
                check(ggml_metal_buffer_park(b), "cannot park fixture");
                const auto parked = ggml_metal_test_get_stats();
                ggml_metal_test_config config{};
                config.failure = failure;
                config.nth = 1;
                ggml_metal_test_configure(config);
                check(!ggml_metal_buffer_unpark(b), "failed residency restore was published as success");
                check(ggml_metal_test_get_stats().injected == 1, "unpark fault was not exercised");
                balanced(parked);
                ggml_metal_test_configure({});
                check(ggml_metal_buffer_unpark(b), "unpark retry did not recover");
                const auto restored = ggml_metal_test_get_stats();
                check(restored.rsets == baseline.rsets + 1 && restored.registered == baseline.registered + 1,
                    "unpark retry skipped residency restore");
                check(*(const unsigned char *)ggml_metal_buffer_get_base(b) == 0x5a, "unpark lost host bytes");
                ggml_metal_buffer_free(b);
                balanced(baseline);
                recovered();
            }
        }
        for (size_t used : {size_t(99), size_t(100), size_t(101)}) {
            ggml_metal_test_config config{};
            config.override_memory = true;
            config.total = 100;
            config.allocated = used;
            ggml_metal_test_configure(config);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            check(total == 100 && free == (used < 100 ? 100 - used : 0), "free memory query underflowed");
        }
        ggml_metal_test_configure({});
        balanced(baseline);
        std::printf("Metal allocation rollback: %d fault/recovery cases and memory-query boundaries passed\n", cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-metal-alloc: %s\n", error.what());
        return 1;
    }
}

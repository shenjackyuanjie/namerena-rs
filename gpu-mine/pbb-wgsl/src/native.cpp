#if defined(_MSC_VER) && !defined(__clang__)
#define __restrict__ __restrict
#endif
#include "pbb-run.hpp"
#include <array>

static_assert(!PBB_HAS_SIMD && N == 256 && M == 128);

extern "C" {
void* pbb_front_open(const U* state, const U* key, const U* setup, const int32_t* gate,
                     U capacity, U block) noexcept;
int pbb_front_run(void* device, Id first, U n, U capture, U* result, U* count, double* seconds) noexcept;
void pbb_front_close(void* device) noexcept;
}

struct WgslDevice {
    const Launch* launch = nullptr;
    U threads = 0;
    void* front = nullptr;
    std::vector<U> readback;

    WgslDevice() = default;
    WgslDevice(const WgslDevice&) = delete;
    ~WgslDevice() { if (front) pbb_front_close(front); }

    void init(const Launch& input, const Options& opts) {
        // 两个后端使用原版 load_prefix 的真实游标和排列。
        std::array<U, 64> state{};
        for (size_t i = 0; i < 256; ++i)
            state[i / 4] |= U(input.prefix_box[i]) << (8 * (i % 4));
        std::array<U, name_limit + 1> key{};
        for (int i = 0; i < input.prefix_len; ++i) key[i] = U(u8_t(input.prefix[i]));
        for (int i = 0; i < encoded_width; ++i) key[input.prefix_len + i] = 256u + U(i);
        for (int i = 0; i < input.suffix_len; ++i) key[input.prefix_len + encoded_width + i] = U(u8_t(input.suffix[i]));
        const std::array<U, 4> setup{input.i_pre, input.j_pre, input.s_pre, U(input.name_len)};
        // 宿主直接从编译用的 PBB 头取权重，顺序为 F/R/S/T。
        #define PLANES(k) GATE_F_##k, GATE_R_##k, GATE_S_##k, GATE_T_##k
        const std::array<int32_t, 45> gate{PLANES(W0), PLANES(W1), PLANES(W2), PLANES(W3),
            PLANES(W4), PLANES(W5), PLANES(W6), PLANES(W7), PLANES(TH), PLANES(REM2), PLANES(REM3), PBB_MIN_P7};
        #undef PLANES
        readback.reserve(opts.batch);
        front = pbb_front_open(state.data(), key.data(), setup.data(), gate.data(), opts.batch, opts.block);
        if (!front) throw std::runtime_error("无法初始化 WGSL 前筛，请检查上方的设备错误");
        launch = &input;
        threads = opts.threads;
    }

    Batch run(Id first, U n, U, bool capture, bool verify_gate_indices = false) {
        readback.resize(n);
        U count = 0;
        double seconds = 0;
        if (pbb_front_run(front, first, n, capture, readback.data(), &count, &seconds) || count > n)
            throw std::runtime_error("WGSL 前筛执行或回读失败");
        std::vector<U> indices;
        if (capture) {
            indices.reserve(count);
            for (U i = 0; i < n; ++i) {
                if (readback[i] > 2) throw std::runtime_error("WGSL 返回了无效的 Q 门档位");
                if (readback[i]) indices.push_back(i);
            }
            if (indices.size() != count) throw std::runtime_error("WGSL 返回的 Q 门计数与捕获档位不一致");
        } else {
            readback.resize(count);
            std::sort(readback.begin(), readback.end());
            if ((!readback.empty() && readback.back() >= n)
                || std::adjacent_find(readback.begin(), readback.end()) != readback.end())
                throw std::runtime_error("WGSL 返回了越界或重复的候选下标");
        }
        auto batch = cpu(*launch, first, n, threads, capture, capture ? &indices : &readback);
        batch.kernel_s = seconds;
        if (capture) batch.gate_levels.swap(readback);
        if (verify_gate_indices) batch.gate_indices.swap(readback);
        return batch;
    }
};

extern "C" int pbb_wgsl_main(int argc, const char* const* argv) noexcept {
    return pbb_main<WgslDevice>(argc, argv, "wgsl");
}

#if defined(_MSC_VER) && !defined(__clang__)
#define __restrict__ __restrict
#endif
#include "pbb-run.hpp"
#include <array>
#include <cstddef>
#include <future>

static_assert(!PBB_HAS_SIMD && N == 256 && M == 128 && K == 64 && skill_cnt == 40);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);
static_assert(sizeof(FillSlot) == 4 * sizeof(U) && alignof(FillSlot) == alignof(U) &&
              offsetof(FillSlot, offset) == 0 && offsetof(FillSlot, width) == sizeof(U) &&
              offsetof(FillSlot, radix) == 2 * sizeof(U) && offsetof(FillSlot, table) == 3 * sizeof(U));

static constexpr auto make_g3() {
    std::array<U, 49> data{};
    const double* planes[] = {GATE_SK_A, GATE_SK_B, GATE_SK_C};
    const double thresholds[] = {GATE_SK_TA, GATE_SK_TB, GATE_SK_TC};
    for (int k = 0; k < 3; ++k) {
        const double t = thresholds[k];
        if (!(t >= 1.0 && t <= 60000.0) || planes[k][43] != 0.0) throw "unsupported G3 plane";
        for (int i = 0; i < 46; ++i) {
            const double w = planes[k][i];
            if (!(w >= 0.0 && w <= 16.0) || w != U(w)) throw "unsupported G3 weight";
            data[i] |= U(w) << (5 * k);
        }
        // floor(T * 65536) - 1 留出 1/65536，远大于原版 f64 的累积舍入误差。
        data[46 + k] = U(t * 65536.0) - 1u;
    }
    // 8 个属性、16 个技能和一个标志；频次至多 106，护盾变换至多 142。
    static_assert((406ULL + 7 * 99 + 16 * 142 + 1) * 16 * 65536 < (1ULL << 32));
    return data;
}

extern "C" {
void* pbb_front_open(const U* state, const U* key, const U* setup, const int32_t* gate,
                     const U* g3, const U* fill_setup, const FillSlot* fill_slots,
                     const U* fill_data, U fill_data_count, U capacity, U block) noexcept;
int pbb_front_run(void* device, Id first, U n, U capture, U* result, U* counts,
                  U* gate_indices, double* seconds) noexcept;
void pbb_front_close(void* device) noexcept;
}

struct WgslDevice {
    const Launch* launch = nullptr;
    U threads = 0;
    void* front = nullptr;
    std::vector<U> readback, indices, q_indices;

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
        for (int i = 0; i < input.fill_len; ++i) key[input.prefix_len + i] = 256u + U(i);
        for (int i = 0; i < input.suffix_len; ++i) key[input.prefix_len + input.fill_len + i] = U(u8_t(input.suffix[i]));
        const std::array<U, 4> setup{input.i_pre, input.j_pre, input.s_pre, U(input.name_len)};
        const std::array<U, 10> fill_setup{
            U(input.fill_len), input.fill_slots_count, input.fill_shuffle,
            input.fill_mask_lo, input.fill_mask_hi, U(input.fill_seed), U(input.fill_seed >> 32),
            U(input.fill_domain), U(input.fill_domain >> 32), input.fill_shift,
        };
        #define PLANES(k) GATE_F_##k, GATE_R_##k, GATE_S_##k, GATE_T_##k
        const std::array<int32_t, 45> gate{PLANES(W0), PLANES(W1), PLANES(W2), PLANES(W3),
            PLANES(W4), PLANES(W5), PLANES(W6), PLANES(W7), PLANES(TH), PLANES(REM2), PLANES(REM3), PBB_MIN_P7};
        #undef PLANES
        static constexpr auto g3 = make_g3();
        readback.resize(opts.batch);
        front = pbb_front_open(state.data(), key.data(), setup.data(), gate.data(), g3.data(),
                               fill_setup.data(), input.fill_slots, input.fill_data, input.fill_data_count,
                               opts.batch, opts.block);
        if (!front) throw std::runtime_error("无法初始化 WGSL 前筛，请检查上方的设备错误");
        launch = &input;
        threads = opts.threads;
    }

    struct GateReadback { U eligible; double seconds; };

    GateReadback read(Id first, U n, bool capture, bool verify_gate_indices) {
        U counts[2]{};
        double seconds = 0;
        if (verify_gate_indices) q_indices.resize(n);
        if (pbb_front_run(front, first, n, capture, readback.data(), counts,
                         verify_gate_indices ? q_indices.data() : nullptr, &seconds) ||
            counts[0] > n || counts[1] > counts[0])
            throw std::runtime_error("WGSL 门控执行或回读失败");
        indices.clear();
        if (capture) {
            indices.reserve(counts[0]);
            for (U i = 0; i < n; ++i) {
                if (readback[i] > 2) throw std::runtime_error("WGSL 返回了无效的 Q 门档位");
                if (readback[i]) indices.push_back(i);
            }
            if (indices.size() != counts[0]) throw std::runtime_error("WGSL 返回的 Q 门计数与捕获档位不一致");
        } else {
            indices.assign(readback.begin(), readback.begin() + counts[1]);
            std::sort(indices.begin(), indices.end());
            if ((!indices.empty() && indices.back() >= n) ||
                std::adjacent_find(indices.begin(), indices.end()) != indices.end())
                throw std::runtime_error("WGSL 返回了越界或重复的候选下标");
        }
        if (verify_gate_indices) q_indices.resize(counts[0]);
        return {counts[0], seconds};
    }

    Batch run(Id first, U n, U, bool capture, bool verify_gate_indices = false) {
        const auto result = read(first, n, capture, verify_gate_indices);
        auto batch = score(first, n, capture, indices, result);
        if (capture) batch.gate_levels.assign(readback.begin(), readback.begin() + n);
        if (verify_gate_indices) {
            std::sort(q_indices.begin(), q_indices.end());
            batch.gate_indices.swap(q_indices);
        }
        return batch;
    }

    Batch score(Id first, U n, bool capture, const std::vector<U>& ids, GateReadback result) const {
        auto batch = cpu(*launch, first, n, threads, capture, &ids);
        batch.counts.eligible = result.eligible;
        batch.kernel_s = result.seconds;
        return batch;
    }

    std::future<Batch> defer(Id first, U n) {
        const auto result = read(first, n, false, false);
        return std::async(std::launch::async, [this, first, n, result, ids = std::move(indices)] {
            return score(first, n, false, ids, result);
        });
    }
};

// 与公共运行器的 start_batch 重载；验证模式仍直接走 run。
static std::future<Batch> start_batch(WgslDevice& gpu, Id first, U n, U) {
    return gpu.defer(first, n);
}

extern "C" int pbb_wgsl_main(int argc, const char* const* argv) noexcept {
    return pbb_main<WgslDevice>(argc, argv, "wgsl");
}

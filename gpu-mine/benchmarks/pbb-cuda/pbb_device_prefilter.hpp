#pragma once
#include "pbb-bench-types.hpp"
#include "pbb_device_name.hpp"
#if defined(__CUDACC__) || defined(__CUDA__)
#include "pbb_device_scoring.hpp"
#endif

#if defined(__CUDACC__) || defined(__CUDA__)
#define PBB_PREFILTER_HD __host__ __device__ __forceinline__
#else
#define PBB_PREFILTER_HD inline
#endif

namespace pbb_device {
namespace prefilter_detail {
static_assert(N == 256 && encoded_width == 18 && name_limit <= 252);
inline constexpr int words_per_thread = N / 4;

PBB_PREFILTER_HD U utf8_word(U c) {
    return (0xe0u | (c >> 12)) | ((0x80u | ((c >> 6) & 63u)) << 8)
                              | ((0x80u | (c & 63u)) << 16);
}

struct Generated {
    U chunks[6];

    PBB_PREFILTER_HD explicit Generated(Id id)
        : chunks{utf8_word(0x4e00u + U((id >> 50) & 0x3fff)),
                 utf8_word(0x4e00u + U((id >> 36) & 0x3fff)),
                 utf8_word(0x4e00u + U((id >> 22) & 0x3fff)),
                 utf8_word(0x4e00u + U((id >> 8) & 0x3fff)),
                 utf8_word(0x4e00u + U((id & 255) << 6)), utf8_word(0x3d01u)} {}

    PBB_PREFILTER_HD U byte_at(int k) const {
        return (chunks[k / 3] >> (8 * (k % 3))) & 255u;
    }
};

// 每个线程独占一个列。Block 为 warp 大小的整数倍时每个 lane 使用独立 bank。
template<int Block>
struct Box {
    static_assert(Block > 0 && Block % 32 == 0);
    U* column;

    // CUDA 为小端；通过 unsigned char 读写字的对象表示，布局仍为每 lane 一个 bank。
    PBB_PREFILTER_HD u8_t& cell(U slot) const {
        return reinterpret_cast<u8_t*>(column)[(slot & ~3u) * Block + (slot & 3u)];
    }

    PBB_PREFILTER_HD void restore(const Launch& input) const {
        for (int i = 0; i < words_per_thread; ++i) {
            const unsigned char* p = input.prefix_box + 4 * i;
            column[i * Block] = U(p[0]) | (U(p[1]) << 8) | (U(p[2]) << 16) | (U(p[3]) << 24);
        }
    }

    PBB_PREFILTER_HD void step(U i, U& sum, U key) const {
        u8_t& a = cell(i);
        const U cur = a;
        sum = (sum + key + cur) & 255u;
        u8_t& b = cell(sum);
        a = b;
        b = u8_t(cur);
    }

    // 读前移：两次交换之间无需等待共享内存写回。
    // 第一换只可能修改 i 和 j，第二次的两个读取用它们的新值做旁路。
    PBB_PREFILTER_HD void step2(U i, U& sum, U key0, U key1) const {
        u8_t& a = cell(i);
        u8_t& c = cell(i + 1);
        const U old_a = a, old_c = c;
        const U j = (sum + key0 + old_a) & 255u;
        const U cur_c = j == i + 1 ? old_a : old_c;
        sum = (j + key1 + cur_c) & 255u;
        u8_t& b = cell(j);
        u8_t& d = cell(sum);
        const U old_b = b, old_d = d;
        const U cur_d = sum == j ? old_a : sum == i ? old_b : old_d;
        a = u8_t(old_b);
        b = u8_t(old_a);
        c = u8_t(cur_d);
        d = u8_t(cur_c);
    }

    PBB_PREFILTER_HD void literal(const char* bytes, int first, int end, int& i, U& sum) const {
        const int count = end - first;
        const int stop = i + count < N ? i + count : N;
        int k = first;
        for (; i + 1 < stop; i += 2, k += 2) step2(i, sum, u8_t(bytes[k]), u8_t(bytes[k + 1]));
        for (; i < stop; ++i, ++k) step(i, sum, u8_t(bytes[k]));
    }

    PBB_PREFILTER_HD void generated(const Generated& key, int first, int& i, U& sum) const {
        if (first == 0 && i <= N - encoded_width) {
#if defined(__CUDACC__) || defined(__CUDA__)
            #pragma unroll
#endif
            for (int c = 0; c < 6; c += 2) {
                const U a = key.chunks[c], b = key.chunks[c + 1];
                step2(i, sum, a & 255u, (a >> 8) & 255u);
                step2(i + 2, sum, a >> 16, b & 255u);
                step2(i + 4, sum, (b >> 8) & 255u, b >> 16);
                i += 6;
            }
        } else {
            for (int k = first; k < encoded_width && i < N; ++i, ++k) step(i, sum, key.byte_at(k));
        }
    }

    // 一个密钥周期依次为固定前缀、生成字节、固定后缀、NUL。
    // 原 KSA 从 name[name_len] 开始，每轮同样截断为 256 个槽；第一轮接原前缀游标。
    PBB_PREFILTER_HD void round(const Launch& input, const Generated& key, int i, int k, U sum) const {
        const int gen_end = input.prefix_len + encoded_width;
        while (i < N) {
            if (k < input.prefix_len) literal(input.prefix, k, input.prefix_len, i, sum);
            if (k < gen_end && i < N)
                generated(key, k > input.prefix_len ? k - input.prefix_len : 0, i, sum);
            if (k < input.name_len && i < N)
                literal(input.suffix, k > gen_end ? k - gen_end : 0, input.suffix_len, i, sum);
            if (i < N) step(i++, sum, 0);
            k = 0;
        }
    }

    PBB_PREFILTER_HD void schedule(const Launch& input, Id id) const {
        restore(input);
        const Generated key(id);
        round(input, key, input.i_pre, input.j_pre, input.s_pre);
        round(input, key, 0, input.name_len, 0);
    }

    PBB_PREFILTER_HD void collect(u8_t (&base)[31]) const {
        int filled = 0;
        for (int i = 0; i < words_per_thread && filled < 31; ++i) {
            const U word = column[i * Block];
#if defined(__CUDACC__) || defined(__CUDA__)
            #pragma unroll
#endif
            for (int k = 0; k < 4; ++k) {
                const U value = ((word >> (8 * k)) * 181 + 160) & 255u;
                if (89 <= value && value < 217 && filled < 31)
                    base[filled++] = u8_t(value & 63u);
            }
        }
    }
};

// 必须在生成版 gate_level_of 前复现原 load_name 的状态；排序的 39 次比较交换不可替换。
PBB_PREFILTER_HD int finish_base(u8_t (&b)[31]) {
    int v = median(b[28], b[29], b[30]);
    if (v < 24) return v;
    v += median(b[13], b[14], b[15]) + median(b[16], b[17], b[18]) + median(b[25], b[26], b[27]);
    if (v < 165) return v;
    v += median(b[10], b[11], b[12]) + median(b[19], b[20], b[21]) + median(b[22], b[23], b[24]);
    if (v < 250) return v;
    sort10(b);
    return v + (154 + b[3] + b[4] + b[5] + b[6]) / 3;
}
} // namespace prefilter_detail

// CPU 对拍也使用这一路；column 指向 threadIdx.x，步长为 Block，需分配 64*Block 个 U。
template<int Block>
PBB_PREFILTER_HD int prefilter_base(const Launch& input, Id id, U* column, u8_t (&base)[31]) {
    const prefilter_detail::Box<Block> box{column};
    box.schedule(input, id);
    box.collect(base);
    return prefilter_detail::finish_base(base);
}

#if defined(__CUDACC__) || defined(__CUDA__)
// launch 由原 prepare 生成，无新增字段；通过的是一级门，完整评分仍由第二个设备核计算。
template<int Block>
__device__ __forceinline__ bool prefilter(const Launch& input, Id id, U* column) {
    u8_t base[31];
    prefilter_base<Block>(input, id, column, base);
    int prop[7];
    return gate_level_of(base, prop) != 0;
}
#endif
} // namespace pbb_device

#undef PBB_PREFILTER_HD

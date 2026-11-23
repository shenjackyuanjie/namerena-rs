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
#if defined(__CUDACC__)
struct PreparedFront {
    U words[words_per_thread];
    unsigned short pairs[16];
};
static __constant__ PreparedFront prepared_front;
#endif

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
#if defined(__CUDA_ARCH__)
            column[i * Block] = prepared_front.words[i];
#else
            const unsigned char* p = input.prefix_box + 4 * i;
            column[i * Block] = U(p[0]) | (U(p[1]) << 8) | (U(p[2]) << 16) | (U(p[3]) << 24);
#endif
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

#if defined(__CUDA_ARCH__)
    __device__ __forceinline__ U shared_address(U slot) const {
        return U(__cvta_generic_to_shared(column)) + (slot & 252u) * Block + (slot & 3u);
    }

    __device__ __forceinline__ void store_byte(U slot, U value) const {
        const U addr = shared_address(slot);
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(addr), "r"(value) : "memory");
    }

    // i 为偶数时两个连续槽可一并读写。先提交随机槽，再用转发后的值写回固定位置。
    __device__ __forceinline__ void step_aligned(U i, U& sum, U key0, U key1) const {
        const U address = shared_address(i);
        U pair;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(pair) : "r"(address) : "memory");
        const U a = pair & 255u;
        const U j = (sum + key0 + a) & 255u;
        const U high = pair >> 8;
        const bool hit_next = j == i+1;
        const U c = hit_next ? a : high;
        // 普通第二游标与 j 并行计算；只有首换命中下个固定槽才转发 a。
        const U predicted = sum + (key0 + key1) + (a + high);
        sum = (hit_next ? j + key1 + a : predicted) & 255u;
        // 无条件发出两个共享读取，避免先等别名判断才读取随机槽。
        U old_b, old_d;
        const U ba = shared_address(j), da = shared_address(sum);
        asm volatile("{ .reg .u32 x, y; ld.shared.u8 x, [%2]; ld.shared.u8 y, [%3]; mov.u32 %0, x; mov.u32 %1, y; }"
            : "=r"(old_b), "=r"(old_d) : "r"(ba), "r"(da) : "memory");
        const U cur_d = sum == j ? a : sum == i ? old_b : old_d;
        asm volatile("st.shared.u8 [%0], %1; st.shared.u8 [%2], %3;"
                     :: "r"(ba), "r"(a), "r"(da), "r"(c) : "memory");
        const unsigned short final_pair = (sum == i ? c : old_b) | ((sum == i+1 ? c : cur_d) << 8);
        asm volatile("st.shared.u16 [%0], %1;" :: "r"(address), "h"(final_pair) : "memory");
    }
#endif

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

#if defined(__CUDA_ARCH__)
    // 第二轮第 i 步前，裁剪版与完整 KSA 的 j、[0,Limit) 以及 [i,256) 完全相同。
    // 随机旧值只会写入本步结束后不再观察的固定槽，无需读取。
    __device__ __forceinline__ void cropped_pair(U i, U& sum, U key0, U key1) const {
        const U address = shared_address(i);
        U word;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(word) : "r"(address) : "memory");

        const U a = word & 255u, b = word >> 8;
        const U j = (sum+key0+a)&255u;
        const bool next = j == i+1;
        const U c = next ? a : b;
        const U predicted = sum + (key0+key1) + (a+b);
        sum = (next ? j + key1 + a : predicted) & 255u;
        // 同组固定槽的变化已转发；写入已退出观察范围的槽无影响，省掉逐 lane 谓词。
        store_byte(j, a);
        store_byte(sum, c);
    }

    __device__ __forceinline__ void cropped_word(U i, U& carry, U k0, U k1, U k2, U k3) const {
        const U x = column[(i >> 2) * Block];
        const U a0 = x & 255u, a1 = (x >> 8) & 255u, a2 = (x >> 16) & 255u, a3 = x >> 24;
        const U j0 = (carry + k0 + a0) & 255u;
        const U r1 = (j0 == i+1 ? a0 : a1);
        const U j1 = ((j0 == i+1 ? j0 + k1 + a0 : carry + (k0+k1) + (a0+a1))) & 255u;
        const U r2 = j1 == i+2 ? r1 : j0 == i+2 ? a0 : a2;
        const U j2 = (j1 + k2 + r2) & 255u;
        const U r3 = j2 == i+3 ? r2 : j1 == i+3 ? r1 : j0 == i+3 ? a0 : a3;
        const U j3 = (j2 + k3 + r3) & 255u;
        store_byte(j0, a0);
        store_byte(j1, r1);
        store_byte(j2, r2);
        store_byte(j3, r3);
        carry = j3;
    }

    __device__ __forceinline__ U fixed_key(const Generated& key, int k) const {
        if (k >= 32) return key.byte_at(k-32);
        const U x = prepared_front.pairs[k/2];
        return (x >> ((k&1)*8)) & 255u;
    }

    template<int Nstep>
    __device__ __forceinline__ void cropped_words(const Generated& key, U start, U& carry) const {
        #pragma unroll
        for (int k=0; k<Nstep; k+=4) {
            cropped_word(start+k, carry, fixed_key(key,k%50),fixed_key(key,(k+1)%50),
                                                fixed_key(key,(k+2)%50),fixed_key(key,(k+3)%50));
        }
    }

    template<int Limit>
    __device__ __forceinline__ void schedule_cropped(const Launch& input, Id id) const {
        static_assert(Limit >= 50 && Limit <= 100 && Limit % 2 == 0);
        restore(input);
        const Generated key(id);
        U j = input.s_pre;
        #pragma unroll
        for (int k=0; k<18; k+=2) step_aligned(32+k,j,key.byte_at(k),key.byte_at(k+1));
        for (int i=50; i+50<=N; i+=50) {
            #pragma unroll 4
            for (int k=0; k<32; k+=2) {
                const U p=prepared_front.pairs[k/2];
                step_aligned(i+k,j,p&255u,p>>8);
            }
            #pragma unroll
            for (int k=0; k<18; k+=2) step_aligned(i+32+k,j,key.byte_at(k),key.byte_at(k+1));
        }
        round(input,key,250,input.name_len,j);
        j=0;
        #pragma unroll 4
        for (int k=0; k<32; k+=2) {
            const U p=prepared_front.pairs[k/2];
            step_aligned(k,j,p&255u,p>>8);
        }
        #pragma unroll
        for (int k=0; k<18; k+=2) step_aligned(32+k,j,key.byte_at(k),key.byte_at(k+1));
        #pragma unroll
        for (int k=0; k<32; k+=2) {
            const U p=prepared_front.pairs[k/2];
            if (50+k < Limit) step_aligned(50+k,j,p&255u,p>>8);
            else cropped_pair(50+k,j,p&255u,p>>8);
        }
        #pragma unroll
        for (int k=0; k<18; k+=2) {
            if (82+k < Limit) step_aligned(82+k,j,key.byte_at(k),key.byte_at(k+1));
            else cropped_pair(82+k,j,key.byte_at(k),key.byte_at(k+1));
        }
        cropped_words<100>(key,100,j);
        cropped_words<56>(key,200,j);
    }
#endif

    PBB_PREFILTER_HD void schedule(const Launch& input, Id id) const {
        restore(input);
        const Generated key(id);
#if defined(__CUDA_ARCH__)
        // 32 字节 NUL+前缀和 18 字节生成段组成偶数长度的圈，任何此前缀内容都能走此路径。
        if (input.prefix_len == 31 && input.suffix_len == 0 && input.i_pre == 32 && input.j_pre == 31) {
            for (int r = 0; r < 2; ++r) {
                int i = r ? 0 : 32;
                U j = r ? 0 : input.s_pre;
                if (!r) {
                    #pragma unroll
                    for (int k = 0; k < 18; k += 2) step_aligned(i+k, j, key.byte_at(k), key.byte_at(k+1));
                    i += 18;
                }
                for (; i + 50 <= N; i += 50) {
                    #pragma unroll 4
                    for (int k = 0; k < 32; k += 2) {
                        const U pair = prepared_front.pairs[k/2];
                        step_aligned(i+k, j, pair & 255u, pair >> 8);
                    }
                    #pragma unroll
                    for (int k = 0; k < 18; k += 2) step_aligned(i+32+k, j, key.byte_at(k), key.byte_at(k+1));
                }
                // 尾部仍从 NUL 开始，沿用通用输入规则。
                round(input, key, i, input.name_len, j);
            }
            return;
        }
#endif
        round(input, key, input.i_pre, input.j_pre, input.s_pre);
        round(input, key, 0, input.name_len, 0);
    }

    template<int Words = words_per_thread>
    PBB_PREFILTER_HD int collect(u8_t (&base)[31]) const {
        int filled = 0;
        for (int i = 0; i < Words && filled < 31; ++i) {
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
        return filled;
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
// 完整排列的整数 Q 门，供隔离验证或前筛测速使用。
template<int Block>
__device__ __forceinline__ bool prefilter(const Launch& input, Id id, U* column) {
    u8_t base[31];
    const prefilter_detail::Box<Block> box{column};
    box.schedule(input, id);
    box.collect(base);
    // Q 门重新完整排序前十项，不需要提前计算旧 V 或重复排序。
    int prop[7];
    return gate_level_of(base, prop) != 0;
}
#endif
} // namespace pbb_device

#undef PBB_PREFILTER_HD

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
// 低四个半字节为 PRMT 源下标（4 指向零），高位为有效字节数。
// 动态取表放普通只读显存，避免 constant bank 在一个 warp 中串行化。
static __device__ const U compact_q[16] = {
    0x04444u, 0x14440u, 0x14441u, 0x24410u,
    0x14442u, 0x24420u, 0x24421u, 0x34210u,
    0x14443u, 0x24430u, 0x24431u, 0x34310u,
    0x24432u, 0x34320u, 0x34321u, 0x43210u,
};
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
    // slot 已截断为 u8。先抵消低两位的跨列偏移，再合成共享地址。
    __device__ __forceinline__ U shared_address(U slot) const {
        const U base = U(__cvta_generic_to_shared(column));
        return slot * Block + (base + (slot & 3u) * U(1 - Block));
    }

    __device__ __forceinline__ U load_byte(U addr) const {
        U value;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(value) : "r"(addr) : "memory");
        return value;
    }

    __device__ __forceinline__ void store_byte(U slot, U value) const {
        const U addr = shared_address(slot);
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(addr), "r"(value) : "memory");
    }

    // 固定位置偶数对可一并读写；先发首个随机读，首换命中下一槽则转发。
    __device__ __forceinline__ void step_aligned(U i, U& sum, U key0, U key1) const {
        const U address = shared_address(i);
        U pair;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(pair) : "r"(address) : "memory");
        // 此处 pair 的高字节只参与已截断的游标和写入，不需要提前清零。
        const U raw_j = sum + key0 + pair;
        const U j = raw_j & 255u;
        const U ba = shared_address(j);
        const U old_b = load_byte(ba);
        const U c = j == i + 1 ? pair : pair >> 8;
        sum = (raw_j + key1 + c) & 255u;
        const U da = shared_address(sum);
        const U old_d = load_byte(da);
        const U cur_d = sum == j ? pair : sum == i ? old_b : old_d;
        asm volatile("st.shared.u8 [%0], %1; st.shared.u8 [%2], %3;"
                     :: "r"(ba), "r"(pair), "r"(da), "r"(c) : "memory");
        const U final_pair = __byte_perm(sum == i ? c : old_b, sum == i + 1 ? c : cur_d, 0x0040);
        asm volatile("st.shared.u16 [%0], %1;" :: "r"(address), "r"(final_pair) : "memory");
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

        const U raw_j = sum + key0 + word;
        const U j = raw_j & 255u;
        const U c = j == i + 1 ? word : word >> 8;
        sum = (raw_j + key1 + c) & 255u;
        // 同组固定槽的变化已转发；写入已退出观察范围的槽无影响，省掉逐 lane 谓词。
        store_byte(j, word);
        store_byte(sum, c);
    }

    __device__ __forceinline__ void cropped_word(U i, U& carry, U k0, U k1, U k2, U k3) const {
        const U x = column[(i >> 2) * Block];
        const U s0 = carry + k0 + x, j0 = s0 & 255u;
        const U r1 = j0 == i+1 ? x : x >> 8;
        const U s1 = s0 + k1 + r1, j1 = s1 & 255u;
        const U r2 = j1 == i+2 ? r1 : j0 == i+2 ? x : x >> 16;
        const U s2 = s1 + k2 + r2, j2 = s2 & 255u;
        const U r3 = j2 == i+3 ? r2 : j1 == i+3 ? r1 : j0 == i+3 ? x : x >> 24;
        const U j3 = (s2 + k3 + r3) & 255u;
        store_byte(j0, x);
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

    __device__ __forceinline__ void aligned_tail(U carry) const {
        // 50 字节键环走完第 5 圈后，尾 6 槽消费 NUL 与前缀首 5 字节。
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            const U pair = prepared_front.pairs[i];
            step_aligned(250 + i * 2, carry, pair & 255u, pair >> 8);
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
        aligned_tail(j);
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

#if defined(__CUDACC__) || defined(__CUDA__)
    // 仅布局满足 31 字节前缀、无后缀且已走完 NUL+前缀时调用。
    __device__ __forceinline__ void schedule_aligned(const Launch& input, Id id) const {
#if defined(__CUDA_ARCH__)
        restore(input);
        const Generated key(id);
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
            aligned_tail(j);
        }
#endif
    }
#endif

    PBB_PREFILTER_HD void schedule(const Launch& input, Id id) const {
#if defined(__CUDA_ARCH__)
        // 32 字节 NUL+前缀和 18 字节生成段组成偶数长度的圈，任何此前缀内容都能走此路径。
        if (input.prefix_len == 31 && input.suffix_len == 0 && input.i_pre == 32 && input.j_pre == 31) {
            schedule_aligned(input, id);
            return;
        }
#endif
        restore(input);
        const Generated key(id);
        round(input, key, input.i_pre, input.j_pre, input.s_pre);
        round(input, key, 0, input.name_len, 0);
    }

#if defined(__CUDA_ARCH__)
    template<int Words>
    __device__ __forceinline__ int collect_packed(U (&base)[8]) const {
        int filled = 0;
        U pending = 0;
        for (int i = 0; i < Words && filled < 31; ++i) {
            const U word = column[i * Block];
            // 两个字节分别占 16 位：乘法与偏置的最大和小于 2^16，不串道。
            const U even = (word & 0x00ff00ffu) * 181u + 0x00470047u;
            const U odd = ((word >> 8) & 0x00ff00ffu) * 181u + 0x00470047u;
            const U joined = __byte_perm(even, odd, 0x6240);
            // u=(181*x+71) mod 256；原 Q 窗口等价于 u<128，载荷是 (u+89) mod 64。
            const U mask = ((~joined & 0x80808080u) * 0x00204081u) >> 28;
            const U values = ((joined & 0x3f3f3f3fu) + 0x19191919u) & 0x3f3f3f3fu;
            const U select = __ldg(compact_q + mask);
            const U bytes = __byte_perm(values, 0u, select);
            const U shift = (filled & 3) * 8u;
            const U merged = pending | (bytes << shift);
            const int next = filled + int(select >> 16);
            const bool full = ((next ^ filled) & 4) != 0;
            if (full) base[filled / 4] = merged;
            pending = full ? __funnelshift_l(bytes, 0u, shift) : merged;
            filled = next;
        }
        if (filled < 32 && (filled & 3)) base[filled / 4] = pending;
        return filled;
    }
#endif

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

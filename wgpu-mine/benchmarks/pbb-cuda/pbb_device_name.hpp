#pragma once
// PBB 标量 Name 的独立 CUDA/CPU 实现。只保留评分需要的状态，不依赖原头文件或权重。
#if defined(__CUDACC__) || defined(__CUDA__)
#define PBB_NAME_HD __host__ __device__ inline
#else
#define PBB_NAME_HD inline
#endif

namespace pbb_device {
using u8_t = unsigned char;
using u64_t = unsigned long long;
inline constexpr int N = 256;
inline constexpr int M = 128;
inline constexpr int K = 64;
inline constexpr int skill_cnt = 40;

PBB_NAME_HD u8_t name_min(u8_t a, u8_t b) { return a < b ? a : b; }
PBB_NAME_HD u8_t name_max(u8_t a, u8_t b) { return a > b ? a : b; }
PBB_NAME_HD u8_t median(u8_t a, u8_t b, u8_t c) {
    return name_max(name_min(a, b), name_min(name_max(a, b), c));
}
PBB_NAME_HD void name_compare_exchange(u8_t& a, u8_t& b) {
    const u8_t mask = u8_t((u8_t(a - b) >> 7) - 1);
    const u8_t change = u8_t((a ^ b) & mask);
    a ^= change;
    b ^= change;
}
PBB_NAME_HD void sort10(u8_t* b) {
    // 保留原 utils.hpp 的实际 39 次比较交换；不能用标准排序或评分侧的 sort10_net 替代。
    // 即使输入只有 0 和 63，原序列也有 6 种局部逆序，调用方观察的是原字节顺序。
    #define PBB_NAME_CE(i, j) name_compare_exchange(b[i], b[j]);
    PBB_NAME_CE(0,5) PBB_NAME_CE(1,6) PBB_NAME_CE(2,7) PBB_NAME_CE(3,8) PBB_NAME_CE(4,9)
    PBB_NAME_CE(0,3) PBB_NAME_CE(1,4) PBB_NAME_CE(5,8) PBB_NAME_CE(6,9)
    PBB_NAME_CE(0,2) PBB_NAME_CE(3,6) PBB_NAME_CE(4,7) PBB_NAME_CE(1,5)
    PBB_NAME_CE(0,1) PBB_NAME_CE(2,4) PBB_NAME_CE(7,9) PBB_NAME_CE(3,5)
    PBB_NAME_CE(2,3) PBB_NAME_CE(6,8) PBB_NAME_CE(4,5)
    PBB_NAME_CE(1,2) PBB_NAME_CE(3,4) PBB_NAME_CE(5,6) PBB_NAME_CE(7,8)
    PBB_NAME_CE(1,3) PBB_NAME_CE(4,6) PBB_NAME_CE(2,5)
    PBB_NAME_CE(0,1) PBB_NAME_CE(2,3) PBB_NAME_CE(4,5) PBB_NAME_CE(6,7) PBB_NAME_CE(8,9)
    PBB_NAME_CE(1,2) PBB_NAME_CE(3,4) PBB_NAME_CE(5,6) PBB_NAME_CE(7,8)
    PBB_NAME_CE(2,3) PBB_NAME_CE(4,5) PBB_NAME_CE(6,7)
    #undef PBB_NAME_CE
}

struct Name {
    u8_t val[N], val_base[N], val_base2[N];
    u8_t name_base[M], freq[16], skill[skill_cnt];
    u8_t p, q, i_pre, j_pre, s_pre;
    int q_len, V;
    int PRELEN = 0; // 按上游约定计 KSA 步数，而不是在此处额外加一。
    int NAMELEN;

    PBB_NAME_HD void load_team(const char* team) {
        for (int i = 0; i < N; ++i) val_base[i] = u8_t(i);
        round(val_base, team, cstr_length(team));
    }

    // 调用方保证 name[name_len] 可读；通常是 NUL。共享模板的 PRELEN 按上游步数设定。
    PBB_NAME_HD void load_prefix(const char* name, int name_len) {
        copy_box(val_base2, val_base);
        NAMELEN = name_len;
        int cursor = name_len;
        u8_t sum = 0;
        span(val_base2, name, name_len, 0, PRELEN, cursor, sum);
        i_pre = u8_t(PRELEN > 0 ? PRELEN : 0);
        // 原标量的保存游标是 u8_t；包括长名字下的截断，逐值保留。
        j_pre = u8_t(i_pre == 0 ? name_len : cursor);
        s_pre = i_pre == 0 ? u8_t(0) : sum;
    }

    // 原标量忽略 name_len_hint。不同长度要先重做 load_prefix。
    PBB_NAME_HD void load_name(const char* name, int /*name_len_hint*/ = 0) {
        copy_box(val, val_base2);
        int cursor = j_pre;
        u8_t sum = s_pre;
        span(val, name, NAMELEN, i_pre, N, cursor, sum);
        round(val, name, NAMELEN);
        properties<true>();
    }

    // 影子独立从队伍状态做双轮 KSA；不看前缀，不改 skill/freq。
    PBB_NAME_HD void loading_name(const char* name) {
        copy_box(val, val_base);
        const int len = cstr_length(name);
        round(val, name, len);
        round(val, name, len);
        properties<false>();
    }

    PBB_NAME_HD u8_t m() {
        p = u8_t(p + 1);
        q = u8_t(q + val[p]);
        exchange(val[p], val[q]);
        return val[(val[p] + val[q]) & 255];
    }

    PBB_NAME_HD int gen() {
        const int hi = m();
        const int lo = m();
        return ((hi << 8) | lo) % skill_cnt;
    }

    PBB_NAME_HD void calc_skills(const char* /*name*/ = nullptr) {
        // 181 与 256 互素；在排列中恰好 128 个结果小于 128，按原顺序写满。
        // 必须从 0 开始覆盖，不能只写技能区；这会恢复 PTT 使用的未排序序列。
        q_len = -1;
        for (int i = 0; i < N; ++i) {
            const u8_t u = u8_t(181 * val[i] + 71);
            if (u < 128) name_base[++q_len] = u8_t((u + 89) & 63);
        }
        for (int i = 0; i < skill_cnt; ++i) skill[i] = u8_t(i);
        p = q = 0;
        int mix = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < skill_cnt; ++j) {
                mix = (mix + gen() + skill[j]) % skill_cnt;
                exchange(skill[j], skill[mix]);
            }
        }
        int last = -1;
        for (int j = 0; j < K / 4; ++j) {
            const u8_t* a = name_base + K + 4 * j;
            const u8_t base = name_min(name_min(a[0], a[1]), name_min(a[2], a[3]));
            freq[j] = 0;
            if (base > 10 && skill[j] < 35) {
                freq[j] = u8_t(base - 10);
                if (skill[j] < 25) last = j;
            }
        }
        if (last >= 0) freq[last] = u8_t(2 * freq[last]);
        for (int j = 14; j < 16; ++j) {
            const int base = 60 + 2 * (j - 14);
            if (j != last && freq[j])
                freq[j] = u8_t(freq[j] + name_min(freq[j], name_min(name_base[base], name_base[base + 1])));
        }
    }

    static PBB_NAME_HD void calc_skills_batch4(Name (&lanes)[4], unsigned active_mask) {
        for (int lane = 0; lane < 4; ++lane)
            if (active_mask & (1u << lane)) lanes[lane].calc_skills();
    }

private:
    static PBB_NAME_HD void exchange(u8_t& a, u8_t& b) {
        const u8_t x = a;
        a = b;
        b = x;
    }

    static PBB_NAME_HD void copy_box(u8_t* dst, const u8_t* src) {
        for (int i = 0; i < N; ++i) dst[i] = src[i];
    }

    static PBB_NAME_HD int cstr_length(const char* s) {
        int len = 0;
        while (s[len] != '\0') ++len;
        return len;
    }

    static PBB_NAME_HD void span(u8_t* box, const char* name, int len, int start, int stop,
                                  int& cursor, u8_t& sum) {
        for (int i = start; i < stop; ++i) {
            sum = u8_t(sum + u8_t(name[cursor]) + box[i]);
            exchange(box[i], box[sum]);
            cursor = cursor == len ? 0 : cursor + 1;
        }
    }

    static PBB_NAME_HD void round(u8_t* box, const char* name, int len) {
        int cursor = len; // NUL 为第一步，随后才由下标 0 消费名字字节。
        u8_t sum = 0;
        span(box, name, len, 0, N, cursor, sum);
    }

    PBB_NAME_HD int med_at(int i) const {
        return median(name_base[i], name_base[i + 1], name_base[i + 2]);
    }

    template<bool Early>
    PBB_NAME_HD void properties() {
        q_len = -1;
        for (int i = 0; i < N && q_len < 30; ++i) {
            const u8_t u = u8_t(181 * val[i] + 160);
            if (89 <= u && u < 217) name_base[++q_len] = u8_t(u & 63);
        }
        V = med_at(28);
        if constexpr (Early) if (V < 24) return;
        V += med_at(13) + med_at(16) + med_at(25);
        if constexpr (Early) if (V < 165) return;
        V += med_at(10) + med_at(19) + med_at(22);
        if constexpr (Early) if (V < 250) return;
        sort10(name_base);
        V += (154 + name_base[3] + name_base[4] + name_base[5] + name_base[6]) / 3;
    }
};
} // namespace pbb_device
#undef PBB_NAME_HD

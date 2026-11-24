#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define PBB_HD __host__ __device__
#else
#define PBB_HD
#endif

// 只跨核运输明确字段，不依赖 PBB CPU 与 GPU 结构体的布局相同。
constexpr int families = 10;
constexpr int name_limit = 252;  // 上游 shadow_name[260] 还需 7 字节后缀及 NUL。
constexpr int encoded_width = 15;
using U = std::uint32_t;
using Id = std::uint64_t;

struct Rules {
    int enabled[families], order[families], minimum[families], potential[families];
    int norder, any, any_gate;
    int collect_mode, eight, seven, hp398, floor;
};
struct FillSlot { U offset, width, radix, table; };
struct Launch {
    Rules rules;
    char prefix[name_limit], suffix[name_limit];
    int prefix_len, suffix_len, name_len, fill_len;
    // domain=0 表示完整 64 位编号空间；有限布局右侧位置先变化。
    Id fill_seed, fill_domain;
    U fill_shuffle, fill_shift, fill_mask_lo, fill_mask_hi, fill_slots_count, fill_data_count;
    const FillSlot* fill_slots;
    const U* fill_data;
    unsigned char team_box[256], prefix_box[256], i_pre, j_pre, s_pre;
};
struct Flags { bool keep, blue, ptt; };
struct Record {
    U index;
    bool passed, keep, blue, ptt;
    int xp, xd, sum, tag, shadow, illusion, pp_ref, qp_ref;
    int scores[families], potential[families], props[8], ids[16], freqs[16];
    double effective, strength, deduction;
};
struct Counts { U written, passed, keep, blue, ptt, eligible; };

PBB_HD inline U fill_hash(U x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}

template<bool Reverse>
PBB_HD inline Id shuffle_id(const Launch& input, Id id) {
    if (!input.fill_shuffle) return id;
    const Id mask = (Id(input.fill_mask_hi) << 32) | input.fill_mask_lo;
    const Id seed = input.fill_seed;
    const U shift = input.fill_shift;
    constexpr Id a = 0x9e3779b1u, b = 0x85ebca77u;
    constexpr Id inv_a = 0xcfa4a56b0e8b2f51ULL, inv_b = 0x59ebd4deb6c92f47ULL;
    static_assert(a * inv_a == 1 && b * inv_b == 1);
    const auto unxor = [shift](Id x) {
        for (U s = shift; s < 64; s *= 2) x ^= x >> s;
        return x;
    };
    // 取最接近的 2 的幂作为置换域。正/反向都沿同一轨道走到下一个有效编号。
    do {
        if (input.fill_shuffle == 1) {
            if constexpr (Reverse) {
                id = (unxor(id) * inv_b) & mask;
                id = (unxor(id) * inv_a) & mask;
                id = (unxor(id) - seed) & mask;
            } else {
                id = (id + seed) & mask;
                id = ((id ^ (id >> shift)) * a) & mask;
                id = ((id ^ (id >> shift)) * b) & mask;
                id ^= id >> shift;
            }
        } else if (input.fill_shuffle == 2) {
            if constexpr (Reverse) id = ((id - seed) * inv_a) & mask;
            else id = (id * a + seed) & mask;
        } else {
            const U ml = U((Id(1) << shift) - 1), mr = U(mask >> shift);
            const U s0 = U(seed), s1 = U(seed >> 32);
            U l = U(id) & ml, r = U(id >> shift);
            if constexpr (Reverse) {
                r ^= fill_hash(l + s0 + 0x3c6ef372u) & mr;
                l ^= fill_hash(r + s1 + 0x9e3779b9u) & ml;
                r ^= fill_hash(l + s1) & mr;
                l ^= fill_hash(r + s0) & ml;
            } else {
                l ^= fill_hash(r + s0) & ml;
                r ^= fill_hash(l + s1) & mr;
                l ^= fill_hash(r + s1 + 0x9e3779b9u) & ml;
                r ^= fill_hash(l + s0 + 0x3c6ef372u) & mr;
            }
            id = (Id(r) << shift) | (l & mask);
        }
    } while (input.fill_domain && id >= input.fill_domain);
    return id;
}

PBB_HD inline void make_fill(const Launch& input, Id id, char* out) {
    id = shuffle_id<false>(input, id);
    if (input.fill_slots_count) {
        for (int i = 0; i < input.fill_len; ++i) out[i] = char(input.fill_data[i]);
        for (U i = input.fill_slots_count; i > 0; --i) {
            const auto slot = input.fill_slots[i - 1];
            const Id q = id / slot.radix;
            const U at = slot.table + U(id - q * slot.radix) * slot.width;
            id = q;
            for (U k = 0; k < slot.width; ++k) out[slot.offset + k] = char(input.fill_data[at + k]);
        }
    } else {
        const U points[] = {
            0x4e00 + U((id >> 50) & 0x3fff), 0x4e00 + U((id >> 36) & 0x3fff),
            0x4e00 + U((id >> 22) & 0x3fff), 0x4e00 + U((id >> 8) & 0x3fff),
            0x4e00 + U((id & 0xff) << 6),
        };
        for (int i = 0; i < 5; ++i) {
            const U c = points[i];
            out[3 * i] = char(0xe0 | (c >> 12));
            out[3 * i + 1] = char(0x80 | ((c >> 6) & 63));
            out[3 * i + 2] = char(0x80 | (c & 63));
        }
    }
}

PBB_HD inline void make_name(const Launch& input, Id id, char* out) {
    for (int i = 0; i < input.prefix_len; ++i) out[i] = input.prefix[i];
    make_fill(input, id, out + input.prefix_len);
    for (int i = 0; i < input.suffix_len; ++i) out[input.prefix_len + input.fill_len + i] = input.suffix[i];
    out[input.name_len] = 0;
}

template<class Plan>
PBB_HD inline Plan make_plan(const Rules& rules) {
    Plan plan{};
    for (int m = 0; m < families; ++m) {
        plan.comp[m] = rules.enabled[m] != 0;
        plan.order[m] = rules.order[m];
        plan.minv[m] = rules.minimum[m];
        plan.pttv[m] = rules.potential[m];
    }
    plan.norder = rules.norder;
    plan.any_mode = rules.any != 0;
    return plan;
}

template<class Name>
PBB_HD inline void init_candidate(Name& name, const Launch& input) {
    // 只复制稳定的前缀快照；工作数组由原算法填充。
    for (int i = 0; i < 256; ++i) {
        name.val_base[i] = input.team_box[i];
        name.val_base2[i] = input.prefix_box[i];
    }
    name.PRELEN = input.prefix_len + 1;
    name.NAMELEN = input.name_len;
    name.i_pre = input.i_pre;
    name.j_pre = input.j_pre;
    name.s_pre = input.s_pre;
}

template<class Result>
PBB_HD inline Flags selection(const Rules& rules, const Result& r) {
    if (!r.flag) return {};
    bool keep = !rules.any_gate;
    bool ptt = false;
    for (int m = 0; m < families; ++m) {
        if (!rules.enabled[m]) continue;
        if (rules.minimum[m] > 0 && r.sc[m] >= rules.minimum[m]) keep = true;
        if (rules.potential[m] > 0 && r.sc_max[m] >= rules.potential[m]) keep = true;
        if (r.sc_max[m] != 0) ptt = true;
    }
    bool blue = false;
    if (rules.collect_mode) {
        const int hp = r.props[7];
        int hl = r.props[0];
        for (int i = 1; i < 7; ++i) if (r.props[i] < hl) hl = r.props[i];
        if (rules.collect_mode == 1) {
            blue = r.sum >= 777 || r.sum * 3 - hp >= 2000 || (hp == 398 && r.sum >= 741) || hl >= 93;
        } else {
            blue = r.sum >= rules.eight || r.sum - hp / 3 >= rules.seven
                || (hp == 398 && r.sum >= rules.hp398) || hl >= rules.floor;
        }
    }
    return {keep, blue, ptt};
}

template<class Result>
PBB_HD inline Record wire(U index, Flags f, const Result& r) {
    Record out{};
    out.index = index;
    out.passed = r.flag;
    out.keep = f.keep;
    out.blue = f.blue;
    out.ptt = f.ptt;
    out.xp = r.xp;
    out.xd = r.xd;
    out.sum = r.sum;
    out.tag = r.flag3;
    out.shadow = r.shadow;
    out.illusion = r.illusion_freq;
    out.pp_ref = r.qp_reference_pp;
    out.qp_ref = r.qp_reference_qp;
    out.effective = r.illusion_effective;
    out.strength = r.shadow_strength;
    out.deduction = r.qp_deduction;
    for (int i = 0; i < families; ++i) { out.scores[i] = r.sc[i]; out.potential[i] = r.sc_max[i]; }
    for (int i = 0; i < 8; ++i) out.props[i] = r.props[i];
    for (int i = 0; i < 16; ++i) { out.ids[i] = r.skills[i]; out.freqs[i] = r.freqs[i]; }
    return out;
}

#undef PBB_HD

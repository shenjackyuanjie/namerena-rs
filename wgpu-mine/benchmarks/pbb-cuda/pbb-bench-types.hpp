#pragma once
#include <cstdint>

// 只跨核运输明确字段，不依赖 PBB CPU 与 GPU 结构体的布局相同。
constexpr int families = 10;
constexpr int name_limit = 252;  // 上游 shadow_name[260] 还需 7 字节后缀及 NUL。
constexpr int encoded_width = 18;
using U = std::uint32_t;
using Id = std::uint64_t;

struct Rules {
    int enabled[families], order[families], minimum[families], potential[families];
    int norder, any, any_gate;
    int collect_mode, eight, seven, hp398, floor;
};
struct Launch {
    Rules rules;
    char prefix[name_limit], suffix[name_limit];
    int prefix_len, suffix_len, name_len;
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

// 这里的名字枚举与原 CUDA 基准相同；PBB 专注自己原版评分，而不是替代它的字符集/UI。
__host__ __device__ inline void make_name(const Launch& input, Id id, char* out) {
    for (int i = 0; i < input.prefix_len; ++i) out[i] = input.prefix[i];
    const U points[6] = {
        0x4e00 + U((id >> 50) & 0x3fff), 0x4e00 + U((id >> 36) & 0x3fff),
        0x4e00 + U((id >> 22) & 0x3fff), 0x4e00 + U((id >> 8) & 0x3fff),
        0x4e00 + U((id & 0xff) << 6), 0x3d01,
    };
    for (int i = 0; i < 6; ++i) {
        U c = points[i];
        char* next = out + input.prefix_len + 3 * i;
        next[0] = char(0xe0 | (c >> 12));
        next[1] = char(0x80 | ((c >> 6) & 63));
        next[2] = char(0x80 | (c & 63));
    }
    for (int i = 0; i < input.suffix_len; ++i) out[input.prefix_len + encoded_width + i] = input.suffix[i];
    out[input.name_len] = 0;
}

template<class Plan>
__host__ __device__ inline Plan make_plan(const Rules& rules) {
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
__host__ __device__ inline void init_candidate(Name& name, const Launch& input) {
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
__host__ __device__ inline Flags selection(const Rules& rules, const Result& r) {
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
__host__ __device__ inline Record wire(U index, Flags f, const Result& r) {
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

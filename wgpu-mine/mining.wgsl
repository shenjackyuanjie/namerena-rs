// 一候选一 lane；转置 Sbox 共占 32KiB，每个 invocation 独占一列。
// 本文件不含跨 lane barrier，因此尾批与属性筛选均可直接 return。
// host 每批清零 Output 的 16B header，records 至少分配 count 个 176B 槽。
// team_state 是 host 对 raw team 执行一次 KSA 后缓存的 256 个字节（各占 u32）。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;

struct NameInfo {
    offset: u32,
    length: u32,
}
@group(0) @binding(1) var<storage, read> names: array<NameInfo>;
// 紧凑 UTF-8：offset/length 均以字节计，每个 u32 低字节在先。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>;

// 严格 44 个 32bit 字，align=4、size/stride=176；不增加 vec3 对齐填充。
struct Record {
    index: u32,                       // 字节偏移 0
    props: array<u32, 8>,              // 字节偏移 4
    skill_ids: array<u32, 16>,         // 字节偏移 36
    skill_freqs: array<u32, 16>,       // 字节偏移 100
    shadow_numerator: i32,            // 字节偏移 164
    xp: f32,                          // 字节偏移 168
    xd: f32,                          // 字节偏移 172
}
struct Output {
    count: atomic<u32>,
    prop_pass: atomic<u32>,
    scored: atomic<u32>,
    shadow_count: atomic<u32>,
    records: array<Record>,           // 字节偏移 16
}
@group(0) @binding(3) var<storage, read_write> output: Output;

// 八个 u32，uniform 共 32B；host 将 prop_threshold 限制到 3*p 不溢出。
struct Params {
    count: u32,
    prop_threshold: u32,
    xp_threshold: u32,
    xd_threshold: u32,
    capture_all: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;
// weights[0] = vec2(BASE, BASE_QD)；weights[k+1] 对应第 k 项 XP/XD 系数。
// 一共 1035 个 vec2<f32>，storage stride=8，共 8280B。
@group(0) @binding(5) var<storage, read> weights: array<vec2<f32>>;

var<workgroup> values: array<u32, 8192>;

fn restore_team(lane: u32) {
    for (var i = 0u; i < 256u; i += 1u) {
        values[i * 32u + lane] = team_state[i / 4u][i % 4u];
    }
}

// 以原始名字字节拼接 ?shadow；绝不解析名字内的 @，也不改 raw team。
fn shadow_byte(index: u32) -> u32 {
    switch index {
        case 0u: { return 63u; }   // ?
        case 1u: { return 115u; }  // s
        case 2u: { return 104u; }  // h
        case 3u: { return 97u; }   // a
        case 4u: { return 100u; }  // d
        case 5u: { return 111u; }  // o
        default: { return 119u; } // w，唯一剩余的合法索引为 6
    }
}

// k=0 是编码前导零；k>0 对应 name 或 name+?shadow 的第 k-1 个字节。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    if (k == 0u) { return 0u; }
    let byte_index = k - 1u;
    if (byte_index < info.length) {
        let address = info.offset + byte_index;
        return (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u;
    }
    return shadow_byte(byte_index - info.length);
}

// raw name 最长 256B，shadow 拼接后内部允许 263B；两轮均重置 s/k。
fn name_ksa(lane: u32, info: NameInfo, with_shadow: bool) {
    let key_length = info.length + select(0u, 7u, with_shadow);
    for (var round = 0u; round < 2u; round += 1u) {
        var s = 0u;
        var k = 0u;
        for (var i = 0u; i < 256u; i += 1u) {
            let address_i = i * 32u + lane;
            let current = values[address_i];
            s = (s + key_byte(info, k) + current) & 255u;
            let address_s = s * 32u + lane;
            values[address_i] = values[address_s];
            values[address_s] = current;
            if (k == key_length) { k = 0u; } else { k += 1u; }
        }
    }
}

// Sbox 是完整排列，仿射映射亦为排列，[89,217) 恰好命中 128 项。
// 按原 Sbox 顺序稳定压缩，不排序，不修改 PRGA 使用的 Sbox。
fn collect_base(lane: u32, base: ptr<function, array<u32, 128>>) {
    var count = 0u;
    for (var i = 0u; i < 256u; i += 1u) {
        let mapped = (values[i * 32u + lane] * 181u + 160u) & 255u;
        if (mapped >= 89u && mapped < 217u) {
            (*base)[count] = mapped & 63u;
            count += 1u;
        }
    }
}

fn median3(x: u32, y: u32, z: u32) -> u32 {
    return max(min(x, y), min(max(x, y), z));
}

fn properties(base: ptr<function, array<u32, 128>>) -> array<u32, 8> {
    // HP 排序只作用于局部副本，保留 skills 所用完整原 base。
    var hp: array<u32, 10>;
    for (var i = 0u; i < 10u; i += 1u) {
        hp[i] = (*base)[i];
    }
    for (var i = 1u; i < 10u; i += 1u) {
        let value = hp[i];
        var j = i;
        loop {
            if (j == 0u) { break; }
            if (hp[j - 1u] <= value) { break; }
            hp[j] = hp[j - 1u];
            j -= 1u;
        }
        hp[j] = value;
    }
    var props: array<u32, 8>;
    props[0] = 154u + hp[3] + hp[4] + hp[5] + hp[6];
    for (var i = 1u; i < 8u; i += 1u) {
        let start = 10u + (i - 1u) * 3u;
        props[i] = median3((*base)[start], (*base)[start + 1u], (*base)[start + 2u]) + 36u;
    }
    return props;
}

fn prga_byte(lane: u32, a: ptr<function, u32>, b: ptr<function, u32>) -> u32 {
    *a = (*a + 1u) & 255u;
    let address_a = *a * 32u + lane;
    let current = values[address_a];
    *b = (*b + current) & 255u;
    let address_b = *b * 32u + lane;
    values[address_a] = values[address_b];
    values[address_b] = current;
    let slot = (values[address_a] + values[address_b]) & 255u;
    return values[slot * 32u + lane];
}

fn finite_score(value: f32) -> bool {
    // 位级检测同时覆盖 NaN 和正负 Inf，避免比较 NaN 时静默丢弃记录。
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(32)
fn mine(@builtin(global_invocation_id) position: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let index = position.x;
    if (index >= params.count) { return; }
    let info = names[index];

    restore_team(lane);
    name_ksa(lane, info, false);
    var base: array<u32, 128>;
    collect_base(lane, &base);
    var props = properties(&base);

    var prop_numerator = props[0];
    for (var i = 1u; i < 8u; i += 1u) {
        prop_numerator += 3u * props[i];
    }
    if (params.capture_all == 0u && prop_numerator <= 3u * params.prop_threshold) {
        return;
    }
    // capture_all 模式统计所有进入技能阶段者，即本批实际候选数。
    atomicAdd(&output.prop_pass, 1u);

    var skill_ids: array<u32, 40>;
    var skill_freqs: array<u32, 16>;
    for (var i = 0u; i < 40u; i += 1u) {
        skill_ids[i] = i;
    }
    // 两轮共同保留 a/b/s；每轮 40 项，每项消耗两个 PRGA 字节，共 160B。
    var a = 0u;
    var b = 0u;
    var s = 0u;
    for (var round = 0u; round < 2u; round += 1u) {
        for (var i = 0u; i < 40u; i += 1u) {
            let high = prga_byte(lane, &a, &b);
            let low = prga_byte(lane, &a, &b);
            let rnd = ((high << 8u) | low) % 40u;
            // 保留 Rust 的 u8 截断再 %40，不可改成直接 %40。
            s = ((s + rnd + skill_ids[i]) & 255u) % 40u;
            let current = skill_ids[i];
            skill_ids[i] = skill_ids[s];
            skill_ids[s] = current;
        }
    }

    var last = -1i;
    for (var i = 0u; i < 16u; i += 1u) {
        let start = 64u + i * 4u;
        let p = min(min(base[start], base[start + 1u]), min(base[start + 2u], base[start + 3u]));
        if (p > 10u && skill_ids[i] < 35u) {
            skill_freqs[i] = p - 10u;
            if (p < 35u) { last = i32(i); }
        } else {
            skill_freqs[i] = 0u;
        }
    }
    if (last >= 0i) {
        skill_freqs[u32(last)] *= 2u;
    }
    // last=14/15 时该槽已翻倍，不再叠加自身的尾槽加成；另一槽照常。
    if (last != 14i && skill_freqs[14] != 0u) {
        skill_freqs[14] += min(min(base[60], base[61]), skill_freqs[14]);
    }
    if (last != 15i && skill_freqs[15] != 0u) {
        skill_freqs[15] += min(min(base[62], base[63]), skill_freqs[15]);
    }

    var features: array<f32, 44>;
    for (var i = 0u; i < 8u; i += 1u) {
        features[i] = f32(props[i]);
    }
    for (var i = 0u; i < 16u; i += 1u) {
        if (skill_freqs[i] != 0u) {
            features[skill_ids[i] + 8u] = f32(skill_freqs[i]);
        }
    }

    var shadow_numerator = 0i;
    if (features[32] > 0.0) {
        atomicAdd(&output.shadow_count, 1u);
        // 技能已结束，直接复用该 lane 的 Sbox/base，原 props/ids/freq 保留。
        restore_team(lane);
        name_ksa(lane, info, true);
        collect_base(lane, &base);
        var shadow_props = properties(&base);
        var raw_sum = 0u;
        for (var i = 1u; i < 8u; i += 1u) {
            raw_sum += shadow_props[i] - 36u;
        }
        shadow_numerator = i32(shadow_props[0] + 3u * raw_sum)
            - 9i * i32(shadow_props[7] - 36u) - 630i;
        features[43] = f32(shadow_numerator) / 3.0 * features[32] / 100.0;
    }
    if (features[42] > 0.0) {
        features[42] += 20.0;
    }

    // 与 poly 完全相同的 44 个线性项及 i=0..43,j=i..43 二次项次序。
    // 不构造 1034 项多项式数组；零特征跳过乘加但仍推进权重索引。
    var score = weights[0];
    var weight_index = 1u;
    for (var i = 0u; i < 44u; i += 1u) {
        if (features[i] != 0.0) {
            score += weights[weight_index] * features[i];
        }
        weight_index += 1u;
    }
    for (var i = 0u; i < 44u; i += 1u) {
        if (features[i] == 0.0) {
            weight_index += 44u - i;
            continue;
        }
        for (var j = i; j < 44u; j += 1u) {
            if (features[j] != 0.0) {
                let term = features[i] * features[j];
                score += weights[weight_index] * term;
            }
            weight_index += 1u;
        }
    }
    atomicAdd(&output.scored, 1u);

    // 10 分安全余量只做保守候选筛选；host 用整数记录重建 f64 后精筛。
    // 非有限结果强制记录，交由 host validation 明确报告，不静默通过/丢弃。
    let invalid_score = !finite_score(score.x) || !finite_score(score.y);
    // Rust as u32 将负分饱和为 0：阈值为 0 时不能丢弃负分候选。
    let possible_hit = (params.xp_threshold == 0u || score.x + 10.0 >= f32(params.xp_threshold))
        && (params.xd_threshold == 0u || score.y + 10.0 >= f32(params.xd_threshold));
    if (params.capture_all == 0u && !invalid_score && !possible_hit) { return; }

    // 每 invocation 最多写一次，故 count <= params.count；无需限容竞争或回退。
    let slot = atomicAdd(&output.count, 1u);
    output.records[slot].index = index;
    output.records[slot].props = props;
    for (var i = 0u; i < 16u; i += 1u) {
        output.records[slot].skill_ids[i] = skill_ids[i];
        output.records[slot].skill_freqs[i] = skill_freqs[i];
    }
    output.records[slot].shadow_numerator = shadow_numerator;
    output.records[slot].xp = score.x;
    output.records[slot].xd = score.y;
}

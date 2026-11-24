// LANES、四张 Q 门平面和阈值在创建管线时由原版 PBB 注入。
struct Setup {
    state: array<u32, 64>,
    key: array<u32, 253>,
    first_step: u32,
    first_cursor: u32,
    first_sum: u32,
    name_len: u32,
    g3_weights: array<u32, 46>,
    g3_minimum: array<u32, 3>,
    fill_mask_lo: u32,
    fill_mask_hi: u32,
    fill_seed_lo: u32,
    fill_seed_hi: u32,
    fill_domain_lo: u32,
    fill_domain_hi: u32,
    fill_shift: u32,
    // 每个 slot 四字（offset、width、radix、table）；随后是每字节一个 u32 的字面量和词表。
    fill: array<u32>,
}
alias FillWords = array<u32, FILL_WORDS>;

struct Params {
    low: u32,
    high: u32,
    count: u32,
    capture: u32,
}
struct Output {
    count: atomic<u32>,
    entries: array<u32>,
}
@group(0) @binding(0) var<storage, read> setup: Setup;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> output: Output;
@group(0) @binding(3) var<storage, read_write> selected: Output;
var<workgroup> sbox: array<u32, 64u * LANES>;

fn utf8(point: u32) -> u32 {
    return (0xe0u | (point >> 12u)) | ((0x80u | ((point >> 6u) & 63u)) << 8u) | ((0x80u | (point & 63u)) << 16u);
}

fn ksa(lane: u32, start: u32, cursor: u32, seed: u32, words: ptr<function, FillWords>) {
    var j = seed;
    var k = cursor;
    for (var i = start; i < 256u; i += 1u) {
        let token = setup.key[k];
        var byte = token;
        if token >= 256u {
            let at = token - 256u;
            byte = ((*words)[at / 3u] >> (8u * (at % 3u))) & 255u;
        }
        k = select(k + 1u, 0u, k == setup.name_len);
        let ai = (i >> 2u) * LANES + lane;
        let a_shift = (i & 3u) * 8u;
        let a_word = sbox[ai];
        let a = (a_word >> a_shift) & 255u;
        j = (j + byte + a) & 255u;
        let bi = (j >> 2u) * LANES + lane;
        let b_shift = (j & 3u) * 8u;
        let b_word = sbox[bi];
        let b = (b_word >> b_shift) & 255u;
        let delta = a ^ b;
        let updated = a_word ^ (delta << a_shift);
        sbox[ai] = updated;
        sbox[bi] = select(b_word, updated, ai == bi) ^ (delta << b_shift);
    }
}

fn median(base: ptr<function, array<u32, 34>>, at: u32) -> i32 {
    let a = (*base)[at];
    let b = (*base)[at + 1u];
    let c = (*base)[at + 2u];
    return i32(max(min(a, b), min(max(a, b), c)));
}

fn compare_swap(hp: ptr<function, array<u32, 10>>, i: u32, j: u32) {
    let a = (*hp)[i];
    let b = (*hp)[j];
    (*hp)[i] = min(a, b);
    (*hp)[j] = max(a, b);
}

fn sorted_hp(base: ptr<function, array<u32, 34>>) -> u32 {
    // Q 门用完整排序网络。旧 V 路径交换这些字节的顺序不影响门中完整排序的结果。
    var hp: array<u32, 10>;
    for (var i = 0u; i < 10u; i += 1u) { hp[i] = (*base)[i]; }
    compare_swap(&hp, 0u, 5u); compare_swap(&hp, 1u, 6u); compare_swap(&hp, 2u, 7u); compare_swap(&hp, 3u, 8u); compare_swap(&hp, 4u, 9u);
    compare_swap(&hp, 0u, 3u); compare_swap(&hp, 1u, 4u); compare_swap(&hp, 5u, 8u); compare_swap(&hp, 6u, 9u);
    compare_swap(&hp, 0u, 2u); compare_swap(&hp, 3u, 6u); compare_swap(&hp, 7u, 9u);
    compare_swap(&hp, 0u, 1u); compare_swap(&hp, 2u, 4u); compare_swap(&hp, 5u, 7u); compare_swap(&hp, 8u, 9u);
    compare_swap(&hp, 1u, 2u); compare_swap(&hp, 3u, 5u); compare_swap(&hp, 4u, 6u); compare_swap(&hp, 7u, 8u);
    compare_swap(&hp, 1u, 3u); compare_swap(&hp, 2u, 5u); compare_swap(&hp, 4u, 7u); compare_swap(&hp, 6u, 8u);
    compare_swap(&hp, 2u, 3u); compare_swap(&hp, 6u, 7u);
    compare_swap(&hp, 3u, 4u); compare_swap(&hp, 5u, 6u);
    compare_swap(&hp, 4u, 5u);
    return 154u + hp[3] + hp[4] + hp[5] + hp[6];
}

fn gate_level(base: ptr<function, array<u32, 34>>) -> u32 {
    let p7 = median(base, 28u);
    if p7 < MIN_P7 { return 0u; }
    let a = W7 * p7 + W2 * median(base, 13u) + W3 * median(base, 16u) + W6 * median(base, 25u);
    if all(a + REM2 < TH) { return 0u; }
    let b = a + W1 * median(base, 10u) + W4 * median(base, 19u) + W5 * median(base, 22u);
    if all(b + REM3 < TH) { return 0u; }

    let accepted = b + W0 * i32(sorted_hp(base)) >= TH;
    if accepted.x { return 2u; }
    return u32(any(accepted.yzw));
}

fn shift_right_id(id: vec2<u32>, shift: u32) -> vec2<u32> {
    if shift == 32u { return vec2(id.y, 0u); }
    return vec2((id.x >> shift) | (id.y << (32u - shift)), id.y >> shift);
}

fn mul_id(id: vec2<u32>, m: u32) -> vec2<u32> {
    // 低 × 低的进位连同两个交叉项，给出低 32 位乘积向高半字的进位。
    let alo = id.x & 65535u;
    let ahi = id.x >> 16u;
    let blo = m & 65535u;
    let bhi = m >> 16u;
    let cross0 = ahi * blo + ((alo * blo) >> 16u);
    let cross1 = alo * bhi + (cross0 & 65535u);
    let carry = ahi * bhi + (cross0 >> 16u) + (cross1 >> 16u);
    return vec2(id.x * m, id.y * m + carry);
}

fn add_id(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2(low, a.y + b.y + u32(low < a.x));
}

fn feistel_hash(value: u32) -> u32 {
    let a = (value ^ (value >> 16u)) * 0x7feb352du;
    let b = (a ^ (a >> 15u)) * 0x846ca68bu;
    return b ^ (b >> 16u);
}

fn feistel_id(id: vec2<u32>, seed: vec2<u32>, mask: vec2<u32>, shift: u32) -> vec2<u32> {
    // 与覆盖掩码取交集，使 domain=1 时两个半区都严格为零。
    let mask_l = mask.x & (0xffffffffu >> (32u - shift));
    let mask_r = shift_right_id(mask, shift).x;
    var l = id.x & mask_l;
    var r = shift_right_id(id, shift).x;
    l ^= feistel_hash(r + seed.x) & mask_l;
    r ^= feistel_hash(l + seed.y) & mask_r;
    l ^= feistel_hash(r + seed.y + 0x9e3779b9u) & mask_l;
    r ^= feistel_hash(l + seed.x + 0x3c6ef372u) & mask_r;
    if shift == 32u { return vec2(l, r); }
    return vec2(l | (r << shift), r >> (32u - shift));
}

fn permute_id(id: vec2<u32>) -> vec2<u32> {
    if FILL_SHUFFLE == 0u { return id; }
    let mask = vec2(setup.fill_mask_lo, setup.fill_mask_hi);
    let seed = vec2(setup.fill_seed_lo, setup.fill_seed_hi);
    let shift = setup.fill_shift;
    let domain = vec2(setup.fill_domain_lo, setup.fill_domain_hi);
    var x = id;
    loop {
        switch FILL_SHUFFLE {
            case 1u: {
                x = add_id(x, seed) & mask;
                x ^= shift_right_id(x, shift);
                x = mul_id(x, 0x9e3779b1u) & mask;
                x ^= shift_right_id(x, shift);
                x = mul_id(x, 0x85ebca77u) & mask;
                x ^= shift_right_id(x, shift);
            }
            case 2u: { x = add_id(mul_id(x, 0x9e3779b1u), seed) & mask; }
            default: { x = feistel_id(x, seed, mask, shift); }
        }
        if all(domain == vec2(0u)) || x.y < domain.y || (x.y == domain.y && x.x < domain.x) {
            break;
        }
    }
    return x;
}

fn div_rem_id(id: vec2<u32>, radix: u32) -> vec3<u32> {
    if id.y == 0u {
        let q = id.x / radix;
        return vec3(q, 0u, id.x - q * radix);
    }
    // 先用 u32 整除高半，再处理低半的两个 16 位 limb。radix <= 65536，拼接的被除数不会溢出。
    let qh = id.y / radix;
    let mid = ((id.y - qh * radix) << 16u) | (id.x >> 16u);
    let qm = mid / radix;
    let low = ((mid - qm * radix) << 16u) | (id.x & 65535u);
    let ql = low / radix;
    return vec3((qm << 16u) | ql, qh, low - ql * radix);
}

fn pack_fill_byte(words: ptr<function, FillWords>, at: u32, byte: u32) {
    (*words)[at / 3u] |= byte << (8u * (at % 3u));
}

fn mixed_words(id: vec2<u32>) -> FillWords {
    var words: FillWords;
    const data = FILL_SLOTS_COUNT * 4u;
    for (var i = 0u; i < FILL_LEN; i += 1u) {
        pack_fill_byte(&words, i, setup.fill[data + i]);
    }
    var q = id;
    // 右端是最低位；先填字面量（变量位置为 0），再写各变量槽。
    for (var i = data; i > 0u; i -= 4u) {
        let at = i - 4u;
        let offset = setup.fill[at];
        let width = setup.fill[at + 1u];
        let parts = div_rem_id(q, setup.fill[at + 2u]);
        q = parts.xy;
        let entry = data + setup.fill[at + 3u] + parts.z * width;
        for (var k = 0u; k < width; k += 1u) {
            pack_fill_byte(&words, offset + k, setup.fill[entry + k]);
        }
    }
    return words;
}

fn name_words(i: u32) -> FillWords {
    let low = params.low + i;
    let high = params.high + u32(low < params.low);
    let id = permute_id(vec2(low, high));
    if FILL_SLOTS_COUNT != 0u { return mixed_words(id); }

    // 保留旧版的前五字；末尾的旧版标记字不参与新名字生成。
    var words: FillWords;
    words[0] = utf8(0x4e00u + ((id.y >> 18u) & 0x3fffu));
    words[1] = utf8(0x4e00u + ((id.y >> 4u) & 0x3fffu));
    words[2] = utf8(0x4e00u + (((id.y << 10u) | (id.x >> 22u)) & 0x3fffu));
    words[3] = utf8(0x4e00u + ((id.x >> 8u) & 0x3fffu));
    words[4] = utf8(0x4e00u + ((id.x & 255u) << 6u));
    return words;
}

fn load_name(lane: u32, words: ptr<function, FillWords>) {
    // 每个 lane 独占并完整恢复其列；不需要工作组同步或初始化。
    for (var w = 0u; w < 64u; w += 1u) { sbox[w * LANES + lane] = setup.state[w]; }
    ksa(lane, setup.first_step, setup.first_cursor, setup.first_sum, words);
    ksa(lane, 0u, setup.name_len, 0u, words);
}

fn push_property(base: ptr<function, array<u32, 34>>, at: u32, raw: u32) -> u32 {
    // 平移 89 后，原区间 [89,217) 恰为 [0,128)。无效字节的暂存槽会被下一项覆盖。
    let mapped = (raw * 181u + 71u) & 255u;
    (*base)[at] = (mapped + 89u) & 63u;
    return at + u32(mapped < 128u);
}

fn collect_properties(lane: u32, limit: u32, base: ptr<function, array<u32, 34>>) -> u32 {
    var filled = 0u;
    for (var w = 0u; w < limit && filled < 31u; w += 1u) {
        let word = sbox[w * LANES + lane];
        // 从最多 30 项进入，四次写入的下标至多为 33；只消费前 31 项。
        filled = push_property(base, filled, word);
        filled = push_property(base, filled, word >> 8u);
        filled = push_property(base, filled, word >> 16u);
        filled = push_property(base, filled, word >> 24u);
    }
    return min(filled, 31u);
}

@compute @workgroup_size(LANES)
fn prefilter(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * LANES;
    if i >= params.count { return; }
    var words = name_words(i);
    load_name(lane, &words);
    var base: array<u32, 34>;
    _ = collect_properties(lane, 64u, &base);
    let tier = gate_level(&base);
    if params.capture != 0u {
        output.entries[i] = tier;
        if tier != 0u { atomicAdd(&output.count, 1u); }
    } else if tier != 0u {
        output.entries[atomicAdd(&output.count, 1u)] = i | (u32(tier == 2u) << 31u);
    }
}

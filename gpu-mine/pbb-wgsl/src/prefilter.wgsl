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
    key_pairs: array<u32, 16>,
}
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

fn ksa(lane: u32, start: u32, cursor: u32, seed: u32, words: ptr<function, array<u32, 6>>) {
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

fn median(base: ptr<function, array<u32, 31>>, at: u32) -> i32 {
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

fn sorted_hp(base: ptr<function, array<u32, 31>>) -> u32 {
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

fn gate_level(base: ptr<function, array<u32, 31>>) -> u32 {
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

fn name_words(i: u32) -> array<u32, 6> {
    let low = params.low + i;
    let high = params.high + u32(low < params.low);
    return array<u32, 6>(
        utf8(0x4e00u + ((high >> 18u) & 0x3fffu)),
        utf8(0x4e00u + ((high >> 4u) & 0x3fffu)),
        utf8(0x4e00u + (((high << 10u) | (low >> 22u)) & 0x3fffu)),
        utf8(0x4e00u + ((low >> 8u) & 0x3fffu)),
        utf8(0x4e00u + ((low & 255u) << 6u)),
        utf8(0x3d01u),
    );
}

fn load_name(lane: u32, words: ptr<function, array<u32, 6>>, crop: bool) {
    // 每个 lane 独占并完整恢复其列；不需要工作组同步或初始化。
    for (var w = 0u; w < 64u; w += 1u) { sbox[w * LANES + lane] = setup.state[w]; }
    if FAST_KEY {
        ksa_fast(lane, words, crop);
    } else {
        ksa(lane, setup.first_step, setup.first_cursor, setup.first_sum, words);
        ksa(lane, 0u, setup.name_len, 0u, words);
    }
}

fn collect_properties(lane: u32, limit: u32, base: ptr<function, array<u32, 31>>) -> u32 {
    var filled = 0u;
    for (var w = 0u; w < limit && filled < 31u; w += 1u) {
        let word = sbox[w * LANES + lane];
        for (var b = 0u; b < 4u && filled < 31u; b += 1u) {
            let value = ((word >> (8u * b)) * 181u + 160u) & 255u;
            if 89u <= value && value < 217u {
                (*base)[filled] = value & 63u;
                filled += 1u;
            }
        }
    }
    return filled;
}

@compute @workgroup_size(LANES)
fn prefilter(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * LANES;
    if i >= params.count { return; }
    var words = name_words(i);
    load_name(lane, &words, FAST_KEY);
    var base: array<u32, 31>;
    if FAST_KEY {
        if collect_properties(lane, 22u, &base) != 31u {
            // 未收齐时重新计算完整排列，不能从裁剪之后的尾部补值。
            load_name(lane, &words, false);
            _ = collect_properties(lane, 64u, &base);
        }
    } else { _ = collect_properties(lane, 64u, &base); }
    let tier = gate_level(&base);
    if params.capture != 0u {
        output.entries[i] = tier;
        if tier != 0u { atomicAdd(&output.count, 1u); }
    } else if tier != 0u {
        output.entries[atomicAdd(&output.count, 1u)] = i | (u32(tier == 2u) << 31u);
    }
}

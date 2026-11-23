// 定长前核：调用方已用 setup.state 恢复本 lane 的列。
// 仅用于 31 字节前缀、无后缀，且 setup.first_step=32、setup.first_cursor=31、setup.name_len=49。

// i 为偶数，两个顺序槽在同一个共享字中。
fn ksa_fast_pair(lane: u32, i: u32, sum: u32, keys: u32) -> u32 {
    let at = (i >> 2u) * LANES + lane;
    let shift = (i & 2u) * 8u;
    let fixed_word = sbox[at];
    let pair = fixed_word >> shift;
    let a = pair & 255u;
    let c0 = (pair >> 8u) & 255u;

    let p0 = sum + (keys & 255u) + a;
    let p1 = p0 + (keys >> 8u);
    let j0 = p0 & 255u;
    let hit_next = j0 == i + 1u;
    let c = select(c0, a, hit_next);
    let j1 = select(p1 + c0, p1 + a, hit_next) & 255u;

    let at0 = (j0 >> 2u) * LANES + lane;
    let at1 = (j1 >> 2u) * LANES + lane;
    // 两个随机字都在别名转发、写回之前读取，不等待第一换提交。
    let word0 = sbox[at0];
    let word1 = sbox[at1];
    let s0 = (j0 & 3u) * 8u;
    let s1 = (j1 & 3u) * 8u;
    let b = (word0 >> s0) & 255u;
    let d0 = (word1 >> s1) & 255u;
    let d_after_j0 = select(d0, a, j1 == j0);
    let d = select(d_after_j0, b, j1 == i);

    // 随机写同字时合并；落到顺序字的随机写一并由顺序字提交。
    let same_word = at0 == at1;
    let updated0 = word0 ^ ((b ^ a) << s0);
    let updated1 = select(word1, updated0, same_word) ^ ((d_after_j0 ^ c) << s1);
    let updated = select(select(fixed_word, updated0, at == at0), updated1, at == at1);
    let final_lo = select(b, c, j1 == i);
    // j1 == i+1 时 d 已是第一换后的顺序值 c。别名处由较晚的写回覆盖。
    sbox[at0] = updated0;
    sbox[at1] = updated1;
    sbox[at] = (updated & ~(65535u << shift)) | ((final_lo | (d << 8u)) << shift);
    return j1;
}

fn ksa_fast_put(word: u32, slot: u32, value: u32) -> u32 {
    let shift = (slot & 3u) * 8u;
    return (word & ~(255u << shift)) | (value << shift);
}

// 先提交一对随机写，后一对从共享内存读到已提交的字。
fn ksa_fast_crop_put_pair(lane: u32, j0: u32, j1: u32, r0: u32, r1: u32) {
    let at0 = (j0 >> 2u) * LANES + lane;
    let at1 = (j1 >> 2u) * LANES + lane;
    let word0 = sbox[at0];
    let word1 = sbox[at1];
    let updated0 = ksa_fast_put(word0, j0, r0);
    sbox[at0] = updated0;
    sbox[at1] = ksa_fast_put(select(word1, updated0, at0 == at1), j1, r1);
}

// i 按字对齐。每组开始时，[0,88) 和未处理区间仍与完整 KSA 一致。
// 顺序槽处理完就不再读取，因此只提交随机写；组内按最后一次命中转发。
fn ksa_fast_crop_word(lane: u32, i: u32, sum: u32, keys01: u32, keys23: u32) -> u32 {
    let x = sbox[(i >> 2u) * LANES + lane];
    let a0 = x & 255u;
    let a1 = (x >> 8u) & 255u;
    let a2 = (x >> 16u) & 255u;
    let a3 = x >> 24u;
    let p0 = sum + (keys01 & 255u) + a0;
    let p1 = p0 + (keys01 >> 8u);
    let j0 = p0 & 255u;
    let hit_next = j0 == i + 1u;
    let r1 = select(a1, a0, hit_next);
    let j1 = select(p1 + a1, p1 + a0, hit_next) & 255u;
    ksa_fast_crop_put_pair(lane, j0, j1, a0, r1);

    let r2 = select(select(a2, a0, j0 == i + 2u), r1, j1 == i + 2u);
    let j2 = (j1 + (keys23 & 255u) + r2) & 255u;
    let r3 = select(select(select(a3, a0, j0 == i + 3u), r1, j1 == i + 3u), r2, j2 == i + 3u);
    let j3 = (j2 + (keys23 >> 8u) + r3) & 255u;
    ksa_fast_crop_put_pair(lane, j2, j3, r2, r3);
    return j3;
}

// 50 步键环：NUL、31 字节固定前缀、18 字节生成段。
// head 是 NUL 和前缀首字节；其余前缀对只读取字面量，不解码 token。
fn ksa_fast_prefix_pair(k: u32) -> u32 {
    return setup.key_pairs[k / 2u];
}

fn ksa_fast_pack(a: u32, b: u32) -> vec3<u32> {
    return vec3<u32>(a & 65535u, ((a >> 16u) & 255u) | ((b & 255u) << 8u), (b >> 8u) & 65535u);
}

fn ksa_fast_full_prefix(lane: u32, i: u32, sum: u32, head: u32, end: u32) -> u32 {
    var j = ksa_fast_pair(lane, i, sum, head);
    // FULL_LITERAL_PAIRS
    return j;
}

fn ksa_fast_full_gen6(lane: u32, i: u32, sum: u32, keys: vec3<u32>) -> u32 {
    let j0 = ksa_fast_pair(lane, i, sum, keys.x);
    let j1 = ksa_fast_pair(lane, i + 2u, j0, keys.y);
    return ksa_fast_pair(lane, i + 4u, j1, keys.z);
}

fn ksa_fast_full_gen(lane: u32, i: u32, sum: u32, g0: vec3<u32>, g1: vec3<u32>, g2: vec3<u32>) -> u32 {
    let j0 = ksa_fast_full_gen6(lane, i, sum, g0);
    let j1 = ksa_fast_full_gen6(lane, i + 6u, j0, g1);
    return ksa_fast_full_gen6(lane, i + 12u, j1, g2);
}

fn ksa_fast_full_round(lane: u32, start: u32, sum: u32, head: u32, g0: vec3<u32>, g1: vec3<u32>, g2: vec3<u32>) {
    var j = sum;
    for (var i = start; i < 250u; i += 50u) {
        j = ksa_fast_full_prefix(lane, i, j, head, 32u);
        j = ksa_fast_full_gen(lane, i + 32u, j, g0, g1, g2);
    }
    _ = ksa_fast_full_prefix(lane, 250u, j, head, 6u);
}

fn ksa_fast_crop_gen12(lane: u32, i: u32, sum: u32, g0: vec3<u32>, g1: vec3<u32>) -> u32 {
    let j0 = ksa_fast_crop_word(lane, i, sum, g0.x, g0.y);
    let j1 = ksa_fast_crop_word(lane, i + 4u, j0, g0.z, g1.x);
    return ksa_fast_crop_word(lane, i + 8u, j1, g1.y, g1.z);
}

// 一个偶数键环，加上下个键环开头两步，整组保持四槽对齐。
fn ksa_fast_crop52(lane: u32, i: u32, sum: u32, head: u32, g0: vec3<u32>, g1: vec3<u32>, g2: vec3<u32>) -> u32 {
    var j = ksa_fast_crop_word(lane, i, sum, head, ksa_fast_prefix_pair(2u));
    // CROPPED_LITERAL_PAIRS
    j = ksa_fast_crop_gen12(lane, i + 32u, j, g0, g1);
    j = ksa_fast_crop_word(lane, i + 44u, j, g2.x, g2.y);
    return ksa_fast_crop_word(lane, i + 48u, j, g2.z, head);
}

// crop=true 时只有前 22 个字是真实结果；收不满属性时须由调用方走完整路径。
fn ksa_fast(lane: u32, words: ptr<function, array<u32, 6>>, crop: bool) {
    let head = setup.key_pairs[0u];
    let g0 = ksa_fast_pack((*words)[0], (*words)[1]);
    let g1 = ksa_fast_pack((*words)[2], (*words)[3]);
    let g2 = ksa_fast_pack((*words)[4], (*words)[5]);
    let first = ksa_fast_full_gen(lane, 32u, setup.first_sum, g0, g1, g2);
    ksa_fast_full_round(lane, 50u, first, head, g0, g1, g2);

    if !crop {
        ksa_fast_full_round(lane, 0u, 0u, head, g0, g1, g2);
        return;
    }

    var j = ksa_fast_full_prefix(lane, 0u, 0u, head, 32u);
    j = ksa_fast_full_gen(lane, 32u, j, g0, g1, g2);
    j = ksa_fast_full_prefix(lane, 50u, j, head, 32u);
    j = ksa_fast_full_gen6(lane, 82u, j, g0); // 第二轮完整跑完 [0,88)。
    j = ksa_fast_crop_gen12(lane, 88u, j, g1, g2);
    j = ksa_fast_crop52(lane, 100u, j, head, g0, g1, g2);
    for (var k = 2u; k < 30u; k += 4u) {
        j = ksa_fast_crop_word(lane, 150u + k, j, ksa_fast_prefix_pair(k), ksa_fast_prefix_pair(k + 2u));
    }
    j = ksa_fast_crop_word(lane, 180u, j, ksa_fast_prefix_pair(30u), g0.x);
    j = ksa_fast_crop_word(lane, 184u, j, g0.y, g0.z);
    j = ksa_fast_crop_gen12(lane, 188u, j, g1, g2);
    j = ksa_fast_crop52(lane, 200u, j, head, g0, g1, g2);
    _ = ksa_fast_crop_word(lane, 252u, j, ksa_fast_prefix_pair(2u), ksa_fast_prefix_pair(4u));
}

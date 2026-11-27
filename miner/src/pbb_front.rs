//! 纯 Rust 的 PBB 一阶 Q 门前核（RWIR）。
//!
//! 管道：ID 置换 → 五字 UTF-8 键字直出（SSE2/AVX2）→ 双轮 KSA（8 路交错 `[u64; 256]`）→
//! AVX2 保留区转置 → 属性窗口压缩（AVX2 + BMI2 `pdep/pext`）→ 四平面整数门控 + 29 路排序网络。
//! 收不齐 31 项属性的极少数候选（实测 1e-5 量级）在 Rust 内按原版完整排列复核，
//! 因此 C++ 侧只在过门（1/2 档）候选上做 1124 维浮点精算 + CLI/CSV。
//!
//! 数据面直接读取 C++ 的 `Launch`（`gpu-mine/benchmarks/pbb-cuda/pbb-bench-types.hpp`），
//! 布局由 `tswn_rust_launch_size()` 与 C++ 侧 `sizeof(Launch)` 在初始化时核对。

use core::arch::x86_64::*;
use core::ffi::c_void;
use core::ptr;

/// 一次 `levels` 调用处理的候选数（= C++ 侧 `front_width`）。
pub const W: usize = 8;
const NAME_LIMIT: usize = 252;
/// 属性压缩只看前 `SLOT_LIMIT` 槽（必须是 8 的倍数，最大 96）。
const SLOT_LIMIT: usize = 80;
/// 保留区解交错后每路缓存大小（向上取整到 32 字节对齐，供 AVX2 安全读入）。
const UNWEAVE_STRIDE: usize = SLOT_LIMIT.div_ceil(32) * 32;
/// 需要收齐的属性字节数。
const NEEDED: usize = 31;

/// 64 字节对齐的 8 路交错 S-box（2KB，整块驻留 L1d）。
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AlignedBox([u64; 256]);

/// 32 字节对齐的保留区解交错缓冲（8 路 × UNWEAVE_STRIDE）。
#[repr(C, align(32))]
struct Unwoven([[u8; UNWEAVE_STRIDE]; W]);

// ===== 与 pbb-bench-types.hpp 逐字段对齐 =====

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FillSlot {
    pub offset: u32,
    pub width: u32,
    pub radix: u32,
    pub table: u32,
}

#[repr(C)]
pub struct Rules {
    pub enabled: [i32; 10],
    pub order: [i32; 10],
    pub minimum: [i32; 10],
    pub potential: [i32; 10],
    pub norder: i32,
    pub any: i32,
    pub any_gate: i32,
    pub collect_mode: i32,
    pub eight: i32,
    pub seven: i32,
    pub hp398: i32,
    pub floor: i32,
}

#[repr(C)]
pub struct Launch {
    pub rules: Rules,
    pub prefix: [u8; NAME_LIMIT],
    pub suffix: [u8; NAME_LIMIT],
    pub prefix_len: i32,
    pub suffix_len: i32,
    pub name_len: i32,
    pub fill_len: i32,
    pub fill_seed: u64,
    pub fill_domain: u64,
    pub fill_shuffle: u32,
    pub fill_shift: u32,
    pub fill_mask_lo: u32,
    pub fill_mask_hi: u32,
    pub fill_slots_count: u32,
    pub fill_data_count: u32,
    pub fill_slots: *const FillSlot,
    pub fill_data: *const u32,
    pub team_box: [u8; 256],
    pub prefix_box: [u8; 256],
    pub i_pre: u8,
    pub j_pre: u8,
    pub s_pre: u8,
}

const _: () = assert!(core::mem::size_of::<FillSlot>() == 16);
const _: () = assert!(core::mem::size_of::<Rules>() == 192);

/// 编号置换参数快照，创建时读一次。
#[derive(Clone, Copy)]
struct Shuffle {
    kind: u32,
    seed: u64,
    domain: u64,
    shift: u32,
    mask: u64,
}

#[derive(Clone, Copy)]
struct Span {
    start: u16,
    end: u16,
    offset: u16,
    varying: bool,
}

pub struct Front {
    initial: AlignedBox,
    initial_scalar: [u8; 256],
    fixed_first: [u64; 256],
    fixed_second: [u64; 256],
    prefix: [u8; NAME_LIMIT],
    suffix: [u8; NAME_LIMIT],
    prefix_len: usize,
    suffix_len: usize,
    name_len: usize,
    i_pre: usize,
    j_pre: usize,
    s_pre: u8,
    first: Box<[Span]>,
    head: Box<[Span]>,
    tail: Box<[Span]>,
    shuffle: Shuffle,
    slots: Box<[FillSlot]>,
    data: Box<[u32]>,
    fill_len: usize,
    plain_fill: bool,
    avx2: bool,
}

// ===== 编号置换 =====

#[inline(always)]
fn fill_hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

#[inline(always)]
fn shuffle_id(s: &Shuffle, mut id: u64) -> u64 {
    if s.kind == 0 {
        return id;
    }
    loop {
        id = match s.kind {
            1 => {
                let x = id.wrapping_add(s.seed) & s.mask;
                let x = (x ^ (x >> s.shift)).wrapping_mul(0x9e37_79b1) & s.mask;
                let x = (x ^ (x >> s.shift)).wrapping_mul(0x85eb_ca77) & s.mask;
                x ^ (x >> s.shift)
            }
            2 => id.wrapping_mul(0x9e37_79b1).wrapping_add(s.seed) & s.mask,
            _ => {
                let ml = ((1u64 << s.shift) - 1) as u32;
                let mr = (s.mask >> s.shift) as u32;
                let s0 = s.seed as u32;
                let s1 = (s.seed >> 32) as u32;
                let mut l = (id as u32) & ml;
                let mut r = (id >> s.shift) as u32;
                l ^= fill_hash(r.wrapping_add(s0)) & ml;
                r ^= fill_hash(l.wrapping_add(s1)) & mr;
                l ^= fill_hash(r.wrapping_add(s1).wrapping_add(0x9e37_79b9)) & ml;
                r ^= fill_hash(l.wrapping_add(s0).wrapping_add(0x3c6e_f372)) & mr;
                ((r as u64) << s.shift) | ((l as u64) & s.mask)
            }
        };
        if s.domain == 0 || id < s.domain {
            return id;
        }
    }
}

/// 五字 Base16384 填充的 5 个码点。
#[inline(always)]
fn plain_points(v: u64) -> [u32; 5] {
    [
        0x4e00 + ((v >> 50) & 0x3fff) as u32,
        0x4e00 + ((v >> 36) & 0x3fff) as u32,
        0x4e00 + ((v >> 22) & 0x3fff) as u32,
        0x4e00 + ((v >> 8) & 0x3fff) as u32,
        0x4e00 + (((v & 0xff) << 6) as u32),
    ]
}

#[inline(always)]
fn point_bytes(c: u32) -> [u8; 3] {
    [
        (0xe0 | (c >> 12)) as u8,
        (0x80 | ((c >> 6) & 63)) as u8,
        (0x80 | (c & 63)) as u8,
    ]
}

/// 单候选连续 15 字节填充（回退路径构造真名字时用）。
#[inline(always)]
fn plain_run(v: u64, out: &mut [u8]) {
    for (i, cp) in plain_points(v).into_iter().enumerate() {
        out[3 * i..3 * i + 3].copy_from_slice(&point_bytes(cp));
    }
}

/// 标量写入单路 15 字节键字到交错 `fills: [u64; NAME_LIMIT]`。
#[inline(always)]
fn plain_lanes_scalar(v: u64, fills: &mut [u64; NAME_LIMIT], lane: usize) {
    let shift = lane * 8;
    for (i, cp) in plain_points(v).into_iter().enumerate() {
        let [b0, b1, b2] = point_bytes(cp);
        fills[3 * i] |= (b0 as u64) << shift;
        fills[3 * i + 1] |= (b1 as u64) << shift;
        fills[3 * i + 2] |= (b2 as u64) << shift;
    }
}

/// SSE2 一次将 8 个候选的五字 UTF-8 键字直出为交错的 15 个 `u64`。
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn pack_bytes_sse2(left: __m128i, right: __m128i) -> u64 {
    let words = _mm_packs_epi32(left, right);
    _mm_cvtsi128_si64(_mm_packus_epi16(words, _mm_setzero_si128())) as u64
}

#[target_feature(enable = "sse2")]
#[inline]
unsafe fn emit_utf8_sse2<const LAST: bool>(
    fills: &mut [u64; NAME_LIMIT],
    at: usize,
    a: __m128i,
    b: __m128i,
) {
    unsafe {
        let base = _mm_set1_epi32(0x4e00);
        let mask = _mm_set1_epi32(63);
        let lead = _mm_set1_epi32(0xe0);
        let cont = _mm_set1_epi32(0x80);
        let ar = _mm_srli_epi32(_mm_add_epi32(a, base), 6);
        let br = _mm_srli_epi32(_mm_add_epi32(b, base), 6);
        *fills.get_unchecked_mut(at) = pack_bytes_sse2(
            _mm_or_si128(_mm_srli_epi32(ar, 6), lead),
            _mm_or_si128(_mm_srli_epi32(br, 6), lead),
        );
        *fills.get_unchecked_mut(at + 1) = pack_bytes_sse2(
            _mm_or_si128(_mm_and_si128(ar, mask), cont),
            _mm_or_si128(_mm_and_si128(br, mask), cont),
        );
        if LAST {
            *fills.get_unchecked_mut(at + 2) = 0x8080_8080_8080_8080;
        } else {
            *fills.get_unchecked_mut(at + 2) = pack_bytes_sse2(
                _mm_or_si128(_mm_and_si128(a, mask), cont),
                _mm_or_si128(_mm_and_si128(b, mask), cont),
            );
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn plain_batch_sse2(vals: &[u64; W], fills: &mut [u64; NAME_LIMIT]) {
    unsafe {
        let al = _mm_set_epi32(vals[3] as i32, vals[2] as i32, vals[1] as i32, vals[0] as i32);
        let bl = _mm_set_epi32(vals[7] as i32, vals[6] as i32, vals[5] as i32, vals[4] as i32);
        let ah = _mm_set_epi32(
            (vals[3] >> 32) as i32,
            (vals[2] >> 32) as i32,
            (vals[1] >> 32) as i32,
            (vals[0] >> 32) as i32,
        );
        let bh = _mm_set_epi32(
            (vals[7] >> 32) as i32,
            (vals[6] >> 32) as i32,
            (vals[5] >> 32) as i32,
            (vals[4] >> 32) as i32,
        );
        let mask = _mm_set1_epi32(0x3fff);
        let tail = _mm_set1_epi32(255);

        emit_utf8_sse2::<false>(fills, 0, _mm_srli_epi32(ah, 18), _mm_srli_epi32(bh, 18));
        emit_utf8_sse2::<false>(
            fills,
            3,
            _mm_and_si128(_mm_srli_epi32(ah, 4), mask),
            _mm_and_si128(_mm_srli_epi32(bh, 4), mask),
        );
        let mid_a = _mm_and_si128(_mm_or_si128(_mm_srli_epi32(al, 22), _mm_slli_epi32(ah, 10)), mask);
        let mid_b = _mm_and_si128(_mm_or_si128(_mm_srli_epi32(bl, 22), _mm_slli_epi32(bh, 10)), mask);
        emit_utf8_sse2::<false>(fills, 6, mid_a, mid_b);
        emit_utf8_sse2::<false>(
            fills,
            9,
            _mm_and_si128(_mm_srli_epi32(al, 8), mask),
            _mm_and_si128(_mm_srli_epi32(bl, 8), mask),
        );
        emit_utf8_sse2::<true>(
            fills,
            12,
            _mm_slli_epi32(_mm_and_si128(al, tail), 6),
            _mm_slli_epi32(_mm_and_si128(bl, tail), 6),
        );
    }
}

// ===== 门控常量（与 scoring_1124.hpp 一致）=====

const P_MAX: i32 = 63;
const X0_MAX: i32 = 406;

const F_W0: i32 = 431;
const F_W1: i32 = 813;
const F_W2: i32 = 807;
const F_W3: i32 = 1000;
const F_W4: i32 = 989;
const F_W5: i32 = 812;
const F_W6: i32 = 966;
const F_W7: i32 = 924;
const F_TH: i32 = 451_667;

const R_W0: i32 = 372;
const R_W1: i32 = 326;
const R_W2: i32 = 791;
const R_W3: i32 = 1000;
const R_W4: i32 = 542;
const R_W5: i32 = 562;
const R_W6: i32 = 906;
const R_W7: i32 = 928;
const R_TH: i32 = 331_652;

const S_W0: i32 = 215;
const S_W1: i32 = 44;
const S_W2: i32 = 486;
const S_W3: i32 = 815;
const S_W4: i32 = 74;
const S_W5: i32 = 86;
const S_W6: i32 = 563;
const S_W7: i32 = 1000;
const S_TH: i32 = 204_325;

const T_W0: i32 = 51;
const T_W1: i32 = 54;
const T_W2: i32 = 372;
const T_W3: i32 = 750;
const T_W4: i32 = -282;
const T_W5: i32 = 42;
const T_W6: i32 = 12;
const T_W7: i32 = 1000;
const T_TH: i32 = 125_128;

const fn pos(w: i32) -> i32 {
    if w > 0 { w } else { 0 }
}

const F_REM2: i32 = F_W1 * P_MAX + F_W4 * P_MAX + F_W5 * P_MAX + F_W0 * X0_MAX;
const R_REM2: i32 = R_W1 * P_MAX + R_W4 * P_MAX + R_W5 * P_MAX + R_W0 * X0_MAX;
const S_REM2: i32 =
    pos(S_W1) * P_MAX + pos(S_W4) * P_MAX + pos(S_W5) * P_MAX + pos(S_W0) * X0_MAX;
const T_REM2: i32 =
    pos(T_W1) * P_MAX + pos(T_W4) * P_MAX + pos(T_W5) * P_MAX + pos(T_W0) * X0_MAX;
const F_REM3: i32 = F_W0 * X0_MAX;
const R_REM3: i32 = R_W0 * X0_MAX;
const S_REM3: i32 = pos(S_W0) * X0_MAX;
const T_REM3: i32 = pos(T_W0) * X0_MAX;

#[inline(always)]
fn median(a: u8, b: u8, c: u8) -> u8 {
    a.max(b).min(c).max(a.min(b))
}

#[inline(always)]
fn cswap(b: &mut [u8; 10], i: usize, j: usize) {
    let (x, y) = (b[i], b[j]);
    b[i] = x.min(y);
    b[j] = x.max(y);
}

/// 29 次比较交换的升序排序网络，与 `sort10_net` 逐值等价。
#[inline(always)]
fn sort10_net(b: &mut [u8; 10]) {
    const PAIRS: [(usize, usize); 29] = [
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (0, 3), (1, 4), (5, 8), (6, 9),
        (0, 2), (3, 6), (7, 9),
        (0, 1), (2, 4), (5, 7), (8, 9),
        (1, 2), (3, 5), (4, 6), (7, 8),
        (1, 3), (2, 5), (4, 7), (6, 8),
        (2, 3), (6, 7),
        (3, 4), (5, 6),
        (4, 5),
    ];
    for (i, j) in PAIRS {
        cswap(b, i, j);
    }
}

/// 四平面整数门控：0 淘汰 / 1 PRE / 2 FREE。与 `gate_level_of` 判定逐位一致。
#[inline(always)]
fn gate_level_of(nb: &[u8; 48]) -> u8 {
    let p7 = median(nb[28], nb[29], nb[30]) as i32;
    if p7 < 27 {
        return 0;
    }
    let p2 = median(nb[13], nb[14], nb[15]) as i32;
    let p3 = median(nb[16], nb[17], nb[18]) as i32;
    let p6 = median(nb[25], nb[26], nb[27]) as i32;
    let fa = F_W7 * p7 + F_W2 * p2 + F_W3 * p3 + F_W6 * p6;
    let ra = R_W7 * p7 + R_W2 * p2 + R_W3 * p3 + R_W6 * p6;
    let sa = S_W7 * p7 + S_W2 * p2 + S_W3 * p3 + S_W6 * p6;
    let ta = T_W7 * p7 + T_W2 * p2 + T_W3 * p3 + T_W6 * p6;
    if fa + F_REM2 < F_TH && ra + R_REM2 < R_TH && sa + S_REM2 < S_TH && ta + T_REM2 < T_TH {
        return 0;
    }
    let p1 = median(nb[10], nb[11], nb[12]) as i32;
    let p4 = median(nb[19], nb[20], nb[21]) as i32;
    let p5 = median(nb[22], nb[23], nb[24]) as i32;
    let fb = fa + F_W1 * p1 + F_W4 * p4 + F_W5 * p5;
    let rb = ra + R_W1 * p1 + R_W4 * p4 + R_W5 * p5;
    let sb = sa + S_W1 * p1 + S_W4 * p4 + S_W5 * p5;
    let tb = ta + T_W1 * p1 + T_W4 * p4 + T_W5 * p5;
    if fb + F_REM3 < F_TH && rb + R_REM3 < R_TH && sb + S_REM3 < S_TH && tb + T_REM3 < T_TH {
        return 0;
    }
    let mut b: [u8; 10] = nb[..10].try_into().unwrap();
    sort10_net(&mut b);
    let x0 = 154 + b[3] as i32 + b[4] as i32 + b[5] as i32 + b[6] as i32;
    if fb + F_W0 * x0 >= F_TH {
        return 2;
    }
    if rb + R_W0 * x0 >= R_TH || sb + S_W0 * x0 >= S_TH || tb + T_W0 * x0 >= T_TH {
        return 1;
    }
    0
}

/// 标量属性压缩（无 AVX2/BMI2 时的回退，逐位等价）。
fn gate_scalar(sbox: &[u8]) -> Option<u8> {
    let mut attrs = [0u8; 48];
    let mut n = 0usize;
    for &byte in &sbox[..SLOT_LIMIT] {
        let window = byte.wrapping_mul(181).wrapping_add(71);
        if window < 128 {
            attrs[n] = window.wrapping_add(89) & 63;
            n += 1;
            if n == NEEDED {
                return Some(gate_level_of(&attrs));
            }
        }
    }
    None
}

/// AVX2 一次扫 32 槽 + BMI2 `pdep/pext` 稳定压缩。
/// 若 `SLOT_LIMIT` 不是 32 的倍数，最后一段通过掩码截断，严禁读入超出 `SLOT_LIMIT` 的脏槽。
#[target_feature(enable = "avx2,bmi2")]
unsafe fn gate_avx2(sbox: &[u8; UNWEAVE_STRIDE]) -> Option<u8> {
    let mul = _mm256_set1_epi16(181);
    let low_mask = _mm256_set1_epi16(0x00ff);
    let add71 = _mm256_set1_epi8(71);
    let add89 = _mm256_set1_epi8(89);
    let mask63 = _mm256_set1_epi8(63);
    let mut attrs = [0u8; 48];
    let mut raw = [0u8; 32];
    let mut n = 0usize;

    let full_blocks = SLOT_LIMIT / 32;
    let rem = SLOT_LIMIT % 32;
    let total_blocks = UNWEAVE_STRIDE / 32;

    for blk in 0..total_blocks {
        let mut valid = unsafe {
            let x = _mm256_load_si256(sbox.as_ptr().add(blk * 32).cast());
            let lo = _mm256_mullo_epi16(x, mul);
            let hi = _mm256_mullo_epi16(_mm256_srli_epi16(x, 8), mul);
            let window = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(lo, low_mask), _mm256_slli_epi16(hi, 8)),
                add71,
            );
            let bytes = _mm256_and_si256(_mm256_add_epi8(window, add89), mask63);
            _mm256_storeu_si256(raw.as_mut_ptr().cast(), bytes);
            !(_mm256_movemask_epi8(window) as u32)
        };
        if blk == full_blocks && rem != 0 {
            valid &= (1u32 << rem) - 1;
        }
        for (chunk_idx, chunk) in raw.as_chunks::<8>().0.iter().enumerate() {
            let bits = (valid >> (chunk_idx * 8)) & 0xff;
            if bits != 0 {
                let word = u64::from_ne_bytes(*chunk);
                let spread = _pdep_u64(bits as u64, 0x0101_0101_0101_0101).wrapping_mul(0xff);
                let packed = _pext_u64(word, spread);
                // attrs 长 48 字节，n < 31，写入 8 字节恒不越界。
                attrs[n..n + 8].copy_from_slice(&packed.to_ne_bytes());
                n += bits.count_ones() as usize;
                if n >= NEEDED {
                    return Some(gate_level_of(&attrs));
                }
            }
        }
    }
    None
}

// ===== 8 路交错保留区解交错（转置 8x8 字节块）=====

/// 用 SSE2 将 16 个交错 `u64`（16 槽 × 8 路）转置为 8 路连续 16 字节。
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn unweave16_lane<const BASE: usize, const SHIFT: i32>(
    out: &mut Unwoven,
    i: usize,
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) {
    unsafe {
        let mask = _mm_set1_epi32(255);
        let ab = _mm_packs_epi32(
            _mm_and_si128(_mm_srli_epi32::<SHIFT>(a), mask),
            _mm_and_si128(_mm_srli_epi32::<SHIFT>(b), mask),
        );
        let cd = _mm_packs_epi32(
            _mm_and_si128(_mm_srli_epi32::<SHIFT>(c), mask),
            _mm_and_si128(_mm_srli_epi32::<SHIFT>(d), mask),
        );
        _mm_storeu_si128(
            out.0.get_unchecked_mut(BASE + (SHIFT as usize) / 8).as_mut_ptr().add(i).cast(),
            _mm_packus_epi16(ab, cd),
        );
    }
}

/// 将 `box_words[..SLOT_LIMIT]` 解交错回 `out.0[lane][..SLOT_LIMIT]`。
/// 一次处理 16 槽；若 `SLOT_LIMIT` 不是 16 的倍数（如 88），最后 8 槽用标量/半宽补齐。
#[target_feature(enable = "sse2")]
unsafe fn unweave_prefix(box_words: &[u64; 256], out: &mut Unwoven) {
    unsafe {
        let mut i = 0usize;
        while i + 16 <= SLOT_LIMIT {
            let p = box_words.as_ptr().add(i);
            let x0 = _mm_load_si128(p.cast());
            let x1 = _mm_load_si128(p.add(2).cast());
            let x2 = _mm_load_si128(p.add(4).cast());
            let x3 = _mm_load_si128(p.add(6).cast());
            let x4 = _mm_load_si128(p.add(8).cast());
            let x5 = _mm_load_si128(p.add(10).cast());
            let x6 = _mm_load_si128(p.add(12).cast());
            let x7 = _mm_load_si128(p.add(14).cast());
            let a = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0x88>(x0), _mm_shuffle_epi32::<0x88>(x1));
            let b = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0x88>(x2), _mm_shuffle_epi32::<0x88>(x3));
            let c = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0x88>(x4), _mm_shuffle_epi32::<0x88>(x5));
            let d = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0x88>(x6), _mm_shuffle_epi32::<0x88>(x7));
            let e = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0xdd>(x0), _mm_shuffle_epi32::<0xdd>(x1));
            let f = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0xdd>(x2), _mm_shuffle_epi32::<0xdd>(x3));
            let g = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0xdd>(x4), _mm_shuffle_epi32::<0xdd>(x5));
            let h = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0xdd>(x6), _mm_shuffle_epi32::<0xdd>(x7));
            unweave16_lane::<0, 0>(out, i, a, b, c, d);
            unweave16_lane::<0, 8>(out, i, a, b, c, d);
            unweave16_lane::<0, 16>(out, i, a, b, c, d);
            unweave16_lane::<0, 24>(out, i, a, b, c, d);
            unweave16_lane::<4, 0>(out, i, e, f, g, h);
            unweave16_lane::<4, 8>(out, i, e, f, g, h);
            unweave16_lane::<4, 16>(out, i, e, f, g, h);
            unweave16_lane::<4, 24>(out, i, e, f, g, h);
            i += 16;
        }
        while i < SLOT_LIMIT {
            let w = *box_words.get_unchecked(i);
            for lane in 0..W {
                *out.0.get_unchecked_mut(lane).get_unchecked_mut(i) = (w >> (lane * 8)) as u8;
            }
            i += 1;
        }
    }
}

// ===== 8 路交错 KSA（方案 1B：SSE2 _mm_add_epi8）=====

/// 双 8 路（16 候选）并行流水交错单步 KSA：两份独立 S-box（共 4KB，驻留 L1d）交替推进，
/// 隐藏 Zen 3 单路游标递推（vpaddb -> vmovq -> load/store）的串行延迟。
#[inline(always)]
unsafe fn exchange_dual<const CROP: bool>(
    b0: &mut [u64; 256],
    j0: &mut __m128i,
    k0: u64,
    b1: &mut [u64; 256],
    j1: &mut __m128i,
    k1: u64,
    i: usize,
) {
    unsafe {
        let a0 = *b0.get_unchecked(i);
        let a1 = *b1.get_unchecked(i);
        let n0 = _mm_add_epi8(
            *j0,
            _mm_add_epi8(_mm_cvtsi64_si128(a0 as i64), _mm_cvtsi64_si128(k0 as i64)),
        );
        let n1 = _mm_add_epi8(
            *j1,
            _mm_add_epi8(_mm_cvtsi64_si128(a1 as i64), _mm_cvtsi64_si128(k1 as i64)),
        );
        *j0 = n0;
        *j1 = n1;
        let x0 = _mm_cvtsi128_si64(n0) as u64;
        let x1 = _mm_cvtsi128_si64(n1) as u64;

        let r0 = b0.as_mut_ptr().cast::<u8>();
        let p00 = r0.add(((x0 & 255) as usize) << 3);
        let p01 = r0.add((((x0 >> 5) & 2040) | 1) as usize);
        let p02 = r0.add((((x0 >> 13) & 2040) | 2) as usize);
        let p03 = r0.add((((x0 >> 21) & 2040) | 3) as usize);
        let p04 = r0.add((((x0 >> 29) & 2040) | 4) as usize);
        let p05 = r0.add((((x0 >> 37) & 2040) | 5) as usize);
        let p06 = r0.add((((x0 >> 45) & 2040) | 6) as usize);
        let p07 = r0.add((((x0 >> 53) & 2040) | 7) as usize);

        if !CROP {
            let v0 = (*p00 as u64)
                | ((*p01 as u64) << 8)
                | ((*p02 as u64) << 16)
                | ((*p03 as u64) << 24)
                | ((*p04 as u64) << 32)
                | ((*p05 as u64) << 40)
                | ((*p06 as u64) << 48)
                | ((*p07 as u64) << 56);
            *b0.get_unchecked_mut(i) = v0;
        }
        *p00 = a0 as u8;
        *p01 = (a0 >> 8) as u8;
        *p02 = (a0 >> 16) as u8;
        *p03 = (a0 >> 24) as u8;
        *p04 = (a0 >> 32) as u8;
        *p05 = (a0 >> 40) as u8;
        *p06 = (a0 >> 48) as u8;
        *p07 = (a0 >> 56) as u8;

        let r1 = b1.as_mut_ptr().cast::<u8>();
        let p10 = r1.add(((x1 & 255) as usize) << 3);
        let p11 = r1.add((((x1 >> 5) & 2040) | 1) as usize);
        let p12 = r1.add((((x1 >> 13) & 2040) | 2) as usize);
        let p13 = r1.add((((x1 >> 21) & 2040) | 3) as usize);
        let p14 = r1.add((((x1 >> 29) & 2040) | 4) as usize);
        let p15 = r1.add((((x1 >> 37) & 2040) | 5) as usize);
        let p16 = r1.add((((x1 >> 45) & 2040) | 6) as usize);
        let p17 = r1.add((((x1 >> 53) & 2040) | 7) as usize);

        if !CROP {
            let v1 = (*p10 as u64)
                | ((*p11 as u64) << 8)
                | ((*p12 as u64) << 16)
                | ((*p13 as u64) << 24)
                | ((*p14 as u64) << 32)
                | ((*p15 as u64) << 40)
                | ((*p16 as u64) << 48)
                | ((*p17 as u64) << 56);
            *b1.get_unchecked_mut(i) = v1;
        }
        *p10 = a1 as u8;
        *p11 = (a1 >> 8) as u8;
        *p12 = (a1 >> 16) as u8;
        *p13 = (a1 >> 24) as u8;
        *p14 = (a1 >> 32) as u8;
        *p15 = (a1 >> 40) as u8;
        *p16 = (a1 >> 48) as u8;
        *p17 = (a1 >> 56) as u8;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn ksa_dual<const CROP: bool>(
    b0: &mut [u64; 256],
    j0: &mut __m128i,
    f0: &[u64; NAME_LIMIT],
    b1: &mut [u64; 256],
    j1: &mut __m128i,
    f1: &[u64; NAME_LIMIT],
    spans: &[Span],
    fixed: &[u64; 256],
) {
    let mut c0 = *j0;
    let mut c1 = *j1;
    for span in spans {
        let start = span.start as usize;
        let end = span.end as usize;
        if span.varying {
            for (k, i) in (span.offset as usize..).zip(start..end) {
                unsafe {
                    let k0 = *f0.get_unchecked(k);
                    let k1 = *f1.get_unchecked(k);
                    exchange_dual::<CROP>(b0, &mut c0, k0, b1, &mut c1, k1, i);
                }
            }
        } else {
            for i in start..end {
                unsafe {
                    let key = *fixed.get_unchecked(i);
                    exchange_dual::<CROP>(b0, &mut c0, key, b1, &mut c1, key, i);
                }
            }
        }
    }
    *j0 = c0;
    *j1 = c1;
}

/// 8 路交错单步 KSA（SSE2 并行 u8 加法 + 树形合并 + 单次 GPR 读）。
/// Safety: 调用方保证 `i < 256`；`p0..p7` 均在 `box_words` 的 2048 字节范围内。
#[inline(always)]
unsafe fn exchange<const CROP: bool>(box_words: &mut [u64; 256], j: &mut __m128i, i: usize, key: u64) {
    unsafe {
        let a = *box_words.get_unchecked(i);
        let next = _mm_add_epi8(
            *j,
            _mm_add_epi8(_mm_cvtsi64_si128(a as i64), _mm_cvtsi64_si128(key as i64)),
        );
        *j = next;
        let x = _mm_cvtsi128_si64(next) as u64;

        let raw = box_words.as_mut_ptr().cast::<u8>();
        let p0 = raw.add(((x & 255) as usize) << 3);
        let p1 = raw.add((((x >> 5) & 2040) | 1) as usize);
        let p2 = raw.add((((x >> 13) & 2040) | 2) as usize);
        let p3 = raw.add((((x >> 21) & 2040) | 3) as usize);
        let p4 = raw.add((((x >> 29) & 2040) | 4) as usize);
        let p5 = raw.add((((x >> 37) & 2040) | 5) as usize);
        let p6 = raw.add((((x >> 45) & 2040) | 6) as usize);
        let p7 = raw.add((((x >> 53) & 2040) | 7) as usize);

        if !CROP {
            let v01 = (*p0 as u64) | ((*p1 as u64) << 8);
            let v23 = (*p2 as u64) | ((*p3 as u64) << 8);
            let v45 = (*p4 as u64) | ((*p5 as u64) << 8);
            let v67 = (*p6 as u64) | ((*p7 as u64) << 8);
            let lo = v01 | (v23 << 16);
            let hi = v45 | (v67 << 16);
            *box_words.get_unchecked_mut(i) = lo | (hi << 32);
        }
        *p0 = a as u8;
        *p1 = (a >> 8) as u8;
        *p2 = (a >> 16) as u8;
        *p3 = (a >> 24) as u8;
        *p4 = (a >> 32) as u8;
        *p5 = (a >> 40) as u8;
        *p6 = (a >> 48) as u8;
        *p7 = (a >> 56) as u8;
    }
}

#[inline(always)]
fn ksa<const CROP: bool>(
    box_words: &mut [u64; 256],
    j: &mut __m128i,
    spans: &[Span],
    fills: &[u64; NAME_LIMIT],
    fixed: &[u64; 256],
) {
    let mut cur = *j;
    for span in spans {
        let start = span.start as usize;
        let end = span.end as usize;
        if span.varying {
            for (k, i) in (span.offset as usize..).zip(start..end) {
                unsafe {
                    let key = *fills.get_unchecked(k);
                    exchange::<CROP>(box_words, &mut cur, i, key);
                }
            }
        } else {
            for i in start..end {
                unsafe {
                    let key = *fixed.get_unchecked(i);
                    exchange::<CROP>(box_words, &mut cur, i, key);
                }
            }
        }
    }
    *j = cur;
}

/// 把模板编成「固定字节段 / 填充段」计划，逐句对应 `pbb-prefilter.hpp::plan()`。
#[allow(clippy::too_many_arguments)]
fn build_plan(
    prefix_len: usize,
    end_fill: usize,
    prefix: &[u8],
    suffix: &[u8],
    name_len: usize,
    fixed: &mut [u64; 256],
    mut i: usize,
    end: usize,
    mut k: usize,
) -> Vec<Span> {
    let mut spans = Vec::with_capacity(8);
    while i < end {
        let start = i;
        if k >= prefix_len && k < end_fill {
            let count = (end - i).min(end_fill - k);
            spans.push(Span {
                start: start as u16,
                end: (i + count) as u16,
                offset: (k - prefix_len) as u16,
                varying: true,
            });
            i += count;
            k += count;
            continue;
        }
        loop {
            if k == name_len {
                fixed[i] = 0;
                i += 1;
                k = 0;
            } else {
                let b = if k < prefix_len {
                    prefix[k]
                } else {
                    suffix[k - end_fill]
                };
                fixed[i] = (b as u64).wrapping_mul(0x0101_0101_0101_0101);
                i += 1;
                k += 1;
            }
            if !(i < end && (k < prefix_len || k >= end_fill)) {
                break;
            }
        }
        spans.push(Span {
            start: start as u16,
            end: i as u16,
            offset: 0,
            varying: false,
        });
    }
    spans
}

impl Front {
    fn new(launch: &Launch) -> Self {
        let fill_len = launch.fill_len.max(0) as usize;
        let prefix_len = launch.prefix_len.max(0) as usize;
        let suffix_len = launch.suffix_len.max(0) as usize;
        let name_len = launch.name_len.max(0) as usize;
        let end_fill = prefix_len + fill_len;

        let mut initial_words = [0u64; 256];
        for i in 0..256 {
            initial_words[i] = (launch.prefix_box[i] as u64).wrapping_mul(0x0101_0101_0101_0101);
        }

        let mut fixed_first = [0u64; 256];
        let mut fixed_second = [0u64; 256];
        let first = build_plan(
            prefix_len,
            end_fill,
            &launch.prefix,
            &launch.suffix,
            name_len,
            &mut fixed_first,
            launch.i_pre as usize,
            256,
            launch.j_pre as usize,
        );
        let head = build_plan(
            prefix_len,
            end_fill,
            &launch.prefix,
            &launch.suffix,
            name_len,
            &mut fixed_second,
            0,
            SLOT_LIMIT,
            name_len,
        );
        let tail_start = (name_len + SLOT_LIMIT) % (name_len + 1);
        let tail = build_plan(
            prefix_len,
            end_fill,
            &launch.prefix,
            &launch.suffix,
            name_len,
            &mut fixed_second,
            SLOT_LIMIT,
            256,
            tail_start,
        );

        let slots = if launch.fill_slots_count == 0 || launch.fill_slots.is_null() {
            Box::new([])
        } else {
            unsafe { core::slice::from_raw_parts(launch.fill_slots, launch.fill_slots_count as usize) }
                .to_vec()
                .into_boxed_slice()
        };
        let data = if launch.fill_data_count == 0 || launch.fill_data.is_null() {
            Box::new([])
        } else {
            unsafe { core::slice::from_raw_parts(launch.fill_data, launch.fill_data_count as usize) }
                .to_vec()
                .into_boxed_slice()
        };

        Self {
            initial: AlignedBox(initial_words),
            initial_scalar: launch.prefix_box,
            fixed_first,
            fixed_second,
            prefix: launch.prefix,
            suffix: launch.suffix,
            prefix_len,
            suffix_len,
            name_len,
            i_pre: launch.i_pre as usize,
            j_pre: launch.j_pre as usize,
            s_pre: launch.s_pre,
            first: first.into_boxed_slice(),
            head: head.into_boxed_slice(),
            tail: tail.into_boxed_slice(),
            shuffle: Shuffle {
                kind: launch.fill_shuffle,
                seed: launch.fill_seed,
                domain: launch.fill_domain,
                shift: launch.fill_shift.clamp(1, 63),
                mask: ((launch.fill_mask_hi as u64) << 32) | launch.fill_mask_lo as u64,
            },
            plain_fill: slots.is_empty() && fill_len == 15,
            slots,
            data,
            fill_len,
            avx2: std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("bmi2"),
        }
    }

    /// 生成 W 个候选的填充键字，按「槽优先」写入交错 `fills[k]`。
    #[inline]
    fn fill_batch(&self, id: u64, fills: &mut [u64; NAME_LIMIT]) {
        if self.plain_fill {
            let mut vals = [0u64; W];
            for (lane, dst) in vals.iter_mut().enumerate() {
                *dst = shuffle_id(&self.shuffle, id.wrapping_add(lane as u64));
            }
            // SAFETY: x86-64 恒具备 SSE2。
            unsafe { plain_batch_sse2(&vals, fills) };
            return;
        }
        fills[..self.fill_len].fill(0);
        for lane in 0..W {
            let v = shuffle_id(&self.shuffle, id.wrapping_add(lane as u64));
            if self.slots.is_empty() {
                plain_lanes_scalar(v, fills, lane);
            } else {
                self.fill_slots_lane(v, fills, lane);
            }
        }
    }

    /// 单候选连续填充（回退路径构造真名字时用）。
    fn fill_one(&self, id: u64, out: &mut [u8]) {
        let v = shuffle_id(&self.shuffle, id);
        if self.plain_fill || self.slots.is_empty() {
            plain_run(v, out);
        } else {
            self.fill_slots_contig(v, out);
        }
    }

    fn fill_slots_lane(&self, mut q: u64, fills: &mut [u64; NAME_LIMIT], lane: usize) {
        let shift = lane * 8;
        let mut buf = [0u8; NAME_LIMIT];
        for (dst, &word) in buf[..self.fill_len].iter_mut().zip(&self.data[..self.fill_len]) {
            *dst = word as u8;
        }
        for slot in self.slots.iter().rev() {
            let radix = slot.radix as u64;
            let d = q / radix;
            let at = slot.table as usize + (q - d * radix) as usize * slot.width as usize;
            q = d;
            let offset = slot.offset as usize;
            for (dst, &word) in buf[offset..offset + slot.width as usize]
                .iter_mut()
                .zip(&self.data[at..at + slot.width as usize])
            {
                *dst = word as u8;
            }
        }
        for (dst, &b) in fills[..self.fill_len].iter_mut().zip(&buf[..self.fill_len]) {
            *dst |= (b as u64) << shift;
        }
    }

    fn fill_slots_contig(&self, mut q: u64, out: &mut [u8]) {
        for (dst, &word) in out[..self.fill_len].iter_mut().zip(&self.data[..self.fill_len]) {
            *dst = word as u8;
        }
        for slot in self.slots.iter().rev() {
            let radix = slot.radix as u64;
            let d = q / radix;
            let at = slot.table as usize + (q - d * radix) as usize * slot.width as usize;
            q = d;
            let offset = slot.offset as usize;
            for (dst, &word) in out[offset..offset + slot.width as usize]
                .iter_mut()
                .zip(&self.data[at..at + slot.width as usize])
            {
                *dst = word as u8;
            }
        }
    }

    #[inline(always)]
    fn levels_with_scratch(
        &self,
        id: u64,
        out: &mut [u32; W],
        fills: &mut [u64; NAME_LIMIT],
        unwoven: &mut Unwoven,
    ) {
        let mut boxes = self.initial;
        self.fill_batch(id, fills);

        let mut cursor = unsafe { _mm_set1_epi8(self.s_pre as i8) };
        ksa::<false>(&mut boxes.0, &mut cursor, &self.first, fills, &self.fixed_first);
        cursor = unsafe { _mm_setzero_si128() };
        ksa::<false>(&mut boxes.0, &mut cursor, &self.head, fills, &self.fixed_second);
        ksa::<true>(&mut boxes.0, &mut cursor, &self.tail, fills, &self.fixed_second);

        // SAFETY: x86-64 恒具备 SSE2。
        unsafe { unweave_prefix(&boxes.0, unwoven) };

        for (lane, sbox) in unwoven.0.iter().enumerate() {
            let level = if self.avx2 {
                // SAFETY: self.avx2 已在初始化时核实 avx2 + bmi2，且 unwoven 32 字节对齐。
                unsafe { gate_avx2(sbox) }
            } else {
                gate_scalar(sbox)
            };
            out[lane] = match level {
                Some(v) => v as u32,
                None => self.exact(id.wrapping_add(lane as u64)) as u32,
            };
        }
    }

    #[inline]
    fn levels_dual_with_scratch(
        &self,
        id: u64,
        out0: &mut [u32; W],
        out1: &mut [u32; W],
        fills0: &mut [u64; NAME_LIMIT],
        fills1: &mut [u64; NAME_LIMIT],
        unwoven: &mut Unwoven,
    ) {
        let mut b0 = self.initial;
        let mut b1 = self.initial;
        self.fill_batch(id, fills0);
        self.fill_batch(id.wrapping_add(W as u64), fills1);

        let init_j = unsafe { _mm_set1_epi8(self.s_pre as i8) };
        let mut c0 = init_j;
        let mut c1 = init_j;
        ksa_dual::<false>(&mut b0.0, &mut c0, fills0, &mut b1.0, &mut c1, fills1, &self.first, &self.fixed_first);
        c0 = unsafe { _mm_setzero_si128() };
        c1 = c0;
        ksa_dual::<false>(&mut b0.0, &mut c0, fills0, &mut b1.0, &mut c1, fills1, &self.head, &self.fixed_second);
        ksa_dual::<true>(&mut b0.0, &mut c0, fills0, &mut b1.0, &mut c1, fills1, &self.tail, &self.fixed_second);

        let eval_half = |box_words: &[u64; 256], base_id: u64, out: &mut [u32; W], unwoven: &mut Unwoven| {
            unsafe { unweave_prefix(box_words, unwoven) };
            for (lane, sbox) in unwoven.0.iter().enumerate() {
                let level = if self.avx2 {
                    unsafe { gate_avx2(sbox) }
                } else {
                    gate_scalar(sbox)
                };
                out[lane] = match level {
                    Some(v) => v as u32,
                    None => self.exact(base_id.wrapping_add(lane as u64)) as u32,
                };
            }
        };
        eval_half(&b0.0, id, out0, unwoven);
        eval_half(&b1.0, id.wrapping_add(W as u64), out1, unwoven);
    }

    pub fn levels(&self, id: u64, out: &mut [u32; W]) {
        let mut fills = [0u64; NAME_LIMIT];
        let mut unwoven = Unwoven([[0u8; UNWEAVE_STRIDE]; W]);
        self.levels_with_scratch(id, out, &mut fills, &mut unwoven);
    }

    /// 原版完整排列复核：真名字 + 两轮 KSA + 全 256 槽压缩。等价 C++ `Front::exact`。
    fn exact(&self, id: u64) -> u8 {
        let mut name = [0u8; NAME_LIMIT + 1];
        name[..self.prefix_len].copy_from_slice(&self.prefix[..self.prefix_len]);
        self.fill_one(id, &mut name[self.prefix_len..self.prefix_len + self.fill_len]);
        let suffix_start = self.prefix_len + self.fill_len;
        name[suffix_start..suffix_start + self.suffix_len]
            .copy_from_slice(&self.suffix[..self.suffix_len]);
        name[self.name_len] = 0;

        let mut sbox = self.initial_scalar;
        let name_len = self.name_len;

        let mut s = self.s_pre;
        let mut j = self.j_pre;
        for i in self.i_pre..256 {
            s = s.wrapping_add(name[j]).wrapping_add(sbox[i]);
            sbox.swap(i, s as usize);
            j = if j == name_len { 0 } else { j + 1 };
        }

        s = 0;
        j = name_len;
        for i in 0..256 {
            s = s.wrapping_add(name[j]).wrapping_add(sbox[i]);
            sbox.swap(i, s as usize);
            j = if j == name_len { 0 } else { j + 1 };
        }

        let mut attrs = [0u8; 48];
        let mut n = 0usize;
        for &byte in &sbox {
            let ual = byte.wrapping_mul(181).wrapping_add(160);
            if (89..217).contains(&ual) {
                attrs[n] = ual & 63;
                n += 1;
                if n == NEEDED {
                    break;
                }
            }
        }
        gate_level_of(&attrs)
    }
}

// ===== C ABI =====

#[unsafe(no_mangle)]
pub extern "C" fn tswn_rust_launch_size() -> usize {
    core::mem::size_of::<Launch>()
}

#[unsafe(no_mangle)]
pub extern "C" fn tswn_rust_front_width() -> u32 {
    W as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tswn_rust_front_create(launch: *const Launch) -> *mut c_void {
    if launch.is_null() {
        return ptr::null_mut();
    }
    let front = Box::new(Front::new(unsafe { &*launch }));
    Box::into_raw(front).cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tswn_rust_front_destroy(front: *mut c_void) {
    if !front.is_null() {
        drop(unsafe { Box::from_raw(front.cast::<Front>()) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tswn_rust_front_levels(front: *const c_void, id: u64, out: *mut u32) {
    let front = unsafe { &*front.cast::<Front>() };
    let out = unsafe { &mut *out.cast::<[u32; W]>() };
    front.levels(id, out);
}

/// 批量扫描 `count` 个连续候选（须为 `W` 的倍数），仅将非零门控写入 `hit_offsets / hit_levels`，返回命中数。
/// 彻底消除每 8 个候选一次的跨语言 FFI 调用和 99.99% 零门控分支。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tswn_rust_front_scan_hits(
    front: *const c_void,
    first_id: u64,
    count: u32,
    hit_offsets: *mut u32,
    hit_levels: *mut u32,
) -> u32 {
    let front = unsafe { &*front.cast::<Front>() };
    let mut hits = 0usize;
    let mut levels0 = [0u32; W];
    let mut levels1 = [0u32; W];
    let mut fills0 = [0u64; NAME_LIMIT];
    let mut fills1 = [0u64; NAME_LIMIT];
    let mut unwoven = Unwoven([[0u8; UNWEAVE_STRIDE]; W]);
    let mut offset = 0u32;
    while (offset as usize) + 2 * W <= count as usize {
        front.levels_dual_with_scratch(
            first_id.wrapping_add(offset as u64),
            &mut levels0,
            &mut levels1,
            &mut fills0,
            &mut fills1,
            &mut unwoven,
        );
        let any0 = levels0[0] | levels0[1] | levels0[2] | levels0[3]
            | levels0[4] | levels0[5] | levels0[6] | levels0[7];
        if any0 != 0 {
            for (lane, &lvl) in levels0.iter().enumerate() {
                if lvl != 0 {
                    unsafe {
                        *hit_offsets.add(hits) = offset + lane as u32;
                        *hit_levels.add(hits) = lvl;
                    }
                    hits += 1;
                }
            }
        }
        let any1 = levels1[0] | levels1[1] | levels1[2] | levels1[3]
            | levels1[4] | levels1[5] | levels1[6] | levels1[7];
        if any1 != 0 {
            for (lane, &lvl) in levels1.iter().enumerate() {
                if lvl != 0 {
                    unsafe {
                        *hit_offsets.add(hits) = offset + (W + lane) as u32;
                        *hit_levels.add(hits) = lvl;
                    }
                    hits += 1;
                }
            }
        }
        offset += (2 * W) as u32;
    }
    while (offset as usize) + W <= count as usize {
        front.levels_with_scratch(first_id.wrapping_add(offset as u64), &mut levels0, &mut fills0, &mut unwoven);
        let any = levels0[0] | levels0[1] | levels0[2] | levels0[3]
            | levels0[4] | levels0[5] | levels0[6] | levels0[7];
        if any != 0 {
            for (lane, &lvl) in levels0.iter().enumerate() {
                if lvl != 0 {
                    unsafe {
                        *hit_offsets.add(hits) = offset + lane as u32;
                        *hit_levels.add(hits) = lvl;
                    }
                    hits += 1;
                }
            }
        }
        offset += W as u32;
    }
    hits as u32
}

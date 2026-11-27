//! 纯 Rust 的 PBB 一阶 Q 门前核（RWIR）。
//!
//! 管道：ID 置换 → 五字 UTF-8 键字直出 → 双轮 KSA（8 路独立 S-box）→
//! 属性窗口压缩（AVX2 + BMI2 `pdep/pext`）→ 四平面整数门控 + 29 路排序网络。
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
/// 属性压缩只看前 96 槽。
const SLOT_LIMIT: usize = 96;
/// 需要收齐的属性字节数。
const NEEDED: usize = 31;

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
    initial: [u8; 256],
    fixed_first: [u8; 256],
    fixed_second: [u8; 256],
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

/// 八路「槽优先」键字写入：`fills[k][lane]`，一圈 KSA 只需一次 8 字节读取。
#[inline(always)]
fn plain_lanes(v: u64, fills: &mut [[u8; W]; NAME_LIMIT], lane: usize) {
    for (i, cp) in plain_points(v).into_iter().enumerate() {
        let [b0, b1, b2] = point_bytes(cp);
        fills[3 * i][lane] = b0;
        fills[3 * i + 1][lane] = b1;
        fills[3 * i + 2][lane] = b2;
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
fn gate_scalar(sbox: &[u8; 256]) -> Option<u8> {
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
#[target_feature(enable = "avx2,bmi2")]
unsafe fn gate_avx2(sbox: &[u8; 256]) -> Option<u8> {
    let mul = _mm256_set1_epi16(181);
    let low_mask = _mm256_set1_epi16(0x00ff);
    let add71 = _mm256_set1_epi8(71);
    let add89 = _mm256_set1_epi8(89);
    let mask63 = _mm256_set1_epi8(63);
    let mut attrs = [0u8; 48];
    let mut raw = [0u8; 32];
    let mut n = 0usize;

    for block in sbox[..SLOT_LIMIT].as_chunks::<32>().0 {
        let valid = unsafe {
            let x = _mm256_loadu_si256(block.as_ptr().cast());
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

// ===== KSA =====

/// 单路一个槽：游标加法 + 交换。`CROP` 为真时省掉顺序槽回读与写回。
///
/// Safety: 调用方保证 `i < 256`（`u8` 范围）；`d` 由 `u8` 转成 `usize` 恒在 `0..256` 内。
#[inline(always)]
unsafe fn ksa_step<const CROP: bool>(sbox: &mut [u8; 256], j: &mut u8, i: usize, key: u8) {
    // SAFETY: i < 256 且 d = next as usize < 256。
    unsafe {
        let a = *sbox.get_unchecked(i);
        let next = j.wrapping_add(a).wrapping_add(key);
        *j = next;
        let d = next as usize;
        if CROP {
            *sbox.get_unchecked_mut(d) = a;
        } else {
            let target = *sbox.get_unchecked(d);
            *sbox.get_unchecked_mut(i) = target;
            *sbox.get_unchecked_mut(d) = a;
        }
    }
}

/// 八路同一槽，键字节相同（固定段）。
#[inline(always)]
unsafe fn ksa_step8_fixed<const CROP: bool>(
    boxes: &mut [[u8; 256]; W],
    j: &mut [u8; W],
    i: usize,
    key: u8,
) {
    unsafe {
        ksa_step::<CROP>(&mut boxes[0], &mut j[0], i, key);
        ksa_step::<CROP>(&mut boxes[1], &mut j[1], i, key);
        ksa_step::<CROP>(&mut boxes[2], &mut j[2], i, key);
        ksa_step::<CROP>(&mut boxes[3], &mut j[3], i, key);
        ksa_step::<CROP>(&mut boxes[4], &mut j[4], i, key);
        ksa_step::<CROP>(&mut boxes[5], &mut j[5], i, key);
        ksa_step::<CROP>(&mut boxes[6], &mut j[6], i, key);
        ksa_step::<CROP>(&mut boxes[7], &mut j[7], i, key);
    }
}

/// 八路同一槽，键字节逐 lane（填充段）。
#[inline(always)]
unsafe fn ksa_step8_varying<const CROP: bool>(
    boxes: &mut [[u8; 256]; W],
    j: &mut [u8; W],
    i: usize,
    keys: [u8; W],
) {
    unsafe {
        ksa_step::<CROP>(&mut boxes[0], &mut j[0], i, keys[0]);
        ksa_step::<CROP>(&mut boxes[1], &mut j[1], i, keys[1]);
        ksa_step::<CROP>(&mut boxes[2], &mut j[2], i, keys[2]);
        ksa_step::<CROP>(&mut boxes[3], &mut j[3], i, keys[3]);
        ksa_step::<CROP>(&mut boxes[4], &mut j[4], i, keys[4]);
        ksa_step::<CROP>(&mut boxes[5], &mut j[5], i, keys[5]);
        ksa_step::<CROP>(&mut boxes[6], &mut j[6], i, keys[6]);
        ksa_step::<CROP>(&mut boxes[7], &mut j[7], i, keys[7]);
    }
}

fn ksa<const CROP: bool>(
    boxes: &mut [[u8; 256]; W],
    j: &mut [u8; W],
    spans: &[Span],
    fills: &[[u8; W]; NAME_LIMIT],
    fixed: &[u8; 256],
) {
    for span in spans {
        let start = span.start as usize;
        let end = (span.end as usize).min(256);
        if span.varying {
            for (k, i) in (span.offset as usize..).zip(start..end) {
                // SAFETY: i < 256，且 build_plan 保证 k < fill_len <= NAME_LIMIT。
                unsafe {
                    let keys = *fills.get_unchecked(k);
                    ksa_step8_varying::<CROP>(boxes, j, i, keys);
                }
            }
        } else {
            for i in start..end {
                // SAFETY: i < 256。
                unsafe {
                    let key = *fixed.get_unchecked(i);
                    ksa_step8_fixed::<CROP>(boxes, j, i, key);
                }
            }
        }
    }
}

/// 把模板编成「固定字节段 / 填充段」计划，逐句对应 `pbb-prefilter.hpp::plan()`。
#[allow(clippy::too_many_arguments)]
fn build_plan(
    prefix_len: usize,
    end_fill: usize,
    prefix: &[u8],
    suffix: &[u8],
    name_len: usize,
    fixed: &mut [u8; 256],
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
                fixed[i] = if k < prefix_len {
                    prefix[k]
                } else {
                    suffix[k - end_fill]
                };
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

        let mut fixed_first = [0u8; 256];
        let mut fixed_second = [0u8; 256];
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
            // SAFETY: Launch 初始化后 fill_slots/fill_data 指向有效只读数组，此处拷贝为自有切片。
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
            initial: launch.prefix_box,
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

    /// 生成 W 个候选的填充键字，按「槽优先」写入 `fills[k][lane]`。
    #[inline]
    fn fill_batch(&self, id: u64, fills: &mut [[u8; W]; NAME_LIMIT]) {
        for lane in 0..W {
            let v = shuffle_id(&self.shuffle, id.wrapping_add(lane as u64));
            if self.plain_fill || self.slots.is_empty() {
                plain_lanes(v, fills, lane);
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

    fn fill_slots_lane(&self, mut q: u64, fills: &mut [[u8; W]; NAME_LIMIT], lane: usize) {
        for (i, &word) in self.data[..self.fill_len].iter().enumerate() {
            fills[i][lane] = word as u8;
        }
        for slot in self.slots.iter().rev() {
            let radix = slot.radix as u64;
            let d = q / radix;
            let at = slot.table as usize + (q - d * radix) as usize * slot.width as usize;
            q = d;
            for (k, &word) in self.data[at..at + slot.width as usize].iter().enumerate() {
                fills[slot.offset as usize + k][lane] = word as u8;
            }
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

    pub fn levels(&self, id: u64, out: &mut [u32; W]) {
        let mut boxes = [self.initial; W];
        let mut fills = [[0u8; W]; NAME_LIMIT];
        self.fill_batch(id, &mut fills);

        let mut cursor = [self.s_pre; W];
        ksa::<false>(&mut boxes, &mut cursor, &self.first, &fills, &self.fixed_first);
        cursor = [0; W];
        ksa::<false>(&mut boxes, &mut cursor, &self.head, &fills, &self.fixed_second);
        ksa::<true>(&mut boxes, &mut cursor, &self.tail, &fills, &self.fixed_second);

        for (lane, sbox) in boxes.iter().enumerate() {
            let level = if self.avx2 {
                // SAFETY: self.avx2 已在初始化时通过 is_x86_feature_detected! 核实 avx2 + bmi2。
                unsafe { gate_avx2(sbox) }
            } else {
                gate_scalar(sbox)
            };
            // 收不齐 31 项属性：原版会把这类候选交回完整排列复核，这里同样在 Rust 内处理。
            out[lane] = match level {
                Some(v) => v as u32,
                None => self.exact(id.wrapping_add(lane as u64)) as u32,
            };
        }
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

        let mut sbox = self.initial;
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

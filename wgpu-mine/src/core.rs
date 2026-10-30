//! 独立、安全的完整 CPU 金标准：名字 KSA、八围、技能与虚评 2.0 XP/QD。
//!
//! 队伍和名字始终按原始 UTF-8 字节处理（包括 `@` 与 NUL），不解析完整名字。
//! 此路径不使用旧 `replace_name` 的固定启发式预筛。所有模型计算均保留 f64，
//! 导出的 f32 权重仅供 GPU 对照使用。

#[allow(clippy::excessive_precision)]
#[path = "../../miner/src/evaluate/xuping/model20.rs"]
mod model20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Thresholds {
    pub prop: u32,
    pub xp: u32,
    pub xd: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub props: [u32; 8],
    pub skill_ids: [u32; 16],
    pub skill_freqs: [u32; 16],
    /// 未启用 ID 24 时为零；启用时是影子特征未乘技能频率前的三倍值。
    pub shadow_numerator: i32,
    pub xp: f64,
    pub xd: f64,
}

impl Evaluation {
    /// 八围严格大于 prop；XP/QD 按 Rust 的饱和、截断 as u32 后比较。
    pub fn matches(&self, thresholds: Thresholds) -> bool {
        u64::from(property_numerator(&self.props)) > 3 * u64::from(thresholds.prop)
            && self.xp as u32 >= thresholds.xp
            && self.xd as u32 >= thresholds.xd
    }
}

/// 八围总值的精确三倍：HP + 3 * 其余七围之和，不提前整除 HP。
/// props 是本模块生成的实际属性（HP <= 406，其他属性 <= 99）。
pub fn property_numerator(props: &[u32; 8]) -> u32 { props[0] + 3 * props[1..].iter().sum::<u32>() }

/// 一轮带隐式前导零的安全 KSA；每轮都重新令 j = 0。
fn ksa_round(state: &mut [u8; 256], bytes: &[u8]) {
    let mut j = 0_u8;
    let mut key_index = 0;
    for i in 0..256 {
        let byte = if key_index == 0 { 0 } else { bytes[key_index - 1] };
        j = j.wrapping_add(byte).wrapping_add(state[i]);
        state.swap(i, usize::from(j));
        key_index = if key_index == bytes.len() { 0 } else { key_index + 1 };
    }
}

fn initial_team_state(team: &str) -> [u8; 256] {
    let mut state = std::array::from_fn(|i| i as u8);
    ksa_round(&mut state, team.as_bytes());
    state
}

/// 从队伍状态执行两轮名字 KSA，返回尚未执行 PRGA 的状态。
/// 不限制内部名字长度，因此 256 字节的原名字仍可附加 7 字节 ?shadow。
/// 每轮实际只读取 KSA 所需的前 255 字节，与原实现的 UTF-8 字节语义一致。
pub fn rc4_name(team_state: &[u8; 256], name: &str) -> [u8; 256] {
    let mut state = *team_state;
    ksa_round(&mut state, name.as_bytes());
    ksa_round(&mut state, name.as_bytes());
    state
}

/// 输入必须是 KSA 后的排列。变换是模 256 的双射，所以恰有 128 项入选。
/// 保持原始状态中的相对顺序，技能必须使用完整 128 项而非仅属性前缀。
fn raw_filter(state: &[u8; 256]) -> [u8; 128] {
    let mut base = [0_u8; 128];
    let mut count = 0;
    for &value in state {
        let transformed = value.wrapping_mul(181).wrapping_add(160);
        if (89..217).contains(&transformed) {
            base[count] = transformed & 63;
            count += 1;
        }
    }
    assert_eq!(count, 128, "RC4 状态必须是 0..255 的排列");
    base
}

fn base_properties(base: &[u8; 128]) -> [u32; 8] {
    let mut hp = [0_u8; 10];
    hp.copy_from_slice(&base[..10]);
    hp.sort_unstable();
    let mut props = [0_u32; 8];
    props[0] = 154 + hp[3..7].iter().map(|&value| u32::from(value)).sum::<u32>();
    for (i, prop) in props.iter_mut().enumerate().skip(1) {
        let start = 10 + (i - 1) * 3;
        let mut triple = [base[start], base[start + 1], base[start + 2]];
        triple.sort_unstable();
        *prop = u32::from(triple[1]) + 36;
    }
    props
}

/// 由两轮名字 KSA 后、PRGA 前的状态求完整八围，不应用任何预筛。
pub fn properties(state: &[u8; 256]) -> [u32; 8] { base_properties(&raw_filter(state)) }

fn prga_byte(state: &mut [u8; 256], a: &mut u8, b: &mut u8) -> u8 {
    *a = a.wrapping_add(1);
    *b = b.wrapping_add(state[usize::from(*a)]);
    state.swap(usize::from(*a), usize::from(*b));
    let index = state[usize::from(*a)].wrapping_add(state[usize::from(*b)]);
    state[usize::from(index)]
}

fn shuffle_skills(state: &mut [u8; 256]) -> [u32; 40] {
    let mut ids = std::array::from_fn(|i| i as u32);
    let (mut a, mut b, mut s) = (0_u8, 0_u8, 0_u8);
    // 两轮共享 a/b/s，连续消耗 160 字节 PRGA。
    for _ in 0..2 {
        for i in 0..40 {
            let high = prga_byte(state, &mut a, &mut b);
            let low = prga_byte(state, &mut a, &mut b);
            let random = ((u32::from(high) << 8) | u32::from(low)) % 40;
            // 显式保留原版先 u8 回绕、再 % 40 的顺序。
            s = s.wrapping_add(random as u8).wrapping_add(ids[i] as u8) % 40;
            ids.swap(i, usize::from(s));
        }
    }
    ids
}

fn skill_frequencies(base: &[u8; 128], ids: &[u32; 40]) -> [u32; 16] {
    let mut freqs = [0_u32; 16];
    let mut last = None;
    for i in 0..16 {
        let start = 64 + i * 4;
        let p = *base[start..start + 4].iter().min().unwrap();
        if p > 10 && ids[i] < 35 {
            freqs[i] = u32::from(p - 10);
            if p < 35 {
                last = Some(i);
            }
        }
    }
    if let Some(i) = last {
        freqs[i] *= 2;
    }
    // 原版四个 match 分支的等价实现：最后一个低频技能翻倍；
    // 第 14/15 槽若没有翻倍，则各自添加两项 base 与自身频率的最小值。
    for i in [14, 15] {
        if last != Some(i) {
            let start = 60 + 2 * (i - 14);
            freqs[i] += u32::from(base[start].min(base[start + 1])).min(freqs[i]);
        }
    }
    freqs
}

fn shadow_numerator(props: &[u32; 8]) -> i32 {
    // raw 围是去掉 36 基值后的七围；HP 不去基值。
    let raw_sum: i32 = props[1..].iter().map(|&prop| prop as i32 - 36).sum();
    props[0] as i32 + 3 * raw_sum - 9 * (props[7] as i32 - 36) - 630
}

fn features(props: &[u32; 8], ids: &[u32; 16], freqs: &[u32; 16], shadow: i32) -> [f64; 44] {
    let mut st = [0.0; 44];
    for (target, &prop) in st[..8].iter_mut().zip(props) {
        *target = f64::from(prop);
    }
    for (&id, &freq) in ids.iter().zip(freqs) {
        if freq != 0 {
            // ID 35..39 是空技能，不进入 35 项技能特征。
            assert!(id < 35, "非零频率的技能 ID 必须小于 35");
            st[id as usize + 8] = f64::from(freq);
        }
    }
    if st[32] > 0.0 {
        st[43] = f64::from(shadow) / 3.0 * st[32] / 100.0;
    }
    if st[42] > 0.0 {
        st[42] += 20.0;
    }
    st
}

fn poly(st: &[f64; 44]) -> [f64; 1034] {
    let mut result = [0.0; 1034];
    result[..44].copy_from_slice(st);
    let mut index = 44;
    for i in 0..44 {
        for j in i..44 {
            result[index] = st[i] * st[j];
            index += 1;
        }
    }
    result
}

/// 同时计算原虚评 2.0 XP 与 QD，使用原始 f64 权重与标量求和顺序。
pub fn scores(props: &[u32; 8], ids: &[u32; 16], freq: &[u32; 16], shadow_numerator: i32) -> [f64; 2] {
    let expanded = poly(&features(props, ids, freq, shadow_numerator));
    let mut result = [model20::BASE, model20::BASE_QD];
    for (i, feature) in expanded.iter().enumerate() {
        result[0] += feature * model20::MODEL[i];
        result[1] += feature * model20::MODEL_QD[i];
    }
    result
}

/// GPU 模型缓冲：第 0 项为常数对，之后为 44 线性 + 990 二次项的系数对。
pub fn model_weights() -> Vec<[f32; 2]> {
    let mut result = Vec::with_capacity(1035);
    result.push([model20::BASE as f32, model20::BASE_QD as f32]);
    result.extend(model20::MODEL.iter().zip(&model20::MODEL_QD).map(|(&xp, &qd)| [xp as f32, qd as f32]));
    result
}

/// 不解析 @：传入的是原始队伍和原始名字，分别使用各自的字节。
pub fn evaluate(team: &str, name: &str) -> Evaluation { evaluate_cached(&initial_team_state(team), name) }

pub fn evaluate_cached(team_state: &[u8; 256], name: &str) -> Evaluation {
    evaluate_state(team_state, rc4_name(team_state, name), name)
}

/// state 必须是该名字两轮 KSA 后、尚未执行 PRGA 的状态。
/// 可先 properties/state 精确预筛，再复用该状态完成技能与评分。
pub fn evaluate_state(team_state: &[u8; 256], mut state: [u8; 256], name: &str) -> Evaluation {
    let base = raw_filter(&state);
    let props = base_properties(&base);
    let all_ids = shuffle_skills(&mut state);
    let skill_freqs = skill_frequencies(&base, &all_ids);
    let mut skill_ids = [0_u32; 16];
    skill_ids.copy_from_slice(&all_ids[..16]);
    let has_shadow = skill_ids.iter().zip(&skill_freqs).any(|(&id, &freq)| id == 24 && freq != 0);
    let shadow_numerator = if has_shadow {
        let mut shadow = String::with_capacity(name.len() + 7);
        shadow.push_str(name);
        shadow.push_str("?shadow");
        shadow_numerator(&properties(&rc4_name(team_state, &shadow)))
    } else {
        0
    };
    let [xp, xd] = scores(&props, &skill_ids, &skill_freqs, shadow_numerator);
    Evaluation {
        props,
        skill_ids,
        skill_freqs,
        shadow_numerator,
        xp,
        xd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 参考路径有意使用分配的密钥、usize 取模与完整 PRGA 字节流，
    // 不调用被测 KSA、属性、洗牌或频率实现。
    fn reference_state(team: &str, name: &str) -> [u8; 256] {
        let mut state = std::array::from_fn(|i| i as u8);
        for text in [team, name, name] {
            let key: Vec<u8> = std::iter::once(0).chain(text.bytes()).collect();
            let mut j = 0;
            for i in 0..256 {
                j = (j + usize::from(state[i]) + usize::from(key[i % key.len()])) % 256;
                state.swap(i, j);
            }
        }
        state
    }

    fn reference_base(state: &[u8; 256]) -> Vec<u8> {
        state
            .iter()
            .map(|&value| (u32::from(value) * 181 + 160) % 256)
            .filter(|&value| (89..217).contains(&value))
            .map(|value| (value & 63) as u8)
            .collect()
    }

    fn reference_props(state: &[u8; 256]) -> [u32; 8] {
        let base = reference_base(state);
        assert_eq!(base.len(), 128);
        let mut sorted = base[..10].to_vec();
        sorted.sort_unstable();
        let mut props = [0; 8];
        props[0] = 154 + u32::from(sorted[3]) + u32::from(sorted[4]) + u32::from(sorted[5]) + u32::from(sorted[6]);
        for i in 0..7 {
            let start = 10 + 3 * i;
            let (x, y, z) = (base[start], base[start + 1], base[start + 2]);
            props[i + 1] = u32::from(x.min(y).max(x.max(y).min(z))) + 36;
        }
        props
    }

    fn reference_freq(base: &[u8], ids: &[u32; 40]) -> [u32; 16] {
        let mut freq = [0; 16];
        let mut last = -1;
        for j in 0..16 {
            let i = 64 + 4 * j;
            let p = base[i].min(base[i + 1]).min(base[i + 2].min(base[i + 3]));
            if p > 10 && ids[j] < 35 {
                freq[j] = u32::from(p - 10);
                if p < 35 {
                    last = j as i32;
                }
            }
        }
        let bonus14 = |freq: u32| u32::from(base[60].min(base[61])).min(freq);
        let bonus15 = |freq: u32| u32::from(base[62].min(base[63])).min(freq);
        match last {
            -1 => {
                freq[14] += bonus14(freq[14]);
                freq[15] += bonus15(freq[15]);
            }
            14 => {
                freq[14] <<= 1;
                freq[15] += bonus15(freq[15]);
            }
            15 => {
                freq[15] <<= 1;
                freq[14] += bonus14(freq[14]);
            }
            x => {
                freq[x as usize] <<= 1;
                freq[14] += bonus14(freq[14]);
                freq[15] += bonus15(freq[15]);
            }
        }
        freq
    }

    fn reference_skills(mut state: [u8; 256]) -> ([u32; 40], [u32; 16], [u8; 256]) {
        let base = reference_base(&state);
        let mut stream = Vec::with_capacity(160);
        let (mut a, mut b) = (0, 0);
        for _ in 0..160 {
            a = (a + 1) % 256;
            b = (b + usize::from(state[a])) % 256;
            state.swap(a, b);
            stream.push(state[(usize::from(state[a]) + usize::from(state[b])) % 256]);
        }
        let mut ids = std::array::from_fn(|i| i as u32);
        let mut s = 0;
        for (step, pair) in stream.chunks_exact(2).enumerate() {
            let i = step % 40;
            let random = (u32::from(pair[0]) * 256 + u32::from(pair[1])) % 40;
            s = ((s + random + ids[i]) % 256) % 40;
            ids.swap(i, s as usize);
        }
        let freq = reference_freq(&base, &ids);
        (ids, freq, state)
    }

    // 原 eval.rs 的逐索引算法，独立于被测三角双循环。
    fn original_poly(st: &[f64; 44]) -> [f64; 1034] {
        std::array::from_fn(|index| {
            let (mut i, mut p, mut q, mut j) = (0_i32, 0_i32, 0_i32, index as i32);
            for _ in 0..45 {
                i += 1;
                if i > 2 {
                    p += 1;
                }
                q = j;
                j = j - 44 + p;
                if j < 0 {
                    break;
                }
            }
            if i == 1 {
                st[q as usize]
            } else {
                st[p as usize] * st[(p + q) as usize]
            }
        })
    }

    fn reference_evaluation(team: &str, name: &str) -> Evaluation {
        let state = reference_state(team, name);
        let props = reference_props(&state);
        let (all_ids, skill_freqs, _) = reference_skills(state);
        let mut skill_ids = [0; 16];
        skill_ids.copy_from_slice(&all_ids[..16]);
        let mut st = [0.0; 44];
        for i in 0..8 {
            st[i] = f64::from(props[i]);
        }
        for i in 0..16 {
            if skill_freqs[i] != 0 {
                st[skill_ids[i] as usize + 8] = f64::from(skill_freqs[i]);
            }
        }
        let mut numerator = 0;
        if st[32] > 0.0 {
            let shadow = reference_props(&reference_state(team, &format!("{name}?shadow")));
            numerator = shadow[0] as i32;
            let mut shadow_sum = f64::from(shadow[0]) / 3.0;
            for &prop in &shadow[1..] {
                shadow_sum += f64::from(prop) - 36.0;
                numerator += 3 * (prop as i32 - 36);
            }
            numerator -= 9 * (shadow[7] as i32 - 36) + 630;
            shadow_sum -= (f64::from(shadow[7]) - 36.0) * 3.0;
            st[43] = (shadow_sum - 210.0) * st[32] / 100.0;
        }
        if st[42] > 0.0 {
            st[42] += 20.0;
        }
        let expanded = original_poly(&st);
        let mut xp = model20::BASE;
        let mut xd = model20::BASE_QD;
        for i in 0..1034 {
            xp += expanded[i] * model20::MODEL[i];
            xd += expanded[i] * model20::MODEL_QD[i];
        }
        Evaluation {
            props,
            skill_ids,
            skill_freqs,
            shadow_numerator: numerator,
            xp,
            xd,
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8, "实际 {actual:.15}，预期 {expected:.15}");
    }

    fn assert_evaluation(actual: &Evaluation, expected: &Evaluation) {
        assert_eq!(actual.props, expected.props);
        assert_eq!(actual.skill_ids, expected.skill_ids);
        assert_eq!(actual.skill_freqs, expected.skill_freqs);
        assert_eq!(actual.shadow_numerator, expected.shadow_numerator);
        close(actual.xp, expected.xp);
        close(actual.xd, expected.xd);
    }

    #[test]
    fn x_at_x_matches_all_original_fixed_arrays() {
        let team = initial_team_state("x");
        let mut state = rc4_name(&team, "x");
        let base = raw_filter(&state);
        assert_eq!(
            base,
            [
                53, 0, 40, 4, 58, 61, 37, 46, 56, 51, 21, 20, 27, 17, 15, 26, 13, 30, 52, 63, 36, 30, 57, 34, 22, 37, 35, 6, 12,
                25, 50, 49, 59, 23, 49, 27, 51, 58, 39, 28, 60, 20, 31, 36, 41, 11, 7, 29, 24, 24, 61, 62, 57, 4, 28, 48, 55, 50,
                38, 29, 10, 40, 42, 15, 23, 47, 42, 62, 47, 1, 60, 5, 43, 21, 1, 46, 45, 9, 9, 14, 38, 13, 56, 0, 31, 59, 39, 6,
                35, 41, 55, 5, 34, 3, 7, 33, 33, 45, 16, 16, 32, 43, 18, 44, 22, 14, 17, 10, 11, 53, 18, 44, 19, 52, 2, 32, 12,
                8, 2, 54, 26, 48, 8, 3, 63, 54, 19, 25,
            ]
        );
        assert_eq!(properties(&state), [344, 57, 53, 66, 72, 70, 71, 61]);
        let ids = shuffle_skills(&mut state);
        assert_eq!(
            ids,
            [
                9, 13, 12, 38, 4, 27, 26, 15, 16, 32, 24, 5, 7, 21, 18, 10, 37, 2, 6, 20, 39, 1, 14, 3, 11, 29, 22, 33, 19, 0,
                30, 31, 17, 28, 34, 35, 23, 8, 25, 36,
            ]
        );
        let mut full_freq = [0; 40];
        full_freq[..16].copy_from_slice(&skill_frequencies(&base, &ids));
        assert_eq!(
            full_freq,
            [
                13, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
            ]
        );
        let evaluation = evaluate("x", "x");
        assert_eq!(evaluation.skill_ids, ids[..16]);
        close(evaluation.xp, 316.6563896305898);
        close(evaluation.xd, 5623.442905977731);
        assert_eq!(evaluation.shadow_numerator, 0); // ID 24 存在但频率为零。
        assert_eq!(property_numerator(&evaluation.props), 1694);
    }

    #[test]
    fn empty_unicode_emoji_nul_and_byte_boundaries_match_independent_reference() {
        let names = [
            String::new(),
            "中文名字".to_owned(),
            "🙂🚀".to_owned(),
            "\0".to_owned(),
            "a\0b".to_owned(),
            "名字@原始后缀".to_owned(),
            "x".repeat(254),
            "x".repeat(255),
            "x".repeat(256),
            format!("{}x", "中".repeat(85)),
            "🙂".repeat(64),
            format!("{}🙂", "a".repeat(252)),
        ];
        for team in ["", "中文队伍", "🙂", "含\0队伍", "原始@队伍"] {
            let cached = initial_team_state(team);
            assert_eq!(cached, crate::cpu::team_state(team));
            for name in &names {
                let state = rc4_name(&cached, name);
                assert_eq!(state, reference_state(team, name), "队伍 {team:?}，名字 {name:?}");
                let expected = reference_evaluation(team, name);
                assert_evaluation(&evaluate(team, name), &expected);
                assert_evaluation(&evaluate_cached(&cached, name), &expected);
                assert_evaluation(&evaluate_state(&cached, state, name), &expected);
                let mut mutated = state;
                let (ids, freq, final_state) = reference_skills(state);
                assert_eq!(shuffle_skills(&mut mutated), ids);
                assert_eq!(mutated, final_state); // 确认恰好 160 字节 PRGA，第二轮没有重置。
                assert_eq!(skill_frequencies(&raw_filter(&state), &ids), freq);
                assert_eq!(cached, initial_team_state(team));
            }
        }
    }

    #[test]
    fn deterministic_corpus_matches_independent_full_reference() {
        for i in 0..256 {
            let team = format!("队伍{}@\0🙂", i % 7);
            let name = format!("测号-{i}-中文🙂\0@{}", i * 7919);
            assert_evaluation(&evaluate(&team, &name), &reference_evaluation(&team, &name));
        }
    }

    #[test]
    fn property_prefilter_has_no_loss_at_550() {
        // 原 replace_name 因智力 raw = 16 < 24 丢弃，但实际八围是 577。
        let evaluation = evaluate("golden-team", "完整金标准-3");
        assert_eq!(evaluation.props, [339, 79, 82, 41, 97, 64, 49, 52]);
        assert_eq!(property_numerator(&evaluation.props), 1731);
        assert!(evaluation.props[7] - 36 < 24);
        assert!(evaluation.matches(Thresholds { prop: 550, xp: 0, xd: 0 }));
        assert!(!evaluation.matches(Thresholds { prop: 577, xp: 0, xd: 0 }));
    }

    #[test]
    fn property_comparison_preserves_fraction_and_large_threshold_does_not_overflow() {
        let mut evaluation = evaluate("x", "x");
        evaluation.props = [301, 75, 75, 75, 75, 50, 50, 50];
        assert_eq!(property_numerator(&evaluation.props), 1651);
        assert!(evaluation.matches(Thresholds { prop: 550, xp: 0, xd: 0 }));
        evaluation.props[0] = 300;
        assert!(!evaluation.matches(Thresholds { prop: 550, xp: 0, xd: 0 }));
        for prop in [u32::MAX, u32::MAX / 3 + 1, 1 << 31] {
            assert!(!evaluation.matches(Thresholds { prop, xp: 0, xd: 0 }));
        }
    }

    #[test]
    fn score_thresholds_use_saturating_truncating_rust_casts() {
        let mut evaluation = evaluate("x", "x");
        for (score, truncated) in [
            (f64::NEG_INFINITY, 0),
            (-1.0, 0),
            (f64::NAN, 0),
            (0.0, 0),
            (9.999, 9),
            (10.0, 10),
            (f64::from(u32::MAX), u32::MAX),
            (4294967296.0, u32::MAX),
            (f64::INFINITY, u32::MAX),
        ] {
            evaluation.xp = score;
            evaluation.xd = score;
            assert!(evaluation.matches(Thresholds {
                prop: 0,
                xp: truncated,
                xd: truncated
            }));
            if truncated < u32::MAX {
                assert!(!evaluation.matches(Thresholds {
                    prop: 0,
                    xp: truncated + 1,
                    xd: 0
                }));
                assert!(!evaluation.matches(Thresholds {
                    prop: 0,
                    xp: 0,
                    xd: truncated + 1
                }));
            }
        }
    }

    #[test]
    fn triangular_poly_matches_original_indexing_for_every_coefficient() {
        for st in [
            std::array::from_fn(|i| i as f64 + 1.0),
            std::array::from_fn(|i| (i as f64 - 20.0) / 3.0),
            std::array::from_fn(|i| if i % 3 == 0 { 0.0 } else { -(i as f64) }),
        ] {
            assert_eq!(poly(&st), original_poly(&st));
        }
    }

    #[test]
    fn raw_filter_boundaries_stability_and_hp_sorting_are_exact() {
        // 157 是 181 在模 256 下的逆元，使变换后按 0..255 顺序排列。
        let state = std::array::from_fn(|i| (((i + 256 - 160) * 157) & 255) as u8);
        let filtered = raw_filter(&state);
        let expected: Vec<u8> = (89..217).map(|i| (i & 63) as u8).collect();
        assert_eq!(filtered.as_slice(), expected);
        assert_eq!(filtered[0], 89 & 63);
        assert_eq!(filtered[127], 216 & 63);
        for value in 0..64 {
            assert_eq!(filtered.iter().filter(|&&item| item == value).count(), 2);
        }
        let mut base = [0; 128];
        base[..10].copy_from_slice(&[63, 0, 9, 8, 7, 6, 5, 4, 3, 2]);
        base[10..13].copy_from_slice(&[63, 1, 2]);
        base[13..16].copy_from_slice(&[0, 63, 63]);
        let unchanged = base;
        assert_eq!(base_properties(&base), [176, 38, 99, 36, 36, 36, 36, 36]);
        assert_eq!(base, unchanged); // HP 的局部排序不能修改技能所用的 base。
        base[31] = 63;
        assert_eq!(base_properties(&base), base_properties(&unchanged));
    }

    #[test]
    fn model_export_contains_both_original_weight_sequences() {
        let weights = model_weights();
        assert_eq!(weights.len(), 1035);
        assert_eq!(weights[0], [model20::BASE as f32, model20::BASE_QD as f32]);
        for i in 0..1034 {
            assert_eq!(weights[i + 1], [model20::MODEL[i] as f32, model20::MODEL_QD[i] as f32]);
        }
    }

    #[test]
    fn skill_frequency_corner_cases_preserve_all_original_branches() {
        let ids = std::array::from_fn(|i| i as u32);
        let mut base = [40_u8; 128];
        base[60..64].copy_from_slice(&[6, 9, 40, 4]);
        let freq = skill_frequencies(&base, &ids);
        assert_eq!(&freq[..14], &[30; 14]);
        assert_eq!(&freq[14..], &[36, 34]); // last = -1，14/15 都加成。
        for last in 0..16 {
            let mut corner = base;
            corner[64 + last * 4..68 + last * 4].fill(11);
            let freq = skill_frequencies(&corner, &ids);
            assert_eq!(freq[last], 2);
            assert_eq!(freq, reference_freq(&corner, &ids));
        }
        for minimum in [0, 10, 11, 34, 35, 63] {
            for id in 0..40 {
                let mut corner_ids = ids;
                corner_ids[15] = id;
                let mut corner = base;
                corner[124..128].fill(minimum);
                let freq = skill_frequencies(&corner, &corner_ids);
                assert_eq!(freq, reference_freq(&corner, &corner_ids));
                if id >= 35 || minimum <= 10 {
                    assert_eq!(freq[15], 0);
                }
            }
        }
        let mut high = [63; 128];
        assert_eq!(skill_frequencies(&high, &ids)[14..], [106, 106]);
        high[60..64].fill(0);
        assert_eq!(skill_frequencies(&high, &ids)[14..], [53, 53]);
    }

    #[test]
    fn features_are_indexed_by_id_and_shadow_and_id34_bonuses_are_exact() {
        let props = [344, 57, 53, 66, 72, 70, 71, 61];
        let mut ids = [39; 16];
        let mut freq = [0; 16];
        ids[0] = 24;
        ids[1] = 34;
        freq[0] = 19;
        freq[1] = 2;
        let st = features(&props, &ids, &freq, 68);
        assert_eq!(st[32], 19.0);
        assert_eq!(st[42], 22.0);
        assert_eq!(st[43], 68.0 / 3.0 * 19.0 / 100.0);
        let expanded = original_poly(&st);
        let mut expected = [model20::BASE, model20::BASE_QD];
        for i in 0..1034 {
            expected[0] += expanded[i] * model20::MODEL[i];
            expected[1] += expanded[i] * model20::MODEL_QD[i];
        }
        assert_eq!(scores(&props, &ids, &freq, 68), expected);
        assert!(features(&props, &ids, &freq, -1)[43] < 0.0);
        freq[0] = 0;
        freq[1] = 0;
        let st = features(&props, &ids, &freq, -999);
        assert_eq!(st[32], 0.0);
        assert_eq!(st[42], 0.0);
        assert_eq!(st[43], 0.0);
    }

    #[test]
    fn shadow_uses_raw_name_and_team_including_at_characters() {
        let evaluation = evaluate("原始@队伍", "原始@名字-8");
        assert_eq!(evaluation.shadow_numerator, -108);
        assert_eq!(evaluation.skill_ids[15], 24);
        assert_eq!(evaluation.skill_freqs[15], 6);
        close(evaluation.xp, 1600.9857424094128);
        close(evaluation.xd, 6221.845397485766);
        assert_evaluation(&evaluation, &reference_evaluation("原始@队伍", "原始@名字-8"));
        let evaluation = evaluate("golden-team", "完整金标准-1");
        assert_eq!(evaluation.shadow_numerator, 68);
        close(evaluation.xp, 1824.4084469516386);
        close(evaluation.xd, 4813.403379853868);
    }

    #[test]
    fn shadow_may_exceed_256_bytes_without_truncating_utf8_semantics() {
        let name = format!("{}x", "中".repeat(85));
        let shadow = format!("{name}?shadow");
        assert_eq!(name.len(), 256);
        assert_eq!(shadow.len(), 263);
        let team = initial_team_state("边界队伍4");
        // 一轮 256 步只读到前导零之后的 255 字节，附加 shadow 不改变此状态。
        assert_eq!(rc4_name(&team, &name), rc4_name(&team, &shadow));
        assert_eq!(rc4_name(&team, &shadow), reference_state("边界队伍4", &shadow));
        let evaluation = evaluate_cached(&team, &name);
        assert_eq!(evaluation.shadow_numerator, -201);
        assert_eq!(evaluation.skill_ids[7], 24);
        assert_eq!(evaluation.skill_freqs[7], 15);
        assert_evaluation(&evaluation, &reference_evaluation("边界队伍4", &name));
        for length in [0, 1, 248, 249, 254, 255, 256] {
            let name = format!("{}?shadow", "a".repeat(length));
            assert_eq!(rc4_name(&team, &name), reference_state("边界队伍4", &name));
        }
        let nul_team = "含\0@队伍🙂";
        let nul_name = "含\0@名字🙂";
        assert_evaluation(&evaluate(nul_team, nul_name), &reference_evaluation(nul_team, nul_name));
    }
}

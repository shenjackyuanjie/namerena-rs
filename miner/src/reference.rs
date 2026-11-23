//! 仅测试使用的独立安全 oracle；参考 GPU 金标准的模运算/完整 PRGA 路径。
//! 不调用生产 KSA、属性提取、技能或多项式映射，也不依赖 GPU crate。
use crate::{
    GenMethods,
    evaluate::xuping::Features20,
    generate::NameGenerator,
    name::{KsaPrefix, Namer, TeamNamer},
};
#[allow(clippy::excessive_precision)]
#[path = "evaluate/xuping/model20.rs"]
mod weights;

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
    for (step, pair) in stream.as_chunks::<2>().0.iter().enumerate() {
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

fn reference_scores(team: &str, name: &str, props: &[u32; 8], ids: &[u32; 40], freqs: &[u32; 16]) -> [f64; 2] {
    let mut st = [0.0; 44];
    for i in 0..8 {
        st[i] = f64::from(props[i]);
    }
    for i in 0..16 {
        if freqs[i] != 0 {
            st[ids[i] as usize + 8] = f64::from(freqs[i]);
        }
    }
    if st[32] > 0.0 {
        let shadow = reference_props(&reference_state(team, &format!("{name}?shadow")));
        let mut sum = f64::from(shadow[0]) / 3.0;
        for &prop in &shadow[1..] {
            sum += f64::from(prop) - 36.0;
        }
        sum -= (f64::from(shadow[7]) - 36.0) * 3.0;
        st[43] = (sum - 210.0) * st[32] / 100.0;
    }
    if st[42] > 0.0 {
        st[42] += 20.0;
    }
    let mut result = [weights::BASE, weights::BASE_QD];
    for (i, term) in original_poly(&st).into_iter().enumerate() {
        result[0] += term * weights::MODEL[i];
        result[1] += term * weights::MODEL_QD[i];
    }
    result
}

fn check(team: &str, name: &str, state: [u8; 256]) {
    let expected = reference_state(team, name);
    assert_eq!(state, expected, "KSA: {name:?}@{team:?}");
    let props = reference_props(&expected);
    let (ids, freqs, after_prga) = reference_skills(expected);
    let mut namer = Namer::new_raw_unchecked("", team);
    namer.replace_state(name, state);
    assert_eq!(namer.name_prop, props);
    let base = reference_base(&expected);
    assert_eq!(namer.name_base[..31], base[..31]);
    assert!(namer.name_base[31..].iter().all(|&value| value == 0));
    namer.update_skill();
    assert_eq!(namer.val, after_prga);
    assert_eq!(namer.name_base.as_slice(), base);
    assert_eq!(namer.skl_id.map(u32::from), ids);
    assert_eq!(namer.skl_freq[..16].iter().map(|&f| u32::from(f)).collect::<Vec<_>>(), freqs);
    assert!(namer.skl_freq[16..].iter().all(|&f| f == 0));
    let scores = reference_scores(team, name, &props, &ids, &freqs);
    let features = Features20::new(&namer);
    for (actual, expected) in [features.xp(), features.xd()].into_iter().zip(scores) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "模型实际 {actual}，预期 {expected}，{name:?}@{team:?}"
        );
        assert_eq!(actual as u32, expected as u32);
    }
}

#[test]
fn independent_safe_oracle_covers_utf8_skills_shadow_and_four_lanes() {
    let mut random = 0x9234_5678_9abc_def0_u64;
    for team in ["", "shenjack", "队@伍\0", &"t".repeat(256)] {
        let team_namer = TeamNamer::new(team).unwrap();
        for algorithm in [
            GenMethods::Base16384,
            GenMethods::Base64Std,
            GenMethods::Base64Url,
            GenMethods::Number,
        ] {
            for template in ["[]", "我也不知道我要测多少 []", "前@缀\0[]后缀[]"] {
                let generator = NameGenerator::new(&algorithm, template);
                let prefix = KsaPrefix::new(&team_namer, generator.prefix());
                let mut names: [String; 4] = std::array::from_fn(|_| String::new());
                for _ in 0..24 {
                    for name in &mut names {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        generator.render(random, name);
                    }
                    let states = prefix.states::<4>(std::array::from_fn(|i| names[i].as_str()));
                    for (name, state) in names.iter().zip(states) {
                        check(team, name, state);
                    }
                }
            }
        }
    }
}

#[test]
fn independent_safe_oracle_covers_ksa_prefix_and_length_edges() {
    let team = TeamNamer::new_unchecked("@队伍");
    for name in [
        String::new(),
        "x".into(),
        "@\0".into(),
        "a".repeat(255),
        "a".repeat(256),
        "测".repeat(85),
        format!("{}@", "测".repeat(85)),
    ] {
        for prefix_len in [0, 1, 10, 254, 255, 256] {
            let prefix_len = prefix_len.min(name.len());
            if !name.is_char_boundary(prefix_len) {
                continue;
            }
            let prefix = KsaPrefix::new(&team, &name[..prefix_len]);
            check(&team.team, &name, prefix.states([name.as_str()])[0]);
        }
    }
}

#[test]
fn four_lane_ksa_matches_reference_across_key_wraps_and_mixed_lengths() {
    let team = TeamNamer::new_unchecked("队@伍\0");
    let near_limit = "字".repeat(85);
    let cases = [
        ("", ["", "", "", ""]),
        ("", ["a", "\0", "b", "c"]),
        ("前", ["前", "前", "前", "前"]),
        ("前", ["前\0中", "前abcd", "前中\0", "前éé"]),
        ("前", ["前9", "前10", "前11", "前12"]),
        ("", ["", "\0", "中", near_limit.as_str()]),
        (near_limit.as_str(), [near_limit.as_str(); 4]),
    ];
    for (prefix, names) in cases {
        let states = KsaPrefix::new(&team, prefix).states(names);
        for (name, state) in names.into_iter().zip(states) {
            assert_eq!(state, reference_state(&team.team, name), "{prefix:?} / {name:?}");
        }
    }
}

#[test]
fn prior_lossy_gate_counterexample_keeps_full_properties() {
    let team = TeamNamer::new_unchecked("golden-team");
    let mut namer = Namer::new_from_team_namer_unchecked(&team, "");
    assert!(namer.replace_name(&team, "完整金标准-3"));
    assert_eq!(namer.name_prop, [339, 79, 82, 41, 97, 64, 49, 52]);
    assert_eq!(namer.get_property(), 577.0);
    assert!(namer.name_prop[7] - 36 < 24);
    check("golden-team", "完整金标准-3", namer.val);
}

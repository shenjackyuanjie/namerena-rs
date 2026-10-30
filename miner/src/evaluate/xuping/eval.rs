//! 虚评 1.3.1

#[cfg(feature = "simd")]
use std::simd::f64x64;
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;

use crate::evaluate::xuping::{model13 as xuping13, model20 as xuping20};
use crate::name::Namer;

pub fn predict_13(name: &Namer) -> f64 {
    let mut sum = xuping13::INTERCEPT;
    /*
    memset(st + 8, 0, 35 * sizeof(int));
    for (int i = 0; i < 16; i++)
        if (context::freq[i]) st[context::skill[i] + 8] = context::freq[i]; */

    let mut combined_skills: [f64; 43] = [0.0; 43];
    // 长度取 64, 方便simd填充
    // for i in 0..8 {
    //     combined_skills[i] = name.name_prop[i] as f64;
    // }
    for (i, prop) in combined_skills.iter_mut().enumerate().take(8) {
        *prop = name.name_prop[i] as f64;
    }
    for i in 0..16 {
        if name.skl_freq[i] != 0 {
            combined_skills[name.skl_id[i] as usize + 8] = name.skl_freq[i] as f64;
        }
    }
    let mut check: [f64; 989] = [0.0; 989];
    // use simd
    #[cfg(feature = "simd")]
    // #[cfg(not(feature = "simd"))]
    {
        // 先准备数据
        let mut target = [0_f64; 989];
        target[0..43].copy_from_slice(&combined_skills[0..43]);
        let mut k = 43;
        // 43 * 43
        for i in 0..43 {
            for j in i..43 {
                target[k] = combined_skills[i] * combined_skills[j];
                k += 1;
            }
        }

        // 准备模型数据
        // 989 整除 64 为 15 余 49
        // let mut aline_target = &mut target[0..15 * 64];
        let simd_module = {
            let mut simd_vec = Vec::with_capacity(15);
            for i in 0..15 {
                let simd = f64x64::from_slice(&xuping13::MODULE[i * 64..(i + 1) * 64]);
                simd_vec.push(simd);
            }
            simd_vec
        };

        let simd_target = {
            let mut simd_vec = Vec::with_capacity(15);
            for i in 0..15 {
                let simd = f64x64::from_slice(&target[i * 64..(i + 1) * 64]);
                simd_vec.push(simd);
            }
            simd_vec
        };

        // 主! 体!
        let mut tmp = f64x64::splat(0.0);
        for i in 0..simd_module.len() {
            tmp += simd_module[i] * simd_target[i];
        }
        sum += tmp.reduce_sum();

        // 最后一个不足 64 的部分
        for i in 15 * 64..989 {
            sum += target[i] * xuping13::MODULE[i];
            check[i] = target[i] * xuping13::MODULE[i];
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        // - st: 名字属性。0~7 是八围，8~42 是技能熟练度。

        let mut cnt = 0;
        for i in 0..43 {
            sum += combined_skills[i] * xuping13::MODULE[cnt];
            cnt += 1;
        }
        for i in 0..43 {
            for j in i..43 {
                sum += combined_skills[i] * combined_skills[j] * xuping13::MODULE[cnt];
                cnt += 1;
            }
        }
    }
    sum
}

/*function Poly(x) {
    var xp = new Array()
    for (let y = 0; y < 1034; y++) {
        var l = 44
        var i = 0, p = 0, q = 0, r = 0
        var j = y
        for (let k = 0; k < 45; k++) {
            i++;
            if (i > 2) p++;
            q = j;
            j = j - l + p;
            if (j < 0) break;
        }
        if (i == 1) r = x[q]
        if (i > 1) {
            r = x[p] * x[p + q]
        }
        xp[y] = r
    }
    return xp
} */
/// 按线性项、上三角二次项的原模型顺序展开，避免每项重新倒推行号。
#[cfg(test)]
pub fn poly(input: &[f64; 44]) -> [f64; 1034] {
    let mut result = [0.0; 1034];
    result[..44].copy_from_slice(input);
    let mut index = 44;
    for i in 0..44 {
        for j in i..44 {
            result[index] = input[i] * input[j];
            index += 1;
        }
    }
    result
}

#[cfg(test)]
fn original_poly(input: &[f64; 44]) -> [f64; 1034] {
    let mut result = [0.0; 1034];
    for index in 0..1034 {
        let l: i32 = 44;
        let mut i: i32 = 0;
        let mut p: i32 = 0;
        let mut q: i32 = 0;
        let mut getter = 0.0;
        let mut j: i32 = index;
        for _ in 0..45 {
            i += 1;
            if i > 2 {
                p += 1;
            }
            q = j;
            j = j - l + p;
            if j < 0 {
                break;
            }
        }
        if i == 1 {
            getter = input[q as usize];
        }
        if i > 1 {
            getter = input[p as usize] * input[(p + q) as usize];
        }
        // println!("{} {} {} {} {}", index, i, p, q, getter);
        result[index as usize] = getter;
    }
    result
}

/*
if (x[32] > 0) {//x[32]>48
    name.load_name(nametmp[0] + '?shadow')
    props = name.calc_props()
    var shadow_sum = props[7] / 3
    for (let j = 0; j < 7; j++)shadow_sum += props[j]
    //更新部分
    shadow_sum -= props[6] * 3
    var shadowi = shadow_sum - 210
    //更新部分
    shadowi = shadowi * x[32] / 100
    x[43] = parseFloat(shadowi.toFixed(3))
} else {
    x[43] = 0
}
if (x[42] > 0) x[42] += 20

 */
#[cfg(test)]
fn predict_20_legacy(name: &Namer) -> f64 {
    let mut st: [f64; 44] = [0.0; 44];
    // for i in 0..8 {
    //     st[i] = name.name_prop[i] as f64;
    // }
    for (i, prop) in st.iter_mut().enumerate().take(8) {
        *prop = name.name_prop[i] as f64;
    }
    for i in 0..16 {
        if name.skl_freq[i] != 0 {
            st[name.skl_id[i] as usize + 8] = name.skl_freq[i] as f64;
        }
    }

    if st[32] > 0.0 {
        let shadow_name = Namer::new_unchecked(&format!("{}?shadow@{}", name.name, name.team));
        let mut shadow_sum = shadow_name.name_prop[0] as f64 / 3.0;

        for j in 1..8 {
            shadow_sum += shadow_name.name_prop[j] as f64 - 36.0;
        }
        shadow_sum -= (shadow_name.name_prop[7] as f64 - 36.0) * 3.0;
        let mut shadowi = shadow_sum - 210.0;

        shadowi = shadowi * st[32] / 100.0;
        st[43] = shadowi;
    } else {
        st[43] = 0.0;
    }

    if st[42] > 0.0 {
        st[42] += 20.0;
    }

    let xp = poly(&st);

    let mut sum = xuping20::BASE;

    unsafe {
        #[cfg(feature = "simd")]
        {
            let mut simd_sum = f64x64::splat(0.0);
            for i in (0..1024).step_by(64) {
                let simd_xp = f64x64::from_slice(xp.get_unchecked(i..));
                let simd_model = f64x64::from_slice(xuping20::MODEL.get_unchecked(i..));
                simd_sum += simd_xp * simd_model;
            }
            sum += simd_sum.reduce_sum();
            // 剩10个
            for i in 0..10 {
                sum += xp[i + 1024] * xuping20::MODEL[i + 1024];
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for (i, xp) in xp.iter().enumerate() {
                sum += xp * xuping20::MODEL.get_unchecked(i);
            }
        }
    }

    sum
}

#[cfg(test)]
fn predict_20_qd_legacy(name: &Namer) -> f64 {
    let mut st: [f64; 44] = [0.0; 44];
    for (i, prop) in st.iter_mut().enumerate().take(8) {
        *prop = name.name_prop[i] as f64;
    }
    for i in 0..16 {
        if name.skl_freq[i] != 0 {
            st[name.skl_id[i] as usize + 8] = name.skl_freq[i] as f64;
        }
    }

    if st[32] > 0.0 {
        let shadow_name = Namer::new_unchecked(&format!("{}?shadow@{}", name.name, name.team));
        let mut shadow_sum = shadow_name.name_prop[0] as f64 / 3.0;

        for j in 1..8 {
            shadow_sum += shadow_name.name_prop[j] as f64 - 36.0;
        }
        shadow_sum -= (shadow_name.name_prop[7] as f64 - 36.0) * 3.0;
        let mut shadowi = shadow_sum - 210.0;

        shadowi = shadowi * st[32] / 100.0;
        st[43] = shadowi;
    } else {
        st[43] = 0.0;
    }

    if st[42] > 0.0 {
        st[42] += 20.0;
    }

    let xp = poly(&st);

    let mut sum = xuping20::BASE_QD;

    unsafe {
        #[cfg(feature = "simd")]
        {
            let mut simd_sum = f64x64::splat(0.0);
            for i in (0..1024).step_by(64) {
                let simd_xp = f64x64::from_slice(xp.get_unchecked(i..));
                let simd_model = f64x64::from_slice(xuping20::MODEL_QD.get_unchecked(i..));
                simd_sum += simd_xp * simd_model;
            }
            sum += simd_sum.reduce_sum();
            // 剩10个
            for i in 0..10 {
                sum += xp[i + 1024] * xuping20::MODEL_QD[i + 1024];
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for (i, xp) in xp.iter().enumerate() {
                sum += xp * xuping20::MODEL_QD.get_unchecked(i);
            }
        }
    }

    sum
}

/// XP/QD 共享同一组 f64 特征，影子名字仅在技能24启用时计算一次。
pub struct Features20([f64; 44]);
impl Features20 {
    pub fn new(name: &Namer) -> Self {
        let mut st = [0.0; 44];
        for (i, &prop) in name.name_prop.iter().enumerate() {
            st[i] = f64::from(prop);
        }
        for i in 0..16 {
            if name.skl_freq[i] != 0 {
                st[name.skl_id[i] as usize + 8] = f64::from(name.skl_freq[i]);
            }
        }
        if st[32] > 0.0 {
            // 原始名字、队伍是独立 UTF-8 字段，不能通过 name@team 重新解析。
            let shadow = Namer::new_raw_unchecked(&format!("{}?shadow", name.name), &name.team);
            let mut sum = f64::from(shadow.name_prop[0]) / 3.0;
            for &prop in &shadow.name_prop[1..] {
                sum += f64::from(prop) - 36.0;
            }
            sum -= (f64::from(shadow.name_prop[7]) - 36.0) * 3.0;
            st[43] = (sum - 210.0) * st[32] / 100.0;
        }
        if st[42] > 0.0 {
            st[42] += 20.0;
        }
        Self(st)
    }

    pub fn xp(&self) -> f64 { self.dot(&xuping20::MODEL, xuping20::BASE) }
    pub fn xd(&self) -> f64 { self.dot(&xuping20::MODEL_QD, xuping20::BASE_QD) }

    fn dot(&self, model: &[f64; 1034], base: f64) -> f64 {
        // 不生成 1034 项临时数组，只跳过数学上精确为零的项。
        // SIMD 档保留旧实现按 index%64 累加和 reduce_sum 的运算顺序，仍是 f64。
        #[cfg(feature = "simd")]
        let mut lanes = [0.0; 64];
        #[cfg(not(feature = "simd"))]
        let mut sum = base;
        #[cfg(feature = "simd")]
        let mut tail = [0.0; 10];
        let mut add = |index: usize, term: f64| {
            let product = term * model[index];
            #[cfg(feature = "simd")]
            if index < 1024 {
                lanes[index & 63] += product;
            } else {
                tail[index - 1024] = product;
            }
            #[cfg(not(feature = "simd"))]
            {
                sum += product;
            }
        };
        for i in 0..44 {
            if self.0[i] != 0.0 {
                add(i, self.0[i]);
            }
        }
        let mut index = 44;
        for i in 0..44 {
            if self.0[i] == 0.0 {
                index += 44 - i;
                continue;
            }
            for j in i..44 {
                if self.0[j] != 0.0 {
                    add(index, self.0[i] * self.0[j]);
                }
                index += 1;
            }
        }
        #[cfg(feature = "simd")]
        {
            let mut sum = base + f64x64::from_array(lanes).reduce_sum();
            for value in tail {
                sum += value;
            }
            sum
        }
        #[cfg(not(feature = "simd"))]
        {
            sum
        }
    }
}

pub fn predict_20(name: &Namer) -> f64 { Features20::new(name).xp() }
pub fn predict_20_qd(name: &Namer) -> f64 { Features20::new(name).xd() }

#[cfg(test)]
mod streaming_tests {
    use super::*;
    #[test]
    fn triangular_expansion_matches_original_index_mapper() {
        let input = std::array::from_fn(|i| i as f64 * 0.125 - 2.0);
        assert_eq!(poly(&input), original_poly(&input));
    }
    #[test]
    fn streaming_f64_matches_legacy_full_model() {
        for team in ["shenjack", "", "中文队伍"] {
            for id in 0..1000 {
                let mut name = Namer::new_raw_unchecked(&format!("我也不知道我要测多少 {id}"), team);
                name.update_skill();
                let features = Features20::new(&name);
                assert!((features.xp() - predict_20_legacy(&name)).abs() < 1e-9);
                assert!((features.xd() - predict_20_qd_legacy(&name)).abs() < 1e-9);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::name::Namer;

    #[test]
    fn xuping_13_test() {
        // let mut namer = Namer::new(&"x@x".to_string()).unwrap();
        let mut namer = Namer::new(&"pi31uXx?shadow@魔".to_string()).unwrap();

        namer.update_skill();

        println!("{:?}", namer.get_info());
        #[cfg(not(feature = "simd"))]
        assert_eq!(predict_13(&namer), 5799.586821819173);
        #[cfg(feature = "simd")]
        assert_eq!(predict_13(&namer), 5799.586821819176);
    }

    #[test]
    fn xuping_20_1015_test() {
        // let mut namer = Namer::new(&"pi31uXx?shadow@魔".to_string()).unwrap();
        let mut namer = Namer::new(&"一一七啺埀㴁@shenjack".to_string()).unwrap();
        // 5971 7226
        namer.update_skill();

        println!("{:?}", namer.get_info());
        #[cfg(not(feature = "simd"))]
        assert_eq!(predict_20(&namer), 3603.4389333619297);
        #[cfg(feature = "simd")]
        assert_eq!(predict_20(&namer), 3603.4389333619315);
    }

    #[test]
    fn xuping_20_1015_qd_test() {
        // let mut namer = Namer::new(&"pi31uXx?shadow@魔".to_string()).unwrap();
        let mut namer = Namer::new(&"一一七啺埀㴁@shenjack".to_string()).unwrap();
        // 5971 7226
        namer.update_skill();

        println!("{:?}", namer.get_info());
        #[cfg(not(feature = "simd"))]
        assert_eq!(predict_20_qd(&namer), 3639.8920896688987);
        #[cfg(feature = "simd")]
        assert_eq!(predict_20_qd(&namer), 3639.8920896689424);
    }
}

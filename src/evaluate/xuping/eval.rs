/// 虚评 1.3.1

#[cfg(feature = "simd")]
use std::simd::f64x64;
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;

use crate::evaluate::xuping::model13 as xuping13;
use crate::name::Namer;

pub fn predict_13(name: &Namer) -> f64 {
    let mut sum = xuping13::INTERCEPT;
    /*
    memset(st + 8, 0, 35 * sizeof(int));
    for (int i = 0; i < 16; i++)
        if (context::freq[i]) st[context::skill[i] + 8] = context::freq[i]; */
    // 长度取 64, 方便simd填充
    let mut st: [f64; 64] = [0.0; 64];
    for i in 0..7 {
        st[i] = name.name_prop[i] as f64;
    }
    for i in 0..16 {
        if name.skl_freq[i] != 0 {
            st[name.skl_id[i] as usize + 8] = name.skl_freq[i] as f64;
        }
    }

    // use simd
    #[cfg(feature = "simd")]
    {
        // 先准备数据
        let mut target = [0_f64; 989];
        target[0..43].copy_from_slice(&st[0..43]);
        let mut k = 43;
        // 43 * 43
        for i in 0..43 {
            for j in i..43 {
                target[k] = st[i] * st[j];
                k += 1;
            }
        }

        // 准备模型数据
        let mut simds = xuping13::MODULE.clone();
        let simd_module = simds.as_simd_mut::<64>();
        let simd_target = target.as_simd_mut::<64>();
        // 前面多出来的
        for i in 0..simd_module.0.len() {
            sum += simd_module.0[i] * simd_target.0[i];
        }
        // 主! 体!
        let mut tmp = f64x64::splat(0.0);
        for i in 0..simd_module.1.len() {
            tmp += simd_module.1[i] * simd_target.1[i];
        }
        sum += tmp.reduce_sum();
        // 后面多出来的
        for i in 0..simd_module.2.len() {
            sum += simd_module.2[i] * simd_target.2[i];
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut cnt = 0;
        for i in 0..43 {
            sum += st[i] * MODEL[cnt];
            cnt += 1;
            for j in i..43 {
                sum += st[i] * st[j] * MODEL[cnt];
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
pub fn poly(name: &Namer) -> [f64; 1034] {
    let mut result = [0.0; 1034];
    
    
    result
}

pub fn predict_20(name: &Namer) -> f64 { 0.0 }

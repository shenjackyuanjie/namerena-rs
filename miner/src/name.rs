use std::cmp::min;
#[cfg(feature = "simd")]
// 考虑到 avx512 的普及性
// 咱还是用 u8x32 吧
use std::simd::u8x32;

use tracing::warn;

#[inline(always)]
pub fn median<T>(x: T, y: T, z: T) -> T
where
    T: std::cmp::Ord + std::marker::Copy,
{
    // std::max(std::min(x, y), std::min(std::max(x, y), z))
    // x.min(y).max(x.max(y).min(z))
    if x < y {
        if y < z {
            y
        } else if x < z {
            z
        } else {
            x
        }
    } else if x < z {
        x
    } else if y < z {
        z
    } else {
        y
    }
}

// #[repr(C)]
#[derive(Debug, Clone)]
pub struct TeamNamer {
    pub team: String,
    pub val: [u8; 256],
}

impl TeamNamer {
    /// 方便使用的 new
    /// 会检查长度是否超过 256
    #[inline(always)]
    pub fn new(team: &str) -> Option<Self> {
        if team.len() > 256 {
            warn!("Team too long({}): {}", team.len(), team);
            return None;
        }
        Some(Self::new_unchecked(team))
    }
    #[inline(always)]
    pub fn new_unchecked(team: &str) -> Self {
        let team_bytes = team.as_bytes();
        let mut val: [u8; 256] = (0..=255).collect::<Vec<u8>>().try_into().unwrap();
        let mut s = 0_u8;
        let t_len = team.len() + 1;
        for i in 0..256 {
            if (i % t_len) != 0 {
                s = s.wrapping_add(team_bytes[(i % t_len) - 1]);
            }
            s = s.wrapping_add(val[i]);
            val.swap(i, s as usize);
        }
        Self {
            team: team.to_string(),
            val,
        }
    }
    #[inline(always)]
    pub fn clone_vals(&self) -> [u8; 256] { self.val }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Namer {
    pub name: String,
    pub team: String,
    pub val: [u8; 256],
    pub name_base: [u8; 128],
    pub name_prop: [u32; 8],
    pub skl_id: [u8; 40],
    pub skl_freq: [u8; 40],
}

#[allow(dead_code)]
impl Namer {
    /// 最完整的、最简单的 new
    /// 可以直接丢一个 name 进来
    #[inline(always)]
    pub fn new(full_name: &String) -> Option<Self> {
        // name@team
        // name
        let (name, team) = full_name.split_once('@').unwrap_or((full_name, full_name));
        if name.len() > 256 {
            warn!("Name too long({}): {}", name.len(), name);
            return None;
        }
        if team.len() > 256 {
            warn!("Team too long({}): {}", team.len(), team);
            return None;
        }
        Some(Self::new_raw_unchecked(name, team))
    }
    /// 用起来也很方便的
    /// 只不过是需要分别输入, 也挺好用的
    /// 包括了长度检测
    #[inline(always)]
    pub fn new_raw(name: &str, team: &str) -> Option<Self> {
        if name.len() > 256 {
            warn!("Name too long({}): {}", name.len(), name);
            return None;
        }
        if team.len() > 256 {
            warn!("Team too long({}): {}", team.len(), team);
            return None;
        }
        Some(Self::new_raw_unchecked(name, team))
    }
    /// 不带检查长度的 new
    /// 依然可以传一个完整的进来
    #[inline(always)]
    pub fn new_unchecked(full_name: &str) -> Self {
        let (name, team) = full_name.split_once('@').unwrap_or((full_name, full_name));
        Self::new_raw_unchecked(name, team)
    }
    /// 大部分情况下的实际调用 p1
    /// p2 是 new_from_team_namer_unchecked
    /// 实际上还是个包装
    #[inline(always)]
    pub fn new_raw_unchecked(name: &str, team: &str) -> Self {
        let team_namer = TeamNamer::new_unchecked(team);
        Self::new_from_team_namer_unchecked(&team_namer, name)
    }
    /// 带检查长度的 from namer
    /// 我其实也不知道为啥要有他, 就带上了吧
    #[inline(always)]
    pub fn new_from_team_namer(team_namer: &TeamNamer, name: &str) -> Option<Self> {
        if name.len() > 256 {
            warn!("Name too long({}): {}", name.len(), name);
            return None;
        }
        Some(Self::new_from_team_namer_unchecked(team_namer, name))
    }
    /// 实际 new 实现
    #[inline(always)]
    pub fn new_from_team_namer_unchecked(team_namer: &TeamNamer, name: &str) -> Self {
        let mut val: [u8; 256] = team_namer.clone_vals();
        let mut name_base = [0_u8; 128];
        let mut name_prop = [0_u32; 8];
        let skl_id = [0_u8; 40];
        let skl_freq = [0_u8; 40];

        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len();
        for _ in 0..2 {
            unsafe {
                let mut s = 0_u8;
                let mut k = 0;
                for i in 0..256 {
                    s = s.wrapping_add(if k == 0 { 0 } else { *name_bytes.get_unchecked(k - 1) });
                    s = s.wrapping_add(*val.get_unchecked(i));
                    val.swap_unchecked(i, s as usize);
                    k = if k == name_len { 0 } else { k + 1 };
                }
            }
        }
        // simd 优化
        #[cfg(feature = "simd")]
        {
            let mut simd_val = val;
            let mut simd_val_b = [0_u8; 256];
            let simd_181 = u8x32::splat(181);
            let simd_160 = u8x32::splat(160);
            let simd_63 = u8x32::splat(63);

            for i in (0..256).step_by(32) {
                // 一次性加载64个数字
                let mut x = u8x32::from_slice(unsafe { simd_val.get_unchecked(i..) });
                x = x * simd_181 + simd_160;
                // 写入到 simd_val
                unsafe {
                    x.copy_to_slice(simd_val.get_unchecked_mut(i..));
                }

                let y = x & simd_63;
                unsafe {
                    y.copy_to_slice(simd_val_b.get_unchecked_mut(i..));
                }
            }

            let mut mod_count = 0;

            for i in 0..96 {
                unsafe {
                    if simd_val.get_unchecked(i) > &88 && simd_val.get_unchecked(i) < &217 {
                        *name_base.get_unchecked_mut(mod_count as usize) = *simd_val_b.get_unchecked(i);
                        mod_count += 1;
                    }
                }
                if mod_count > 30 {
                    break;
                }
            }
            if mod_count < 31 {
                for i in 96..256 {
                    unsafe {
                        if simd_val.get_unchecked(i) > &88 && simd_val.get_unchecked(i) < &217 {
                            *name_base.get_unchecked_mut(mod_count as usize) = *simd_val_b.get_unchecked(i);
                            mod_count += 1;
                        }
                    }
                    if mod_count > 30 {
                        break;
                    }
                }
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            let mut s = 0;
            for i in 0..256 {
                let m = ((val[i] as u32 * 181) + 160) % 256;
                if m >= 89 && m < 217 {
                    name_base[s as usize] = (m & 63) as u8;
                    s += 1;
                }
            }
        }

        // 计算 name_prop
        // let mut prop_name = name_base[0..32].to_vec();
        // prop_name[0..10].sort_unstable();
        // name_prop[0] = 154
        //     + prop_name[3] as u32
        //     + prop_name[4] as u32
        //     + prop_name[5] as u32
        //     + prop_name[6] as u32;
        // name_prop[1] = median(prop_name[10], prop_name[11], prop_name[12]) as u32 + 36;
        // name_prop[2] = median(prop_name[13], prop_name[14], prop_name[15]) as u32 + 36;
        // name_prop[3] = median(prop_name[16], prop_name[17], prop_name[18]) as u32 + 36;
        // name_prop[4] = median(prop_name[19], prop_name[20], prop_name[21]) as u32 + 36;
        // name_prop[5] = median(prop_name[22], prop_name[23], prop_name[24]) as u32 + 36;
        // name_prop[6] = median(prop_name[25], prop_name[26], prop_name[27]) as u32 + 36;
        // name_prop[7] = median(prop_name[28], prop_name[29], prop_name[30]) as u32 + 36;
        // 疯狂的 unsafe 优化（确信
        unsafe {
            let mut prop_name = [0_u8; 32];
            prop_name.copy_from_slice(name_base.get_unchecked(0..32));
            prop_name.get_unchecked_mut(0..10).sort_unstable();
            *name_prop.get_unchecked_mut(0) = 154
                + *prop_name.get_unchecked(3) as u32
                + *prop_name.get_unchecked(4) as u32
                + *prop_name.get_unchecked(5) as u32
                + *prop_name.get_unchecked(6) as u32;

            *name_prop.get_unchecked_mut(1) = median(
                *prop_name.get_unchecked(10),
                *prop_name.get_unchecked(11),
                *prop_name.get_unchecked(12),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(2) = median(
                *prop_name.get_unchecked(13),
                *prop_name.get_unchecked(14),
                *prop_name.get_unchecked(15),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(3) = median(
                *prop_name.get_unchecked(16),
                *prop_name.get_unchecked(17),
                *prop_name.get_unchecked(18),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(4) = median(
                *prop_name.get_unchecked(19),
                *prop_name.get_unchecked(20),
                *prop_name.get_unchecked(21),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(5) = median(
                *prop_name.get_unchecked(22),
                *prop_name.get_unchecked(23),
                *prop_name.get_unchecked(24),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(6) = median(
                *prop_name.get_unchecked(25),
                *prop_name.get_unchecked(26),
                *prop_name.get_unchecked(27),
            ) as u32
                + 36;
            *name_prop.get_unchecked_mut(7) = median(
                *prop_name.get_unchecked(28),
                *prop_name.get_unchecked(29),
                *prop_name.get_unchecked(30),
            ) as u32
                + 36;
        }

        Self {
            name: name.to_string(),
            team: team_namer.team.clone(),
            val,
            name_base,
            name_prop,
            skl_id,
            skl_freq,
        }
    }

    /// 更新当前的名字
    #[inline(always)]
    pub fn replace_name(&mut self, team_namer: &TeamNamer, name: &str) -> bool {
        self.val = team_namer.clone_vals();
        self.name = name.to_string();

        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len();

        unsafe {
            let val_ptr = self.val.as_mut_ptr();
            for _ in 0..2 {
                let mut s = 0_u8;
                let mut k = 0;
                for i in 0..256 {
                    s = s.wrapping_add(if k == 0 { 0 } else { *name_bytes.get_unchecked(k - 1) });
                    s = s.wrapping_add(*self.val.get_unchecked(i));
                    std::ptr::swap(val_ptr.add(i), val_ptr.add(s as usize));
                    k = if k == name_len { 0 } else { k + 1 };
                }
            }
        }
        // simd!
        #[cfg(feature = "simd")]
        {
            let mut simd_val = [0_u8; 256];
            let mut simd_val_b = [0_u8; 256];
            let simd_181 = u8x32::splat(181);
            let simd_160 = u8x32::splat(160);
            let simd_63 = u8x32::splat(63);

            for i in (0..256).step_by(32) {
                unsafe {
                    let mut x = u8x32::from_slice(self.val.get_unchecked(i..));
                    x = x * simd_181 + simd_160;
                    x.copy_to_slice(simd_val.get_unchecked_mut(i..));
                    let y = x & simd_63;
                    y.copy_to_slice(simd_val_b.get_unchecked_mut(i..));
                }
            }
            let mut mod_count = 0;
            for i in 0..96 {
                unsafe {
                    if *simd_val.get_unchecked(i) > 88 && *simd_val.get_unchecked(i) < 217 {
                        *self.name_base.get_unchecked_mut(mod_count as usize) = *simd_val_b.get_unchecked(i);
                        mod_count += 1;
                    }
                }
                if mod_count > 30 {
                    break;
                }
            }
            if mod_count < 31 {
                for i in 96..256 {
                    unsafe {
                        if simd_val.get_unchecked(i) > &88 && simd_val.get_unchecked(i) < &217 {
                            *self.name_base.get_unchecked_mut(mod_count as usize) = *simd_val_b.get_unchecked(i);
                            mod_count += 1;
                        }
                    }
                    if mod_count > 30 {
                        break;
                    }
                }
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut s = 0;
            for i in 0..256 {
                let m = ((self.val[i] as u32 * 181) + 160) % 256;
                if m >= 89 && m < 217 {
                    self.name_base[s as usize] = (m & 63) as u8;
                    s += 1;
                }
            }
        }
        // 计算 name_prop
        unsafe {
            let mut full = 0;
            let mut prop_name = [0_u8; 32];
            prop_name.copy_from_slice(self.name_base.get_unchecked(0..32));
            // 加一些特殊检测
            *self.name_prop.get_unchecked_mut(7) = median(
                *prop_name.get_unchecked(28),
                *prop_name.get_unchecked(29),
                *prop_name.get_unchecked(30),
            ) as u32
                + 36;
            full += self.name_prop.get_unchecked(7) - 36;
            if full < 24 {
                return false;
            }
            *self.name_prop.get_unchecked_mut(2) = median(
                *prop_name.get_unchecked(13),
                *prop_name.get_unchecked(14),
                *prop_name.get_unchecked(15),
            ) as u32
                + 36;
            *self.name_prop.get_unchecked_mut(3) = median(
                *prop_name.get_unchecked(16),
                *prop_name.get_unchecked(17),
                *prop_name.get_unchecked(18),
            ) as u32
                + 36;
            *self.name_prop.get_unchecked_mut(6) = median(
                *prop_name.get_unchecked(25),
                *prop_name.get_unchecked(26),
                *prop_name.get_unchecked(27),
            ) as u32
                + 36;
            full += self.name_prop.get_unchecked(2) + self.name_prop.get_unchecked(3) + self.name_prop.get_unchecked(6) - 108;
            if full < 165 {
                return false;
            }
            *self.name_prop.get_unchecked_mut(1) = median(
                *prop_name.get_unchecked(10),
                *prop_name.get_unchecked(11),
                *prop_name.get_unchecked(12),
            ) as u32
                + 36;
            *self.name_prop.get_unchecked_mut(4) = median(
                *prop_name.get_unchecked(19),
                *prop_name.get_unchecked(20),
                *prop_name.get_unchecked(21),
            ) as u32
                + 36;
            *self.name_prop.get_unchecked_mut(5) = median(
                *prop_name.get_unchecked(22),
                *prop_name.get_unchecked(23),
                *prop_name.get_unchecked(24),
            ) as u32
                + 36;
            full += self.name_prop.get_unchecked(1) + self.name_prop.get_unchecked(4) + self.name_prop.get_unchecked(5) - 108;
            if full < 250 {
                return false;
            }
            prop_name.get_unchecked_mut(0..10).sort_unstable();
            *self.name_prop.get_unchecked_mut(0) = 154
                + *prop_name.get_unchecked(3) as u32
                + *prop_name.get_unchecked(4) as u32
                + *prop_name.get_unchecked(5) as u32
                + *prop_name.get_unchecked(6) as u32;
            full += self.name_prop.get_unchecked(0) / 3 + 154;
            if full < 380 {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    pub fn update_skill(&mut self) {
        self.skl_id = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
        ];

        #[cfg(feature = "simd")]
        {
            let mut simd_val = [0_u8; 256];
            let mut simd_val_b = [0_u8; 256];
            let simd_181 = u8x32::splat(181);
            let simd_160 = u8x32::splat(160);
            let simd_63 = u8x32::splat(63);

            for i in (0..256).step_by(32) {
                unsafe {
                    let mut x = u8x32::from_slice(self.val.get_unchecked(i..));
                    x = x * simd_181 + simd_160;
                    x.copy_to_slice(simd_val.get_unchecked_mut(i..));
                    let y = x & simd_63;
                    y.copy_to_slice(simd_val_b.get_unchecked_mut(i..));
                }
            }

            let mut mod_count = 0;
            for i in 0..256 {
                unsafe {
                    if *simd_val.get_unchecked(i) > 88 && *simd_val.get_unchecked(i) < 217 {
                        *self.name_base.get_unchecked_mut(mod_count as usize) = *simd_val_b.get_unchecked(i);
                        mod_count += 1;
                    }
                }
            }
            // const int N = 256, M = 128, K = 64, skill_cnt = 40, max_len = 25;
            let mut a: u8 = 0;
            let mut b: u8 = 0;
            let mut s: u8 = 0;
            for _ in 0..2 {
                for i in 0..40 {
                    let rnd = unsafe {
                        a += 1;
                        b = b.wrapping_add(*self.val.get_unchecked(a as usize));
                        self.val.swap_unchecked(a as usize, b as usize);
                        let u: u8 = *self.val.get_unchecked(
                            ((*self.val.get_unchecked(a as usize) as u16 + *self.val.get_unchecked(b as usize) as u16) & 255)
                                as usize,
                        );
                        a += 1;
                        b = b.wrapping_add(*self.val.get_unchecked(a as usize));
                        self.val.swap_unchecked(a as usize, b as usize);
                        let t: u8 = *self.val.get_unchecked(
                            ((*self.val.get_unchecked(a as usize) as u16 + *self.val.get_unchecked(b as usize) as u16) & 255)
                                as usize,
                        );
                        ((((u as u32) << 8) | t as u32) % 40) as u8
                    };
                    unsafe {
                        s = (s as u16 + rnd as u16 + *self.skl_id.get_unchecked(i as usize) as u16) as u8 % 40;
                        self.skl_id.swap_unchecked(i as usize, s as usize);
                    }
                }
            }
            let mut last = -1;
            for (j, i) in (64..128).step_by(4).enumerate() {
                let p = unsafe {
                    min(
                        min(*self.name_base.get_unchecked(i), *self.name_base.get_unchecked(i + 1)),
                        min(*self.name_base.get_unchecked(i + 2), *self.name_base.get_unchecked(i + 3)),
                    )
                };
                unsafe {
                    if p > 10 && *self.skl_id.get_unchecked(j) < 35 {
                        *self.skl_freq.get_unchecked_mut(j) = p - 10;
                        if p < 35 {
                            last = j as i8;
                        }
                    } else {
                        *self.skl_freq.get_unchecked_mut(j) = 0;
                    }
                }
            }
            match last {
                -1 => unsafe {
                    // 判断 14, 15 去
                    if *self.skl_freq.get_unchecked(14) != 0 {
                        *self.skl_freq.get_unchecked_mut(14) += min(
                            min(*self.name_base.get_unchecked(60), *self.name_base.get_unchecked(61)),
                            *self.skl_freq.get_unchecked(14),
                        )
                    }
                    if *self.skl_freq.get_unchecked(15) != 0 {
                        *self.skl_freq.get_unchecked_mut(15) += min(
                            min(*self.name_base.get_unchecked(62), *self.name_base.get_unchecked(63)),
                            *self.skl_freq.get_unchecked(15),
                        )
                    }
                },
                14 => unsafe {
                    // 判断 15
                    *self.skl_freq.get_unchecked_mut(14) <<= 1;
                    if *self.skl_freq.get_unchecked(15) != 0 {
                        *self.skl_freq.get_unchecked_mut(15) += min(
                            min(*self.name_base.get_unchecked(62), *self.name_base.get_unchecked(63)),
                            *self.skl_freq.get_unchecked(15),
                        )
                    }
                },
                15 => unsafe {
                    // 判断 14
                    *self.skl_freq.get_unchecked_mut(15) <<= 1;
                    if *self.skl_freq.get_unchecked(14) != 0 {
                        *self.skl_freq.get_unchecked_mut(14) += min(
                            min(*self.name_base.get_unchecked(60), *self.name_base.get_unchecked(61)),
                            *self.skl_freq.get_unchecked(14),
                        )
                    }
                },
                x => unsafe {
                    *self.skl_freq.get_unchecked_mut(x as usize) <<= 1;
                    if *self.skl_freq.get_unchecked(14) != 0 {
                        *self.skl_freq.get_unchecked_mut(14) += min(
                            min(*self.name_base.get_unchecked(60), *self.name_base.get_unchecked(61)),
                            *self.skl_freq.get_unchecked(14),
                        )
                    }
                    if *self.skl_freq.get_unchecked(15) != 0 {
                        *self.skl_freq.get_unchecked_mut(15) += min(
                            min(*self.name_base.get_unchecked(62), *self.name_base.get_unchecked(63)),
                            *self.skl_freq.get_unchecked(15),
                        )
                    }
                },
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            let mut valb = self.val;

            for val in valb.iter_mut() {
                *val = ((*val as u32 * 181 + 160) % 256) as u8;
            }
            let mut mod_count = 0;
            for i in 0..256 {
                if valb[i] > 88 && valb[i] < 217 {
                    self.name_base[mod_count] = valb[i] & 63;
                    mod_count += 1;
                }
            }

            // const int N = 256, M = 128, K = 64, skill_cnt = 40, max_len = 25;
            let mut a: u8 = 0;
            let mut b: u8 = 0;
            let mut s: u8 = 0;
            for _ in 0..2 {
                for i in 0..40 {
                    let rnd = {
                        a += 1;
                        b = b.wrapping_add(self.val[a as usize]);
                        self.val.swap(a as usize, b as usize);
                        let u: u8 = self.val[((self.val[a as usize] as u16 + self.val[b as usize] as u16) & 255) as usize];
                        a += 1;
                        b = b.wrapping_add(self.val[a as usize]);
                        self.val.swap(a as usize, b as usize);
                        let t: u8 = self.val[((self.val[a as usize] as u16 + self.val[b as usize] as u16) & 255) as usize];
                        (((u as u32) << 8 | t as u32) % 40) as u8
                    };
                    s = (s as u16 + rnd as u16 + self.skl_id[i as usize] as u16) as u8 % 40;
                    self.skl_id.swap(i as usize, s as usize);
                }
            }

            let mut last = -1;
            for (j, i) in (64..128).step_by(4).enumerate() {
                let p = min(
                    min(self.name_base[i], self.name_base[i + 1]),
                    min(self.name_base[i + 2], self.name_base[i + 3]),
                );
                if p > 10 && self.skl_id[j] < 35 {
                    self.skl_freq[j] = p - 10;
                    if p < 35 {
                        last = j as i8;
                    }
                } else {
                    self.skl_freq[j] = 0;
                }
            }
            match last {
                // 判断 14, 15 去
                -1 => {
                    if self.skl_freq[14] != 0 {
                        self.skl_freq[14] += min(
                            min(self.name_base[60], self.name_base[61]),
                            self.skl_freq[14],
                        )
                    }
                    if self.skl_freq[15] != 0 {
                        self.skl_freq[15] += min(
                            min(self.name_base[62], self.name_base[63]),
                            self.skl_freq[15],
                        )
                    }
                },
                14 => {
                    self.skl_freq[14] <<= 1;
                    if self.skl_freq[15] != 0 {
                        self.skl_freq[15] += min(
                            min(self.name_base[62], self.name_base[63]),
                            self.skl_freq[15],
                        )
                    }
                },
                15 => {
                    self.skl_freq[15] <<= 1;
                    if self.skl_freq[14] != 0 {
                        self.skl_freq[14] += min(
                            min(self.name_base[60], self.name_base[61]),
                            self.skl_freq[14],
                        )
                    }
                },
                x => {
                    self.skl_freq[x as usize] <<= 1;
                    if self.skl_freq[14] != 0 {
                        self.skl_freq[14] += min(
                            min(self.name_base[60], self.name_base[61]),
                            self.skl_freq[14],
                        )
                    }
                    if self.skl_freq[15] != 0 {
                        self.skl_freq[15] += min(
                            min(self.name_base[62], self.name_base[63]),
                            self.skl_freq[15],
                        )
                    }
                },
            }
        }
    }

    #[inline(always)]
    pub fn get_property(&self) -> f32 {
        unsafe {
            self.name_prop.get_unchecked(1..=7).iter().sum::<u32>() as f32 + (*self.name_prop.get_unchecked(0) as f32 / 3_f32)
        }
    }

    pub fn get_info(&self) -> String {
        let main = format!(
            "name: {}, team: {} HP|{} 攻|{} 防|{} 速|{} 敏|{} 魔|{} 抗|{} 智|{} 八围:{}",
            self.name,
            self.team,
            self.name_prop[0],
            self.name_prop[1],
            self.name_prop[2],
            self.name_prop[3],
            self.name_prop[4],
            self.name_prop[5],
            self.name_prop[6],
            self.name_prop[7],
            self.get_property()
        );
        let skills = {
            let mut base = "".to_string();
            let skill_names = [
                "火球", "冰冻", "雷击", "地裂", "吸血", "投毒", "连击", "会心", "瘟疫", "命轮", "狂暴", "魅惑", "加速", "减速",
                "诅咒", "治愈", "苏生", "净化", "铁壁", "蓄力", "聚气", "潜行", "血祭", "分身", "幻术", "防御", "守护", "反弹",
                "护符", "护盾", "反击", "吞噬", "亡灵", "垂死", "隐匿", "啧", "啧", "啧", "啧", "啧",
            ];
            // 后处理
            let mut skills = [0; 40];
            for i in 0..40 {
                if self.skl_freq[i] != 0 {
                    skills[self.skl_id[i] as usize] = self.skl_freq[i];
                }
            }
            for (i, v) in skills.iter().enumerate() {
                if *v > 0 {
                    base.push_str(format!("{}: {}|", skill_names[i], v).as_str());
                }
            }
            base
        };
        format!("{}|{}", main, skills)
    }

    pub fn get_info_csv(&self) -> String {
        let main = format!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            self.name,
            self.team,
            self.name_prop[0],
            self.name_prop[1],
            self.name_prop[2],
            self.name_prop[3],
            self.name_prop[4],
            self.name_prop[5],
            self.name_prop[6],
            self.name_prop[7],
            self.get_property()
        );
        let skills = {
            let mut base = "".to_string();
            let skill_names = [
                "火球", "冰冻", "雷击", "地裂", "吸血", "投毒", "连击", "会心", "瘟疫", "命轮", "狂暴", "魅惑", "加速", "减速",
                "诅咒", "治愈", "苏生", "净化", "铁壁", "蓄力", "聚气", "潜行", "血祭", "分身", "幻术", "防御", "守护", "反弹",
                "护符", "护盾", "反击", "吞噬", "亡灵", "垂死", "隐匿", "啧", "啧", "啧", "啧", "啧",
            ];
            // 后处理
            let mut skills = [0; 40];
            for i in 0..40 {
                if self.skl_freq[i] != 0 {
                    skills[self.skl_id[i] as usize] = self.skl_freq[i];
                }
            }
            for (i, v) in skills.iter().enumerate() {
                if *v > 0 {
                    base.push_str(format!("{}-{},", skill_names[i], v).as_str());
                }
            }
            base
        };
        format!("{},{}", main, skills)
    }

    pub fn get_fullname(&self) -> String {
        format!("{}@{}", self.name, if self.team.is_empty() { &self.name } else { &self.team })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new_test() {
        let namer = Namer::new(&"x@x".to_string());
        assert!(namer.is_some());
        let namer = namer.unwrap();
        assert_eq!(namer.name, "x");
        assert_eq!(namer.team, "x");
    }

    #[test]
    fn val_test() {
        let team = TeamNamer::new_unchecked("x");

        let team_val: Vec<u8> = vec![
            0, 59, 5, 27, 36, 119, 125, 25, 127, 133, 143, 18, 88, 163, 214, 134, 72, 85, 55, 103, 58, 80, 12, 3, 197, 86, 112,
            227, 37, 180, 202, 184, 126, 15, 169, 96, 35, 149, 160, 90, 130, 10, 207, 254, 31, 253, 194, 150, 198, 111, 161, 87,
            128, 129, 24, 73, 238, 34, 245, 168, 52, 42, 232, 142, 152, 135, 201, 132, 200, 144, 203, 138, 154, 141, 109, 47,
            155, 93, 145, 82, 185, 113, 186, 69, 217, 166, 210, 92, 131, 183, 151, 171, 251, 40, 98, 195, 41, 7, 244, 30, 26, 65,
            179, 16, 100, 218, 211, 120, 45, 237, 182, 8, 137, 107, 67, 61, 176, 249, 190, 123, 122, 243, 70, 247, 174, 241, 209,
            148, 228, 50, 208, 14, 51, 146, 117, 68, 162, 234, 102, 104, 62, 11, 76, 84, 9, 139, 224, 235, 108, 44, 78, 222, 242,
            167, 89, 172, 114, 115, 221, 233, 140, 193, 255, 191, 4, 188, 66, 29, 6, 64, 204, 192, 178, 77, 216, 196, 57, 164,
            95, 75, 229, 212, 223, 215, 97, 136, 220, 38, 219, 94, 158, 252, 226, 21, 239, 157, 32, 39, 124, 46, 105, 175, 230,
            156, 248, 106, 116, 43, 101, 250, 231, 28, 147, 173, 177, 187, 20, 181, 205, 22, 91, 83, 225, 1, 81, 99, 199, 246,
            63, 48, 54, 13, 118, 79, 19, 170, 153, 165, 213, 110, 23, 121, 206, 159, 236, 2, 33, 71, 49, 56, 74, 60, 189, 53,
            240, 17,
        ];
        assert_eq!(team.clone_vals().to_vec(), team_val);

        let namer = Namer::new_from_team_namer_unchecked(&team, "x");

        let val_vec: Vec<u8> = vec![
            225, 96, 49, 232, 20, 47, 115, 245, 234, 23, 111, 178, 231, 100, 118, 197, 42, 98, 137, 196, 209, 86, 114, 184, 167,
            129, 164, 239, 205, 211, 82, 173, 189, 153, 198, 67, 4, 3, 90, 52, 128, 134, 176, 145, 85, 9, 250, 30, 63, 247, 240,
            17, 215, 200, 78, 188, 132, 117, 10, 45, 162, 79, 123, 73, 109, 91, 57, 210, 22, 175, 107, 203, 103, 32, 83, 70, 242,
            75, 220, 140, 148, 15, 138, 44, 228, 43, 105, 199, 99, 116, 97, 69, 80, 172, 230, 25, 224, 33, 31, 135, 235, 74, 193,
            238, 233, 88, 216, 204, 24, 163, 141, 6, 201, 26, 38, 21, 186, 237, 101, 206, 212, 76, 144, 219, 149, 169, 202, 110,
            41, 166, 139, 194, 168, 34, 142, 147, 187, 108, 223, 94, 5, 243, 226, 60, 40, 102, 51, 87, 61, 236, 46, 159, 64, 227,
            113, 190, 81, 127, 65, 8, 183, 253, 150, 249, 229, 37, 156, 182, 180, 246, 124, 244, 174, 122, 89, 120, 160, 35, 143,
            11, 14, 151, 133, 27, 177, 251, 221, 207, 58, 29, 131, 119, 171, 157, 93, 185, 48, 112, 192, 191, 66, 106, 39, 59,
            92, 19, 1, 155, 254, 84, 222, 165, 54, 121, 13, 50, 36, 130, 95, 161, 213, 170, 28, 241, 71, 53, 68, 218, 0, 252, 16,
            136, 179, 158, 248, 2, 154, 12, 125, 126, 255, 18, 146, 104, 77, 152, 208, 214, 72, 55, 195, 62, 7, 217, 56, 181,
        ];
        assert_eq!(namer.val.to_vec(), val_vec);
    }

    #[test]
    fn more_val_test() {
        let team = TeamNamer::new_unchecked("y");
        let namer = Namer::new_from_team_namer_unchecked(&team, "y");

        let val_vec: Vec<u8> = vec![
            172, 201, 232, 76, 55, 209, 7, 109, 135, 97, 137, 96, 238, 117, 175, 52, 72, 207, 136, 16, 69, 108, 116, 11, 37, 79,
            169, 46, 244, 57, 8, 67, 156, 21, 48, 43, 162, 185, 191, 141, 40, 65, 233, 211, 54, 42, 208, 28, 87, 251, 167, 53,
            82, 160, 66, 125, 183, 128, 225, 168, 78, 252, 0, 81, 199, 236, 145, 177, 178, 131, 12, 27, 189, 62, 151, 182, 204,
            83, 63, 202, 123, 148, 30, 41, 218, 196, 157, 77, 224, 229, 246, 104, 24, 93, 22, 216, 180, 171, 215, 144, 99, 193,
            176, 17, 158, 44, 213, 103, 29, 70, 18, 120, 2, 231, 61, 50, 51, 74, 106, 247, 105, 166, 133, 205, 152, 235, 248, 71,
            139, 237, 245, 38, 170, 60, 195, 119, 142, 84, 88, 13, 25, 33, 234, 154, 223, 110, 243, 203, 164, 227, 68, 138, 58,
            146, 184, 39, 173, 26, 90, 242, 161, 89, 149, 126, 132, 80, 214, 147, 222, 219, 174, 121, 49, 220, 85, 230, 23, 15,
            111, 32, 102, 75, 35, 98, 253, 56, 186, 9, 198, 118, 91, 1, 150, 112, 187, 190, 254, 10, 197, 212, 31, 113, 181, 153,
            130, 240, 241, 155, 143, 210, 115, 228, 64, 159, 165, 114, 45, 94, 20, 221, 6, 192, 200, 92, 100, 127, 163, 179, 59,
            34, 107, 124, 206, 250, 194, 36, 217, 73, 95, 86, 239, 129, 101, 134, 47, 226, 3, 14, 255, 4, 188, 249, 122, 5, 140,
            19,
        ];
        assert_eq!(namer.val.to_vec(), val_vec);

        let team = TeamNamer::new_unchecked("z");
        let namer = Namer::new_from_team_namer_unchecked(&team, "z");

        let val_vec: Vec<u8> = vec![
            0, 68, 111, 26, 132, 249, 229, 20, 250, 160, 63, 201, 94, 3, 46, 254, 79, 204, 248, 167, 6, 228, 102, 104, 246, 183,
            37, 12, 75, 177, 140, 83, 240, 209, 124, 51, 117, 66, 251, 95, 152, 137, 33, 133, 197, 76, 70, 28, 222, 166, 99, 113,
            27, 87, 82, 161, 34, 121, 212, 141, 143, 244, 35, 169, 30, 59, 31, 47, 13, 89, 236, 86, 192, 92, 237, 231, 8, 148,
            206, 72, 194, 14, 149, 4, 50, 221, 112, 61, 71, 131, 190, 78, 90, 45, 128, 165, 198, 203, 239, 180, 118, 32, 25, 24,
            15, 10, 19, 219, 200, 88, 146, 42, 1, 208, 185, 129, 179, 225, 154, 67, 130, 168, 245, 186, 178, 188, 44, 120, 126,
            242, 110, 247, 49, 252, 211, 163, 193, 144, 241, 7, 91, 145, 199, 189, 115, 73, 226, 127, 184, 157, 138, 175, 40,
            233, 218, 58, 147, 223, 36, 21, 106, 142, 43, 65, 255, 56, 39, 125, 220, 217, 109, 23, 151, 173, 156, 224, 230, 176,
            53, 74, 158, 136, 62, 22, 119, 48, 29, 243, 52, 238, 2, 213, 77, 101, 116, 80, 114, 202, 187, 93, 54, 139, 135, 171,
            196, 84, 108, 64, 174, 38, 16, 55, 96, 234, 182, 103, 195, 207, 235, 69, 122, 97, 100, 214, 18, 232, 205, 191, 162,
            155, 57, 41, 98, 60, 123, 85, 164, 9, 81, 150, 253, 210, 227, 159, 134, 105, 107, 17, 5, 215, 153, 181, 172, 170, 11,
            216,
        ];
        assert_eq!(namer.val.to_vec(), val_vec);
    }

    #[test]
    fn base_name_test() {
        let team = TeamNamer::new_unchecked("x");
        let mut namer = Namer::new_from_team_namer_unchecked(&team, "x");
        let base_name_vec: Vec<u8> = vec![
            53, 0, 40, 4, 58, 61, 37, 46, 56, 51, 21, 20, 27, 17, 15, 26, 13, 30, 52, 63, 36, 30, 57, 34, 22, 37, 35, 6, 12, 25,
            50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let full_base_name_vec: Vec<u8> = vec![
            53, 0, 40, 4, 58, 61, 37, 46, 56, 51, 21, 20, 27, 17, 15, 26, 13, 30, 52, 63, 36, 30, 57, 34, 22, 37, 35, 6, 12, 25,
            50, 49, 59, 23, 49, 27, 51, 58, 39, 28, 60, 20, 31, 36, 41, 11, 7, 29, 24, 24, 61, 62, 57, 4, 28, 48, 55, 50, 38, 29,
            10, 40, 42, 15, 23, 47, 42, 62, 47, 1, 60, 5, 43, 21, 1, 46, 45, 9, 9, 14, 38, 13, 56, 0, 31, 59, 39, 6, 35, 41, 55,
            5, 34, 3, 7, 33, 33, 45, 16, 16, 32, 43, 18, 44, 22, 14, 17, 10, 11, 53, 18, 44, 19, 52, 2, 32, 12, 8, 2, 54, 26, 48,
            8, 3, 63, 54, 19, 25,
        ];
        assert_eq!(namer.name_base.to_vec(), base_name_vec);
        namer.update_skill();
        // update skill 之后才会是完整的 name

        assert_eq!(namer.name_base.to_vec(), full_base_name_vec);
    }

    #[test]
    fn skill_prop_test() {
        let team = TeamNamer::new_unchecked("x");
        let mut namer = Namer::new_from_team_namer_unchecked(&team, "x");

        namer.update_skill();
        let skill_freq_vec: Vec<u8> = vec![
            13, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
        ];
        assert_eq!(namer.skl_freq.to_vec(), skill_freq_vec);
    }

    #[test]
    fn skill_id_test() {
        let team = TeamNamer::new_unchecked("x");
        let mut namer = Namer::new_from_team_namer_unchecked(&team, "x");

        namer.update_skill();
        let skill_id_vec: Vec<u8> = vec![
            9, 13, 12, 38, 4, 27, 26, 15, 16, 32, 24, 5, 7, 21, 18, 10, 37, 2, 6, 20, 39, 1, 14, 3, 11, 29, 22, 33, 19, 0, 30,
            31, 17, 28, 34, 35, 23, 8, 25, 36,
        ];
        assert_eq!(namer.skl_id.to_vec(), skill_id_vec);
    }

    #[test]
    fn prop_test() {
        let team = TeamNamer::new_unchecked("x");
        let namer = Namer::new_from_team_namer_unchecked(&team, "x");

        let prop_vec: Vec<u32> = vec![344, 57, 53, 66, 72, 70, 71, 61];
        assert_eq!(namer.name_prop.to_vec(), prop_vec);
        assert_eq!(namer.get_property(), 564.6667_f32);
    }

    #[test]
    fn shadow_test() {
        let name = Namer::new_unchecked("一一七啺埀㴁?shadow");
        let prop_vec: Vec<u32> = vec![240, 89, 69, 82, 65, 75, 49, 49];

        assert_eq!(name.name_prop.to_vec(), prop_vec);
        assert_eq!(name.get_property(), 558_f32);
    }

    #[test]
    fn update_name_test() {
        // 先创建一个正常的 namer
        // 然后更新名字
        let team = TeamNamer::new_unchecked("shenjack");
        let mut namer = Namer::new_from_team_namer_unchecked(&team, "x");

        let update_name = "一一一丑堀㴁";
        namer.replace_name(&team, update_name);

        let mut none_update_name = Namer::new_from_team_namer_unchecked(&team, update_name);
        none_update_name.update_skill();
        namer.update_skill();

        assert_eq!(namer.name_base.to_vec(), none_update_name.name_base.to_vec());
        assert_eq!(namer.name_prop.to_vec(), none_update_name.name_prop.to_vec());
        assert_eq!(namer.val.to_vec(), none_update_name.val.to_vec());
        assert_eq!(namer.skl_id.to_vec(), none_update_name.skl_id.to_vec());
        assert_eq!(namer.skl_freq.to_vec(), none_update_name.skl_freq.to_vec());
    }
}

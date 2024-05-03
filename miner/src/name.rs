use std::cmp::min;
#[cfg(feature = "simd")]
use std::simd::u8x64;

use tracing::warn;

#[inline(always)]
pub fn median<T>(x: T, y: T, z: T) -> T
where
    T: std::cmp::Ord + std::marker::Copy,
{
    // std::max(std::min(x, y), std::min(std::max(x, y), z))
    // x.max(y).max(x.min(y).min(z))
    x.min(y).max(x.max(y).min(z))
}

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
        let b_name_len = name_len + 1;
        for _ in 0..2 {
            let mut s = 0_u8;
            unsafe {
                val.swap_unchecked(s as usize, 0);
                let mut k = 0;
                for i in 0..256 {
                    s = s.wrapping_add(if k == 0 { 0 } else { *name_bytes.get_unchecked(k - 1) });
                    s = s.wrapping_add(*val.get_unchecked(i));
                    val.swap_unchecked(i, s as usize);
                    k = if k == b_name_len - 1 { 0 } else { k + 1 };
                }
            }
        }
        // simd 优化
        #[cfg(feature = "simd")]
        {
            let mut simd_val = val;
            let mut simd_val_b = [0_u8; 256];
            let simd_181 = u8x64::splat(181);
            let simd_160 = u8x64::splat(160);
            let simd_63 = u8x64::splat(63);

            for i in (0..256).step_by(64) {
                // 一次性加载64个数字
                let mut x = u8x64::from_slice(unsafe { simd_val.get_unchecked(i..) });
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
    pub fn replace_name(&mut self, team_namer: &TeamNamer, name: &str) {
        self.val = team_namer.clone_vals();
        
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len();
        let b_name_len = name_len + 1;
        
        for _ in 0..2 {
            let mut s = 0_u8;
            unsafe {
                self.val.swap_unchecked(s as usize, 0);
                let mut k = 0;
                for i in 0..256 {
                    s = s.wrapping_add(if k == 0 { 0 } else { *name_bytes.get_unchecked(k - 1) });
                    s = s.wrapping_add(*self.val.get_unchecked(i));
                    self.val.swap_unchecked(i, s as usize);
                    k = if k == b_name_len - 1 { 0 } else { k + 1 };
                }
            }
        }
        // simd!
        #[cfg(feature = "simd")]
        {
            let mut simd_val = [0_u8; 256];
            let mut simd_val_b = [0_u8; 256];
            let simd_181 = u8x64::splat(181);
            let simd_160 = u8x64::splat(160);
            let simd_63 = u8x64::splat(63);

            for i in (0..256).step_by(64) {
                unsafe {
                    let mut x = u8x64::from_slice(self.val.get_unchecked(i..i+64));
                    x = x * simd_181 + simd_160;
                    x.copy_to_slice(simd_val.get_unchecked_mut(i..i+64));
                    let y = x & simd_63;
                    y.copy_to_slice(simd_val_b.get_unchecked_mut(i..i+64));   
                }
            }
            let mut mod_count = 0;
            for i in 0..96 {
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
            let mut prop_name = [0_u8; 32];
            prop_name.copy_from_slice(self.name_base.get_unchecked(0..32));
            prop_name.get_unchecked_mut(0..10).sort_unstable();
            *self.name_prop.get_unchecked_mut(0) = 154
                + *prop_name.get_unchecked(3) as u32
                + *prop_name.get_unchecked(4) as u32
                + *prop_name.get_unchecked(5) as u32
                + *prop_name.get_unchecked(6) as u32;
            
            *self.name_prop.get_unchecked_mut(1) = median(
                *prop_name.get_unchecked(10),
                *prop_name.get_unchecked(11),
                *prop_name.get_unchecked(12),
            ) as u32
                + 36;
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
            *self.name_prop.get_unchecked_mut(6) = median(
                *prop_name.get_unchecked(25),
                *prop_name.get_unchecked(26),
                *prop_name.get_unchecked(27),
            ) as u32
                + 36;
            *self.name_prop.get_unchecked_mut(7) = median(
                *prop_name.get_unchecked(28),
                *prop_name.get_unchecked(29),
                *prop_name.get_unchecked(30),
            ) as u32
                + 36;
        }

    }

    #[inline(always)]
    pub fn update_skill(&mut self) {
        let skill_id = self.skl_id.as_mut();
        for i in 0..40 {
            unsafe {
                *skill_id.get_unchecked_mut(i) = i as u8;
            }
        }

        #[cfg(feature = "simd")]
        {
            let mut simd_val = self.val;
            let mut simd_val_b = self.val;
            let simd_181 = u8x64::splat(181);
            let simd_199 = u8x64::splat(199);
            let simd_128 = u8x64::splat(128);
            let simd_53 = u8x64::splat(53);
            let simd_63 = u8x64::splat(63);
            let simd_32 = u8x64::splat(32);

            for i in (0..256).step_by(64) {
                unsafe {
                    let mut x = u8x64::from_slice(simd_val.get_unchecked(i..));
                    let mut y = u8x64::from_slice(simd_val_b.get_unchecked(i..));
                    x = (x * simd_181 + simd_199) & simd_128;
                    y = (y * simd_53) & simd_63 ^ simd_32;
                    x.copy_to_slice(simd_val.get_unchecked_mut(i..));
                    y.copy_to_slice(simd_val_b.get_unchecked_mut(i..));
                }
            }

            let mut mod_count = 0;
            for i in 0..256 {
                unsafe {
                    if simd_val.get_unchecked(i) != &0 {
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
                        (((u as u32) << 8 | t as u32) % 40) as u8
                    };
                    unsafe {
                        s = (s as u16 + rnd as u16 + *skill_id.get_unchecked(i as usize) as u16) as u8 % 40;
                        skill_id.swap_unchecked(i as usize, s as usize);
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
                if p > 10 && skill_id[j] < 35 {
                    self.skl_freq[j] = p - 10;
                    if skill_id[j] < 25 {
                        last = j as i32;
                    };
                } else {
                    self.skl_freq[j] = 0
                }
            }
            if last != -1 {
                self.skl_freq[last as usize] <<= 1;
                // *= 2
            }
            if (self.skl_freq[14] != 0) && (last != 14) {
                self.skl_freq[14] += min(min(self.name_base[60], self.name_base[61]), self.skl_freq[14]);
            }
            if (self.skl_freq[15] != 0) && (last != 15) {
                self.skl_freq[15] += min(min(self.name_base[62], self.name_base[63]), self.skl_freq[15]);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            todo!("none simd 还没写呢")
        }
    }

    #[inline(always)]
    pub fn get_property(&self) -> f32 {
        let sum1 = self.name_prop[1..=7].iter().sum::<u32>();
        let sum2 = self.name_prop[0];
        sum1 as f32 + (sum2 as f32 / 3_f32)
    }

    pub fn get_净化(&self) -> u8 {
        // self.skl_freq[17]
        for (i, v) in self.skl_freq.iter().enumerate() {
            if *v != 0 && self.skl_id[i] == 17 {
                return *v;
            }
        }
        0
    }

    pub fn get_分身(&self) -> u8 {
        // self.skl_freq[23]
        for (i, v) in self.skl_freq.iter().enumerate() {
            if *v != 0 && self.skl_id[i] == 23 {
                return *v;
            }
        }
        0
    }

    pub fn get_幻术(&self) -> u8 {
        // self.skl_freq[24]
        for (i, v) in self.skl_freq.iter().enumerate() {
            if *v != 0 && self.skl_id[i] == 24 {
                return *v;
            }
        }
        0
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
        if self.team.is_empty() {
            self.name.clone()
        } else {
            format!("{}@{}", self.name, self.team)
        }
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
        let skill_prop_vec: Vec<u8> = vec![
            13, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
        ];
        assert_eq!(namer.skl_freq.to_vec(), skill_prop_vec);
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
    }

    #[test]
    fn shadow_test() {
        let name = Namer::new_unchecked("一一七啺埀㴁?shadow");
        let prop_vec: Vec<u32> = vec![240, 89, 69, 82, 65, 75, 49, 49];

        assert_eq!(name.name_prop.to_vec(), prop_vec);
    }

    #[test]
    fn update_name_test() {
        // 先创建一个正常的 namer
        // 然后更新名字
        let team = TeamNamer::new_unchecked("x");
        let mut namer = Namer::new_from_team_namer_unchecked(&team, "x");

        let update_name = "k";
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

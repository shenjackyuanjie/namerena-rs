use tracing::warn;

#[inline(always)]
pub fn median<T>(x: T, y: T, z: T) -> T
where
    T: std::cmp::Ord + std::marker::Copy,
{
    x.max(y).max(x.min(y).min(z))
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
    pub fn clone_vals(&self) -> [u8; 256] {
        self.val.clone()
    }
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

impl Namer {
    /// 最完整的、最简单的 new
    /// 可以直接丢一个 name 进来
    #[inline(always)]
    pub fn new(full_name: &String) -> Option<Self> {
        // name@team
        // name
        let (name, team) = full_name.split_once('@').unwrap_or((full_name, ""));
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
        let (name, team) = full_name.split_once('@').unwrap_or((full_name, ""));
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
        // let mut val: [u8; 256] = (0..=255).collect::<Vec<u8>>().try_into().unwrap();
        let mut name_base = [0_u8; 128];
        let mut name_prop = [0_u32; 8];
        let skl_id = [0_u8; 40];
        let skl_freq = [0_u8; 40];

        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len();
        // 转到 256 长度 的 u8 数组
        let name_bytes = {
            let mut bytes = [0_u8; 256];
            for i in 0..name_len {
                bytes[i + 1] = name_bytes[i];
            }
            bytes
        };
        // // 计算
        // for i in 0..256 {
        //     s += team_bytes[i % (team_len + 1)] as u32 + val[i] as u32;
        //     s %= 256;
        //     let tmp = val[i];
        //     val[i] = val[s as usize];
        //     val[s as usize] = tmp;
        // }
        // s = 0;
        /*
        for (int _ = 0; _ < 2; _++) {
            for (int i = s = 0, j = 0; i < N; i++, j++) {
                s += name[j];
                s += val[i];
                std::swap(val[i], val[s]);
                if (j == len) j = -1;
            }
        } */
        let mut s = 0_u32;
        for _ in 0..2 {
            for j in 0..256 {
                s += name_bytes[j % (name_len + 1)] as u32 + val[j] as u32;
                s %= 256;
                let tmp = val[j];
                val[j] = val[s as usize];
                val[s as usize] = tmp;
            }
            s = 0;
        }
        // for i in 0..256 {
        //     let m = ((val[i] as u32 * 181) + 160) % 256;
        //     if m >= 89 && m < 217 {
        //         name_base[s as usize] = (m & 63) as u8;
        //         s += 1;
        //     }
        // }
        // for _ in 0..2 {
        //     for i in 0..256 {
        //         if i < name_len {
        //             s = s.overflowing_add(name_bytes[i]).0;
        //         }
        //         s = s.overflowing_add(val[i]).0;
        //         val.swap(i, s as usize);
        //     }
        // }
        /*
        #define median(x, y, z) std::max(std::min(x, y), std::min(std::max(x, y), z))
        #define LIM 96
        #define WK(x) val[i + x] = val[i + x] * 181 + 160;
        #define a name_base 
	    for (int i = 0; i < LIM; i += 8) {
	    	WK(0) WK(1) WK(2) WK(3) WK(4) WK(5) WK(6) WK(7)
	    }
	    for (int i = 0; i < LIM && q_len < 30; i++)
	    	if (val[i] >= 89 && val[i] < 217) a[++q_len] = val[i] & 63;

	    if (q_len < 30) {
	    	for (int i = LIM; i < N; i += 8) {
	    		WK(0) WK(1) WK(2) WK(3) WK(4) WK(5) WK(6) WK(7)
	    	}
	    	for (int i = LIM; i < N && q_len < 30; i++)
	    		if (val[i] >= 89 && val[i] < 217) a[++q_len] = val[i] & 63;
	    }*/
        s = 0;
        for i in 0..256 {
            let m = ((val[i] as u32 * 181) + 160) % 256;
            if m >= 89 && m < 217 {
                name_base[s as usize] = (m & 63) as u8;
                s += 1;
            }
        }
        // let mut q_len = 0;
        // for i in (0..256).step_by(8) {
        //     val[i] = ((val[i] as u32 * 181 + 160) % 256) as u8;
        //     val[i+1] = ((val[i+1] as u32 * 181 + 160) % 256) as u8;
        //     val[i+2] = ((val[i+2] as u32 * 181 + 160) % 256) as u8;
        //     val[i+3] = ((val[i+3] as u32 * 181 + 160) % 256) as u8;
        //     val[i+4] = ((val[i+4] as u32 * 181 + 160) % 256) as u8;
        //     val[i+5] = ((val[i+5] as u32 * 181 + 160) % 256) as u8;
        //     val[i+6] = ((val[i+6] as u32 * 181 + 160) % 256) as u8;
        //     val[i+7] = ((val[i+7] as u32 * 181 + 160) % 256) as u8;
        // }
        // for i in 0..256 {
        //     if val[i] >= 89 && val[i] < 217 {
        //         name_base[q_len] = (val[i] & 63) as u8;
        //         q_len += 1;
        //         if q_len >= 30 {
        //             break;
        //         }
        //     }
        // }
        // if q_len < 30 {
        //     for i in (96..256).step_by(8) {
        //         val[i] = ((val[i] as u32 * 181 + 160) % 256) as u8;
        //         val[i+1] = ((val[i+1] as u32 * 181 + 160) % 256) as u8;
        //         val[i+2] = ((val[i+2] as u32 * 181 + 160) % 256) as u8;
        //         val[i+3] = ((val[i+3] as u32 * 181 + 160) % 256) as u8;
        //         val[i+4] = ((val[i+4] as u32 * 181 + 160) % 256) as u8;
        //         val[i+5] = ((val[i+5] as u32 * 181 + 160) % 256) as u8;
        //         val[i+6] = ((val[i+6] as u32 * 181 + 160) % 256) as u8;
        //         val[i+7] = ((val[i+7] as u32 * 181 + 160) % 256) as u8;
        //     }
        //     for i in 96..256 {
        //         if val[i] >= 89 && val[i] < 217 {
        //             name_base[q_len] = (val[i] & 63) as u8;
        //             q_len += 1;
        //             if q_len >= 30 {
        //                 break;
        //             }
        //         }
        //     }
        // }

        // 计算 name_prop
        let mut prop_cnt = 0;
        let mut r = name_base[0..32].to_vec();
        for i in (10..31).step_by(3) {
            r[i..i + 3].sort_unstable();
            let med = median(r[i], r[i + 1], r[i + 2]);
            name_prop[prop_cnt] = med as u32;
            prop_cnt += 1;
        }
        r[0..10].sort_unstable();
        name_prop[prop_cnt] = 154;
        prop_cnt += 1;
        for i in 3..7 {
            name_prop[prop_cnt - 1] += r[i] as u32;
        }
        for i in 0..7 {
            name_prop[i] += 36;
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

    #[inline(always)]
    pub fn get_property(&self) -> f32 {
        // 除 prop[7] 外 加起来  + prop[7] / 3
        let sum1 = self.name_prop[0..7].iter().sum::<u32>();
        let sum2 = self.name_prop[7] as u32;
        sum1 as f32 + (sum2 as f32 / 3_f32)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_new_test() {
        let team = TeamNamer::new_unchecked("x");
        let namer = Namer::new_from_team_namer_unchecked(&team, "x");
        // let namer = Namer::new(&"x@x".to_string());

        // assert!(namer.is_some());

        // let namer = namer.unwrap();
        println!("team val: {:?}", team.val);
        println!("namer: {:?}", crate::show_name(&namer));

        let base_name_vec: Vec<u8> = vec![
            53, 0, 40, 4, 58, 61, 37, 46, 56, 51, 21, 20, 27, 17, 15, 26, 13, 30, 52, 63, 36, 30,
            57, 34, 22, 37, 35, 6, 12, 25, 50, 49, 59, 23, 49, 27, 51, 58, 39, 28, 60, 20, 31, 36,
            41, 11, 7, 29, 24, 24, 61, 62, 57, 4, 28, 48, 55, 50, 38, 29, 10, 40, 42, 15, 23, 47,
            42, 62, 47, 1, 60, 5, 43, 21, 1, 46, 45, 9, 9, 14, 38, 13, 56, 0, 31, 59, 39, 6, 35,
            41, 55, 5, 34, 3, 7, 33, 33, 45, 16, 16, 32, 43, 18, 44, 22, 14, 17, 10, 11, 53, 18,
            44, 19, 52, 2, 32, 12, 8, 2, 54, 26, 48, 8, 3, 63, 54, 19, 25,
        ];
        let val_vec: Vec<u8> = vec![
            225, 96, 49, 232, 20, 47, 115, 245, 234, 23, 111, 178, 231, 100, 118, 197, 42, 98, 137,
            196, 209, 86, 114, 184, 167, 129, 164, 239, 205, 211, 82, 173, 189, 153, 198, 67, 4, 3,
            90, 52, 128, 134, 176, 145, 85, 9, 250, 30, 63, 247, 240, 17, 215, 200, 78, 188, 132,
            117, 10, 45, 162, 79, 123, 73, 109, 91, 57, 210, 22, 175, 107, 203, 103, 32, 83, 70,
            242, 75, 220, 140, 148, 15, 138, 44, 228, 43, 105, 199, 99, 116, 97, 69, 80, 172, 230,
            25, 224, 33, 31, 135, 235, 74, 193, 238, 233, 88, 216, 204, 24, 163, 141, 6, 201, 26,
            38, 21, 186, 237, 101, 206, 212, 76, 144, 219, 149, 169, 202, 110, 41, 166, 139, 194,
            168, 34, 142, 147, 187, 108, 223, 94, 5, 243, 226, 60, 40, 102, 51, 87, 61, 236, 46,
            159, 64, 227, 113, 190, 81, 127, 65, 8, 183, 253, 150, 249, 229, 37, 156, 182, 180,
            246, 124, 244, 174, 122, 89, 120, 160, 35, 143, 11, 14, 151, 133, 27, 177, 251, 221,
            207, 58, 29, 131, 119, 171, 157, 93, 185, 48, 112, 192, 191, 66, 106, 39, 59, 92, 19,
            1, 155, 254, 84, 222, 165, 54, 121, 13, 50, 36, 130, 95, 161, 213, 170, 28, 241, 71,
            53, 68, 218, 0, 252, 16, 136, 179, 158, 248, 2, 154, 12, 125, 126, 255, 18, 146, 104,
            77, 152, 208, 214, 72, 55, 195, 62, 7, 217, 56, 181,
        ];
        let prop_vec: Vec<u32> = vec![57, 53, 66, 72, 70, 71, 61, 344];
        assert_eq!(namer.name, "x");
        assert_eq!(namer.team, "x");
        assert_eq!(namer.val.to_vec(), val_vec);
        assert_eq!(namer.name_prop.to_vec(), prop_vec);
        assert_eq!(namer.name_base.to_vec(), base_name_vec);
    }
}

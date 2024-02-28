use tracing::warn;

pub fn median<T>(x: T, y: T, z: T) -> T
where
    T: std::cmp::Ord + std::marker::Copy,
{
    // std::max(std::min(x, y), std::min(std::max(x, y), z))
    x.max(y).max(x.min(y).min(z))
}

#[derive(Debug, Clone)]
pub struct Namer {
    pub name: String,
    pub team: String,
    pub val: [u8; 256],
    pub name_base: [u8; 128],
    pub name_bytes: [u8; 256],
    pub team_bytes: [u8; 256],
    pub name_prop: [u32; 8],
    pub skl_id: [u8; 40],
    pub skl_freq: [u8; 40],
}

impl Namer {
    pub fn new(raw_name: &String) -> Option<Self> {
        let mut val = [0_u8; 256];
        for i in 0..256 {
            val[i] = i as u8;
        }
        let mut name_base = [0_u8; 128];
        let mut name_prop = [0_u32; 8];
        let mut skl_id = [0_u8; 40];
        let mut skl_freq = [0_u8; 40];

        // name@team
        // name
        let (name, team) = raw_name.split_once('@').unwrap_or((raw_name, ""));
        // len < 256
        if name.len() > 256 {
            warn!("Name too long({}): {}", name.len(), name);
            return None;
        }
        let name_len = name.len();
        if team.len() > 256 {
            warn!("Team too long({}): {}", team.len(), team);
            return None;
        }
        let team_len = team.len();

        let name_bytes = name.as_bytes();
        let team_bytes = team.as_bytes();
        // 转到 256 长度 的 u8 数组
        let name_bytes = {
            let mut bytes = [0_u8; 256];
            for i in 0..name_len {
                bytes[i + 1] = name_bytes[i];
            }
            bytes
        };
        let team_bytes = {
            let mut bytes = [0_u8; 256];
            for i in 0..team_len {
                bytes[i + 1] = team_bytes[i];
            }
            bytes
        };

        // 计算
        let mut s = 0_u32;
        for i in 0..256 {
            s += team_bytes[i % (team_len + 1)] as u32 + val[i] as u32;
            s %= 256;
            let tmp = val[i];
            val[i] = val[s as usize];
            val[s as usize] = tmp;
        }
        for _ in 0..2 {
            s = 0;
            for j in 0..256 {
                s += name_bytes[j % (name_len + 1)] as u32 + val[j] as u32;
                s %= 256;
                let tmp = val[j];
                val[j] = val[s as usize];
                val[s as usize] = tmp;
            }
        }
        s = 0;
        for i in 0..256 {
            let m = ((val[i] as u32 * 181) + 160) % 256;
            if m >= 89 && m < 217 {
                name_base[s as usize] = (m & 63) as u8;
				s +=1;
            }
        }

		// 计算 name_prop
        let mut prop_cnt = 0;
        let mut r = name_base[0..32].to_vec();
        for i in (10..31).step_by(3) {
            name_prop[prop_cnt] = median(r[i], r[i + 1], r[i + 2]) as u32;
            prop_cnt += 1;
        }
		println!("r:{:?}", r);
        r[0..10].sort();
		println!("r:{:?} {}", r, prop_cnt);
        name_prop[prop_cnt] = 154;
        prop_cnt += 1;
		println!("name_prop:{:?}", name_prop);
        for i in 3..7 {
            name_prop[prop_cnt - 1] += r[i] as u32;
        }
		println!("name_prop:{:?}", name_prop);
        for i in 0..7 {
            name_prop[i] += 36;
        }

        Some(Self {
            name: name.to_string(),
            team: team.to_string(),
            val,
            name_base,
            name_bytes,
            team_bytes,
            name_prop,
            skl_id,
            skl_freq,
        })
    }

    pub fn name_len(&self) -> usize {
        self.name.len() + 1
    }

    pub fn team_len(&self) -> usize {
        self.team.len() + 1
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_new_test() {
        let namer = Namer::new(&"x@x".to_string());
        assert!(namer.is_some());
        let namer = namer.unwrap();
        // println!("{:#?}", namer);
        assert_eq!(namer.name, "x");
        assert_eq!(namer.team, "x");
        println!("val: {:?}", namer.val);
        println!("name_base: {:?}", namer.name_base);
        println!("name_bytes: {:?}", namer.name_bytes);
        println!("team_bytes: {:?}", namer.team_bytes);
        println!("name_prop: {:?}", namer.name_prop);
        println!("skl_id: {:?}", namer.skl_id);
        println!("skl_freq: {:?}", namer.skl_freq);
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
        assert_eq!(namer.val.to_vec(), val_vec);
        // 57 53 66 72 70 71 61 344
        let prop_vel: Vec<u32> = vec![57, 53, 66, 72, 70, 71, 61, 344];
        assert_eq!(namer.name_prop.to_vec(), prop_vel);
    }
}

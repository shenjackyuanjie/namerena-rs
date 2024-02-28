use tracing::warn;

use crate::name;


pub fn median(x: u8, y: u8, z: u8) -> u8 {
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
	pub name_prop: [u8; 8],
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
		let mut name_prop = [0_u8; 8];
		let mut skl_id = [0_u8; 40];
		let mut skl_freq = [0_u8; 40];

		// name@team
		// name
		let (name, team) = raw_name.split_once('@').unwrap_or((raw_name, ""));
		// len < 256
		if name.len() > 255 {
			warn!("Name too long({}): {}", name.len(), name);
			return None;
		}
		let name_len = name.len();
		if team.len() > 255 {
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
				bytes[i] = name_bytes[i];
			}
			bytes
		};
		let team_bytes = {
			let mut bytes = [0_u8; 256];
			for i in 0..team_len {
				bytes[i] = team_bytes[i];
			}
			bytes
		};

		let mut s = 0_u32;
		for i in 0..256 {
			s += name_bytes[i] as u32 + val[i] as u32;
			s %= 256;
			let tmp = val[i];
			val[i] = val[s as usize];
			val[s as usize] = tmp;
		}
		for _ in 0..2 {
			s = 0;
			for j in 0..256 {
				s += name_bytes[j % name_len] as u32 + val[j] as u32;
				s %= 256;
				let tmp = val[j];
				val[j] = val[s as usize];
				val[s as usize] = tmp;
			}
		}
		s = 0;
		for i in 0..256 {
			let m = ((val[i] as u32 * 181 + 160) % 256) as u8;
			if m >= 89 && m < 217 {
				name_base[s as usize] = m & 63;
			}
		}

		let mut prop_cnt = 0;
		let mut r = name_base[0..32].to_vec();
		for i in (10..31).step_by(3) {
			r[i..i + 3].sort();
			name_prop[prop_cnt] = r[i + 1];
			prop_cnt += 1;
		}
		r[0..10].sort();
		name_prop[prop_cnt] = 154;
		prop_cnt += 1;
		for i in 3..7 {
			name_prop[prop_cnt - 1] += r[i];
		}
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

	pub fn name_len(&self) -> usize { self.name.len() + 1 }

	pub fn team_len(&self) -> usize { self.team.len() + 1 }
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
	}
}

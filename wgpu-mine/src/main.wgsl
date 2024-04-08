@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

fn collatz_iterations(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        }
        else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }

            n = 3u * n + 1u;
        }
        i = i + 1u;
    }
    return i;
}

// @compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}

@compute
@workgroup_size(1)
// 输入 team_bytes: 队伍名转化成的字节数组
fn team_bytes(team_bytes: array<u32, 256>, team_len: u32) -> array<u32, 256> {
    var val: array<u32, 256> = team_bytes;
    var s: u32 = 0u;
    for (var i: u32 = 0; i < 256; i = i + 1) {
        if (i % team_len != 0) {
            s = s + val[(i % team_len) - 1];
        }
        s = s + val[i];
        // team_bytes.swap(i, s);
        var n = val[i];
        val[i] = val[s];
        val[s] = n;
    }
    return val;
}

/*
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
    pub fn clone_vals(&self) -> [u8; 256] { self.val.clone() }
}
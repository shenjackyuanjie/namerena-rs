
pub const PROFILE_START: u32 = 33554431;

pub mod runners {
    use crate::name::Player;

    pub struct PlayerGroup {
        players: Vec<Player>,
    }

    pub struct Runner {
        /// 应该是一个 Rc4 实例类似物
        seed: u32,
        /// 所有玩家 (包括 boss)
        players: Vec<PlayerGroup>,
    }
}

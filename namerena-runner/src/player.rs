pub struct Player {
    team: String,
    name: String,
    weapon: String,
    player_type: PlayerType,
}

pub const BOSS_NAMES: [&str; 11] = [
    "mario", "sonic", "mosquito", "yuri", "slime", "ikaruga", "conan", "aokiji", "lazy", "covid", "saitama",
];

#[derive(Default)]
pub enum PlayerType {
    #[default]
    Normal,
    /// 种子玩家
    ///
    /// # marker: `seed:`
    Seed,
    /// 被克隆的玩家
    ///
    /// 似乎有个三种?
    Clone,
    /// Boss 玩家
    /// 其实应该是一大堆
    Boss,
    /// 被特殊增强的玩家
    ///
    /// 有一堆
    Boost,
    /// 标准测号用靶子
    ///
    /// # marker: `\u0002`
    Test1,
    /// 没用到的测号用玩家
    ///
    /// # marker: `\u0003`
    Test2,
    /// 比标准测号再强一点的测号用靶子
    ///
    /// # marker: `!`
    TestEx,
}

impl Player {
    pub fn new(team: String, name: String, weapon: String) -> Self {
        let player_type = {
            if name.starts_with("seed:") {
                // 种子
                PlayerType::Seed
            } else {
                match team.as_str() {
                    "!" => {
                        if BOSS_NAMES.contains(&name.as_str()) {
                            PlayerType::Boss
                        } else {
                            // 高强度测号用靶子
                            PlayerType::TestEx
                        }
                    }
                    "\u{0002}" => PlayerType::Test1,
                    "\u{0003}" => PlayerType::Test2,
                    _ => PlayerType::Normal,
                }
            }
        };
        Player {
            team,
            name,
            weapon,
            player_type,
        }
    }

    pub fn update_player(&mut self) {}
}

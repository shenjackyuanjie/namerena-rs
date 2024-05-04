
pub struct Player {
    pub team: String,
    pub name: String,
    pub weapon: String,
    pub player_type: PlayerType,
}

#[derive(Default)]
pub enum PlayerType {
    #[default]
    Normal,
    /// 种子玩家
    /// 
    /// # marker: `seed:`
    Seed,
    /// 被克隆的玩家
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
    pub fn new(team: String, name: String, weapon: String, player_type: Option<PlayerType>) -> Self {
        Player {
            team,
            name,
            weapon,
            player_type: player_type.unwrap_or_default(),
        }
    }
}

pub mod skills;

use std::fmt::Display;

use crate::rc4::RC4;

pub struct PlayerStatus {
    /// 是否被冻结
    frozen: bool,
    /// 是否存活
    alive: bool,
    /// 血量
    hp: u32,
    /// 分数
    point: u32,
}

impl Default for PlayerStatus {
    fn default() -> Self {
        PlayerStatus {
            frozen: false,
            alive: true,
            hp: 0,
            point: 0,
        }
    }
}

impl Display for PlayerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PlayerStatus{{{},{} hp: {}, point: {} }}",
            // 冻结/正常
            // 存活/死亡
            if self.frozen { "冻结" } else { "正常" },
            if self.alive { "存货" } else { "死亡" },
            self.hp,
            self.point
        )
    }
}

pub struct Player {
    /// 队伍
    team: String,
    /// 玩家名
    name: String,
    /// 武器
    weapon: Option<String>,
    /// 玩家类型
    player_type: PlayerType,
    /// skl id
    skil_id: Vec<u32>,
    /// skl prop
    skil_prop: Vec<u32>,
    /// 玩家状态
    ///
    /// 主要是我懒得加一大堆字段
    status: PlayerStatus,
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
    /// 有一堆玩家都被增强了
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
    pub fn new(team: String, name: String, weapon: Option<String>) -> Self {
        let player_type = {
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
                _ => {
                    if name.starts_with("seed:") {
                        PlayerType::Seed
                    } else {
                        PlayerType::Normal
                    }
                }
            }
        };
        Player {
            team,
            name,
            weapon,
            player_type,
            skil_id: vec![],
            skil_prop: vec![],
            status: PlayerStatus::default(),
        }
    }

    /// 直接从一个名竞的原始输入创建一个 Player
    /// 
    /// # 要求
    /// 不许有 \n
    /// 
    /// 可能的输入格式:
    /// - <name>
    /// - <name>@<team>
    /// - <name>+<weapon>
    /// - <name>+<weapon>+diy{xxxxx}
    /// - <name>@<team>+<weapon>
    /// - <name>@<team>+<weapon>+diy{xxxxx}
    pub fn new_from_namerena_raw(raw_name: String) -> Self {
        // 先判断是否有 + 和 @
        if !raw_name.contains("@") && !raw_name.contains("+") {
            return Player::new(raw_name.clone(), raw_name.clone(), None);
        }
        // 区分队伍名
        let name: &str;
        let mut team: &str;
        let weapon: Option<&str>;
        if raw_name.contains("@") {
            (name, team) = raw_name.split_once("@").unwrap();
            // 判定武器
            if team.contains("+") {
                (team, weapon) = team.split_once("+").unwrap();
            } else {
                weapon = None;
            }
        } else {
            // 没有队伍名, 直接是武器
            if team.contains("+") {
                (name, weapon) = raw_name.split_once("+").unwrap();
                team = name;
            } else {
                name = raw_name.as_str();
                team = name;
                weapon = None;
            }
        }
        Player::new(name.to_string(), team.to_string(), weapon.map(|s| s.to_string()))
    }

    pub fn update_player(&mut self) {}

    pub fn step(&mut self, randomer: &mut RC4) {}
}

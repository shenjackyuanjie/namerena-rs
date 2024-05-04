
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
    Seed
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

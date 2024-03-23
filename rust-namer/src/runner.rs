use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use thirtyfour::prelude::*;

const INSERT_JS: &str = include_str!("../insert.js");

#[derive(Debug, Serialize, Deserialize)]
pub struct WinData {
    pub all: Vec<Vec<Vec<String>>>,
    pub winners: Vec<String>,
    pub pic: String, // base64 img
    #[serde(rename = "firstKill")]
    pub first_kill: Option<String>,
}

impl WinData {
    pub fn str_without_pic(&self) -> String {
        format!(
            "Winners: {:?}\nFirst Kill: {:?}\nAll: {:?}",
            self.winners, self.first_kill, self.all
        )
    }
}

#[derive(Debug)]
pub struct TeamRunner {
    pub time_out: Duration,
    /// 每个队伍的成员, 队伍名
    pub teams: Vec<(String, Vec<String>)>,
}

impl TeamRunner {
    // pub fn builder
}

pub struct WebDriverRunner {
    pub driver: WebDriver,
}

impl std::fmt::Display for WebDriverRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Runner")
    }
}

impl WebDriverRunner {
    pub async fn init(target_url: impl AsRef<str>) -> Result<Self> {
        let caps = DesiredCapabilities::edge();
        let driver = WebDriver::new("http://localhost:9515", caps).await?;
        driver.goto(target_url.as_ref()).await?;
        driver.execute(INSERT_JS, vec![]).await?;
        // insert.js
        // 预备环境

        Ok(Self { driver })
    }

    pub async fn raw_flight(&self, teams: String) -> Result<WinData> {
        let done_target = self.driver.find(By::Id("done_target")).await?;
        let go_btn = self.driver.find(By::ClassName("goBtn")).await?;
        let fast_forward_btn = self.driver.find(By::Id("fastBtn")).await?;
        let name_input = self.driver.find(By::Id("input_name")).await?;

        name_input.send_keys(teams).await?;

        go_btn.click().await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await; // 等一会
        fast_forward_btn.click().await?;

        done_target
            .wait_until()
            .has_attribute("done", "true")
            .await?;

        let win_data = self
            .driver
            .execute("return arguments[0].win_data", vec![done_target.to_json()?])
            .await?;
        let win_data: WinData = serde_json::from_value(win_data.json().to_owned())?;
        Ok(win_data)
    }

    pub async fn flight(&self, teams: Vec<Vec<String>>) -> Result<WinData> {
        let done_target = self.driver.find(By::Id("done_target")).await?;
        let go_btn = self.driver.find(By::ClassName("goBtn")).await?;
        let fast_forward_btn = self.driver.find(By::Id("fastBtn")).await?;
        let name_input = self.driver.find(By::Id("input_name")).await?;
        todo!("flight")
    }

    pub async fn quit(self) -> Result<()> {
        self.driver.quit().await?;
        Ok(())
    }
}

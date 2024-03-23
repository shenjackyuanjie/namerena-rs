use anyhow::Result;

mod runner;

#[tokio::main]
async fn main() -> Result<()> {
    let web_runner = runner::WebDriverRunner::init("https://shenjack.top:82/md5").await?;

    let result = web_runner.raw_flight("aaaaaaa\nnnnnn".to_string()).await?;

    println!("{}", result.str_without_pic());

    web_runner.quit().await?;

    Ok(())
}

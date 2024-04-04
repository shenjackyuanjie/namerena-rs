use anyhow::Result;
use clap::Parser;

mod runner;

const ABOUT: &str = "名竞-CLI 版 by shenjack";
const LONG_ABOUT: &str = r#"名竞-CLI 版 by shenjack
基于 msedge webdriver"#;
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug, Clone)]
#[command(version = VERSION, about = ABOUT, long_about = LONG_ABOUT, name = "namerena-cli-webderiver")]
pub struct CliArg {
    #[arg(
        short,
        long = "target-url",
        default_value = "https://shenjack.top:82/md5",
        help = "使用的名竞 URL",
        long_help = "使用的名竞 URL, 可以用于指定到不同的名竞服务端点"
    )]
    pub target_url: String,
    #[arg(
        short,
        long = "input",
        default_value = "aaaaaaa\\nnnnnn",
        help = "输入的队伍名, 以\\n分隔",
        long_help = "输入要对战的队伍名称, 用 \\n 分割, 会自动替换为换行符\n请不要输入任何!test!相关的测号内容, 会导致报错"
    )]
    pub input: String,
    #[arg(short = 'c',
        long = "cli",
        default_value = None,
        help = "是否输出为 CLI 格式",
        long_help = "是否将输出切换为 json 的 CLI 格式, 默认为 false\n切换后可用于cli调用"
    )]
    pub is_cli: bool,
    #[arg(
        short = 'd',
        long = "driver",
        default_value = "http://localhost:9515",
        help = "webdriver 的地址",
        long_help = "使用的 msedge webdriver 地址"
    )]
    pub driver_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArg::parse();

    let replaced_input = args.input.replace("\\n", "\n");

    let web_runner = runner::WebDriverRunner::init(&args).await?;

    let result = web_runner.raw_flight(replaced_input).await?;

    if args.is_cli {
        println!("{}", result.cli_str());
    } else {
        println!("{}", result.str_without_pic());
    }

    web_runner.quit().await?;

    Ok(())
}

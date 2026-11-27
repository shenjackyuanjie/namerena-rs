#![cfg_attr(
    feature = "simd",
    feature(portable_simd, slice_swap_unchecked)
)]

mod cacluate;
mod evaluate;
mod generate;
mod interrupt;
mod name;
#[cfg(feature = "native-pbb")]
mod pbb;
#[cfg(feature = "native-pbb")]
mod pbb_front;
#[cfg(test)]
mod reference;

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use tracing::{info, warn};

use crate::cacluate::CacluateConfig;

#[derive(Parser, Debug, Clone)]
#[command(after_help = "PBB 模式：tswn pbb --help（需要 native-pbb 特性）")]
pub struct Command {
    /// 开始的 id
    #[arg(long, default_value_t = 0)]
    pub start: u64,
    /// 结束的 id
    #[arg(long, default_value_t = u64::MAX)]
    pub end: u64,
    /// 线程数
    #[arg(long, short = 't', default_value_t = 10)]
    pub thread_count: u32,
    /// 八围预期值
    #[arg(long = "prop-expected", short = 'p', default_value_t = 600)]
    pub prop_expect: u32,
    /// xp 预期值
    #[arg(long = "xp-exp", short = 'x', default_value_t = 4900)]
    pub xp_expect: u32,
    /// xd 预测值
    #[arg(long = "xd-exp", short = 'd', default_value_t = 5600)]
    pub xd_expect: u32,
    /// 队伍名称
    #[arg(long)]
    pub team: String,
    /// 如果指定, 则根据线程的实时速度*时间为单位作为 batch 大小
    #[arg(long, short = 'r')]
    pub report_interval: Option<u64>,
    /// 如果指定, 则使用固定的数量作为 batch 大小
    #[arg(long, short = 'b')]
    pub batch_size: Option<u64>,
    /// 单线程模式模式下的核心亲和性核心号 (从 0 开始)
    #[arg(long = "core-pick")]
    pub pick_core: Option<usize>,
    /// 是否为 debug 模式
    #[arg(long)]
    pub debug: bool,
    /// 完整计算和 CSV 序列化基准，结果写入空 sink，不创建结果文件
    #[arg(long)]
    pub bench: bool,
    /// 名字生成算法, 默认为 base 16384
    #[arg(long, short = 'n', value_enum, default_value_t = GenMethods::Base16384)]
    pub name_gen: GenMethods,
    /// 名字模板, 默认为 [], 将使用 生成的内容替换 []
    #[arg(long, short = 'm', default_value_t = String::from("[]"))]
    pub name_template: String,
}

use std::fmt::Display;

/// 名字生成算法
#[derive(Debug, Clone, ValueEnum)]
pub enum GenMethods {
    /// Base 16384
    Base16384,
    /// base 64 std
    Base64Std,
    /// base 64 url safe encode
    Base64Url,
    /// 直接 xxxxx
    Number,
}

impl Display for GenMethods {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenMethods::Base16384 => write!(f, "Base16384"),
            GenMethods::Base64Std => write!(f, "Base64Std"),
            GenMethods::Base64Url => write!(f, "Base64Url"),
            GenMethods::Number => write!(f, "Number"),
        }
    }
}

impl Command {
    pub fn as_cacl_config(&self) -> CacluateConfig {
        CacluateConfig {
            prop_expect: self.prop_expect,
            xp_expect: self.xp_expect,
            xd_expect: self.xd_expect,
            team: self.team.clone(),
            core_affinity: self.pick_core.and_then(|core| 1_usize.checked_shl(core as u32)),
            algorithm: self.name_gen.clone(),
            template: self.name_template.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.thread_count == 0 {
            return Err("线程数必须大于 0".into());
        }
        if self.end < self.start {
            return Err("--end 必须大于或等于 --start（右边界不包含）".into());
        }
        if self.report_interval == Some(0) || self.batch_size == Some(0) {
            return Err("batch 大小和报告间隔必须大于 0".into());
        }
        if !self.batch_in_time() && !self.batch_in_count() {
            return Err("请通过 -r 或 -b 指定 batch".into());
        }
        if !self.name_template.contains("[]") {
            return Err("名字模板必须含有 []".into());
        }
        if self.team.len() > 256 {
            return Err("队伍名不能超过 256 个 UTF-8 字节".into());
        }
        let max_suffix = match self.name_gen {
            GenMethods::Number => 20,
            GenMethods::Base16384 => 18,
            _ => 12,
        };
        let placeholders = self.name_template.matches("[]").count();
        if self.name_template.len().saturating_add(placeholders.saturating_mul(max_suffix - 2)) > 256 {
            return Err("生成的名字可能超过 256 个 UTF-8 字节".into());
        }
        if self.pick_core.is_some_and(|core| core >= usize::BITS as usize) {
            return Err("核心号超出本平台亲和性掩码宽度".into());
        }
        Ok(())
    }

    pub fn batch_size_for(&self, speed: f64) -> u64 {
        if let Some(seconds) = self.report_interval {
            if speed <= 0.0 {
                10_000
            } else {
                (speed as u64).saturating_mul(seconds).max(1)
            }
        } else {
            self.batch_size.unwrap_or(10_000).max(1)
        }
    }

    /// 动态尾段按剩余量均分上限，防止首个预热完成的 worker 独占全部剩余范围。
    pub fn balanced_batch_size(&self, speed: f64, remaining: u64) -> u64 {
        let batch = self.batch_size_for(speed);
        if self.batch_in_time() {
            batch.min(remaining.div_ceil(u64::from(self.thread_count)).max(1))
        } else {
            batch
        }
    }

    pub fn batch_in_time(&self) -> bool { self.report_interval.is_some() }

    pub fn batch_in_count(&self) -> bool { self.batch_size.is_some() }

    pub fn display_info(&self) -> String {
        format!(
            "开始: {} 结尾: {} 总计: {}\n线程数: {}\n八围预期: {}\n虚评阈值: {}|虚单阈值: {}\n队伍名: {}\n{}\n名字生成算法: {}\n名字模板: {}",
            self.start,
            self.end,
            self.end - self.start,
            self.thread_count,
            self.prop_expect,
            self.xp_expect,
            self.xd_expect,
            self.team,
            if !self.batch_in_time() {
                format!("固定 batch 大小: {}", self.batch_size.unwrap())
            } else {
                format!("时间 batch 大小: {}s", self.report_interval.unwrap())
            },
            self.name_gen,
            self.name_template,
        )
    }
}

pub fn set_thread2core(cores: usize) {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};

        let thread_id = GetCurrentThread();
        let core_mask = cores;
        match SetThreadAffinityMask(thread_id, core_mask) {
            0 => warn!("设置线程亲和性 {cores} 失败 {}", std::io::Error::last_os_error()),
            x => info!("设置线程亲和性 {cores} 成功 {}", x),
        }
    }
    #[cfg(unix)]
    {
        warn!("Linux 下不支持设置线程亲和性 (未实现) {}", cores)
    }
}

pub fn set_process_cores(cores: usize) {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessAffinityMask};
        let process = GetCurrentProcess();
        let core_mask = cores;
        match SetProcessAffinityMask(process, core_mask) {
            0 => warn!("设置进程亲和性 {cores} 失败 {}", std::io::Error::last_os_error()),
            x => info!("设置进程亲和性 {cores} 成功 {}", x),
        }
    }
    #[cfg(unix)]
    {
        warn!("Linux 下不支持设置进程亲和性 (未实现) {}", cores)
    }
}

pub fn set_process_priority_to_high() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::HIGH_PRIORITY_CLASS;
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass};
        let process = GetCurrentProcess();
        match SetPriorityClass(process, HIGH_PRIORITY_CLASS) {
            0 => warn!("设置进程优先级为高失败 {}", std::io::Error::last_os_error()),
            _ => info!("设置进程优先级为高成功"),
        }
    }
    #[cfg(unix)]
    {
        warn!("Linux 下不支持设置进程优先级为高 (未实现)")
    }
}

fn main() {
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("pbb")) {
        #[cfg(feature = "native-pbb")]
        std::process::exit(pbb::run());
        #[cfg(not(feature = "native-pbb"))]
        {
            eprintln!("PBB 后端未构建：cargo build -p tswn --features native-pbb --release");
            std::process::exit(2);
        }
    }
    let cli_arg = Command::parse();
    tracing_subscriber::fmt()
        .with_max_level(if cli_arg.debug {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .init();
    if let Err(error) = cli_arg.validate() {
        eprintln!("参数错误: {error}");
        std::process::exit(2);
    }
    if cli_arg.batch_in_count() && cli_arg.batch_in_time() {
        warn!("两个 batch 选项都指定了，将使用时间为准");
    }

    if let Err(error) = interrupt::install() {
        eprintln!("安装安全退出处理器失败: {error}");
        std::process::exit(1);
    }
    if cli_arg.bench {
        match cacluate::benchmark(&cli_arg) {
            Ok(summary) => {
                if summary.interrupted {
                    std::process::exit(130);
                }
            }
            Err(error) => {
                eprintln!("基准失败: {error}");
                std::process::exit(1);
            }
        }
        return;
    }
    let now = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S-%f").to_string();
    // namerena-<team>-<time>.csv
    // <time>: %Y-%m-%d-%H-%M-%S
    let safe_team: String = cli_arg
        .team
        .chars()
        .map(|c| {
            if c.is_control() || r#"<>:"/\|?*"#.contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let output_filename = format!("namerena-{}-{}.csv", safe_team, now);
    let out_path = PathBuf::from(format!("./namerena/{}", output_filename));
    info!("输出文件: {:?}", out_path);
    // 先创建文件夹
    if let Err(e) = std::fs::create_dir_all(out_path.parent().unwrap()) {
        warn!("创建文件夹失败: {}", e);
        std::process::exit(1);
    }

    info!("{}", cli_arg.display_info());
    set_process_priority_to_high();
    match cacluate::start_main(cli_arg, out_path) {
        Ok(summary) => {
            if summary.interrupted {
                std::process::exit(130);
            }
        }
        Err(error) => {
            eprintln!("计算失败: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_has_no_conflicts() { Command::command().debug_assert(); }

    #[test]
    fn parses_basic_mining_command() {
        let args = Command::try_parse_from([
            "tswn",
            "-t",
            "2",
            "-p",
            "550",
            "-x",
            "4000",
            "-d",
            "5000",
            "--team",
            "shenjack",
            "-r",
            "10",
            "-m",
            "我也不知道我要测多少 []",
        ])
        .unwrap();
        assert_eq!(args.thread_count, 2);
        assert_eq!(args.prop_expect, 550);
        assert_eq!(args.xp_expect, 4000);
        assert_eq!(args.xd_expect, 5000);
        assert_eq!(args.team, "shenjack");
        assert_eq!(args.report_interval, Some(10));
        assert_eq!(args.name_template, "我也不知道我要测多少 []");
        assert!(!args.debug);
    }

    #[test]
    fn debug_is_long_only_and_separate_from_xd() {
        let args = Command::try_parse_from(["tswn", "--team", "shenjack", "-d", "5000", "--debug"]).unwrap();
        assert_eq!(args.xd_expect, 5000);
        assert!(args.debug);
    }

    #[test]
    fn validates_bad_ranges_batches_templates_and_affinity() {
        let cli = Command::parse_from(["tswn", "--team", "x", "-b", "10"]);
        assert!(cli.validate().is_ok());
        let mut bad = cli.clone();
        bad.thread_count = 0;
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.start = 10;
        bad.end = 9;
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.batch_size = Some(0);
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.report_interval = Some(0);
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.team = "中".repeat(86);
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.name_template = "a".repeat(250) + "[]";
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.name_template = "没有占位符".into();
        assert!(bad.validate().is_err());
        let mut bad = cli.clone();
        bad.pick_core = Some(usize::BITS as usize);
        assert!(bad.validate().is_err());
        let mut dynamic = cli.clone();
        dynamic.report_interval = Some(u64::MAX);
        assert_eq!(dynamic.batch_size_for(f64::MAX), u64::MAX);
        assert_eq!(dynamic.batch_size_for(0.5), 1);
        assert_eq!(dynamic.batch_size_for(0.0), 10_000);
        dynamic.thread_count = 6;
        assert_eq!(dynamic.balanced_batch_size(1_000_000.0, 600_000), 100_000);
        assert_eq!(dynamic.balanced_batch_size(0.0, 60), 10);
    }
}

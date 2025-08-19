#![feature(portable_simd)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(slice_swap_unchecked)]

mod cacluate;
mod evaluate;
mod generate;
mod name;

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use tracing::{info, warn};

use crate::cacluate::CacluateConfig;

#[derive(Parser, Debug, Clone)]
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
    #[arg(long, short = 'd')]
    pub debug: bool,
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
    pub fn as_cacl_config(&self, path: &PathBuf) -> CacluateConfig {
        if !self.name_template.contains("[]") {
             panic!("Name template must contain a placeholder for the name ([])");
        }
        CacluateConfig {
            thread_id: 0,
            thread_count: self.thread_count,
            prop_expect: self.prop_expect,
            xp_expect: self.xp_expect,
            xd_expect: self.xd_expect,
            team: self.team.clone(),
            time_based: self.batch_in_time(),
            core_affinity: self.pick_core.map(|x| 1 << x),
            out_file: path.clone(),
            algorithm: self.name_gen.clone(),
            template: self.name_template.clone()
        }
    }

    pub fn is_single_thread(&self) -> bool { self.thread_count == 1 }

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
            if self.batch_in_count() {
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
    let mut cli_arg = Command::parse();
    tracing_subscriber::fmt()
        .with_max_level(if cli_arg.debug {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .init();
    // 先验证参数
    // batch 至少要是 size 或者 count 之一
    if !cli_arg.batch_in_count() && !cli_arg.batch_in_time() {
        warn!("必须指定 batch 大小, 请使用 -r 或者 -b 选项");
        return;
    }
    // 如果俩都指定了, 则使用时间为准
    if cli_arg.batch_in_count() && cli_arg.batch_in_time() {
        warn!("两个 batch 选项都指定了, 将使用时间为准");
    }

    // 将数据量处理成可被 thread_count 整除
    let left = cli_arg.start % cli_arg.thread_count as u64;
    cli_arg.end = cli_arg.end.wrapping_add(left);

    let now = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    // namerena-<team>-<time>.csv
    // <time>: %Y-%m-%d-%H-%M-%S
    let output_filename = format!("namerena-{}-{}.csv", cli_arg.team, now);
    let out_path = PathBuf::from(format!("./namerena/{}", output_filename));
    info!("输出文件: {:?}", out_path);
    // 先创建文件夹
    if let Err(e) = std::fs::create_dir_all(out_path.parent().unwrap()) {
        warn!("创建文件夹失败: {}", e);
        return;
    }

    info!("{}", cli_arg.display_info());
    set_process_priority_to_high();
    cacluate::start_main(cli_arg, out_path);
}

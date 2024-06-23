#![feature(portable_simd)]
#![feature(slice_swap_unchecked)]

mod cacluate;
mod evaluate;
mod generate;
mod name;

use std::path::PathBuf;

use clap::Parser;
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
    #[arg(long = "prop-expected", short = 'p', default_value_t = 640)]
    pub prop_expect: u32,
    /// xp 预期值
    #[arg(long = "xp-expected", short = 'x', default_value_t = 0)]
    pub xp_expect: u32,
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
}

impl Command {
    pub fn as_cacl_config(&self, path: &PathBuf) -> CacluateConfig {
        CacluateConfig {
            thread_id: 0,
            thread_count: self.thread_count,
            prop_expect: self.prop_expect,
            xp_expect: self.xp_expect,
            team: self.team.clone(),
            time_based: self.batch_in_time(),
            core_affinity: self.pick_core.map(|x| 1 << x),
            out_file: path.clone(),
        }
    }

    pub fn is_single_thread(&self) -> bool { self.thread_count == 1 }

    pub fn batch_in_time(&self) -> bool { self.report_interval.is_some() }

    pub fn batch_in_count(&self) -> bool { self.batch_size.is_some() }

    pub fn display_info(&self) -> String {
        format!(
            "开始: {} 结尾: {} 总计: {}\n线程数: {}\n八围预期: {}\n强评/强单最低值: {}\n队伍名: {}\n{}",
            self.start,
            self.end,
            self.end - self.start,
            self.thread_count,
            self.prop_expect,
            self.xp_expect,
            self.team,
            if self.batch_in_count() {
                format!("固定 batch 大小: {}", self.batch_size.unwrap())
            } else {
                format!("时间 batch 大小: {}s", self.report_interval.unwrap())
            }
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
        warn!("Linux 下不支持设置线程亲和性 (未实现) {}", core)
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

fn main() {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    let mut cli_arg = Command::parse();
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

    cacluate::start_main(cli_arg, out_path);
}

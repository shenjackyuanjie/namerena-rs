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
    /// qp 预期值
    #[arg(long = "qp-expected", short = 'q', default_value_t = 0)]
    pub qp_expect: u32,
    /// 队伍名称
    #[arg(long)]
    pub team: String,
    /// 预期状态输出时间间隔 (秒)
    #[arg(long, short = 'r', default_value_t = 10)]
    pub report_interval: u64,
    /// 单线程模式模式下的核心亲和性核心号 (从 0 开始)
    #[arg(long = "core-pick", default_value_t = 0)]
    pub pick_core: usize,
    /// 是否是子进程
    #[arg(short = 's')]
    pub is_sub_process: bool,
}

impl Command {
    pub fn as_cacl_config(&self) -> CacluateConfig {
        CacluateConfig {
            start: self.start,
            end: self.end,
            thread_id: 0,
            prop_expect: self.prop_expect,
            qp_expect: self.qp_expect,
            team: self.team.clone(),
            report_interval: self.report_interval,
            core_affinity: if self.thread_count == 1 { Some(1 << self.pick_core) } else { None },
        }
    }

    pub fn is_single_core(&self) -> bool {
        self.thread_count == 1
    }
}

pub fn set_thread2core(core: usize) {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};

        let thread_id = GetCurrentThread();
        let core_mask = core;
        match SetThreadAffinityMask(thread_id, core_mask) {
            0 => warn!("设置线程亲和性失败 {}", std::io::Error::last_os_error()),
            x => info!("设置线程亲和性成功 {}", x),
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
            0 => warn!("设置进程亲和性失败 {}", std::io::Error::last_os_error()),
            x => info!("设置进程亲和性成功 {}", x),
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

    info!("开始: {} 结尾: {}", cli_arg.start, cli_arg.end);
    info!("线程数: {}", cli_arg.thread_count);
    info!("八围预期: {}", cli_arg.prop_expect);
    info!("队伍名: {}", cli_arg.team);
    info!("输出文件名: {:?}", out_path);
    info!("预期状态输出时间间隔: {} 秒", cli_arg.report_interval);

    cacluate::start_main(cli_arg, out_path);

}

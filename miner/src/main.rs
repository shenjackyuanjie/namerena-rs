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
    ///  Windows 下会强制单线程, 且设置线程亲和性为核心 0
    #[arg(long = "bench", default_value_t = false)]
    pub bench: bool,
    /// benchmark 模式下的核心亲和性核心号 (从 0 开始)
    #[arg(long = "bench-core", default_value_t = 0)]
    pub bench_core: usize,
}

impl Command {
    pub fn as_cacl_config(&self) -> CacluateConfig {
        CacluateConfig {
            start: self.start,
            end: self.end,
            thread_count: self.thread_count,
            prop_expect: self.prop_expect,
            qp_expect: self.qp_expect,
            team: self.team.clone(),
            report_interval: self.report_interval,
            core_affinity: if self.bench {
                Some(1 << self.bench_core)
            } else {
                None
            },
        }
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
        use windows_sys::Win32::System::Threading::{SetProcessAffinityMask, GetCurrentProcess};
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

    let mut threads = Vec::with_capacity(cli_arg.thread_count as usize);
    let now = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    // namerena-<team>-<time>.csv
    // <time>: %Y-%m-%d-%H-%M-%S
    let output_filename = format!("namerena-{}-{}.csv", cli_arg.team, now);
    let out_path = PathBuf::from(format!("./namerena/{}", output_filename));
    info!("输出文件: {:?}", out_path);
    // 先创建文件夹
    if let Err(e) = std::fs::create_dir_all(out_path.parent().unwrap()) {
        warn!("创建文件夹失败: {}", e);
    }

    info!("开始: {} 结尾: {}", cli_arg.start, cli_arg.end);
    info!("线程数: {}", cli_arg.thread_count);
    info!("八围预期: {}", cli_arg.prop_expect);
    info!("队伍名: {}", cli_arg.team);
    info!("输出文件名: {:?}", out_path);
    info!("预期状态输出时间间隔: {} 秒", cli_arg.report_interval);
    info!("是否启动 benchmark 模式: {}", cli_arg.bench);

    if cli_arg.bench {
        info!("开始 benchmark");
        let mut config = cli_arg.as_cacl_config();
        config.core_affinity = Some(1 << cli_arg.bench_core);
        set_process_cores(config.core_affinity.unwrap());
        cacluate::cacl(config, 1, &out_path);
    } else {
        let mut n = 0;
        let mut cores = 0;
        for i in 0..cli_arg.thread_count {
            n += 1;
            let mut config = cli_arg.as_cacl_config();
            // 核心亲和性: n, n+1
            config.core_affinity = Some((1 << i) + (1 << (i + 1)));
            cores |= (1 << i) + (1 << (i + 1));
            let out_path = out_path.clone();
            let thread_name = format!("thread_{}", n);
            threads.push(std::thread::spawn(move || {
                info!("线程 {} 开始计算", thread_name);
                cacluate::cacl(config, n, &out_path);
                info!("线程 {} 结束计算", thread_name);
            }));
        }
        set_process_cores(cores);
    }

    for t in threads {
        t.join().unwrap();
    }
}

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


#[allow(non_upper_case_globals)]
const allow_d: u32 = 10;

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
    #[arg(long = "prop-expected", short = 'p', default_value_t = 740)]
    pub prop_expect: u32,
    /// 队伍名称
    #[arg(long)]
    pub team: String,
    /// 预期状态输出时间间隔 (秒)
    #[arg(long, short = 'r', default_value_t = 10)]
    pub report_interval: u64,
}

impl Command {
    pub fn as_cacl_config(&self) -> CacluateConfig {
        CacluateConfig {
            start: self.start,
            end: self.end,
            thread_count: self.thread_count,
            prop_expect: self.prop_expect,
            prop_allow: allow_d,
            team: self.team.clone(),
            report_interval: self.report_interval,
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let mut cli_arg = Command::parse();

    // 将数据量处理成可被 thread_count 整除
    let left = cli_arg.start % cli_arg.thread_count as u64;
    cli_arg.end = cli_arg.end.wrapping_add(left);

    let mut n = 0;
    let mut threads = Vec::with_capacity(cli_arg.thread_count as usize);
    let now = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    // namerena-<team>-<time>.txt
    // <time>: %Y-%m-%d-%H-%M-%S
    let output_filename = format!("namerena-{}-{}.txt", cli_arg.team, now);
    let out_path = PathBuf::from(format!("./namerena/{}", output_filename));
    info!("输出文件: {:?}", out_path);
    // 先创建文件夹
    if let Err(e) = std::fs::create_dir_all(&out_path.parent().unwrap()) {
        warn!("创建文件夹失败: {}", e);
    }

    info!("开始: {} 结尾: {}", cli_arg.start, cli_arg.end);
    info!("线程数: {}", cli_arg.thread_count);
    info!("八围预期: {}", cli_arg.prop_expect);
    info!("队伍名: {}", cli_arg.team);
    info!("输出文件名: {:?}", out_path);

    for i in 0..cli_arg.thread_count {
        n += 1;
        let config = cli_arg.as_cacl_config();
        let out_path = out_path.clone();
        let thread_name = format!("thread_{}", i);
        threads.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacluate::cacl(config, n, &out_path);
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    info!("开始计算");

    for t in threads {
        t.join().unwrap();
    }
}

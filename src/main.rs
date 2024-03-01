#![feature(portable_simd)]

mod name;

use std::{io::Write, path::PathBuf};

use base16384::Base16384Utf8;
use clap::Parser;
use colored::Colorize;
use tracing::{info, warn};

/// 根据 u64 生成对应的 name
/// 转换成 base 16384
/// 禁用:
/// U00 ~ U1F ，换行，制表符 等
/// ? , 问号
/// U2000 - U202F , unicode特殊空格 等
/// 不可以空格开头
#[inline(always)]
pub fn gen_name(id: u64) -> String {
    // u64 -> [u8]
    let id_bytes = id.to_be_bytes();
    Base16384Utf8::encode(id_bytes.as_slice())
}

pub fn show_name(namer: &name::Namer) -> String {
    // var attributeNames = ["HP", "攻", "防", "速", "敏", "魔", "抗", "智"]
    format!(
        "HP|{} 攻|{} 防|{} 速|{} 敏|{} 魔|{} 抗|{} 智|{} 八围:{}",
        namer.name_prop[7],
        namer.name_prop[0],
        namer.name_prop[1],
        namer.name_prop[2],
        namer.name_prop[3],
        namer.name_prop[4],
        namer.name_prop[5],
        namer.name_prop[6],
        namer.get_property()
    )
}

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
    #[arg(long, default_value_t = 10)]
    pub report_interval: u64,
}

/// 大概的预计速度
/// 来自 5600X 的运行效率
pub const GUESS_SPEED: u64 = 623772;

#[inline(always)]
fn cacl(config: Command, id: u64, outfile: &PathBuf) {
    // 初始猜测的时间间隔
    let mut report_interval = config.report_interval * GUESS_SPEED;
    let mut run_speed = GUESS_SPEED as f64;
    let mut start_time = std::time::Instant::now();
    let mut k: u64 = 0;
    // 提前准备好 team_namer
    let team_namer = name::TeamNamer::new_unchecked(&config.team);

    for i in (config.start + id..config.end).step_by(config.thread_count as usize) {
        let name = gen_name(i as u64);
        let namer = name::Namer::new_from_team_namer_unchecked(&team_namer, name.as_str());
        let prop = namer.get_property();

        if (prop + allow_d as f32) > config.prop_expect as f32 {
            let name = gen_name(i as u64);
            let full_name = format!("{}@{}", name, config.team);
            info!("Id:{:>15}|{}|{}", i, full_name, show_name(&namer));
            // 写入 (写到最后一行)
            match std::fs::OpenOptions::new()
                .append(true)
                .open(outfile)
                .and_then(|mut file| file.write(format!("{}\n", full_name).as_bytes()))
            {
                Ok(_) => {}
                Err(e) => {
                    warn!("写入文件<{:?}>失败: {}", outfile, e);
                }
            }
        }
        k += 1;
        if k >= report_interval as u64 {
            let now = std::time::Instant::now();
            let d_t: std::time::Duration = now.duration_since(start_time);
            let new_run_speed = k as f64 / d_t.as_secs_f64();
            // 预估剩余时间
            let wait_time = (config.end - i) / config.thread_count as u64 / new_run_speed as u64;
            let wait_time = chrono::Duration::seconds(wait_time as i64);
            // 转换成 时:分:秒
            // 根据实际运行速率来调整 report_interval
            report_interval = config.report_interval * new_run_speed as u64;
            info!(
                "|{:>2}|Id:{:>15}|{:6.2}/s {:>3.3}E/d {:>5.2} {} 预计:{}:{}:{}|",
                id,
                i,
                new_run_speed,
                new_run_speed * 8.64 / 1_0000.0,
                d_t.as_secs_f64(),
                // 根据对比上一段运行速度 输出 emoji
                // ⬆️ ➡️ ⬇️
                if new_run_speed > run_speed {
                    "⬆️".green()
                } else if new_run_speed < run_speed {
                    // 橙色
                    "⬇️".red()
                } else {
                    "➡️".blue()
                },
                wait_time.num_hours(),
                wait_time.num_minutes() % 60,
                wait_time.num_seconds() % 60
            );
            run_speed = new_run_speed;
            start_time = std::time::Instant::now();
            k = 0;
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
    // 再创建文件
    if let Err(e) = std::fs::File::create(&out_path) {
        warn!("创建文件失败: {}", e);
    }

    info!("开始: {} 结尾: {}", cli_arg.start, cli_arg.end);
    info!("线程数: {}", cli_arg.thread_count);
    info!("八围预期: {}", cli_arg.prop_expect);
    info!("队伍名: {}", cli_arg.team);
    info!("输出文件名: {:?}", out_path);

    for i in 0..cli_arg.thread_count {
        n += 1;
        let cli = cli_arg.clone();
        let out_path = out_path.clone();
        let thread_name = format!("thread_{}", i);
        threads.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacl(cli, n, &out_path);
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    info!("开始计算");

    for t in threads {
        t.join().unwrap();
    }
}

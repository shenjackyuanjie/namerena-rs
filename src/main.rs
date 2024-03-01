#![feature(portable_simd)]

mod name;

use base16384::Base16384Utf8;
use clap::Parser;
use tracing::info;

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

#[allow(non_upper_case_globals)]
const report_interval: u64 = 1_00_0000;

#[derive(Parser, Debug)]
pub struct Command {
    #[arg(long, default_value_t = 0)]
    pub start: u64,
    #[arg(long, default_value_t = u64::MAX)]
    pub end: u64,
    #[arg(long, short = 't', default_value_t = 10)]
    pub thread_count: u32,
    #[arg(long, default_value_t = 750)]
    pub top: u32,
    #[arg(long)]
    pub team: String,
}

#[inline(always)]
fn cacl(start: u64, max: u64, step: usize, top: u32, id: u64, team: &String) {
    let mut start_time = std::time::Instant::now();
    let mut k: u64 = 0;
    let mut top = top;
    let team_namer = name::TeamNamer::new_unchecked(team.as_str());
    for i in (start + id..max).step_by(step) {
        let name = gen_name(i as u64);
        let namer = name::Namer::new_from_team_namer_unchecked(&team_namer, name.as_str());
        let prop = namer.get_property();
        if (prop + allow_d as f32) > top as f32 {
            if prop > top as f32 {
                info!("新的最高属性 {}", prop);
                top = prop as u32;
            }
            let name = gen_name(i as u64);
            let full_name = format!("{}@{}", name, team);
            info!("{:>15}|{}|{}", i, full_name, show_name(&namer));
        }
        k += 1;
        if k >= report_interval as u64 {
            let now = std::time::Instant::now();
            let d_t: std::time::Duration = now.duration_since(start_time);
            let speed = k as f64 / d_t.as_secs_f64();
            info!(
                "count:{:>15} {:>2} {:.2}/s {:.2}E/d",
                i,
                id,
                speed,
                speed * 8.64 / 1_0000.0
            );
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
    let now = std::time::Instant::now();
    // namerena-<team>-<start>-<end>-<time>.txt
    // <time>: %Y-%m-%d-%H-%M-%S
    let output_filename = format!(
        "namerena-{}-{}-{}-{:?}.txt",
        cli_arg.team,
        cli_arg.start,
        cli_arg.end,
        now
    );
    info!("输出文件: {}", output_filename);
    
    for i in 0..cli_arg.thread_count {
        let top = cli_arg.top;
        let max = cli_arg.end;
        let start = cli_arg.start;
        n += 1;
        let thread_name = format!("thread_{}", i);
        let thread_count = cli_arg.thread_count;
        let team = cli_arg.team.clone();
        threads.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacl(
                start,
                max as u64,
                thread_count as usize,
                top as u32,
                n as u64,
                &team,
            );
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    info!("开始计算");

    for t in threads {
        t.join().unwrap();
    }
}

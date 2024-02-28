mod data_struct;
mod name;

use base16384::Base16384Utf8;
use tracing::info;

/// 根据 u64 生成对应的 name
/// 转换成 base 16384
/// 禁用:
/// U00 ~ U1F ，换行，制表符 等
/// ? , 问号
/// U2000 - U202F , unicode特殊空格 等
/// 不可以空格开头
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
const report_interval: u32 = 1_000_0000;

fn cacl(max: u64, step: usize, top: u32, id: u64) {
    let start_time = std::time::Instant::now();
    let mut k: u64 = 0;
    let mut top = top;
    for i in (0+id..max).step_by(step) {
        // let name = gen_name(i as u64);
        let full_name = format!("{}@shenjack", i);
        let namer = name::Namer::new(&full_name);
        if let Some(namer) = namer {
            let prop = namer.get_property();
            if (prop + allow_d as f32) > top as f32 {
                if prop > top as f32 {
                    info!("新的最高属性 {}", prop);
                    top = prop as u32;
                }
                info!("{:>10}|{}|{}", i, full_name, show_name(&namer));
            }
        }
        k += 1;
        if k > report_interval as u64 {
            let now = std::time::Instant::now();
            info!("{} {} {}/s", k, id, k / now.duration_since(start_time).as_secs());
            k = 0;
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    // let team = "shenjack";
    // cli 获取 max
    let max = match std::env::args().nth(1) {
        Some(arg) => arg.parse().unwrap_or(i64::MAX),
        None => i64::MAX,
    };
    // cli 获取线程数量
    let thread_count = match std::env::args().nth(2) {
        Some(arg) => arg.parse().unwrap_or(10),
        None => 10,
    };

    // 新建线程计算
    let top: u32 = 750;
    // 将数据量处理成可被 thread_count 整除
    let left = max % thread_count;
    // max += left;
    
    let mut n = 0;
    let mut threads = Vec::with_capacity(thread_count as usize);
    for i in 0..thread_count {
        let top = top;
        let max = max / thread_count + if i < left { 1 } else { 0 };
        n += 1;
        let thread_name = format!("thread_{}", i);
        threads.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacl(max as u64, thread_count as usize, top as u32, n as u64);
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    info!("开始计算");

    for t in threads {
        t.join().unwrap();
    }
}

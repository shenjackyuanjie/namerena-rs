mod data_struct;
mod name;

use std::sync::{atomic::AtomicU32, Arc};

use base16384::Base16384Utf8;
use tracing::{info, warn};

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
const allow_d: u32 = 100;

fn main() {
    tracing_subscriber::fmt::init();
    let team = "shenjack";
    // cli 获取 max
    let max = match std::env::args().nth(1) {
        Some(arg) => arg.parse().unwrap_or(1_0000_0000),
        None => 1_0000_0000,
    };
    // cli 获取线程数量
    let thread_count = match std::env::args().nth(2) {
        Some(arg) => arg.parse().unwrap_or(10),
        None => 10,
    };

    // 新建线程计算
    let top = AtomicU32::new(100);
    let arc_top = Arc::new(top);
    // 将数据量处理成可被 thread_count 整除
    // 在主线程计算剩余的
    let left = max % thread_count;
    
    let threads = (0..thread_count).map(|i| {
        let top = arc_top.clone();
        let max = max / thread_count + if i < left { 1 } else { 0 };
        std::thread::spawn(move || {
            for i in 0..max+1 {
                let name = gen_name(i as u64);
                let full_name = format!("{}@{}", name, team);
                let namer = name::Namer::new(&full_name);
                if let Some(namer) = namer {
                    let prop = namer.get_property();
                    let tmp_top = top.as_ref();
                    let top = tmp_top.load(std::sync::atomic::Ordering::Relaxed);
                    if (prop - allow_d as f32) > top as f32 {
                        if prop > top as f32 {
                            warn!("新的最高属性 {}", top);
                            tmp_top.store(prop as u32, std::sync::atomic::Ordering::Relaxed);
                        }
                        info!("{}|{}", full_name, show_name(&namer));
                    }
                }
            }
        })
    });
    info!("开始计算");

    for i in max-left..max+1 {
        let name = gen_name(i as u64);
        let full_name = format!("{}@{}", name, team);
        let namer = name::Namer::new(&full_name);
        if let Some(namer) = namer {
            let prop = namer.get_property();
            let tmp_top = arc_top.as_ref();
            let top = tmp_top.load(std::sync::atomic::Ordering::Relaxed);
            if (prop - allow_d as f32) > top as f32 {
                if prop > top as f32 {
                    warn!("新的最高属性 {}", top as f32);
                    tmp_top.store(prop as u32, std::sync::atomic::Ordering::Relaxed);
                }
                info!("{}|{}", full_name, show_name(&namer));
            }
        }
    }

    for t in threads {
        t.join().unwrap();
    }
}

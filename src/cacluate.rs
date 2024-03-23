use crate::{
    evaluate::NamerEvaluater,
    name::{Namer, TeamNamer},
};

use std::{io::Write, path::PathBuf};

use base16384::Base16384Utf8;
use colored::Colorize;
use tracing::{debug, info, warn};

pub fn show_name(namer: &Namer) -> String {
    format!(
        "HP|{} 攻|{} 防|{} 速|{} 敏|{} 魔|{} 抗|{} 智|{} 八围:{}",
        namer.name_prop[0],
        namer.name_prop[1],
        namer.name_prop[2],
        namer.name_prop[3],
        namer.name_prop[4],
        namer.name_prop[5],
        namer.name_prop[6],
        namer.name_prop[7],
        namer.get_property()
    )
}

/// 根据 u64 生成对应的 name
/// 转换成 base 16384
/// 禁用:
/// U00 ~ U1F ，换行，制表符 等
/// ? , 问号
/// U2000 - U202F , unicode特殊空格 等
/// 不可以空格开头
#[inline(always)]
pub fn gen_name(id: u64) -> String {
    let id_bytes = id.to_be_bytes();
    Base16384Utf8::encode(id_bytes.as_slice())
}

pub struct CacluateConfig {
    /// 开始的 id
    pub start: u64,
    /// 结束的 id
    pub end: u64,
    /// 线程数
    pub thread_count: u32,
    /// 八围预期值
    pub prop_expect: u32,
    /// 八围允许范围
    pub prop_allow: u32,
    /// 队伍名称
    pub team: String,
    /// 预期状态输出时间间隔 (秒)
    pub report_interval: u64,
}

#[inline(always)]
pub fn cacl(config: CacluateConfig, id: u64, outfile: &PathBuf) {
    // 初始猜测的时间间隔
    let mut report_interval = 10000; // 第一次猜测测 1w 次, 获取初始数据
    let mut run_speed = 0.0;
    let mut start_time = std::time::Instant::now();
    let mut k: u64 = 0;
    let mut get_count: u32 = 0;
    let xuping = crate::evaluate::xuping::XuPing1_3_1::new(5000.0);
    // 提前准备好 team_namer
    let team_namer = TeamNamer::new(&config.team).unwrap();

    for i in (config.start + id..config.end).step_by(config.thread_count as usize) {
        let name = gen_name(i as u64);
        let mut namer = Namer::new_from_team_namer_unchecked(&team_namer, name.as_str());
        let prop = namer.get_property();

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
                "|{:>2}|Id:{:>15}|{:6.2}/s {:>3.3}E/d {:>5.2}{}|{:<3}|预计:{}:{}:{}|",
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
                get_count,
                wait_time.num_hours(),
                wait_time.num_minutes() % 60,
                wait_time.num_seconds() % 60
            );
            run_speed = new_run_speed;
            start_time = std::time::Instant::now();
            k = 0;
        }

        if (prop + config.prop_allow as f32) > config.prop_expect as f32 {
            let name = gen_name(i as u64);
            let full_name = format!("{}@{}", name, config.team);
            // 虚评
            // if crate::evaluate::xuping::XuPing1_3_1::evaluate(&namer) {
            //     continue;
            // }
            // let xu = crate::evaluate::xuping::XuPing1_3_1::evaluate(&namer);

            // // debug!("Id:{:>15}|{:>5}|{}|{}", i, full_name, xu, show_name(&namer));
            // if xu < 5000.0 {
            //     continue;
            // }

            namer.update_skill();
            let skill_sum: u32 = {
                let mut sum: u32 = 0;
                for i in namer.skl_freq.iter() {
                    sum += *i as u32;
                }
                sum
            };
            if namer.get_净化() < 70 {
                continue;
            }
            if namer.get_幻术() < 20 {
                continue;
            }
            if skill_sum < 150 {
                continue;
            }

            get_count += 1;
            info!("Id:{:>15}|{}|{}", i, full_name, namer.get_info());
            // 写入 (写到最后一行)
            match std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(outfile)
                .and_then(|mut file| file.write(format!("{}\n", full_name).as_bytes()))
            {
                Ok(_) => {}
                Err(e) => {
                    warn!("写入文件<{:?}>失败: {}", outfile, e);
                }
            }
        }
    }
}

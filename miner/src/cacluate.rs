use crate::{
    evaluate::NamerEvaluater,
    name::{Namer, TeamNamer},
    Command,
};

use std::{io::Write, ops::Range, path::PathBuf, time::Instant};

use base16384::Base16384Utf8;
use colored::Colorize;
use crossbeam::channel::{bounded, Receiver, Sender};
use tracing::{info, warn};

/// 根据 u64 生成对应的 name
/// 转换成 base 16384
/// 禁用:
/// U00 ~ U1F ，换行，制表符 等
/// ? , 问号
/// U2000 - U202F , unicode特殊空格 等
/// 不可以空格开头
pub fn gen_name(id: u64) -> String {
    // 手动把 u64 转换成 8 个 u8
    Base16384Utf8::encode(&id.to_be_bytes())
}

pub struct CacluateConfig {
    /// 开始
    pub start: u64,
    /// 结束
    pub end: u64,
    /// 线程 id
    pub thread_id: u32,
    /// 线程数
    pub thread_count: u32,
    /// 八围预期值
    pub prop_expect: u32,
    /// qp 预期值
    pub xp_expect: u32,
    /// 队伍名称
    pub team: String,
    /// 预期状态输出时间间隔 (秒)
    pub report_interval: u64,
    /// 可能的设置指定核心亲和性
    pub core_affinity: Option<usize>,
    /// 输出文件名
    pub out_file: PathBuf,
}

pub type WorkInfo = Option<(u32, u64)>;

/// 用于在先成之间共享的运行状态
/// 正常状态下是会在多个线程之间共享的
/// 单线程状态下就直接在主线程里面
/// 用于记录当前各个线程的计算状态
pub struct ComputeStatus {
    /// 总计算数
    pub start: u64,
    /// 总计算数
    pub end: u64,
    /// 当前各个线程的计算速度
    pub thread_speed: Vec<u64>,
    /// 当前各个线程是否在运算
    pub thread_running: Vec<bool>,
}

impl ComputeStatus {
    pub fn new(config: &Command) -> Self {
        ComputeStatus {
            start: config.start,
            end: config.end,
            thread_speed: vec![0; config.thread_count as usize],
            thread_running: vec![false; config.thread_count as usize],
        }
    }

    pub fn get_first_stoped(&self) -> Option<usize> { self.thread_running.iter().position(|&x| !x) }
}

pub fn start_main(cli_arg: Command, out_path: PathBuf) {
    if cli_arg.is_single_thread() {
        // 单线程运行的时候也是让他放在主线程跑
        let config = cli_arg.as_cacl_config(&out_path);
        crate::set_process_cores(config.core_affinity.unwrap());
        // cacl(config);
        todo!("单线程模式下的调度逻辑没写完呢, 为了保证性能")
    } else {
        schdule_threads(cli_arg, out_path);
    }
}

/// 描述一下思路吧
///
/// 首先, 本地有几个信息
/// sended: 一个数组, 用于记录每个线程是否已经发送了消息
/// shared_status: 一个共享的状态, 用于记录每个线程的状态
/// threads: 一个线程数组, 用于 hold 住线程
///
/// 其中, shared_status 里面记录了每个线程的状态
/// 里面的 thread_running 用于在分发任务的时候判断是否有线程空闲和哪个线程空闲
///
/// 初始化分发的时候的逻辑如下:
/// 1. 初始化一个 0 大小的 bounded channel
///
/// 分发任务(消息)的时候的逻辑如下:
/// - 如果是 固定大小 的 batch
/// 1. 每次直接发送一个 id 为 -1 (即任意线程都可以接收的) 的消息
/// - 如果是 动态大小 的 batch
/// 0. 等待回返的 request work 的消息
/// 1. 遍历 sended 中 true 的部分, 检查对应 thread_running 是否为 true
/// 2. 如果为 true, 则将对应 sended 置为 false
/// 3. 找到 sended 中 第一个为 false 的线程, 根据 thread_speed 计算出一个合适的 batch
/// 4. 发送一个对应线程 id 的消息
///
/// 最后结尾的时候的逻辑如下:
/// - 如果是 固定大小 的 batch
/// 1.每次发送之前检测是不是快完事了 ( batch size > 剩余 work size )
/// 2.如果是, 则发送剩余的 work, 并且把 ended 置为 true
/// 3.ended 为 true 的时候, 再发送消息的时候直接发送 None
/// - 如果是 动态大小 的 batch
pub fn schdule_threads(cli_arg: Command, out_path: PathBuf) {
    if cli_arg.batch_in_time() {
        schdule_count_batch(cli_arg, out_path);
    } else {
        schdule_time_batch(cli_arg, out_path);
    }
}

/// 简单的部分
///
/// 固定大小的 batch 的分发函数
pub fn schdule_count_batch(cli_arg: Command, out_path: PathBuf) {
    let mut n = 0;
    let mut cores = 0;
    let mut threads = vec![];
    let mut shared_status = ComputeStatus::new(&cli_arg);
    let mut sended = vec![false; cli_arg.thread_count as usize];
    let (sender, receiver) = bounded::<Option<Range<u64>>>(0);
    for i in 0..cli_arg.thread_count {
        n += 1;
        let mut config = cli_arg.as_cacl_config(&out_path);
        // 核心亲和性: n
        config.core_affinity = Some(1 << i);
        cores |= 1 << i;
        let thread_name = format!("thread_{}", n);
        threads.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacl(config, &shared_status, receiver.clone());
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    crate::set_process_cores(cores);
    for t in threads {
        t.join().unwrap();
    }
}

/// 麻烦的要死的部分
///
/// 动态大小的 batch 的分发函数
pub fn schdule_time_batch(cli_arg: Command, out_path: PathBuf) {}

/// 所有的状态输出都在子线程, 也就是这里
///
/// 1. 通过 `Receiver` 获取到主线程的数据
/// 获取到数据后, 开始计算
/// 计算完一个 batch 后, 输出一次状态
/// 这里的状态是在所有运算线程中共享的一个状态
/// 每一个线程运算完一个 batch 后, 都会更新这个状态
/// 输出的时候顺带输出其他线程的状态
#[inline(always)]
pub fn cacl(config: CacluateConfig, status: &ComputeStatus, receiver: Receiver<WorkInfo>, work_sender: Sender<u32>) {
    // 初始猜测的时间间隔
    // 设置线程亲和性
    if let Some(core_affinity) = config.core_affinity {
        crate::set_thread2core(1 << core_affinity)
    }

    // 提前准备好 team_namer
    let team_namer = TeamNamer::new(&config.team).unwrap();
    let mut main_namer = Namer::new_from_team_namer_unchecked(&team_namer, "dummy");

    let mut start_time = std::time::Instant::now();

    let mut report_interval = 100000; // 第一次猜测测 10w 次, 获取初始数据
    let mut run_speed = 0.0;
    let mut k: u64 = 0;
    // k += 1;
    // if k >= report_interval {
    //     let now = std::time::Instant::now();
    //     let d_t: std::time::Duration = now.duration_since(start_time);
    //     let new_run_speed = k as f64 / d_t.as_secs_f64();
    //     // 预估剩余时间
    //     let wait_time = (range.end - i) / new_run_speed as u64;
    //     let wait_time = chrono::Duration::seconds(wait_time as i64);
    //     // 转换成 时:分:秒
    //     // 根据实际运行速率来调整 report_interval
    //     report_interval = config.report_interval * new_run_speed as u64;
    //     info!(
    //         "|{:>2}|Id:{:>15}|{:6.2}/s {:>3.3}E/d {:>5.2}{}|{:<3}|预计:{}:{}:{}|",
    //         config.thread_id,
    //         i,
    //         new_run_speed,
    //         new_run_speed * 8.64 / 1_0000.0,
    //         d_t.as_secs_f64(),
    //         // 根据对比上一段运行速度 输出 emoji
    //         // ⬆️ ➡️ ⬇️
    //         // 两个值 相差 0.1 之内都是 ➡️
    //         if new_run_speed > run_speed + 0.1 {
    //             "⬆️".green()
    //         } else if new_run_speed < run_speed - 0.1 {
    //             // 橙色
    //             "⬇️".red()
    //         } else {
    //             "➡️".blue()
    //         },
    //         get_count,
    //         wait_time.num_hours(),
    //         wait_time.num_minutes() % 60,
    //         wait_time.num_seconds() % 60
    //     );
    //     run_speed = new_run_speed;
    //     range_time = std::time::Instant::now();
    //     k = 0;
    // }
}

/// 固定 batch 的计算函数
pub fn count_batch_cacl(config: CacluateConfig, status: &ComputeStatus, receiver: Receiver<Option<u64>>) {}

/// 动态 batch 的计算函数
pub fn time_batch_cacl(
    config: CacluateConfig,
    status: &ComputeStatus,
    receiver: Receiver<Option<u64>>,
    work_sender: Sender<u32>,
) {
}

/// 每一个 batch 的具体运算
/// 不负责状态统计
/// 状态统计的最小颗粒度是整个 batch
pub fn inner_cacl(
    config: &CacluateConfig,
    range: Range<u64>,
    main_namer: &mut Namer,
    team_namer: &TeamNamer,
    get_count: &mut u64,
) {
    for i in range {
        // 这堆操作放在这边了, 保证统计没问题
        let name = gen_name(i);
        // 新加的提前检测
        if !main_namer.replace_name(&team_namer, &name) {
            continue;
        }
        let prop = main_namer.get_property();

        if prop > config.prop_expect as f32 {
            let name = gen_name(i);
            let full_name = format!("{}@{}", name, config.team);
            // 虚评
            main_namer.update_skill();

            let xu = crate::evaluate::xuping::XuPing2_0_1015::evaluate(&main_namer);
            let xu_qd = crate::evaluate::xuping::XuPing2_0_1015_QD::evaluate(&main_namer);

            if xu < config.xp_expect as f64 || xu_qd < config.xp_expect as f64 {
                continue;
            }

            *get_count += 1;
            info!("Id:{:>15}|{}|{:.4}|{:.4}|{}", i, full_name, xu, xu_qd, main_namer.get_info());
            // 写入文件
            let write_in = format!(
                // <full_name>,<id>,<xu>,<xuqd>,<main_namer.get_info()>
                "{},{:>15},{:.4},{:.4},{}\n",
                main_namer.get_fullname(),
                i,
                xu,
                xu_qd,
                main_namer.get_info_csv()
            );

            // 写入 (写到最后一行)
            match std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&config.out_file)
                .and_then(|mut file| file.write(write_in.as_bytes()))
            {
                Ok(_) => {}
                Err(e) => {
                    warn!("写入文件<{:?}>失败: {}", config.out_file, e);
                }
            }
        }
    }
}

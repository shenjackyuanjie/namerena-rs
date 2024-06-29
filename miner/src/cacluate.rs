use crate::{
    evaluate::NamerEvaluater,
    name::{Namer, TeamNamer},
    Command,
};

use std::{
    intrinsics::{likely, unlikely},
    io::Write,
    ops::Range,
    path::PathBuf,
    time::Instant,
};

use base16384::Base16384Utf8;
use colored::Colorize;
use crossbeam::channel::{bounded, Receiver, Sender};
use tracing::{debug, info, warn};

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
    /// 是否基于时间
    pub time_based: bool,
    /// 可能的设置指定核心亲和性
    pub core_affinity: Option<usize>,
    /// 输出文件名
    pub out_file: PathBuf,
}

pub type WorkInfo = Option<(ThreadId, Range<u64>)>;

/// 用于在先成之间共享的运行状态
/// 正常状态下是会在多个线程之间共享的
/// 单线程状态下就直接在主线程里面
/// 用于记录当前各个线程的计算状态
pub struct ComputeStatus {
    /// 总计算数
    pub start: u64,
    /// 总计算数
    pub end: u64,
    /// top
    pub top_id: u64,
    /// 当前各个线程的计算速度
    pub thread_speed: Vec<u64>,
    /// 当前各个线程是否在运算
    pub thread_running: Vec<bool>,
    /// 各个线程筛到了几个
    pub thread_get_count: Vec<u64>,
}

impl ComputeStatus {
    pub fn new(config: &Command) -> Self {
        ComputeStatus {
            start: config.start,
            end: config.end,
            top_id: config.start,
            thread_speed: vec![0; config.thread_count as usize],
            thread_running: vec![false; config.thread_count as usize],
            thread_get_count: vec![0; config.thread_count as usize],
        }
    }

    pub fn get_idle_thread(&self) -> Option<usize> { self.thread_running.iter().position(|&x| !x) }
    pub fn all_stoped(&self) -> bool { self.thread_running.iter().all(|&x| !x) }
    pub fn update_speed(&mut self, thread_id: ThreadId, speed: u64) { self.thread_speed[thread_id as usize] = speed; }
    pub fn update_running(&mut self, thread_id: ThreadId, running: bool) { self.thread_running[thread_id as usize] = running; }
    pub fn add_get_count(&mut self, thread_id: ThreadId, count: u64) { self.thread_get_count[thread_id as usize] += count; }
    pub fn count_speed(&self) -> u64 { self.thread_speed.iter().sum() }
    pub fn get_sum_count(&self) -> u64 { self.thread_get_count.iter().sum() }
    pub fn predict_time(&self) -> chrono::Duration {
        let speed = self.count_speed();
        let remain = self.end - self.top_id;
        chrono::Duration::milliseconds((remain as f64 / speed as f64 * 1000.0) as i64)
    }
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

pub type ThreadId = u32;

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
/// 1. 每次发送之前检测是不是快完事了 ( batch size > 剩余 work size )
/// 2. 如果是, 则发送剩余的 work, 并且把 ended 置为 true
/// 3. ended 为 true 的时候, 再发送消息的时候直接发送 None
/// - 如果是 动态大小 的 batch
pub fn schdule_threads(cli_arg: Command, out_path: PathBuf) {
    let mut cores = 0;
    let mut thread = vec![];
    let mut shared_status = ComputeStatus::new(&cli_arg);
    let (work_sender, work_receiver) = bounded::<WorkInfo>(0);
    let (work_requester, thread_waiter) = bounded::<(ThreadId, u32)>(0);
    for i in 0..cli_arg.thread_count {
        // 每一个线程
        let mut config = cli_arg.as_cacl_config(&out_path);
        config.thread_id = i;
        let work_receiver = work_receiver.clone();
        let work_requester = work_requester.clone();
        let shared_status: &mut ComputeStatus = unsafe {
            // 直接获取一个共享状态的引用
            // 这个对象是可以在多个线程之间共享的
            // 但是 rust 不让我这么干
            // 所以, unsafe == trust_me
            std::mem::transmute::<&mut ComputeStatus, &mut ComputeStatus>(&mut shared_status)
        };
        config.core_affinity = Some(1 << i);
        cores |= 1 << i;
        let thread_name = format!("thread_{}", i);
        thread.push(std::thread::spawn(move || {
            info!("线程 {} 开始计算", thread_name);
            cacl(config, shared_status, work_receiver, work_requester);
            info!("线程 {} 结束计算", thread_name);
        }));
    }
    crate::set_process_cores(cores);
    // 任务分发
    // 判断是否所有 work 都分发完了
    // 当前分发到的 work 的 最大 index
    let full_start_time = Instant::now();
    if cli_arg.batch_in_time() {
        info!("开始分发任务(动态 batch)");
        let mut sended = vec![false; cli_arg.thread_count as usize];
        loop {
            // 等待一个 request work
            // 大部分时间在这里等待
            let latest_speed = match thread_waiter.recv() {
                Ok(info) => info,
                Err(_) => {
                    // 如果接收到了错误, 则说明所有线程都结束了, 或者怎么着出毛病了
                    // 退出
                    break;
                }
            };
            // 遍历 sended 中 true 的部分, 检查对应 thread_running 是否为 true
            for (i, sended) in sended.iter_mut().enumerate() {
                if *sended {
                    if shared_status.thread_running[i] {
                        // 如果为 true, 则将对应 sended 置为 false
                        *sended = false;
                    }
                }
            }
            // 根据 latest_speed 计算出一个合适的 batch
            if latest_speed.1 == 0 {
                // 如果速度为 0, 则说明刚刚开始
                // 直接发送一个 1w 的 batch
                let _ = work_sender.send(Some((latest_speed.0, shared_status.top_id..shared_status.top_id + 10000)));
                shared_status.top_id += 10000;
            } else {
                // 计算出一个合适的 batch
                let batch = latest_speed.1 as u64 * cli_arg.report_interval.unwrap();
                // 判断是否快完事了
                // 如果是, 则发送剩余的 work, 然后直接发送 None
                if shared_status.top_id + batch > cli_arg.end {
                    let _ = work_sender.send(Some((latest_speed.0, shared_status.top_id..cli_arg.end)));
                    info!("最后一个 batch({}..{}) 已发送", shared_status.top_id, cli_arg.end);
                    loop {
                        let _ = thread_waiter.try_recv();
                        let _ = work_sender.try_send(None);
                        if thread.iter().all(|t| t.is_finished()) {
                            break;
                        }
                    }
                    break;
                } else {
                    // 如果不是, 则发送一个对应线程 id 的消息
                    let _ = work_sender.send(Some((latest_speed.0, shared_status.top_id..shared_status.top_id + batch)));
                    shared_status.top_id += batch;
                }
            }
        }
    } else {
        info!("开始分发任务(固定 batch)");
        loop {
            // 等待一个 request work
            // 大部分时间在这里等待
            if thread_waiter.recv().is_err() {
                // 如果接收到了错误, 则说明所有线程都结束了
                // 退出
                break;
            }
            // work 没分发完
            // 获取第一个空闲的线程
            // 这里不确定是不是会有问题, 先用 unwarp 看看
            let thread_id = shared_status.get_idle_thread().unwrap();
            // 先检测是否快结束了
            if shared_status.top_id + cli_arg.batch_size.unwrap() >= cli_arg.end {
                // 如果快结束了, 则发送剩余的 work 然后发送 None
                let _ = work_sender.send(Some((thread_id as u32, shared_status.top_id..cli_arg.end)));
                info!("最后一个 batch({}..{}) 已发送", shared_status.top_id, cli_arg.end);
                loop {
                    let _ = thread_waiter.try_recv();
                    let _ = work_sender.try_send(None);
                    if thread.iter().all(|t| t.is_finished()) {
                        break;
                    }
                }
                break;
            } else {
                // 如果没有结束, 则发送一个 batch
                let _ = work_sender.send(Some((
                    thread_id as u32,
                    shared_status.top_id..shared_status.top_id + cli_arg.batch_size.unwrap(),
                )));
            }
            // 更新 top_i
            shared_status.top_id += cli_arg.batch_size.unwrap();
        }
    }
    let full_end_time = Instant::now();
    info!("所有任务已完成, 耗时: {:?}", full_end_time - full_start_time);
    info!("各个线程获取数量: {:?}", shared_status.thread_get_count);
    if shared_status.get_sum_count() != 0 {
        info!(
            "总计获取: {}, 效率: {}/s",
            shared_status.get_sum_count(),
            (shared_status.get_sum_count() as f64) / (full_end_time - full_start_time).as_secs_f64()
        );
    } else {
        info!("这真是太悲伤了呢, 干了这么久的活, 一条数据都没获取到");
    }
    info!("输出文件名: {:?}", out_path);
}

/// 所有的状态输出都在子线程, 也就是这里
///
/// 1. 通过 `Receiver` 获取到主线程的数据
/// 
/// 获取到数据后, 开始计算
/// 计算完一个 batch 后, 输出一次状态
/// 这里的状态是在所有运算线程中共享的一个状态
/// 每一个线程运算完一个 batch 后, 都会更新这个状态
/// 输出的时候顺带输出其他线程的状态
pub fn cacl(
    config: CacluateConfig,
    status: &mut ComputeStatus,
    receiver: Receiver<WorkInfo>,
    work_sender: Sender<(ThreadId, u32)>,
) {
    if let Some(core_affinity) = config.core_affinity {
        crate::set_thread2core(core_affinity);
    }
    // 提前准备好 team_namer
    let team_namer = TeamNamer::new(&config.team).unwrap();
    let mut main_namer = Namer::new_from_team_namer_unchecked(&team_namer, "dummy");
    let mut get_count = 0;
    let mut run_speed = 0.0;
    // 开始之前, 先发送一个 request
    let _ = work_sender.send((config.thread_id, 0));
    loop {
        // 先 request 一个 work
        let work = match receiver.recv() {
            Ok(work) => match work {
                Some(work) => {
                    if work.0 == config.thread_id {
                        work.1
                    } else {
                        // 如果不是自己的 work, 则再次发送一个 request
                        let _ = work_sender.send((config.thread_id, if run_speed == 0.0 { 0 } else { run_speed as u32 }));
                        // 然后进入下一次循环
                        continue;
                    }
                }
                None => {
                    // 如果接收到了 None, 则说明活都干完了, 退出
                    return;
                }
            },
            Err(_) => {
                // 如果接收到了错误, 则说明主线程已经结束了, 退出
                return;
            }
        };
        // 开始计算
        let count = work.end - work.start;
        let top = work.end;
        let start_time = std::time::Instant::now();
        // 计算
        let new_get = inner_cacl(&config, work, &mut main_namer, &team_namer);
        get_count += new_get;
        status.add_get_count(config.thread_id, new_get);
        // 完事, 统计
        let now = std::time::Instant::now();
        let d_t: std::time::Duration = now.duration_since(start_time);
        let new_run_speed = count as f64 / d_t.as_secs_f64();
        // 预估剩余时间
        // 先更新自己的状态上去
        status.update_speed(config.thread_id, new_run_speed as u64);
        // 获取一个全局速度预测
        let predict_time = status.predict_time();
        debug!("{:?}", status.thread_speed);
        // 输出状态
        info!(
            // thread_id, top, 当前线程速度, 当前batch用时, emoji, 全局速度, 全局E/d 速度, 算到几个, 进度, 预计时间
            "|{:>2}|Id:{:>15}|{:6.2}E/d {:>5.2}s{}|{:>4.3}E/d|{:<3}|{:>2.3}% {}:{}:{:>2}|",
            config.thread_id,
            top,
            new_run_speed * 8.64 / 1_0000.0,
            d_t.as_secs_f64(),
            // 如果速度差 1k 以上, 则输出emoji
            if new_run_speed > run_speed + 1000.0 {
                "↑".green()
            } else if new_run_speed < run_speed - 1000.0 {
                "↓".red()
            } else {
                "→".blue()
            },
            status.count_speed() as f64 * 8.64 / 1_0000.0,
            get_count,
            (status.top_id - status.start) as f64 / (status.end - status.start) as f64 * 100.0,
            predict_time.num_hours(),
            predict_time.num_minutes() % 60,
            predict_time.num_seconds() % 60
        );
        run_speed = new_run_speed;
        // 然后是调度相关
        status.update_running(config.thread_id, false);
        // 请求一个新的 work
        let _ = work_sender.send((config.thread_id, if run_speed == 0.0 { 0 } else { run_speed as u32 }));
    }
}

/// 每一个 batch 的具体运算
/// 不负责状态统计
/// 状态统计的最小颗粒度是整个 batch
pub fn inner_cacl(config: &CacluateConfig, range: Range<u64>, main_namer: &mut Namer, team_namer: &TeamNamer) -> u64 {
    let mut get_count = 0;
    for i in range {
        // 这堆操作放在这边了, 保证统计没问题
        let name = gen_name(i);
        // 新加的提前检测
        if likely(!main_namer.replace_name(team_namer, &name)) {
            continue;
        }
        let prop = main_namer.get_property();

        if unlikely(prop > config.prop_expect as f32) {
            let name = gen_name(i);
            let full_name = format!("{}@{}", name, config.team);
            // 虚评
            main_namer.update_skill();

            let xu;
            let xu_qd = crate::evaluate::xuping::XuPing2_0_1015_QD::evaluate(main_namer);
            if likely((xu_qd as u32) < config.xp_expect) {
                xu = crate::evaluate::xuping::XuPing2_0_1015::evaluate(main_namer);
                if likely((xu as u32) < config.xp_expect) {
                    continue;
                }
            } else {
                xu = crate::evaluate::xuping::XuPing2_0_1015::evaluate(main_namer);
            }
            get_count += 1;
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
    get_count
}

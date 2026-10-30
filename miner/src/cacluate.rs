use crate::{
    Command, GenMethods,
    evaluate::{
        NamerEvaluater,
        xuping::{XuPing2_0_1015, XuPing2_0_1015_QD},
    },
    name::{Namer, TeamNamer},
};
use crossbeam::channel::{Receiver, Sender, bounded};
use std::{
    io::{self, BufWriter, Write},
    ops::Range,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tracing::info;

#[derive(Clone)]
pub struct CacluateConfig {
    pub prop_expect: u32,
    pub xp_expect: u32,
    pub xd_expect: u32,
    pub team: String,
    pub core_affinity: Option<usize>,
    pub algorithm: GenMethods,
    pub template: String,
}

#[derive(Debug, Default)]
pub struct RunSummary {
    pub computed: u64,
    pub hits: u64,
    pub elapsed: Duration,
}

/// 只有调度线程拥有游标。先求剩余量，绝不执行可能溢出的 top + batch。
struct WorkCursor {
    next: u64,
    end: u64,
}
impl WorkCursor {
    fn take(&mut self, batch: u64) -> Option<Range<u64>> {
        if self.next >= self.end {
            return None;
        }
        let end = self.next + batch.max(1).min(self.end - self.next);
        let range = self.next..end;
        self.next = end;
        Some(range)
    }
}

enum Event {
    Hit(String),
    Done {
        worker: usize,
        computed: u64,
        elapsed: Duration,
    },
}

pub fn start_main(cli: Command, out_path: PathBuf) -> io::Result<RunSummary> {
    cli.validate().map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    // 文件在 worker 启动之前创建；零命中和空范围也有完整表头。绝不覆盖已有结果。
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&out_path)?;
    let mut writer = BufWriter::new(file);
    let summary = schedule(&cli, &mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    info!(
        "实际计算: {}，命中: {}，耗时: {:.6}s，速度: {:.2}/s，输出: {:?}",
        summary.computed,
        summary.hits,
        summary.elapsed.as_secs_f64(),
        summary.computed as f64 / summary.elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        out_path
    );
    Ok(summary)
}

/// 每个 worker 一个私有任务通道；状态和输出仅由主线程修改。
/// 写入错误时取消后续计算，但继续接收事件并 join 所有 worker，避免有界通道死锁。
fn schedule(cli: &Command, writer: &mut impl Write) -> io::Result<RunSummary> {
    writeln!(writer, "{}", csv_header())?;
    writer.flush()?;
    let start = Instant::now();
    let cancelled = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = bounded((cli.thread_count as usize).saturating_mul(2).max(1));
    let mut cursor = WorkCursor {
        next: cli.start,
        end: cli.end,
    };
    let mut work_senders = Vec::new();
    let mut handles = Vec::new();
    let mut failure = None;
    for worker in 0..cli.thread_count as usize {
        let (tx, rx) = bounded(1);
        let mut config = cli.as_cacl_config();
        // 保留用户的可选亲和性语义；不指定时由操作系统调度，不强制绑定相邻 SMT。
        config.core_affinity = cli.pick_core.and_then(|core| 1_usize.checked_shl(core as u32));
        let events = events_tx.clone();
        let cancel = cancelled.clone();
        match std::thread::Builder::new().name(format!("miner-{worker}")).spawn(move || {
            worker_loop(worker, config, rx, events, cancel);
        }) {
            Ok(handle) => {
                handles.push(handle);
                if let Some(range) = cursor.take(cli.batch_size_for(0.0)) {
                    tx.send(range).map_err(|_| io::Error::other("worker 启动失败"))?;
                    work_senders.push(Some(tx));
                } else {
                    work_senders.push(None);
                }
            }
            Err(error) => {
                failure = Some(error);
                cancelled.store(true, Ordering::Relaxed);
                work_senders.clear();
                break;
            }
        }
    }
    drop(events_tx);
    let mut summary = RunSummary::default();
    for event in events_rx {
        match event {
            Event::Hit(row) => {
                if failure.is_none() {
                    match writer.write_all(row.as_bytes()) {
                        Ok(()) => summary.hits += 1,
                        Err(error) => {
                            failure = Some(error);
                        }
                    }
                }
            }
            Event::Done {
                worker,
                computed,
                elapsed,
            } => {
                summary.computed += computed;
                let speed = computed as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
                info!(
                    "worker {worker}: 已完成 {}，本批 {:.2}/s，累计命中 {}",
                    summary.computed, speed, summary.hits
                );
                if failure.is_none() {
                    if let Err(error) = writer.flush() {
                        failure = Some(error);
                    }
                }
                if failure.is_none() {
                    if let Some(range) = cursor.take(cli.batch_size_for(speed)) {
                        if work_senders[worker].as_ref().unwrap().send(range).is_err() {
                            failure = Some(io::Error::other("worker 意外退出"));
                        }
                    } else {
                        work_senders[worker] = None;
                    }
                }
            }
        }
        if failure.is_some() {
            cancelled.store(true, Ordering::Relaxed);
            work_senders.clear();
        }
    }
    drop(work_senders);
    for handle in handles {
        if handle.join().is_err() && failure.is_none() {
            failure = Some(io::Error::other("worker panic"));
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    writer.flush()?;
    summary.elapsed = start.elapsed();
    if summary.computed != cli.end - cli.start {
        return Err(io::Error::other("实际计算数量与请求范围不一致"));
    }
    Ok(summary)
}

fn worker_loop(
    worker: usize,
    config: CacluateConfig,
    work: Receiver<Range<u64>>,
    events: Sender<Event>,
    cancelled: Arc<AtomicBool>,
) {
    if let Some(affinity) = config.core_affinity {
        crate::set_thread2core(affinity);
    }
    let team = TeamNamer::new_unchecked(&config.team);
    let mut namer = Namer::new_from_team_namer_unchecked(&team, "");
    for range in work {
        let start = Instant::now();
        let mut computed = 0;
        for id in range {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            let name = config.algorithm.gen_name(&config.template, id);
            namer.replace_name(&team, &name);
            computed += 1;
            if !property_matches(&namer.name_prop, config.prop_expect) {
                continue;
            }
            namer.update_skill();
            let xd = XuPing2_0_1015_QD::evaluate(&namer);
            if (xd as u32) < config.xd_expect {
                continue;
            }
            let xp = XuPing2_0_1015::evaluate(&namer);
            if (xp as u32) < config.xp_expect {
                continue;
            }
            if events.send(Event::Hit(csv_row(id, &namer, xp, xd))).is_err() {
                return;
            }
        }
        if events
            .send(Event::Done {
                worker,
                computed,
                elapsed: start.elapsed(),
            })
            .is_err()
        {
            return;
        }
    }
}

/// 属性阈值严格 >，整数三倍值避免 f32 舍入，也适用于 u32::MAX 阈值。
fn property_matches(props: &[u32; 8], threshold: u32) -> bool {
    u64::from(props[0]) + 3 * props[1..].iter().map(|&p| u64::from(p)).sum::<u64>() > 3 * u64::from(threshold)
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn csv_header() -> String {
    let mut header =
        String::from("full_name,id,xp,xd,name,team,hp,attack,defense,speed,agility,magic,resistance,wisdom,property");
    for skill in 0..40 {
        header.push_str(&format!(",skill_{skill}"));
    }
    header
}

fn csv_row(id: u64, name: &Namer, xp: f64, xd: f64) -> String {
    let mut row = format!(
        "{},{id},{xp:.4},{xd:.4},{},{}",
        csv_escape(&name.get_fullname()),
        csv_escape(&name.name),
        csv_escape(&name.team)
    );
    for prop in name.name_prop {
        row.push_str(&format!(",{prop}"));
    }
    row.push_str(&format!(",{}", name.get_property()));
    let mut skills = [0; 40];
    for (&id, &freq) in name.skl_id.iter().zip(&name.skl_freq) {
        skills[id as usize] = freq;
    }
    for freq in skills {
        row.push_str(&format!(",{freq}"));
    }
    row.push('\n');
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn command(threads: u32, start: u64, end: u64, dynamic: bool) -> Command {
        let mut cli = Command::parse_from([
            "tswn", "--team", "shenjack", "-b", "7", "-n", "number", "-p", "0", "-x", "0", "-d", "0",
        ]);
        cli.thread_count = threads;
        cli.start = start;
        cli.end = end;
        if dynamic {
            cli.report_interval = Some(10);
        }
        cli
    }

    #[test]
    fn cursor_clamps_and_never_overflows() {
        let mut cursor = WorkCursor {
            next: u64::MAX - 3,
            end: u64::MAX,
        };
        assert_eq!(cursor.take(u64::MAX), Some(u64::MAX - 3..u64::MAX));
        assert_eq!(cursor.take(1), None);
        let mut cursor = WorkCursor { next: 2, end: 4 };
        assert_eq!(cursor.take(0), Some(2..3));
        assert_eq!(cursor.take(9), Some(3..4));
    }

    #[test]
    fn all_worker_modes_visit_each_id_once() {
        for threads in [1, 2, 6] {
            for dynamic in [false, true] {
                for (start, end) in [(0, 0), (5, 8), (17, 70), (u64::MAX - 9, u64::MAX)] {
                    let cli = command(threads, start, end, dynamic);
                    let mut csv = Vec::new();
                    let summary = schedule(&cli, &mut csv).unwrap();
                    assert_eq!(summary.computed, end - start);
                    assert_eq!(summary.hits, end - start);
                    let csv = String::from_utf8(csv).unwrap();
                    let mut ids: Vec<u64> =
                        csv.lines().skip(1).map(|line| line.split(',').nth(1).unwrap().parse().unwrap()).collect();
                    ids.sort_unstable();
                    assert_eq!(ids, (start..end).collect::<Vec<_>>());
                    assert!(csv.lines().all(|line| line.split(',').count() == 55));
                }
            }
        }
    }

    #[test]
    fn strict_property_threshold_and_csv_quoting() {
        assert!(!property_matches(&[300, 0, 0, 0, 0, 0, 0, 0], 100));
        assert!(property_matches(&[301, 0, 0, 0, 0, 0, 0, 0], 100));
        assert!(!property_matches(&[406, 99, 99, 99, 99, 99, 99, 99], u32::MAX));
        assert_eq!(
            csv_escape(
                "a,\"b\"
c"
            ),
            "\"a,\"\"b\"\"
c\""
        );
        assert_eq!(csv_escape("中文"), "中文");
    }

    #[test]
    fn output_failure_cancels_and_joins_workers() {
        struct BrokenWriter {
            remaining: usize,
        }
        impl Write for BrokenWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.remaining == 0 {
                    return Err(io::Error::other("测试写失败"));
                }
                let len = bytes.len().min(self.remaining);
                self.remaining -= len;
                Ok(len)
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        assert!(schedule(&command(6, 0, 1000, false), &mut BrokenWriter { remaining: 650 }).is_err());
    }
}

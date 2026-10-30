use crate::{
    Command, GenMethods,
    evaluate::xuping::Features20,
    generate::NameGenerator,
    name::{KsaPrefix, Namer, TeamNamer},
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
    pub interrupted: bool,
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

/// 测量仍执行完整生成、属性、必要技能/模型、命中 CSV 序列化，仅输出设备换为 sink。
pub fn benchmark(cli: &Command) -> io::Result<RunSummary> {
    cli.validate().map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let summary = schedule(cli, &mut io::sink())?;
    info!(
        "完整 CPU 基准: 实际计算 {}，命中 {}，耗时 {:.6}s，速度 {:.2}/s，中断 {}",
        summary.computed,
        summary.hits,
        summary.elapsed.as_secs_f64(),
        summary.computed as f64 / summary.elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        summary.interrupted
    );
    Ok(summary)
}

/// 每个 worker 一个私有任务通道；状态和输出仅由主线程修改。
/// 写入错误时取消后续计算，但继续接收事件并 join 所有 worker，避免有界通道死锁。
fn schedule(cli: &Command, writer: &mut impl Write) -> io::Result<RunSummary> {
    schedule_with_cancel(cli, writer, Arc::new(AtomicBool::new(false)))
}

fn schedule_with_cancel(cli: &Command, writer: &mut impl Write, cancelled: Arc<AtomicBool>) -> io::Result<RunSummary> {
    writeln!(writer, "{}", csv_header())?;
    writer.flush()?;
    let start = Instant::now();
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
                if let Some(range) = cursor.take(cli.balanced_batch_size(0.0, cursor.end - cursor.next)) {
                    if tx.send(range).is_err() {
                        failure = Some(io::Error::other("worker 启动失败"));
                    }
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
                if failure.is_none() && !cancelled.load(Ordering::Relaxed) && !crate::interrupt::requested() {
                    if let Some(range) = cursor.take(cli.balanced_batch_size(speed, cursor.end - cursor.next)) {
                        if work_senders[worker].as_ref().unwrap().send(range).is_err() {
                            failure = Some(io::Error::other("worker 意外退出"));
                        }
                    } else {
                        work_senders[worker] = None;
                    }
                }
            }
        }
        if failure.is_some() || cancelled.load(Ordering::Relaxed) || crate::interrupt::requested() {
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
    summary.interrupted = cancelled.load(Ordering::Relaxed) || crate::interrupt::requested();
    if !summary.interrupted && summary.computed != cli.end - cli.start {
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
    let team = TeamNamer::new(&config.team).expect("队伍长度已在启动前验证");
    let mut namer = Namer::new_from_team_namer_unchecked(&team, "");
    let generator = NameGenerator::new(&config.algorithm, &config.template);
    let prefix = KsaPrefix::new(&team, generator.prefix());
    const LANES: usize = 4;
    let mut names: [String; LANES] = std::array::from_fn(|_| String::with_capacity(256));
    for range in work {
        let start = Instant::now();
        let mut computed = 0;
        let mut id = range.start;
        while id < range.end {
            if cancelled.load(Ordering::Relaxed) || crate::interrupt::requested() {
                break;
            }
            let count = (range.end - id).min(LANES as u64) as usize;
            for (lane, name) in names.iter_mut().enumerate().take(count) {
                generator.render(id + lane as u64, name);
            }
            if count == LANES {
                let states = prefix.states::<LANES>(std::array::from_fn(|lane| names[lane].as_str()));
                for (lane, state) in states.into_iter().enumerate() {
                    computed += 1;
                    if !evaluate_candidate(id + lane as u64, &names[lane], state, &mut namer, &config, &events) {
                        return;
                    }
                }
            } else {
                // 尾部用真实单候选路径，不补齐、不重复、不越过 end。
                for (lane, name) in names.iter().enumerate().take(count) {
                    computed += 1;
                    if !evaluate_candidate(
                        id + lane as u64,
                        name,
                        prefix.states([name.as_str()])[0],
                        &mut namer,
                        &config,
                        &events,
                    ) {
                        return;
                    }
                }
            }
            id += count as u64;
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

fn evaluate_candidate(
    id: u64,
    name: &str,
    state: [u8; 256],
    namer: &mut Namer,
    config: &CacluateConfig,
    events: &Sender<Event>,
) -> bool {
    namer.replace_state(name, state);
    if !property_matches(&namer.name_prop, config.prop_expect) {
        return true;
    }
    namer.update_skill();
    let features = Features20::new(namer);
    let xd = features.xd();
    if (xd as u32) < config.xd_expect {
        return true;
    }
    let xp = features.xp();
    if (xp as u32) < config.xp_expect {
        return true;
    }
    events.send(Event::Hit(csv_row(id, namer, xp, xd))).is_ok()
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

    #[test]
    fn cooperative_cancel_preserves_rows_and_joins_all_workers() {
        struct CancelWriter {
            bytes: Vec<u8>,
            cancelled: Arc<AtomicBool>,
        }
        impl Write for CancelWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                if self.bytes.len() > csv_header().len() + 800 {
                    self.cancelled.store(true, Ordering::Relaxed);
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut writer = CancelWriter {
            bytes: Vec::new(),
            cancelled: cancelled.clone(),
        };
        let summary = schedule_with_cancel(&command(6, 0, 1_000_000, true), &mut writer, cancelled).unwrap();
        assert!(summary.interrupted);
        assert!(summary.computed < 1_000_000);
        let csv = String::from_utf8(writer.bytes).unwrap();
        assert_eq!(csv.lines().count() as u64, summary.hits + 1);
        assert!(csv.lines().all(|row| row.split(',').count() == 55));
    }

    #[test]
    fn short_writes_zero_hits_and_empty_output_have_complete_header() {
        struct ShortWriter(Vec<u8>);
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let len = bytes.len().min(3);
                self.0.extend_from_slice(&bytes[..len]);
                Ok(len)
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let mut cli = command(2, 10, 19, false);
        cli.prop_expect = u32::MAX;
        let mut writer = ShortWriter(Vec::new());
        assert_eq!(schedule(&cli, &mut writer).unwrap().hits, 0);
        assert_eq!(String::from_utf8(writer.0).unwrap(), format!("{}\n", csv_header()));
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let path = std::env::temp_dir().join(format!(
            "tswn-cpu-{}-{}.csv",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        start_main(command(1, 0, 0, false), path.clone()).unwrap();
        let expected = std::fs::read(&path).unwrap();
        assert_eq!(
            start_main(command(1, 0, 1, false), path.clone()).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        std::fs::remove_file(path).unwrap();
    }
}

mod rc4_bench;

use clap::{Parser, ValueEnum};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use wgpu_mine::{
    core::{self, Thresholds},
    input::{batch_ranges, Encoding, Generator},
    mining_gpu::GpuMiner,
    output::CsvOutput,
    pool::CpuPool,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Gpu,
    Cpu,
}
impl Backend {
    fn label(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "E2E GPU 测号：生成 → RC4 → 属性/技能 → XP/XD → CSV")]
struct Args {
    #[arg(long)]
    team: String,
    #[arg(
        short = 'm',
        long = "name-template",
        alias = "template",
        default_value = "[]"
    )]
    template: String,
    #[arg(
        short = 'n',
        long = "name-gen",
        alias = "encoding",
        value_enum,
        default_value = "base16384"
    )]
    encoding: Encoding,
    #[arg(long, default_value_t = 0)]
    start: u64,
    /// 结束 ID（不含）；不指定时运行到 u64::MAX 或 Ctrl+C。
    #[arg(long, conflicts_with = "count")]
    end: Option<u64>,
    #[arg(long, conflicts_with = "end")]
    count: Option<u64>,
    /// GPU 模式的名字生成线程数；CPU 模式同时用于完整计算。
    #[arg(
        short = 't',
        long = "thread-count",
        alias = "cpu-threads",
        default_value_t = 6
    )]
    threads: usize,
    #[arg(short = 'p', long = "prop-expected", default_value_t = 600)]
    prop: u32,
    #[arg(short = 'x', long = "xp-exp", default_value_t = 4900)]
    xp: u32,
    #[arg(short = 'd', long = "xd-exp", default_value_t = 5600)]
    xd: u32,
    #[arg(short = 'b', long = "batch-size", default_value_t = 65536)]
    batch_size: usize,
    /// 进度报告秒数；不改变范围或筛选语义。
    #[arg(short = 'r', long = "report-interval", default_value_t = 10)]
    report: u64,
    #[arg(long, value_enum, default_value = "gpu")]
    backend: Backend,
    /// 指定新输出文件；禁止覆盖。- 表示 CSV 写到标准输出。
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long, default_value = "namerena")]
    output_dir: PathBuf,
    /// 对最终 GPU 命中额外做完整 CPU 属性/技能复核。
    #[arg(long)]
    verify: bool,
    /// 对指定数量候选做完整 GPU/CPU 对拍并统计模型误差，不执行测号输出。
    #[arg(long)]
    validate: Option<u64>,
    /// 在相同范围内交替执行 GPU/CPU，包含生成、CSV创建、格式化和flush。
    #[arg(long)]
    e2e_bench: bool,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// 验证或 E2E 基准的指标文件；禁止覆盖。
    #[arg(long)]
    metrics: Option<PathBuf>,
}

#[derive(Default, Clone)]
struct Stats {
    processed: u64,
    hits: u64,
    prop_pass: u64,
    shadows: u64,
    next: u64,
    elapsed: Duration,
    generation: Duration,
    compute: Duration,
    output: Duration,
    kernel: Duration,
    max_error: [f64; 2],
}
impl Stats {
    fn rate(&self) -> f64 { self.processed as f64 / self.elapsed.as_secs_f64().max(1e-9) }
}

fn fresh_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("创建输出失败（不覆盖已有文件）{}：{e}", path.display()))
}
fn run_tag() -> String {
    format!(
        "{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
        std::process::id()
    )
}
fn range_end(args: &Args) -> Result<u64, String> {
    if let Some(count) = args.count {
        args.start.checked_add(count).ok_or_else(|| "start+count 超出 u64 范围".to_owned())
    } else {
        Ok(args.end.unwrap_or(u64::MAX))
    }
}
fn thresholds(args: &Args) -> Thresholds {
    Thresholds {
        prop: args.prop,
        xp: args.xp,
        xd: args.xd,
    }
}
fn check_options(args: &Args) -> Result<(), String> {
    if args.threads == 0 || args.threads > 256 || args.report == 0 || args.batch_size == 0 || args.batch_size > 524287 {
        return Err("线程数须为1..=256，报告间隔须大于零，批大小须为1..=524287".to_owned());
    }
    if args.team.len() > 256 {
        return Err("队伍名超过256个UTF-8字节".to_owned());
    }
    let end = range_end(args)?;
    if args.start > end {
        return Err("start 不能大于 end".to_owned());
    }
    if args.e2e_bench
        && (args.rounds == 0 || (args.end.is_none() && args.count.is_none()) || args.output.is_some() || args.validate.is_some())
    {
        return Err("E2E基准须指定有限end或count、rounds>0，使用output-dir而非output，不能同时validate".to_owned());
    }
    Generator::new(args.template.clone(), args.encoding)?.generate(args.start)?;
    Ok(())
}

fn run_once(
    args: &Args,
    backend: Backend,
    pool: &mut CpuPool,
    mut gpu: Option<&mut GpuMiner>,
    path: &Path,
    cancel: &AtomicBool,
) -> Result<Stats, String> {
    let start = Instant::now();
    let writer: Box<dyn Write> = if path == Path::new("-") {
        Box::new(std::io::stdout())
    } else {
        Box::new(fresh_file(path)?)
    };
    let mut output = CsvOutput::new(writer)?;
    let mut stats = Stats {
        next: args.start,
        ..Default::default()
    };
    let mut reported = Instant::now();
    let end = range_end(args)?;
    for range in batch_ranges(args.start, end, args.batch_size)? {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let generated = Instant::now();
        let names = pool.generate(range.clone())?;
        stats.generation += generated.elapsed();
        let computed = Instant::now();
        let records = match backend {
            Backend::Cpu => {
                let batch = pool.evaluate(Arc::clone(&names), thresholds(args), false)?;
                stats.prop_pass += batch.prop_pass;
                stats.shadows += batch.shadows;
                batch.records.into_iter().map(|r| (r.index, r.value)).collect::<Vec<_>>()
            }
            Backend::Gpu => {
                let batch = gpu.as_deref_mut().ok_or("GPU尚未初始化")?.run(&names, thresholds(args), false)?;
                stats.prop_pass += u64::from(batch.prop_pass);
                stats.shadows += u64::from(batch.shadows);
                stats.kernel += batch.kernel_time.unwrap_or_default();
                for i in 0..2 {
                    stats.max_error[i] = stats.max_error[i].max(batch.max_error[i]);
                }
                if args.verify {
                    for record in &batch.records {
                        let wanted = core::evaluate(&args.team, &names[record.index]);
                        compare_values(&record.value, &wanted)?;
                    }
                }
                batch.records.into_iter().map(|r| (r.index, r.value)).collect::<Vec<_>>()
            }
        };
        stats.compute += computed.elapsed();
        let written = Instant::now();
        for (index, value) in records {
            let id = range.start + index as u64;
            output.write_record(id, &names[index], &args.team, &value)?;
            stats.hits += 1;
        }
        stats.output += written.elapsed();
        stats.processed += range.end - range.start;
        stats.next = range.end;
        if reported.elapsed() >= Duration::from_secs(args.report) {
            eprintln!(
                "[{}] 完成={} 命中={} 下一ID={} E2E={:.0}名/s",
                backend.label(),
                stats.processed,
                stats.hits,
                stats.next,
                stats.processed as f64 / start.elapsed().as_secs_f64()
            );
            reported = Instant::now();
        }
    }
    let flushed = Instant::now();
    output.flush()?;
    stats.output += flushed.elapsed();
    stats.elapsed = start.elapsed();
    eprintln!(
        "[{}] {}：完成={} 命中={} 下一ID={} E2E={:.3}s {:.0}名/s，生成={:.3}s 计算/回读={:.3}s 输出={:.3}s GPU内核={:.3}s",
        backend.label(),
        if cancel.load(Ordering::Relaxed) {
            "已停止并flush"
        } else {
            "完成"
        },
        stats.processed,
        stats.hits,
        stats.next,
        stats.elapsed.as_secs_f64(),
        stats.rate(),
        stats.generation.as_secs_f64(),
        stats.compute.as_secs_f64(),
        stats.output.as_secs_f64(),
        stats.kernel.as_secs_f64()
    );
    Ok(stats)
}

fn compare_values(actual: &core::Evaluation, expected: &core::Evaluation) -> Result<(), String> {
    if actual.props != expected.props
        || actual.skill_ids != expected.skill_ids
        || actual.skill_freqs != expected.skill_freqs
        || actual.shadow_numerator != expected.shadow_numerator
        || (actual.xp - expected.xp).abs() > 1e-7
        || (actual.xd - expected.xd).abs() > 1e-7
    {
        return Err(format!("GPU/CPU完整结果不一致：GPU={actual:?} CPU={expected:?}"));
    }
    Ok(())
}

fn validate(args: &Args, pool: &mut CpuPool, gpu: &mut GpuMiner, count: u64, cancel: &AtomicBool) -> Result<(), String> {
    let end = args.start.checked_add(count).ok_or("验证范围溢出")?;
    let start = Instant::now();
    let mut errors = [Vec::<f64>::new(), Vec::<f64>::new()];
    let mut checked = 0_u64;
    let mut shadows = 0_u64;
    for range in batch_ranges(args.start, end, args.batch_size)? {
        if cancel.load(Ordering::Relaxed) {
            return Err("验证被取消，尚未完成全部样本".to_owned());
        }
        let names = pool.generate(range)?;
        let expected = pool.evaluate(Arc::clone(&names), Thresholds::default(), true)?;
        let actual = gpu.run(&names, Thresholds::default(), true)?;
        if actual.records.len() != expected.records.len() || actual.records.len() != names.len() {
            return Err("完整验证漏号或重复号".to_owned());
        }
        shadows += u64::from(actual.shadows);
        for (record, wanted) in actual.records.iter().zip(&expected.records) {
            if record.index != wanted.index {
                return Err("完整验证候选顺序不一致".to_owned());
            }
            compare_values(&record.value, &wanted.value)?;
            for (i, expected_score) in [wanted.value.xp, wanted.value.xd].iter().enumerate() {
                let error = (f64::from(record.approximate[i]) - expected_score).abs();
                if !error.is_finite() || error >= 10.0 {
                    return Err(format!("模型{i}绝对误差不满足<10：{error}"));
                }
                errors[i].push(error);
            }
            checked += 1;
        }
    }
    let mut metric = if let Some(path) = &args.metrics {
        Some(csv::Writer::from_writer(fresh_file(path)?))
    } else {
        None
    };
    if let Some(writer) = &mut metric {
        writer
            .write_record([
                "model",
                "samples",
                "shadows",
                "max_abs_error",
                "mean_abs_error",
                "p99_abs_error",
                "seconds",
            ])
            .map_err(|e| e.to_string())?;
    }
    for (i, model) in ["xp", "xd"].iter().enumerate() {
        errors[i].sort_unstable_by(f64::total_cmp);
        let max = errors[i].last().copied().unwrap_or_default();
        let mean = errors[i].iter().sum::<f64>() / checked.max(1) as f64;
        let p99 = errors[i].get(errors[i].len().saturating_sub(1) * 99 / 100).copied().unwrap_or_default();
        eprintln!("验证 {model}：样本={checked} shadow={shadows} max={max:.8} mean={mean:.8} p99={p99:.8}，要求<10");
        if let Some(writer) = &mut metric {
            writer
                .write_record([
                    model.to_string(),
                    checked.to_string(),
                    shadows.to_string(),
                    max.to_string(),
                    mean.to_string(),
                    p99.to_string(),
                    start.elapsed().as_secs_f64().to_string(),
                ])
                .map_err(|e| e.to_string())?;
        }
    }
    if let Some(writer) = &mut metric {
        writer.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn same_csv(a: &Path, b: &Path) -> Result<(), String> {
    let mut a = csv::Reader::from_path(a).map_err(|e| e.to_string())?;
    let mut b = csv::Reader::from_path(b).map_err(|e| e.to_string())?;
    if a.headers().map_err(|e| e.to_string())? != b.headers().map_err(|e| e.to_string())? {
        return Err("对照CSV表头不一致".to_owned());
    }
    let mut ra = a.records();
    let mut rb = b.records();
    let mut index = 0;
    loop {
        match (ra.next(), rb.next()) {
            (None, None) => return Ok(()),
            (Some(Ok(a)), Some(Ok(b))) if a == b => {}
            (Some(Err(e)), _) | (_, Some(Err(e))) => return Err(e.to_string()),
            _ => return Err(format!("GPU/CPU E2E输出第{index}条记录不一致")),
        }
        index += 1;
    }
}

async fn run(args: Args, cancel: Arc<AtomicBool>) -> Result<(), String> {
    check_options(&args)?;
    let setup = Instant::now();
    let generator = Generator::new(args.template.clone(), args.encoding)?;
    let mut pool = CpuPool::new(generator, &args.team, args.threads)?;
    let needs_gpu = args.e2e_bench || args.validate.is_some() || matches!(args.backend, Backend::Gpu);
    let mut gpu = if needs_gpu {
        Some(GpuMiner::new(&args.team, args.batch_size).await?)
    } else {
        None
    };
    if let Some(gpu) = &gpu {
        eprintln!("GPU：{}", gpu.adapter_description);
    }
    let initialization = setup.elapsed();
    eprintln!(
        "初始化={:.3}s；范围=[{}, {})；线程={}；批大小={}；评分GPU误差余量10，候选使用f64精筛",
        initialization.as_secs_f64(),
        args.start,
        range_end(&args)?,
        args.threads,
        args.batch_size
    );
    if let Some(count) = args.validate {
        return validate(&args, &mut pool, gpu.as_mut().ok_or("验证需要GPU")?, count, &cancel);
    }
    if args.e2e_bench {
        fs::create_dir_all(&args.output_dir).map_err(|e| e.to_string())?;
        let tag = run_tag();
        let mut metrics = if let Some(path) = &args.metrics {
            Some(csv::Writer::from_writer(fresh_file(path)?))
        } else {
            None
        };
        if let Some(writer) = &mut metrics {
            writer
                .write_record([
                    "round",
                    "backend",
                    "threads",
                    "batch_size",
                    "processed",
                    "hits",
                    "seconds",
                    "names_per_second",
                    "generation_s",
                    "compute_s",
                    "output_s",
                    "gpu_kernel_s",
                    "xp_max_error",
                    "xd_max_error",
                    "setup_s",
                ])
                .map_err(|e| e.to_string())?;
        }
        let mut times = [Vec::<f64>::new(), Vec::<f64>::new()];
        for round in 0..args.warmup + args.rounds {
            let mut paths = Vec::new();
            let mut stats = Vec::new();
            let order = if round % 2 == 0 {
                [Backend::Gpu, Backend::Cpu]
            } else {
                [Backend::Cpu, Backend::Gpu]
            };
            for backend in order {
                let path = args.output_dir.join(format!("e2e-{tag}-{round}-{}.csv", backend.label()));
                let value = run_once(&args, backend, &mut pool, gpu.as_mut(), &path, &cancel)?;
                if cancel.load(Ordering::Relaxed) {
                    return Err("E2E基准被取消".to_owned());
                }
                if round >= args.warmup {
                    let i = usize::from(matches!(backend, Backend::Cpu));
                    times[i].push(value.elapsed.as_secs_f64());
                    if let Some(writer) = &mut metrics {
                        writer
                            .write_record([
                                (round - args.warmup).to_string(),
                                backend.label().to_owned(),
                                args.threads.to_string(),
                                args.batch_size.to_string(),
                                value.processed.to_string(),
                                value.hits.to_string(),
                                value.elapsed.as_secs_f64().to_string(),
                                value.rate().to_string(),
                                value.generation.as_secs_f64().to_string(),
                                value.compute.as_secs_f64().to_string(),
                                value.output.as_secs_f64().to_string(),
                                value.kernel.as_secs_f64().to_string(),
                                value.max_error[0].to_string(),
                                value.max_error[1].to_string(),
                                initialization.as_secs_f64().to_string(),
                            ])
                            .map_err(|e| e.to_string())?;
                    }
                }
                paths.push(path);
                stats.push(value);
            }
            same_csv(&paths[0], &paths[1])?;
            if stats[0].processed != stats[1].processed {
                return Err("E2E对照范围不一致".to_owned());
            }
            eprintln!("第{round}轮：GPU/CPU CSV逐字段一致");
        }
        for values in &mut times {
            values.sort_unstable_by(f64::total_cmp);
        }
        let gpu_time = times[0][times[0].len() / 2];
        let cpu_time = times[1][times[1].len() / 2];
        eprintln!(
            "E2E中位数：GPU={gpu_time:.6}s CPU={cpu_time:.6}s，加速比={:.3}x（含名字生成、完整输出；初始化另列）",
            cpu_time / gpu_time
        );
        if let Some(writer) = &mut metrics {
            writer.flush().map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let path = if let Some(path) = &args.output {
        path.clone()
    } else {
        fs::create_dir_all(&args.output_dir).map_err(|e| e.to_string())?;
        args.output_dir.join(format!("namerena-{}-{}.csv", args.backend.label(), run_tag()))
    };
    eprintln!("输出：{}", path.display());
    let stats = run_once(&args, args.backend, &mut pool, gpu.as_mut(), &path, &cancel)?;
    eprintln!(
        "含初始化总耗时={:.3}s，已完成范围=[{}, {})",
        setup.elapsed().as_secs_f64(),
        args.start,
        stats.next
    );
    Ok(())
}

fn main() -> ExitCode {
    if std::env::args_os().any(|arg| arg == "--bench") {
        return rc4_bench::entry();
    }
    let args = Args::parse();
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    if let Err(error) = ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed)) {
        eprintln!("注册Ctrl+C失败：{error}");
        return ExitCode::FAILURE;
    }
    match pollster::block_on(run(args, Arc::clone(&cancel))) {
        Ok(()) => {
            if cancel.load(Ordering::Relaxed) {
                ExitCode::from(130)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("测号失败：{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    #[test]
    fn original_basic_command_parses() {
        Args::command().debug_assert();
        let args = Args::try_parse_from([
            "wgpu-mine",
            "-t",
            "2",
            "-p",
            "550",
            "-x",
            "4000",
            "-d",
            "5000",
            "--team",
            "shenjack",
            "-r",
            "10",
            "-m",
            "我也不知道我要测多少 []",
        ])
        .unwrap();
        assert_eq!(
            (args.threads, args.prop, args.xp, args.xd, args.report),
            (2, 550, 4000, 5000, 10)
        );
        assert!(check_options(&args).is_ok());
    }
    #[test]
    fn invalid_and_overflow_ranges_are_rejected() {
        for options in [
            vec!["--thread-count", "0"],
            vec!["--batch-size", "0"],
            vec!["--report-interval", "0"],
            vec!["--start", "10", "--end", "9"],
            vec!["--start", "18446744073709551615", "--count", "1"],
        ] {
            let args = Args::try_parse_from([vec!["wgpu-mine", "--team", "x"], options].concat()).unwrap();
            assert!(check_options(&args).is_err());
        }
    }
}

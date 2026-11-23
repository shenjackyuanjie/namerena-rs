use crate::cuda;
use clap::{Parser, ValueEnum};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use wgpu_mine::{
    core::{self, Evaluation, Thresholds},
    gpu_input::GpuInput,
    input::{batch_ranges, Encoding, Generator},
    mining_gpu::{GpuBatch, GpuMiner, GpuOptions, GpuRecord},
    output::CsvOutput,
};

const BATCH: u32 = 524_287;
const THRESHOLDS: Thresholds = Thresholds {
    prop: 550,
    xp: 4000,
    xd: 5000,
};
const WARMUP: u64 = 10_000_000;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Mode {
    Split = 0,
    Full  = 1,
    Sort10 = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    Wgsl,
    Cuda,
}

#[derive(Parser)]
#[command(about = "同区间 WGSL/Vulkan 和 CUDA 全链路 A/B（Base16384, p550/x4000/d5000）")]
struct Args {
    #[arg(long, default_value = "target/cuda-ab/cuda-mine.dll")]
    dll: PathBuf,
    /// 必须为不存在的目录；上级目录需已存在。
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    start: u64,
    #[arg(long, default_value_t = 300_000_000)]
    count: u64,
    /// 每个后端实跑指定秒数；到点时收完已经开始的批次。
    #[arg(long, conflicts_with = "count", value_parser = clap::value_parser!(u64).range(1..))]
    seconds: Option<u64>,
    /// 0 只对拍，不跑吞吐。
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// 适用于与独立正式 WGSL 二进制组成三路反向交替。
    #[arg(long)]
    cuda_first: bool,
    #[arg(long, default_value_t = 32)]
    cuda_block: u32,
    /// WGSL 工作组大小；与 CUDA block 可分别扫描。
    #[arg(long, default_value_t = 32)]
    wgsl_block: u32,
    /// 两后端的候选批次容量；大批次要求设备和内存支持。
    #[arg(long, default_value_t = BATCH, value_parser = clap::value_parser!(u32).range(65_537..=2_097_120))]
    batch_size: u32,
    /// 只计时指定后端；仍用两边与 CPU 做正解验证。
    #[arg(long, value_enum)]
    only: Option<Backend>,
    #[arg(long, value_enum, default_value = "full")]
    cuda_mode: Mode,
    #[arg(long, default_value = "shenjack")]
    team: String,
    #[arg(long, default_value = "我也不知道我要测多少 []")]
    name_template: String,
}

struct Workload {
    generator: Generator,
    team_state: [u8; 256],
    args: Args,
}

#[derive(Default, PartialEq, Eq, Debug)]
struct Counts {
    prop_pass: u64,
    scored: u64,
    shadows: u64,
    hits: u64,
    refined: u64,
}
impl Counts {
    fn add(&mut self, batch: &GpuBatch) {
        self.prop_pass += u64::from(batch.prop_pass);
        self.scored += u64::from(batch.scored);
        self.shadows += u64::from(batch.shadows);
        self.hits += batch.records.len() as u64;
        self.refined += batch.refined as u64;
    }
}

#[derive(Clone, Copy)]
enum Budget {
    Candidates(u64),
    Time(Duration),
}

type Batcher = dyn FnMut(u64, u32, Thresholds, bool) -> Result<GpuBatch, String>;

pub fn run() -> Result<(), String> {
    let args = Args::parse();
    let budget = match args.seconds {
        Some(seconds) => Budget::Time(Duration::from_secs(seconds)),
        None => Budget::Candidates(args.count),
    };
    if ![32, 64, 128].contains(&args.cuda_block) || args.start == u64::MAX {
        return Err("起始 ID 和 CUDA block 无效；block 须为 32/64/128".into());
    }
    if let Budget::Candidates(count) = budget {
        if count == 0 || args.start.checked_add(count).is_none() {
            return Err("要求非空有限区间".into());
        }
    }
    let team = &args.team;
    let generator = Generator::new(args.name_template.clone(), Encoding::Base16384)?;
    let input = GpuInput::new(team, &args.name_template, Encoding::Base16384)?.ok_or("缺少固定输入")?;
    fs::create_dir(&args.out).map_err(|e| format!("创建新结果目录 {}：{e}", args.out.display()))?;

    let init = Instant::now();
    let mut gpu = pollster::block_on(GpuMiner::with_generated(
        team,
        args.batch_size as usize,
        GpuOptions {
            workgroup_size: args.wgsl_block,
            ..GpuOptions::default()
        },
        input.clone(),
    ))?;
    if !gpu.adapter_description.contains("Vulkan") || !gpu.adapter_description.contains("NVIDIA") {
        return Err(format!("基准要求 NVIDIA / Vulkan，当前为 {}", gpu.adapter_description));
    }
    println!(
        "WGSL adapter={} init_s={:.3}",
        gpu.adapter_description,
        init.elapsed().as_secs_f64()
    );
    gpu.exact_scores = true;
    let init = Instant::now();
    let mut cuda = cuda::Miner::new(&args.dll, &input, args.batch_size, args.cuda_block, args.cuda_mode as u32)?;
    println!(
        "CUDA mode={:?} block={} init_s={:.3}",
        args.cuda_mode,
        args.cuda_block,
        init.elapsed().as_secs_f64()
    );

    let work = Workload {
        generator,
        team_state: wgpu_mine::cpu::team_state(team),
        args,
    };
    let mut wgsl = move |id, n, t, capture| gpu.run_range(id, n as usize, t, capture);
    let mut cuda = move |id, n, t, capture| cuda.run(id, n, t, capture);
    for (first, n, t, capture) in [
        (0, 65_537, THRESHOLDS, true),
        (u64::from(u32::MAX) - 16, 257, THRESHOLDS, true),
        (u64::MAX - 256, 257, THRESHOLDS, true),
        (
            0,
            32_771,
            Thresholds {
                prop: 550,
                xp: 2000,
                xd: 3000,
            },
            false,
        ),
    ] {
        let a = wgsl(first, n, t, capture)?;
        let b = cuda(first, n, t, capture)?;
        let mut cpu = Counts::default();
        let mut hits = 0;
        for i in 0..n as usize {
            let expected = work.reference(first + i as u64)?;
            if capture || u64::from(core::property_numerator(&expected.props)) > 3 * u64::from(t.prop) {
                cpu.prop_pass += 1;
                cpu.scored += 1;
                cpu.shadows += u64::from(expected.skill_ids.iter().zip(expected.skill_freqs).any(|(&id, f)| id == 24 && f != 0));
            }
            if capture || expected.matches(t) {
                for batch in [&a, &b] {
                    let got = batch.records.get(hits).ok_or_else(|| format!("验证缺少候选 {i}"))?;
                    if got.index != i {
                        return Err(format!("验证候选下标不符：{} != {i}", got.index));
                    }
                    compare(got, &expected, first + i as u64)?;
                }
                hits += 1;
            }
        }
        for batch in [&a, &b] {
            if (
                batch.records.len(),
                u64::from(batch.prop_pass),
                u64::from(batch.scored),
                u64::from(batch.shadows),
            ) != (hits, cpu.prop_pass, cpu.scored, cpu.shadows)
            {
                return Err("验证结果条数或阶段计数与 CPU 不符".into());
            }
        }
        println!(
            "验证 first={first} count={n} capture={capture} hits={hits} shadow={} WGSL误差={:?} CUDA误差={:?}",
            cpu.shadows, a.max_error, b.max_error
        );
    }
    if work.args.rounds == 0 {
        return Ok(());
    }

    let mut metrics = csv::Writer::from_path(work.args.out.join("metrics.csv")).map_err(|e| e.to_string())?;
    metrics
        .write_record([
            "round",
            "backend",
            "candidates",
            "hits",
            "prop_pass",
            "scored",
            "shadows",
            "refined",
            "e2e_s",
            "kernel_s",
            "e2e_names_s",
            "kernel_names_s",
        ])
        .map_err(|e| e.to_string())?;
    if let Budget::Candidates(count) = budget {
        for (label, f) in [("wgsl", &mut wgsl as &mut Batcher), ("cuda", &mut cuda as &mut Batcher)] {
            work.measure(
                f,
                Budget::Candidates(count.min(WARMUP)),
                &work.args.out.join(format!("warmup-{label}.csv")),
            )?;
        }
    }
    for round in 1..=work.args.rounds {
        let mut order: [(&str, &mut Batcher); 2] = [("wgsl", &mut wgsl), ("cuda", &mut cuda)];
        if (round % 2 == 0) ^ work.args.cuda_first {
            order.reverse();
        }
        let mut expected: Option<(u64, Counts, Vec<u8>)> = None;
        for (label, f) in order {
            if work
                .args
                .only
                .is_some_and(|b| b != if label == "wgsl" { Backend::Wgsl } else { Backend::Cuda })
            {
                continue;
            }
            let path = work.args.out.join(format!("{round}-{label}.csv"));
            if let Budget::Time(target) = budget {
                work.measure(
                    f,
                    Budget::Time(Duration::from_secs(2)),
                    &work.args.out.join(format!("{round}-{label}-warmup.csv")),
                )?;
                println!("轮次={round} {label} 开始计时，目标 {:.0} 秒", target.as_secs_f64());
            }
            let (n, counts, e2e, kernel) = work.measure(f, budget, &path)?;
            let e2e_names = n as f64 / e2e.as_secs_f64();
            let kernel_names = n as f64 / kernel.as_secs_f64();
            println!("轮次={round} {label} 候选={n} 命中={} 属性通过={} E2E={:.6}s {e2e_names:.0}名/s 内核={:.6}s {kernel_names:.0}名/s",
                     counts.hits, counts.prop_pass, e2e.as_secs_f64(), kernel.as_secs_f64());
            metrics
                .serialize((
                    round,
                    label,
                    n,
                    counts.hits,
                    counts.prop_pass,
                    counts.scored,
                    counts.shadows,
                    counts.refined,
                    e2e.as_secs_f64(),
                    kernel.as_secs_f64(),
                    e2e_names,
                    kernel_names,
                ))
                .map_err(|e| e.to_string())?;
            metrics.flush().map_err(|e| e.to_string())?;
            let csv = fs::read(path).map_err(|e| e.to_string())?;
            if let Some((n_a, counts_a, csv_a)) = &expected {
                if *n_a == n {
                    // f32 舍入可能改变余量内回传条数；f64 后的命中和已决阶段必须相同。
                    if (counts_a.prop_pass, counts_a.scored, counts_a.shadows, counts_a.hits)
                        != (counts.prop_pass, counts.scored, counts.shadows, counts.hits)
                        || *csv_a != csv
                    {
                        return Err(format!("第 {round} 轮阶段计数或完整 CSV 不一致"));
                    }
                    println!("轮次={round} 阶段计数及 CSV 逐字节一致");
                } else {
                    if matches!(budget, Budget::Candidates(_)) {
                        return Err("同量基准处理数不一致".into());
                    }
                    let shared = (*n_a).min(n);
                    let end = work.args.start + shared;
                    if csv_prefix(csv_a, end)? != csv_prefix(&csv, end)? {
                        return Err(format!("第 {round} 轮定时基准的公共 ID 区间 CSV 不一致"));
                    }
                    println!("轮次={round} 公共区间 {shared} 个候选的完整命中一致");
                }
            } else {
                expected = Some((n, counts, csv));
            }
        }
    }
    Ok(())
}

impl Workload {
    fn reference(&self, id: u64) -> Result<Evaluation, String> {
        Ok(core::evaluate_cached(&self.team_state, &self.generator.generate(id)?))
    }

    fn measure(&self, run: &mut Batcher, budget: Budget, path: &Path) -> Result<(u64, Counts, Duration, Duration), String> {
        let started = Instant::now();
        let writer = fs::File::create_new(path).map_err(|e| e.to_string())?;
        let mut output = CsvOutput::new(writer)?;
        let mut counts = Counts::default();
        let mut kernel = Duration::ZERO;
        let mut processed = 0;
        let end = match budget {
            Budget::Candidates(n) => self.args.start + n,
            Budget::Time(_) => u64::MAX,
        };
        for range in batch_ranges(self.args.start, end, self.args.batch_size as usize)? {
            if matches!(budget, Budget::Time(target) if started.elapsed() >= target) {
                break;
            }
            let count = range.end - range.start;
            let batch = run(range.start, count as u32, THRESHOLDS, false)?;
            kernel += batch.kernel_time.ok_or("没有 GPU 时间戳")?;
            counts.add(&batch);
            for record in batch.records {
                let id = range.start + record.index as u64;
                let name = self.generator.generate(id)?;
                compare(&record, &core::evaluate_cached(&self.team_state, &name), id)?;
                output.write_record(id, &name, &self.args.team, &record.value)?;
            }
            processed += count;
        }
        output.flush()?;
        if processed == 0 || matches!(budget, Budget::Time(target) if started.elapsed() < target) {
            return Err("有效候选不足以完成指定的计时预算".into());
        }
        Ok((processed, counts, started.elapsed(), kernel))
    }
}

fn csv_prefix(data: &[u8], end: u64) -> Result<(csv::StringRecord, Vec<csv::StringRecord>), String> {
    let mut reader = csv::Reader::from_reader(data);
    let header = reader.headers().map_err(|e| e.to_string())?.clone();
    let mut rows = Vec::new();
    for row in reader.records() {
        let row = row.map_err(|e| e.to_string())?;
        let id: u64 = row.get(1).ok_or("CSV 缺少 ID")?.parse::<u64>().map_err(|e| e.to_string())?;
        if id < end {
            rows.push(row);
        }
    }
    Ok((header, rows))
}

fn compare(got: &GpuRecord, expected: &Evaluation, id: u64) -> Result<(), String> {
    let v = &got.value;
    if v.props != expected.props
        || v.skill_ids != expected.skill_ids
        || v.skill_freqs != expected.skill_freqs
        || v.shadow_numerator != expected.shadow_numerator
        || (v.xp - expected.xp).abs() > 1e-7
        || (v.xd - expected.xd).abs() > 1e-7
        || (f64::from(got.approximate[0]) - expected.xp).abs() >= 10.0
        || (f64::from(got.approximate[1]) - expected.xd).abs() >= 10.0
    {
        return Err(format!("CPU 完整对拍失败：ID {id}, GPU={v:?}, CPU={expected:?}"));
    }
    Ok(())
}

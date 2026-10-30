use wgpu_mine::{cpu, gpu};

use clap::{Parser, ValueEnum};
use std::{
    fs::File,
    hint::black_box,
    io::Write,
    path::PathBuf,
    process::ExitCode,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Encoding {
    Base16384,
    Number,
}

#[derive(Debug, Parser)]
#[command(about = "RC4 GPU 移植实验：只计算两轮名字 KSA，不计算属性或评分")]
struct Args {
    /// 运行 CPU/GPU 正确性与性能对照；不指定时只展示两个名字的 RC4 状态。
    #[arg(long)]
    bench: bool,
    /// private 保留每 invocation 私有 Sbox；shared 使用转置共享 Sbox。
    #[arg(long, value_enum, default_value = "shared")]
    kernel: gpu::Kernel,
    #[arg(long, default_value = "shenjack")]
    team: String,
    /// 生成名字时替换所有 []，按 UTF-8 字节上传。
    #[arg(short = 'm', long, default_value = "我也不知道我要测多少 []")]
    template: String,
    #[arg(long, value_enum, default_value = "base16384")]
    encoding: Encoding,
    #[arg(long, default_value_t = 0)]
    start: u64,
    /// 依次测试不同批大小；设备和最大容量缓冲只初始化一次。
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "256,4096,65536"
    )]
    batch_sizes: Vec<usize>,
    /// 多线程 CPU 基线的 worker 数，所有线程数共用同一通用路径。
    #[arg(short = 't', long, default_value_t = 6)]
    cpu_threads: usize,
    /// 每个批大小的正式重复次数。
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// 可选的 CSV 实测结果文件。
    #[arg(long)]
    csv: Option<PathBuf>,
}

fn make_names(args: &Args, count: usize) -> Result<Arc<[String]>, String> {
    if !args.template.contains("[]") {
        return Err("名字模板必须包含 []".to_owned());
    }
    let mut names = Vec::with_capacity(count);
    for offset in 0..count {
        let id = args
            .start
            .checked_add(u64::try_from(offset).map_err(|_| "候选编号溢出")?)
            .ok_or("候选编号溢出")?;
        let value = match args.encoding {
            Encoding::Base16384 => base16384::Base16384Utf8::encode(&id.to_be_bytes()),
            Encoding::Number => id.to_string(),
        };
        let name = args.template.replace("[]", &value);
        if name.len() > gpu::MAX_NAME_BYTES {
            return Err(format!("编号 {id} 的名字超过 256 个 UTF-8 字节"));
        }
        names.push(name);
    }
    Ok(names.into())
}

fn median(samples: &[Duration]) -> Duration {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    if samples.len().is_multiple_of(2) {
        samples[samples.len() / 2 - 1] / 2 + samples[samples.len() / 2] / 2
    } else {
        samples[samples.len() / 2]
    }
}

fn throughput(count: usize, duration: Duration) -> f64 { count as f64 / duration.as_secs_f64() / 1e6 }

fn check_equal(cpu: &[u8], gpu: &[u8]) -> Result<(), String> {
    if cpu.len() != gpu.len() {
        return Err("CPU/GPU 返回状态长度不一致".to_owned());
    }
    if let Some(offset) = cpu.iter().zip(gpu).position(|(a, b)| a != b) {
        return Err(format!(
            "CPU/GPU 不一致：名字 {}，状态下标 {}，CPU={} GPU={}",
            offset / gpu::STATE_SIZE,
            offset % gpu::STATE_SIZE,
            cpu[offset],
            gpu[offset]
        ));
    }
    Ok(())
}

async fn run(args: Args) -> Result<(), String> {
    if args.cpu_threads == 0 || args.rounds == 0 || args.batch_sizes.is_empty() || args.batch_sizes.contains(&0) {
        return Err("CPU 线程数、正式重复次数与批大小必须大于零".to_owned());
    }
    if !args.bench {
        let names = make_names(&args, 2)?;
        let mut gpu = gpu::GpuRc4::new(&args.team, 2, args.kernel).await?;
        let result = gpu.run(&names)?;
        println!("GPU：{}；内核：{}", gpu.adapter_description, args.kernel.label());
        println!("以下仅为两轮名字 RC4 KSA 后的完整置换，不是测号筛选结果：");
        for (name, state) in names.iter().zip(result.states.as_chunks::<{ gpu::STATE_SIZE }>().0) {
            assert_eq!(*state, cpu::reference(&args.team, name));
            println!("{name:?}: {state:?}");
        }
        return Ok(());
    }
    let initialization = Instant::now();
    let capacity = *args.batch_sizes.iter().max().unwrap();
    let mut gpu = gpu::GpuRc4::new(&args.team, capacity, args.kernel).await?;
    let mut cpu = cpu::CpuRc4::new(&args.team, args.cpu_threads)?;
    println!("GPU：{}；内核：{}", gpu.adapter_description, args.kernel.label());
    println!(
        "初始化：{:.3}s（不计入下表），GPU 时间戳：{}",
        initialization.elapsed().as_secs_f64(),
        gpu.has_timestamps()
    );
    println!(
        "固定队伍 {:?}；CPU {} workers；{} 次预热、{} 次正式测量取中位数",
        args.team, args.cpu_threads, args.warmup, args.rounds
    );
    println!("运算：缓存队伍 KSA 后，执行两轮名字 KSA；每名返回完整 256 字节状态。");
    println!("端到端：GPU 包含名字打包、上传、命令提交、状态回读与复制；CPU 包含持久 worker 派发、计算和结果归并。");
    println!("两者均不包含名字生成、设备/管线/线程初始化；不含属性、PRGA、技能或虚评。");
    println!("批大小,名字字节数,CPU毫秒,GPU端到端毫秒,GPU内核毫秒,GPU准备毫秒,CPU百万名每秒,GPU端到端百万名每秒,GPU内核百万名每秒,端到端加速比");
    let mut csv = if let Some(path) = &args.csv {
        let mut file = File::create(path).map_err(|error| format!("创建 CSV 失败：{error}"))?;
        writeln!(file, "kernel,batch_size,name_bytes,cpu_threads,rounds,cpu_ms,gpu_wall_ms,gpu_kernel_ms,gpu_prepare_ms,cpu_mnames_s,gpu_wall_mnames_s,gpu_kernel_mnames_s,speedup,verified")
            .map_err(|error| error.to_string())?;
        Some(file)
    } else {
        None
    };
    for &count in &args.batch_sizes {
        let generation_start = Instant::now();
        let names = make_names(&args, count)?;
        let generation_time = generation_start.elapsed();
        let name_bytes: usize = names.iter().map(String::len).sum();
        // 全量正确性检查在正式计时之前；后续每轮也在计时结束后逐字节校验。
        let expected = cpu.run(Arc::clone(&names))?;
        check_equal(&expected, &gpu.run(&names)?.states)?;
        for _ in 0..args.warmup {
            black_box(cpu.run(Arc::clone(&names))?);
            black_box(gpu.run(&names)?);
        }
        let mut cpu_times = Vec::with_capacity(args.rounds);
        let mut gpu_times = Vec::with_capacity(args.rounds);
        let mut kernel_times = Vec::with_capacity(args.rounds);
        let mut preparation_times = Vec::with_capacity(args.rounds);
        for _ in 0..args.rounds {
            let start = Instant::now();
            let cpu_result = cpu.run(Arc::clone(&names))?;
            cpu_times.push(start.elapsed());
            let gpu_result = gpu.run(&names)?;
            gpu_times.push(gpu_result.wall_time);
            preparation_times.push(gpu_result.preparation_time);
            if let Some(kernel_time) = gpu_result.kernel_time {
                kernel_times.push(kernel_time);
            }
            check_equal(black_box(&cpu_result), black_box(&gpu_result.states))?;
        }
        let cpu_time = median(&cpu_times);
        let gpu_time = median(&gpu_times);
        let prepare = median(&preparation_times);
        let kernel = (!kernel_times.is_empty()).then(|| median(&kernel_times));
        let cpu_rate = throughput(count, cpu_time);
        let gpu_rate = throughput(count, gpu_time);
        let kernel_ms = kernel.map(|time| format!("{:.6}", time.as_secs_f64() * 1e3)).unwrap_or_default();
        let kernel_rate = kernel.map(|time| format!("{:.6}", throughput(count, time))).unwrap_or_default();
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!(
            "{count},{name_bytes},{:.6},{:.6},{kernel_ms},{:.6},{cpu_rate:.6},{gpu_rate:.6},{kernel_rate},{speedup:.4}",
            cpu_time.as_secs_f64() * 1e3,
            gpu_time.as_secs_f64() * 1e3,
            prepare.as_secs_f64() * 1e3
        );
        eprintln!(
            "批 {count}：全量逐项一致；名字生成 {:.3}ms（不计时）；读回 {} 字节/轮",
            generation_time.as_secs_f64() * 1e3,
            count * gpu::STATE_SIZE
        );
        if let Some(file) = &mut csv {
            writeln!(file, "{},{count},{name_bytes},{},{},{:.6},{:.6},{kernel_ms},{:.6},{cpu_rate:.6},{gpu_rate:.6},{kernel_rate},{speedup:.6},true",
                args.kernel.label(), args.cpu_threads, args.rounds, cpu_time.as_secs_f64() * 1e3, gpu_time.as_secs_f64() * 1e3, prepare.as_secs_f64() * 1e3)
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    match pollster::block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("RC4 实验失败：{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn experiment_options_and_template_are_validated() {
        Args::command().debug_assert();
        let args = Args::try_parse_from([
            "wgpu-mine",
            "--bench",
            "--batch-sizes",
            "1,65,513",
            "-t",
            "2",
            "-m",
            "中文 [] []",
        ])
        .unwrap();
        assert_eq!(args.batch_sizes, [1, 65, 513]);
        let names = make_names(&args, 2).unwrap();
        let encoded = base16384::Base16384Utf8::encode(&0_u64.to_be_bytes());
        assert_eq!(names[0], format!("中文 {encoded} {encoded}"));
    }

    #[test]
    fn generation_rejects_invalid_lengths_and_overflow() {
        let mut args = Args::parse_from(["wgpu-mine"]);
        args.start = u64::MAX;
        assert!(make_names(&args, 2).is_err());
        args.start = 0;
        args.template = "无占位符".to_owned();
        assert!(make_names(&args, 1).is_err());
        args.template = format!("{}[]", "中".repeat(86));
        assert!(make_names(&args, 1).is_err());
    }

    #[test]
    fn median_and_mismatch_reporting() {
        assert_eq!(
            median(&[Duration::from_millis(4), Duration::from_millis(2)]),
            Duration::from_millis(3)
        );
        assert!(check_equal(&[1], &[2]).unwrap_err().contains("状态下标 0"));
        assert!(check_equal(&[], &[0]).is_err());
    }
}

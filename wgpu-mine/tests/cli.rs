use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use wgpu_mine::core;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct OwnedDir(PathBuf);
impl OwnedDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "namerena-e2e-{}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf { self.0.join(name) }
}
impl Drop for OwnedDir {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}
fn run(args: &[String]) -> std::process::Output { Command::new(env!("CARGO_BIN_EXE_wgpu-mine")).args(args).output().unwrap() }
fn base(path: &Path) -> Vec<String> {
    [
        "--backend",
        "cpu",
        "--team",
        "shenjack",
        "-p",
        "0",
        "-x",
        "0",
        "-d",
        "0",
        "-n",
        "number",
        "-t",
        "2",
        "-o",
    ]
    .into_iter()
    .map(str::to_owned)
    .chain([path.display().to_string()])
    .collect()
}
fn rows(path: &Path) -> Vec<csv::StringRecord> {
    csv::Reader::from_path(path).unwrap().records().collect::<Result<_, _>>().unwrap()
}
fn success(result: &std::process::Output) {
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
}

#[test]
fn cpu_cli_covers_ids_and_csv_special_characters() {
    let dir = OwnedDir::new();
    let path = dir.path("all.csv");
    let mut args = base(&path);
    let template = format!(
        "中文😀,双引号{}与换行{}{} [] []",
        char::from(34),
        char::from(13),
        char::from(10)
    );
    let team = format!("队,伍{}{}{}😀", char::from(34), char::from(13), char::from(10));
    args[3] = team.to_owned();
    args.extend([
        "--start".to_owned(),
        "7".to_owned(),
        "--end".to_owned(),
        "520".to_owned(),
        "-b".to_owned(),
        "257".to_owned(),
        "-m".to_owned(),
        template.to_owned(),
    ]);
    success(&run(&args));
    let values = rows(&path);
    assert_eq!(values.len(), 513);
    for (index, row) in values.iter().enumerate() {
        let id = 7 + index as u64;
        let name = template.replace("[]", &id.to_string());
        assert_eq!(row.len(), 16);
        assert_eq!(&row[0], format!("{name}@{team}"));
        assert_eq!(&row[1], id.to_string());
        assert_eq!(&row[2], name);
        assert_eq!(&row[3], team);
        let expected = core::evaluate(&team, &name);
        for (column, prop) in expected.props.iter().enumerate() {
            assert_eq!(row[4 + column].parse::<u32>().unwrap(), *prop);
        }
        assert!((row[13].parse::<f64>().unwrap() - expected.xp).abs() < 1e-5);
        assert!((row[14].parse::<f64>().unwrap() - expected.xd).abs() < 1e-5);
    }
}

#[test]
fn empty_range_and_no_hits_still_have_header() {
    let dir = OwnedDir::new();
    for (index, options) in [vec!["--start", "17", "--end", "17"], vec!["--count", "257"]].into_iter().enumerate() {
        let path = dir.path(&format!("empty-{index}.csv"));
        let mut args = base(&path);
        if index == 1 {
            args[5] = u32::MAX.to_string();
        }
        args.extend(options.into_iter().map(str::to_owned));
        success(&run(&args));
        assert!(rows(&path).is_empty());
        let mut reader = csv::Reader::from_path(&path).unwrap();
        assert_eq!(reader.headers().unwrap().len(), 16);
    }
}

#[test]
fn maximum_exclusive_range_and_thread_counts_are_consistent() {
    let dir = OwnedDir::new();
    let mut outputs = Vec::new();
    for threads in [1, 2, 6] {
        let path = dir.path(&format!("tail-{threads}.csv"));
        let mut args = base(&path);
        args.extend([
            "--start".to_owned(),
            (u64::MAX - 5).to_string(),
            "--end".to_owned(),
            u64::MAX.to_string(),
            "-b".to_owned(),
            "3".to_owned(),
            "--thread-count".to_owned(),
            threads.to_string(),
        ]);
        // clap拒绝重复同一参数；替换基础参数中的线程值。
        args.truncate(args.len() - 2);
        args[13] = threads.to_string();
        success(&run(&args));
        outputs.push(rows(&path));
    }
    assert_eq!(outputs[0].len(), 5);
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[1], outputs[2]);
    assert_eq!(outputs[0][4][1].parse::<u64>().unwrap(), u64::MAX - 1);
}

#[test]
fn invalid_arguments_and_existing_output_fail_without_overwrite() {
    let dir = OwnedDir::new();
    for (index, options) in [
        vec!["-b", "0"],
        vec!["-t", "0"],
        vec!["-r", "0"],
        vec!["--start", "10", "--end", "9"],
        vec!["--start", "18446744073709551615", "--count", "1"],
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path(&format!("invalid-{index}.csv"));
        let mut args = vec![
            "--backend".to_owned(),
            "cpu".to_owned(),
            "--team".to_owned(),
            "x".to_owned(),
            "-o".to_owned(),
            path.display().to_string(),
        ];
        args.extend(options.into_iter().map(str::to_owned));
        assert!(!run(&args).status.success());
        assert!(!path.exists());
    }
    let path = dir.path("existing.csv");
    fs::write(&path, b"keep-user-data").unwrap();
    let mut args = base(&path);
    args.extend(["--count".to_owned(), "1".to_owned()]);
    assert!(!run(&args).status.success());
    assert_eq!(fs::read(&path).unwrap(), b"keep-user-data");
}

#[test]
fn stdout_csv_has_no_log_pollution() {
    let path = Path::new("-");
    let mut args = base(path);
    args.extend(["--count".to_owned(), "33".to_owned()]);
    let result = run(&args);
    success(&result);
    let mut csv = csv::Reader::from_reader(result.stdout.as_slice());
    assert_eq!(csv.headers().unwrap().len(), 16);
    assert_eq!(csv.records().count(), 33);
    assert!(!result.stderr.is_empty());
}

#[test]
#[ignore = "需要GPU：真实CLI的范围、评分、CSV和CPU对照"]
fn gpu_cli_output_and_e2e_benchmark_match_cpu() {
    let dir = OwnedDir::new();
    let cpu = dir.path("cpu.csv");
    let gpu = dir.path("gpu.csv");
    let mut args = base(&cpu);
    args.extend([
        "--start".to_owned(),
        "9".to_owned(),
        "--count".to_owned(),
        "513".to_owned(),
        "-b".to_owned(),
        "65".to_owned(),
        "-m".to_owned(),
        format!("中文,{}[]{}", char::from(34), char::from(34)),
    ]);
    success(&run(&args));
    args[1] = "gpu".to_owned();
    args[15] = gpu.display().to_string();
    args.push("--verify".to_owned());
    success(&run(&args));
    assert_eq!(rows(&cpu), rows(&gpu));
    let metrics = dir.path("metrics.csv");
    let bench = vec![
        "--team".to_owned(),
        "shenjack".to_owned(),
        "--e2e-bench".to_owned(),
        "--count".to_owned(),
        "129".to_owned(),
        "--rounds".to_owned(),
        "1".to_owned(),
        "--warmup".to_owned(),
        "0".to_owned(),
        "-p".to_owned(),
        "0".to_owned(),
        "-x".to_owned(),
        "0".to_owned(),
        "-d".to_owned(),
        "0".to_owned(),
        "-b".to_owned(),
        "65".to_owned(),
        "--metrics".to_owned(),
        metrics.display().to_string(),
        "--output-dir".to_owned(),
        dir.path("bench").display().to_string(),
    ];
    success(&run(&bench));
    let records = rows(&metrics);
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|row| &row[4] == "129" && &row[5] == "129"));
}

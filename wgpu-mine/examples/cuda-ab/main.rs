#[cfg(all(windows, target_arch = "x86_64"))]
mod bench;
#[cfg(all(windows, target_arch = "x86_64"))]
mod cuda;

fn main() -> std::process::ExitCode {
    #[cfg(all(windows, target_arch = "x86_64"))]
    match bench::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("CUDA A/B：{e}");
            std::process::ExitCode::FAILURE
        }
    }
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    {
        eprintln!("此独立 CUDA 基准需要 Windows x86_64。");
        std::process::ExitCode::FAILURE
    }
}

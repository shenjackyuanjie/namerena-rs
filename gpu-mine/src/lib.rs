//! GPU 测号核心与独立 CPU 金标准。
pub mod core;
pub mod cpu;
#[cfg(all(windows, target_arch = "x86_64"))]
pub mod cuda;
pub mod gpu;
pub mod gpu_input;
pub mod input;
pub mod mining_gpu;
pub mod output;
pub mod pool;

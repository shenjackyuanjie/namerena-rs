//! 控制台中断仅设置原子标志，I/O、日志与 join 始终在正常线程执行。
use std::sync::atomic::{AtomicBool, Ordering};
static REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn requested() -> bool { REQUESTED.load(Ordering::Relaxed) }

#[cfg(windows)]
pub fn install() -> std::io::Result<()> {
    // Win32 控制台回调不依赖额外 Cargo feature；签名对应 SetConsoleCtrlHandler。
    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    unsafe extern "system" fn handler(event: u32) -> i32 {
        if event == 0 || event == 1 {
            REQUESTED.store(true, Ordering::Relaxed);
            1
        } else {
            0
        }
    }
    // 回调为静态函数；它只访问静态 AtomicBool，不持有生命周期或锁。
    if unsafe { SetConsoleCtrlHandler(Some(handler), 1) } == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
pub fn install() -> std::io::Result<()> {
    unsafe extern "C" {
        fn signal(number: i32, handler: usize) -> usize;
    }
    extern "C" fn handler(_: i32) { REQUESTED.store(true, Ordering::Relaxed); }
    // SIGINT/SIGTERM 的处理器只做无锁原子 store，保留默认的其他信号行为。
    for number in [2, 15] {
        if unsafe { signal(number, handler as *const () as usize) } == usize::MAX {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(not(any(windows, unix)))]
pub fn install() -> std::io::Result<()> { Ok(()) }

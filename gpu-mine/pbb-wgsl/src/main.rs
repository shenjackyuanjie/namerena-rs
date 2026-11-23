mod front;

use std::{ffi::CString, os::raw::c_char, process::ExitCode};

extern "C" {
    fn pbb_wgsl_main(argc: i32, argv: *const *const c_char) -> i32;
}

fn run() -> Result<u8, String> {
    let args: Vec<CString> = std::env::args_os()
        .map(|arg| {
            let arg = arg.into_string().map_err(|_| "PBB 参数需要 UTF-8")?;
            CString::new(arg).map_err(|_| "PBB 参数不能包含 NUL")
        })
        .collect::<Result<_, _>>()?;
    let pointers: Vec<_> = args.iter().map(|arg| arg.as_ptr()).collect();
    let argc = i32::try_from(pointers.len()).map_err(|_| "命令行参数过多")?;
    // 参数及字符串在调用期间有效，C++ 只读取它们。
    let status = unsafe { pbb_wgsl_main(argc, pointers.as_ptr()) };
    u8::try_from(status).map_err(|_| format!("PBB 退出状态无效：{status}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(status) => ExitCode::from(status),
        Err(error) => {
            eprintln!("PBB WGSL：{error}");
            ExitCode::FAILURE
        }
    }
}

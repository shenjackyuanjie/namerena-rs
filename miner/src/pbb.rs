use std::{ffi::CString, os::raw::c_char};

unsafe extern "C" {
    fn tswn_pbb_main(argc: i32, argv: *const *const c_char) -> i32;
}

pub(crate) fn run() -> i32 {
    match dispatch() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("PBB CPU：{error}");
            2
        }
    }
}

fn dispatch() -> Result<i32, &'static str> {
    // 子命令名 pbb 作为 argv[0]，其余参数保持与 GPU 入口相同。
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| {
            let arg = arg.into_string().map_err(|_| "参数需要 UTF-8")?;
            CString::new(arg).map_err(|_| "参数不能含 NUL")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let argv: Vec<_> = args.iter().map(|arg| arg.as_ptr()).collect();
    let argc = i32::try_from(argv.len()).map_err(|_| "命令行参数过多")?;
    // 参数及指针数组在 C++ 入口返回前一直有效，不跨线程借用或转移所有权。
    Ok(unsafe { tswn_pbb_main(argc, argv.as_ptr()) })
}

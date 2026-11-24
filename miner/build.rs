fn main() {
    #[cfg(feature = "native-pbb")]
    build_pbb();
}

#[cfg(feature = "native-pbb")]
fn build_pbb() {
    use std::{env, fs, path::PathBuf};

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let upstream = env::var_os("PBB_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../../PBB/source/pbb/native/src"));
    let shared = root.join("../gpu-mine/benchmarks/pbb-cuda");
    let local = root.join("native");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    assert_eq!(arch, "x86_64", "native-pbb 目前需要 x86-64");
    assert!(
        upstream.join("scoring_1124.hpp").is_file(),
        "未找到原版 PBB 评分头。请设置 PBB_SOURCE_DIR 指向 PBB 的 native/src（当前为 {}）",
        upstream.display()
    );

    println!("cargo:rerun-if-env-changed=PBB_SOURCE_DIR");
    for path in ["pbb-run.hpp", "pbb-config.hpp", "pbb-bench-types.hpp"] {
        println!("cargo:rerun-if-changed={}", shared.join(path).display());
    }
    println!("cargo:rerun-if-changed={}", local.display());
    for entry in fs::read_dir(&upstream).expect("读取原版 PBB 目录失败") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "hpp") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let mut cpp = cc::Build::new();
    cpp.cpp(true).std("c++17").include(upstream).include(shared).file(local.join("pbb.cpp"));
    // 只对整数 Q 门提速；浮点评分沿用 CUDA 对照的标量累加、取整语义。
    if cpp.get_compiler().is_like_msvc() {
        cpp.flag("/utf-8").flag("/fp:strict");
    } else {
        cpp.flag("-fno-fast-math").flag("-ffp-contract=off");
    }
    cpp.compile("tswn_pbb_native");
}

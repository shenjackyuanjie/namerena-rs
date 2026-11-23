fn main() {
    #[cfg(feature = "native-pbb")]
    build_pbb();
}

#[cfg(feature = "native-pbb")]
fn build_pbb() {
    use std::{env, fs, path::PathBuf};

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let pbb = env::var_os("PBB_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("../../../PBB/source/pbb/native/src"));
    let common = root.join("../benchmarks/pbb-cuda");
    let scorer = pbb.join("scoring_1124.hpp");
    let source = fs::read_to_string(&scorer).unwrap_or_else(|error| {
        panic!(
            "读取 {} 失败：{error}。请将 PBB_SOURCE_DIR 指向 PBB 的 native/src。",
            scorer.display()
        );
    });
    let gate = source.split_once("static inline int gate_level_of(").expect("PBB 一级 Q 门签名变化").1;
    let cutoff: u8 = gate
        .split_once("if (P7 < ")
        .and_then(|(_, rest)| rest.split_once(") return 0;"))
        .and_then(|(value, _)| value.parse().ok())
        .filter(|&value| value <= 63)
        .expect("PBB 一级 Q 门 P7 截止值变化");

    println!("cargo:rerun-if-env-changed=PBB_SOURCE_DIR");
    for file in fs::read_dir(&pbb).expect("读取 PBB 头文件目录失败") {
        let file = file.unwrap().path();
        if file.extension().is_some_and(|ext| ext == "hpp") {
            println!("cargo:rerun-if-changed={}", file.display());
        }
    }
    for file in ["pbb-run.hpp", "pbb-config.hpp", "pbb-bench-types.hpp"] {
        println!("cargo:rerun-if-changed={}", common.join(file).display());
    }
    println!("cargo:rerun-if-changed=src/native.cpp");
    let cutoff = cutoff.to_string();
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .include(common)
        .include(pbb)
        .file(root.join("src/native.cpp"))
        .define("PBB_MIN_P7", cutoff.as_str());
    if cpp.get_compiler().is_like_msvc() {
        cpp.flag("/utf-8").flag("/fp:strict");
    } else {
        cpp.flag("-fno-fast-math").flag("-ffp-contract=off");
    }
    cpp.compile("pbb_wgsl_native");
}

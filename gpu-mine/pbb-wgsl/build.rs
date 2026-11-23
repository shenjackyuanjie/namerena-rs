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
    guard_upstream(
        &source,
        &fs::read_to_string(pbb.join("name.hpp")).unwrap(),
        &fs::read_to_string(pbb.join("utils.hpp")).unwrap(),
    );
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
    println!("cargo:rerun-if-changed=src/prefilter.wgsl");
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

#[cfg(feature = "native-pbb")]
fn guard_upstream(scoring: &str, name: &str, utils: &str) {
    // 这些是定点判定的证明前提，不是模型数据哈希。忽略注释、空白；上游语义改动时拒绝误筛。
    fn check(src: &str, label: &str, begin: &str, end: &str, last: bool, expected: u64) {
        let at = if last { src.rfind(begin) } else { src.find(begin) }.unwrap_or_else(|| panic!("PBB {label} 的起始标记变化"));
        let src = &src[at..];
        let end = src.split_once(end).unwrap_or_else(|| panic!("PBB {label} 的结束标记变化")).0;
        let hash = end
            .lines()
            .flat_map(|line| line.split("//").next().unwrap().bytes())
            .filter(|b| !b.is_ascii_whitespace())
            .fold(0xcbf29ce484222325u64, |a, b| (a ^ u64::from(b)).wrapping_mul(0x100000001b3));
        assert_eq!(hash, expected, "PBB {label} 有变化，须重新验证 WGSL G3 的精度边界");
    }
    check(
        scoring,
        "G3 比较",
        "static inline bool skill_ok_of(",
        "\n}",
        false,
        0x5778896b97dafc54,
    );
    check(
        scoring,
        "HP 分支",
        "if (name_obj.V < 250) sort10_net",
        "  int V = name_obj.V;",
        false,
        0xce115a85c010d6b2,
    );
    check(
        scoring,
        "SP1 特征",
        "  double xp_x[46] = {};",
        "  // ---- Step 4b:",
        false,
        0x543b3949e6fcb540,
    );
    check(
        scoring,
        "G3 输入与档位",
        "  // 隐匿(xp_x[42]",
        "  const unsigned fam_mask",
        false,
        0xa6ef959fad476b73,
    );
    check(
        name,
        "PRGA",
        "inline u8_t m()",
        "  // ===== load_team()",
        false,
        0x223f53ec6ea558b1,
    );
    check(
        name,
        "标量技能",
        "void calc_skills(const char *name) {",
        "  static void calc_skills_batch4",
        true,
        0x468190051a593292,
    );
    check(
        utils,
        "字节比较交换",
        "static inline void cond_swap(",
        "\n}",
        false,
        0x11ce102e57e5b74d,
    );

    // 对每个二值输入核对中间四项和。min/max 可和二值化交换；逐阈值求和推广到 0..63。
    fn pairs(src: &str, begin: &str, prefix: &str) -> Vec<(usize, usize)> {
        let lines = src.split_once(begin).unwrap().1.split_once("#undef").unwrap().0;
        let ops: Vec<_> = lines
            .split(prefix)
            .skip(2)
            .map(|item| {
                let (a, b) = item.split_once(')').unwrap().0.split_once(',').unwrap();
                (a.trim().parse().unwrap(), b.trim().parse().unwrap())
            })
            .collect();
        assert!(!ops.is_empty());
        ops
    }
    let original = pairs(utils, "static inline void sort10(u8_t* arr)", "SWAP(");
    let oracle = pairs(scoring, "static inline void sort10_net(u8_t* b)", "CEN(");
    assert_eq!((original.len(), oracle.len()), (39, 29));
    for mask in 0u32..1024 {
        let initial: [u32; 10] = std::array::from_fn(|i| (mask >> i) & 1);
        let mut sorted = initial;
        sorted.sort_unstable();
        for net in [&original, &oracle] {
            let mut out = initial;
            for &(i, j) in net {
                if out[i] > out[j] {
                    out.swap(i, j);
                }
            }
            assert_eq!(out[3..7].iter().sum::<u32>(), sorted[3..7].iter().sum(), "PBB HP 网络变化");
        }
    }
    // 着色器中的 Q 排序必须和同一个已审查网络一致。
    let shader = include_str!("src/prefilter.wgsl");
    let actual: Vec<(usize, usize)> = shader
        .split("compare_swap(&hp,")
        .skip(1)
        .map(|item| {
            let (a, b) = item.split_once(')').unwrap().0.split_once(',').unwrap();
            (
                a.trim().trim_end_matches('u').parse().unwrap(),
                b.trim().trim_end_matches('u').parse().unwrap(),
            )
        })
        .collect();
    assert_eq!(actual, oracle, "WGSL 的 HP 网络与 PBB 不同");
}

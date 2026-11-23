"""从本机 PBB 原始评分头生成 CUDA device 评分头；生成物不入库。"""

import argparse
import hashlib
import re
from pathlib import Path

SCORE_MODELS = ("MODEL_QP0", "MODEL_QD0")
EXTRA_MODELS = (
    "MODEL_PP0", "MODEL_PD0", "MODEL_CQD0", "MODEL_CQD3", "MODEL_CP0",
    "MODEL_FP0", "MODEL_WC0", "MODEL_FS0", "MODEL_PJ0",
)
DEVICE_FUNCTIONS = {
    "qp_shadow_deduction", "score_normal_pass", "score_output_keep", "dot46",
    "build_x2", "sp_eval_plan", "ptt_model_of", "ptt_pass",
    "ptt_fixed_of", "get43_from_base", "shadowi2_of", "ptt_score_max", "skill_ok_of",
    "fam_t1_of", "fam_gate_ok", "fam_pass_mask", "sort10_net", "gate_level_of", "score_full",
}


def exactly(pattern, replacement, text, count=1, flags=re.M):
    """上游形状变化时停止，避免静默漏改或少移植一段。"""
    result, actual = re.subn(pattern, replacement, text, flags=flags)
    if actual != count:
        raise ValueError(f"匹配 {pattern!r} 应为 {count} 项，实为 {actual} 项")
    return result


def find_once(text, needle):
    if text.count(needle) != 1:
        raise ValueError(f"上游标记不唯一：{needle!r}")
    return text.index(needle)


def remove_between(text, begin, end):
    i, j = find_once(text, begin), find_once(text, end)
    if i >= j:
        raise ValueError(f"上游分区倒序：{begin!r} 到 {end!r}")
    return text[:i] + text[j:]


def check_numbers(body, expected, context):
    tokens = [token.strip() for token in body.split(",") if token.strip()]
    limit = float.fromhex("0x1.fffffep127")  # 模型权重会被存成有限 binary32。
    if len(tokens) != expected or not all(abs(float(token)) <= limit for token in tokens):
        raise ValueError(f"{context} 数组不再包含 {expected} 个有限数值")


def device_models(source, expected):
    models = list(re.finditer(
        r"^inline const float (MODEL_[A-Z0-9_]+)\[1124\] = \{\n(.*?)^\};$",
        source, re.M | re.S,
    ))
    if tuple(m[1] for m in models) != expected:
        raise ValueError(f"PBB 模型列表变化：{[m[1] for m in models]}")
    for m in models:
        check_numbers(m[2], 1124, m[1])
    return exactly(
        r"^inline const float (?=MODEL_[A-Z0-9_]+\[1124\] = \{)",
        "static __device__ const float ", source, len(expected),
    )


def replace_min(match):
    args = [arg.strip() for arg in match[1].split(",")]
    if len(args) == 3:
        a, b, c = args
        return f"scoring_byte_min(scoring_byte_min({a}, {b}), {c})"
    if len(args) == 4:
        a, b, c, d = args
        return f"scoring_byte_min(scoring_byte_min({a}, {b}), scoring_byte_min({c}, {d}))"
    raise ValueError(f"未支持的字节最小值组：{match[0]}")


def sparse_polynomial(scoring):
    # 先确认标量核的实际计算，再用原表生成权重索引；模型重排时不能静默沿用此变换。
    find_once(scoring, "xp[y] = e.is_pair ? x[e.p] * x[e.q] : x[e.p];")
    find_once(scoring, "double s = m[0];\n  for (int i = 0; i < 1123; i++) s += xa[i] * m[i + 1];")
    table = re.search(r"POLY_TABLE_1124\[1123\] = \{(.*?)^\};", scoring, re.M | re.S)
    if not table:
        raise ValueError("未找到原版多项式项表")
    terms = [tuple(map(int, t)) for t in re.findall(r"\{(\d+),(\d+),(\d+)\}", table[1])]
    pairs = terms[46:]
    if len(terms) != 1123 or terms[:46] != [(0, i, 0) for i in range(46)]:
        raise ValueError("原版多项式一阶索引变化")
    if pairs != sorted(set(pairs)) or any(t != 1 or not 0 <= i <= j < 46 for t, i, j in pairs):
        raise ValueError("原版多项式二阶索引顺序变化")
    rows = [[0] * 46 for _ in range(46)]
    for k, (_, i, j) in enumerate(pairs, 47):  # bias 和 46 项一阶项之后
        rows[i][j] = k
    indices = "static __device__ const unsigned short sparse_index[46][46] = {\n"
    indices += "".join("  {" + ",".join(map(str, row)) + "},\n" for row in rows) + "};\n"
    helper = r'''
// 未出现在原项表中的交互项对应 0。两个循环均按原特征下标递增，
// 仅略过零贡献；QP/QD 每项仍以 double 相乘、累加后收回 float。
template<class Accum>
static __device__ inline Accum sparse_dot(const double* x, const float* w) {
  u64_t active = 0;
  for (int i = 0; i < 46; ++i) if (x[i] != 0.0) active |= 1ULL << i;
  Accum sum = w[0];
  for (u64_t m = active; m; m &= m - 1) {
    const int i = __ffsll(m) - 1;
    sum += x[i] * w[i + 1];
  }
  for (u64_t m = active; m; m &= m - 1) {
    const int i = __ffsll(m) - 1;
    const double a = x[i];
    for (u64_t later = m; later; later &= later - 1) {
      const int j = __ffsll(later) - 1;
      const int at = sparse_index[i][j];
      if (at) sum += (a * x[j]) * w[at];
    }
  }
  return sum;
}

'''
    start = find_once(scoring, "// ===== PolyEntry1124:")
    stop = find_once(scoring, "// ---- 数据块 2/4:")
    scoring = scoring[:start] + indices + helper + scoring[stop:]
    scoring = remove_between(scoring, "// ===== poly_1124:", "// ===== 46 维点积")

    # 普通评分保留所有族条件、取整与 any 短路；只把展开后的数组换成原特征。
    scoring = remove_between(scoring, "  double a1[1123], a2[1123];", "  auto C = [](double v)")
    for weights, incr, ptr in [("MODEL_QP0", "++i", "* a"), ("W", "i++", " *a")]:
        dense = f"        const double{ptr} = get1();\n        float s = {weights}[0];\n"
        dense += f"        for (int i = 0; i < 1123; {incr}) s += a[i] * {weights}[i + 1];"
        scoring = exactly(re.escape(dense), f"        float s = sparse_dot<float>(xp_x, {weights});", scoring)
    scoring = exactly(r"\bget1\(\)", "xp_x", scoring, 3)
    scoring = exactly(r"\bget2\(\)", "x2", scoring, 5)

    # PTT 仍按原顺序枚举同一组扰动、重判同一组族门、取最大值。
    scoring = exactly(r"^  double xa\[1123\], xc\[50\];$", "  double xc[50];", scoring)
    scoring = exactly(r"^  poly_1124\(xa, x\);\n  double best = dot_1124_f\(xa, W\);",
                      "  double best = sparse_dot<double>(x, W);", scoring)
    scoring = exactly(r"^( +)poly_1124\(xa, xc\);\n\1(double s|s) = dot_1124_f\(xa, W\);",
                      r"\1\2 = sparse_dot<double>(xc, W);", scoring, 3)
    return exactly(r"\bdot_1124_f\(", "sparse_dot<double>(", scoring, 9)


def convert(scoring, extra):
    # 两个输入文件只读取不改动。默认计划、旧测试包装与 CPU batch4 无设备调用者。
    scoring = scoring[:find_once(scoring, "// ===== score_full_loaded_batch4:")]
    scoring = remove_between(scoring, "// 默认计划:", "// ===== ScoreResult:")
    scoring = remove_between(scoring, "// ===== sp_eval_extra:", "// ===== Q 门控")
    scoring = exactly(r'^#pragma once\n', '', scoring)
    scoring = exactly(r'^#include .+\n', '', scoring, 4)
    extra = exactly(r'^#pragma once\n', '', extra)
    scoring = exactly(
        r'^#if PBB_HAS_AVX512\n.*?^#else\n(.*?)^#endif\n',
        r'\1', scoring, 2, flags=re.M | re.S,
    )
    scoring = exactly(
        r"^// plan: 评分计划 \(nullptr ⇒ plan_default\(\), 即全 10 族\)。$",
        "// 设备端 plan 由调用方给定，必须非空。", scoring,
    )
    scoring = exactly(r"const ScorePlan\* plan = nullptr\) \{", "const ScorePlan* plan) {", scoring)
    scoring = exactly(r"^  const ScorePlan& pl = plan \? \*plan : plan_default\(\);$",
                      "  const ScorePlan& pl = *plan;", scoring)

    # 所有模型和门控动态下标表仍为同源常量，放 GPU 全局只读对象，不用 __constant__。
    scoring = device_models(scoring, SCORE_MODELS)
    extra = device_models(extra, EXTRA_MODELS)
    scoring = sparse_polynomial(scoring)
    scoring = exactly(r"^constexpr double (?=(?:GATE_SK_|FAM_W_)[A-Z0-9_]+\[46\])",
                      "static __device__ const double ", scoring, 19)
    scoring = exactly(r"^static const FamGate (?=FAM_GATE\[M_N\])",
                      "static __device__ const FamGate ", scoring)
    para = r"^  static const double skill_para\[25\] = (\{\n.*?^  \});\n"
    tables = re.findall(para, scoring, re.M | re.S)
    if len(tables) != 2 or re.sub(r"\s+", "", tables[0]) != re.sub(r"\s+", "", tables[1]):
        raise ValueError("SP2 两处 skill_para 参数不再完全一致")
    check_numbers(tables[0][1:-1], 25, "SP2 skill_para")
    extra += "\nstatic __device__ const double scoring_sp2_skill_para[25] = " + tables[0] + ";\n"
    scoring = exactly(para, "", scoring, 2, flags=re.M | re.S)
    scoring = exactly(r"\bskill_para\[sk\]", "scoring_sp2_skill_para[sk]", scoring, 2)

    # GPU 不使用宿主专用的 initializer_list 或 std 数学函数。替换不改数值口径。
    scoring = exactly(r"std::min\(\{([^{}]+)\}\)", replace_min, scoring, 4)
    scoring = exactly(r"std::round\(", "::round(", scoring, 3)
    fn = r"^(static )?inline ((?:const )?(?:void|bool|double|int|unsigned|ScoreResult|float)(?:\*)? ([A-Za-z_]\w*)\()"
    matches = list(re.finditer(fn, scoring, re.M))
    if {m[3] for m in matches} != DEVICE_FUNCTIONS or len(matches) != len(DEVICE_FUNCTIONS):
        raise ValueError(f"PBB 评分函数签名变化：{[m[3] for m in matches]}")
    scoring = exactly(fn, lambda m: (m[1] or "") + "__device__ inline " + m[2], scoring, len(DEVICE_FUNCTIONS))
    # 注释保留上游解释；只检查最终有效 C++ 没有宿主专用符号。
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", scoring + extra, flags=re.S)
    if re.search(r"\b(_mm\w*|PBB_HAS_\w+|plan_default|score_full_loaded_batch4|poly_1124|dot_1124_f|POLY_TABLE_1124)\b|std::|__constant__", code):
        raise ValueError("仍存在宿主专用调用或非预期常量存储")
    return extra, scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="实际 PBB/source/pbb/native/src")
    parser.add_argument("--out", required=True, type=Path, help="隔离构建目录内的 pbb_device_scoring.hpp")
    args = parser.parse_args()
    src = args.source_dir.resolve(strict=True)
    if not src.is_dir():
        parser.error("--source-dir 必须是 PBB native src 目录")
    paths = [src / "scoring_1124.hpp", src / "sp_models_extra.hpp"]
    try:
        raw = [p.read_bytes() for p in paths]
        scoring, extra = (b.decode("utf-8-sig").replace("\r\n", "\n") for b in raw)
        model_header, body = convert(scoring, extra)
    except (OSError, ValueError) as err:
        parser.error(str(err))
    digests = [hashlib.sha256(b).hexdigest() for b in raw]
    banner = "\n".join(f"// {p.name} SHA-256: {d}" for p, d in zip(paths, digests))
    header = f'''#pragma once
// 从本机 PBB 源数据生成；输出只供构建，不应提交版本库。
{banner}
// 与 PBB 标量路径作精确对照时，编译 CUDA 请用 --fmad=false。
#include <stdint.h>
#include <cmath>
#include <cstring>
#include "pbb_device_name.hpp"

namespace pbb_device {{

static __device__ inline u8_t scoring_byte_min(u8_t a, u8_t b) {{ return b < a ? b : a; }}

{model_header}
{body}
}}  // namespace pbb_device
'''
    out = args.out.resolve()
    if out in paths or out.name != "pbb_device_scoring.hpp":
        parser.error("--out 必须指向隔离生成的 pbb_device_scoring.hpp")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header, encoding="utf-8", newline="\n")
    print(f"已生成 {out}（11 张评分模型、19 张门控表，使用 CUDA 全局只读存储）")


if __name__ == "__main__":
    main()

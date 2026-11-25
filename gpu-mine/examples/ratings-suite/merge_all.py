"""把多轮测号开箱结果合并成一张大 CSV。

每个输入目录需要：
  scored.jsonl    单人 pf 评分（qp/pp/pd/qd/sum）
  scan/results.csv  原始候选（用来对齐 id 与顺序）
可选：
  cqp.jsonl       超强评（单人组固定靶池）
  duo.jsonl       双人 cqp
  ratings.jsonl   五套配队评分

用 full_name 作主键：不同轮次 / 不同 seed 的 id 区间会重叠，名字不会。
"""

import argparse, csv, json, sys
from decimal import Decimal
from pathlib import Path

PF = [("qp", "强评"), ("qd", "强单"), ("pp", "普评"), ("pd", "普单"), ("sum", "四项总分")]
PAIR_KEYS = ["双人cqp", "刺评", "辅评", "无刺评", "分身评", "配件评"]


def load_jsonl(p, key_alt=None):
    out = {}
    if not Path(p).exists():
        return out
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        name = d.get("full_name") or d.get("label")
        if name:
            out[name] = d
    return out


def num(v):
    if v is None or v == "":
        return ""
    try:
        d = Decimal(str(v))
    except Exception:
        return str(v)
    return str(int(d)) if d == d.to_integral_value() else format(d, "f").rstrip("0").rstrip(".")


def main():
    ap = argparse.ArgumentParser(description="合并多轮开箱结果")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    ap.add_argument("dirs", nargs="+", help="各轮输出目录（含 scored.jsonl）")
    ap.add_argument("--ratings-suffix", default="", help="若配队/双人结果在子目录，填它的名字")
    ap.add_argument("--ratings", default="", help="单独的双人评分目录（含 duo.jsonl / ratings.jsonl）")
    args = ap.parse_args()

    global_duo = load_jsonl(Path(args.ratings).resolve() / "duo.jsonl") if args.ratings else {}
    global_pair = load_jsonl(Path(args.ratings).resolve() / "ratings.jsonl") if args.ratings else {}

    rows = []
    for raw in args.dirs:
        path_part, _, label = raw.partition("#")
        d = Path(path_part).resolve()
        scored = load_jsonl(d / "scored.jsonl")
        if not scored:
            print(f"跳过（没有 scored.jsonl）：{d.name}", file=sys.stderr)
            continue
        cqp = load_jsonl(d / "cqp.jsonl")
        rdir = d / args.ratings_suffix if args.ratings_suffix else d
        duo = load_jsonl(rdir / "duo.jsonl")
        pair = load_jsonl(rdir / "ratings.jsonl")
        for name, sc in scored.items():
            row = {"来源": label or d.name, "id": sc.get("id"), "名字": name}
            for key, col in PF:
                row[col] = num(sc.get(key))
            c = cqp.get(name) or {}
            row["超强评"] = num(c.get("cqp") or c.get("avg_win_rate"))
            du = duo.get(name) or global_duo.get(name) or {}
            row["双人cqp"] = num(du.get("duo_cqp"))
            pr = pair.get(name) or global_pair.get(name) or {}
            for k in PAIR_KEYS[1:]:
                row[k] = num(pr.get(k))
            rows.append(row)

    fields = ["来源", "id", "名字", "强评", "强单", "普评", "普单", "四项总分", "超强评"] + PAIR_KEYS
    outp = Path(args.out).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"rows": len(rows), "dirs": len(args.dirs), "out": str(outp)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

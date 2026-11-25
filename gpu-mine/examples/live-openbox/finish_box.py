"""扫描停下后，把 results.csv 里还没开箱的候选补齐 pf 评分。

用法：live_openbox.py 因为各种原因提前结束时，用这个脚本把剩下的候选补完，
再决定要不要对头部跑精评、双人 cqp 或配队评分。
"""

import argparse, csv, json, subprocess as sp, sys, time
from decimal import Decimal
from pathlib import Path

METRICS = ("qp", "pp", "pd", "qd", "sum")
CREATE_FLAGS = getattr(sp, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def load_names(path):
    return [json.loads(l)["full_name"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def scan_candidates(csv_path):
    rows = []
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("keep") == "1":
                rows.append({"id": int(r["id"]), "full_name": r["full_name"],
                             "model_qp": r.get("qp"), "model_qd": r.get("qd")})
    return rows


def pf(cli, names, n, threads):
    cmd = [cli, "namer-pf", "-n", str(n), "-t", str(threads), "--precision", "6"]
    for m in METRICS:
        cmd += ["--metric", m]
    ans = sp.run(cmd, input="\n".join(names) + "\n", encoding="utf-8",
                 capture_output=True, timeout=3600, check=True, creationflags=CREATE_FLAGS)
    got = {name: {} for name in names}
    for line in ans.stdout.splitlines():
        if " " not in line or ":" not in line:
            continue
        name, score = line.rsplit(" ", 1)
        key, value = score.split(":", 1)
        if name in got and key in METRICS:
            got[name][key] = value
    for name in names:
        if set(got[name]) != set(METRICS):
            raise RuntimeError(f"pf 输出缺少指标：{name}")
    return got


def main():
    ap = argparse.ArgumentParser(description="补齐未开箱候选的 pf 评分")
    ap.add_argument("--scan-csv", required=True, help="pbb-cuda 的 results.csv")
    ap.add_argument("--scored", required=True, help="已有的 scored.jsonl")
    ap.add_argument("--candidates", required=True, help="已有的 candidates.jsonl")
    ap.add_argument("--tswn-cli", required=True)
    ap.add_argument("--n", type=int, default=10000, help="场次；10000 = 精评")
    ap.add_argument("--thread", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    have = set(load_names(args.scored))
    rows = scan_candidates(args.scan_csv)
    todo = [r for r in rows if r["full_name"] not in have]
    print(f"候选 {len(rows)}，已评分 {len(have & {r['full_name'] for r in rows})}，待补 {len(todo)}", flush=True)

    appended = 0
    t0 = time.time()
    with Path(args.scored).open("a", encoding="utf-8", buffering=1) as out:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            names = [r["full_name"] for r in chunk]
            got = pf(args.tswn_cli, names, args.n, args.thread)
            for r in chunk:
                vals = got[r["full_name"]]
                out.write(json.dumps({"id": r["id"], "full_name": r["full_name"],
                                      "model_qp": r["model_qp"], "model_qd": r["model_qd"],
                                      **{k: vals[k] for k in METRICS}}, ensure_ascii=False) + "\n")
                appended += 1
            print(f"{appended}/{len(todo)}  {round(time.time()-t0,1)}s", flush=True)
    print(json.dumps({"candidates": len(rows), "appended": appended,
                      "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

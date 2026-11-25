"""把一份 scored.jsonl 里的所有候选按指定场次重跑 pf，写新文件。

用于把早期跑过粗评（5000 场）的批次补齐成精评（10000 场），
不改原文件，输出到 --out。
"""

import argparse, json, subprocess as sp, sys, time
from pathlib import Path

METRICS = ("qp", "pp", "pd", "qd", "sum")
CREATE_FLAGS = getattr(sp, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def pf(cli, names, n, threads, flags=CREATE_FLAGS):
    cmd = [cli, "namer-pf", "-n", str(n), "-t", str(threads), "--precision", "6"]
    for m in METRICS:
        cmd += ["--metric", m]
    ans = sp.run(cmd, input="\n".join(names) + "\n", encoding="utf-8",
                 capture_output=True, timeout=3600, check=True, creationflags=flags)
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
            raise RuntimeError("pf 输出缺少指标：" + name)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tswn-cli", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--thread", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--normal-priority", action="store_true", help="不降优先级，独占机器时更快")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scored).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    if out.exists():
        raise SystemExit("输出已存在：" + str(out))
    done = 0
    t0 = time.time()
    with out.open("x", encoding="utf-8", buffering=1) as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            got = pf(args.tswn_cli, [r["full_name"] for r in chunk], args.n, args.thread,
                    0 if args.normal_priority else CREATE_FLAGS)
            for r in chunk:
                vals = got[r["full_name"]]
                f.write(json.dumps({"id": r["id"], "full_name": r["full_name"],
                                    "model_qp": r.get("model_qp"), "model_qd": r.get("model_qd"),
                                    **{k: vals[k] for k in METRICS}}, ensure_ascii=False) + "\n")
                done += 1
            print(str(done) + "/" + str(len(rows)) + "  " + str(round(time.time() - t0, 1)) + "s", flush=True)
    print(json.dumps({"rows": len(rows), "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

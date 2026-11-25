"""对一份名单跑双人评分：双人 cqp + 五套配队评分（pair）。

双人 cqp：把每个选手与固定基准号用 + 拼成二人组，对「双人组」靶池
（settings.toml 里 id=2）求加权平均胜率，0..100 百分比。

配队评分：选手 × 预设队友组合，对靶池求加权平均胜率，乘队友 factor 后
取前 head 个求和。五套预设来自 settings.toml：刺评 / 辅评 / 无刺评 /
分身评 / 配件评。

用法见同目录 README.md。
"""

import argparse, json, subprocess as sp, sys, time
from decimal import Decimal
from pathlib import Path

PRESETS = ("刺评", "辅评", "无刺评", "分身评", "配件评")


def score_lines(text):
    """openbox-cli 的输出版式：不缩进的首行是最终分，缩进行是队友明细。"""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        head, _, rest = line.partition(" ")
        try:
            value = Decimal(head)
        except Exception:
            continue
        out.append((value, rest.strip()))
    return out


def run_duo(ob, workdir, players, base, preset, n, threads):
    lines = [f"{p}+{base}" for p in players]
    tmp = Path(workdir) / "_duo_input.txt"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ans = sp.run([ob, "cqd", "--target-preset", str(preset), "-p", str(tmp),
                  "-n", str(n), "-t", str(threads), "--no-show-matchups"],
                 cwd=workdir, encoding="utf-8", capture_output=True, check=True)
    got = {}
    for value, label in score_lines(ans.stdout):
        for p in players:
            if label == f"{p}+{base}":
                got[p] = str(value)
                break
    return got


def run_pair(ob, workdir, players, preset, head, n, threads):
    tmp = Path(workdir) / "_pair_input.txt"
    tmp.write_text("\n".join(players) + "\n", encoding="utf-8")
    ans = sp.run([ob, "pair", "-p", str(tmp), "--teammate-preset", preset,
                  "--head", str(head), "-n", str(n), "-t", str(threads), "--detail", "top"],
                 cwd=workdir, encoding="utf-8", capture_output=True, check=True)
    return {label: str(value) for value, label in score_lines(ans.stdout)}


def main():
    ap = argparse.ArgumentParser(description="双人评分：cqp + 五套 pair")
    ap.add_argument("--players", required=True, help="每行一个完整名字")
    ap.add_argument("--base", required=True, help="双人 cqp 的固定基准号（完整名字）")
    ap.add_argument("--openbox", required=True, help="openbox-cli 可执行文件")
    ap.add_argument("--workdir", required=True, help="含 setting/settings.toml 的目录")
    ap.add_argument("--out", required=True, help="输出目录，必须不存在")
    ap.add_argument("--n", type=int, default=10000, help="场次；10000 = 精评")
    ap.add_argument("--thread", type=int, default=6)
    ap.add_argument("--target-preset", default="2", help="双人组靶池预设 id，默认 2")
    ap.add_argument("--head", type=int, default=4, help="pair 取前几组队友")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--skip-duo", action="store_true")
    ap.add_argument("--skip-pair", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}")
    out.mkdir(parents=True)
    players = [l.strip() for l in Path(args.players).read_text(encoding="utf-8").splitlines() if l.strip()]
    ob = str(Path(args.openbox).resolve())
    work = str(Path(args.workdir).resolve())
    stats = {"players": len(players), "n": args.n, "thread": args.thread}

    duo = {}
    if not args.skip_duo:
        t0 = time.time()
        duo = run_duo(ob, work, players, args.base, args.target_preset, args.n, args.thread)
        stats["duo_seconds"] = round(time.time() - t0, 1)
        with (out / "duo.jsonl").open("w", encoding="utf-8") as f:
            for p in players:
                f.write(json.dumps({"full_name": p, "duo_cqp": duo.get(p)}, ensure_ascii=False) + "\n")
        print(f"双人 cqp：{len(duo)}/{len(players)} 个，{stats['duo_seconds']} 秒", flush=True)

    pairs = {}
    if not args.skip_pair:
        for preset in [x for x in args.presets.split(",") if x.strip()]:
            t0 = time.time()
            got = run_pair(ob, work, players, preset.strip(), args.head, args.n, args.thread)
            pairs[preset.strip()] = got
            print(f"pair {preset}：{len(got)}/{len(players)} 个，{round(time.time()-t0,1)} 秒", flush=True)

    rows = []
    for p in players:
        row = {"full_name": p, "duo_cqp": duo.get(p)}
        for k, v in pairs.items():
            row[k] = v.get(p)
        rows.append(row)
    (out / "ratings.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    stats["records"] = len(rows)
    (out / "status.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())

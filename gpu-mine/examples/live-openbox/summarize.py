"""把 scored.jsonl / cqp.jsonl 汇成一张排序表。

用法：python summarize.py --out target/live-openbox-1t --top 50
"""

import argparse, csv, json
from decimal import Decimal
from pathlib import Path


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def label(v):
    d = Decimal(v)
    return str(int(d)) if d == d.to_integral_value() else format(d, 'f').rstrip('0').rstrip('.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='live_openbox.py 的输出目录')
    p.add_argument('--top', type=int, default=50)
    p.add_argument('--qp-min', default='6000')
    args = p.parse_args()
    base = Path(args.out)
    rows = load(base / 'scored.jsonl')
    # cqp.jsonl 有两种来源：live_openbox.py 自己写的 full_name/cqp，
    # 以及 openbox-cli --log 直接写的 label/avg_win_rate。
    cqp = {}
    for r in load(base / 'cqp.jsonl'):
        name = r.get('full_name') or r.get('label')
        value = r.get('cqp', r.get('avg_win_rate'))
        if name and value is not None:
            cqp[name] = value
    if not rows:
        raise SystemExit('没有 scored.jsonl')
    rows.sort(key=lambda r: (-Decimal(r['qp']), -Decimal(r['sum']), r['id']))
    gate = Decimal(args.qp_min)
    hits = [r for r in rows if Decimal(r['qp']) >= gate]
    lines = [
        f'# {base.name} 汇总',
        '',
        f'- 候选 {len(rows)} 个，全部跑满 pf 100%。',
        f'- pf QP ≥ {label(gate)}：**{len(hits)} 个**。',
        f'- 最高 QP {label(rows[0]["qp"])}，sum {label(rows[0]["sum"])}。',
        '',
        '| # | 实战 QP | sum | pp | pd | qd | 模型 QP | CQP | ID | 名字 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for i, r in enumerate(rows[:args.top], 1):
        lines.append(
            f'| {i} | {label(r["qp"])} | {label(r["sum"])} | {label(r["pp"])} | {label(r["pd"])} | '
            f'{label(r["qd"])} | {r.get("model_qp", "")} | {cqp.get(r["full_name"], "")} | '
            f'{r["id"]} | {r["full_name"]} |')
    (base / '汇总.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (base / '汇总.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'full_name', 'model_qp', 'model_qd',
                                          'qp', 'pp', 'pd', 'qd', 'sum', 'cqp'])
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, '') for k in w.fieldnames if k != 'cqp'},
                        'cqp': cqp.get(r['full_name'], '')})
    print(json.dumps({'candidates': len(rows), 'hits': len(hits),
                      'best_qp': label(rows[0]['qp']), 'md': str(base / '汇总.md')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

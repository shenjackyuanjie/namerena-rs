"""PBB CUDA 扫描 + 实时开箱（pf 满 100% + 可选 cqp 单人组）。

一个进程同时做三件事：
  1. 起 pbb-cuda 扫描，读 results.csv 新增行；
  2. 每个 keep=1 的候选立刻送 tswn-cli namer-pf 跑满 100% 场次；
  3. 按规则把候选送 openbox-cli cqd（单人组靶池）跑 cqp。

用法见同目录 README.md。所有产物落在一个新目录，不覆盖已有记录。
"""

import argparse, bisect, csv, ctypes, json, os, re, signal, subprocess as sp, sys, threading, time
from collections import deque

# 开箱子进程一律降到 BelowNormal：实测把对 GPU 扫描的干扰从 20% 压到 10.5%，
# 比单纯减线程有效得多（6 线程 normal 51.63 T/天 → BelowNormal 57.61 T/天）。
CREATE_FLAGS = getattr(sp, "BELOW_NORMAL_PRIORITY_CLASS", 0) if os.name == "nt" else 0
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

PF_METRICS = ('qp', 'pp', 'pd', 'qd', 'sum')


def now():
    return datetime.now().astimezone()


def dump(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def append(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    f.flush()


def miner_tail(log):
    if not log.exists():
        return '', {}
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size - 8192))
        lines = f.read().decode('utf-8', errors='replace').splitlines()
    last = next((l for l in reversed(lines) if 'next_id=' in l), '')
    return last, {k: int(v) for k, v in re.findall(r'\b(next_id|count|keep)=(\d+)', last)}


def pf_batch(cli, names, n, threads):
    """跑一批候选的 pf 满场实战，返回每个名字的指标字典。"""
    cmd = [str(cli), 'namer-pf', '-n', str(n), '-t', str(threads), '--precision', '6']
    for m in PF_METRICS:
        cmd += ['--metric', m]
    ans = sp.run(cmd, input='\n'.join(names) + '\n', encoding='utf-8',
                 capture_output=True, timeout=1800, check=True,
                 creationflags=CREATE_FLAGS)
    got = {name: {} for name in names}
    for line in ans.stdout.splitlines():
        if ' ' not in line or ':' not in line:
            continue
        name, score = line.rsplit(' ', 1)
        key, value = score.split(':', 1)
        if name in got and key in PF_METRICS:
            got[name][key] = value
    for name in names:
        if set(got[name]) != set(PF_METRICS):
            raise RuntimeError(f'pf 输出缺少指标：{name}')
    return got


def classify_cli_args(argv=None):
    p = argparse.ArgumentParser(description='PBB CUDA 扫描 + 实时开箱')
    p.add_argument('--miner', required=True, help='pbb-cuda 可执行文件')
    p.add_argument('--tswn-cli', required=True, help='tswn-cli 可执行文件（namer-pf）')
    p.add_argument('--openbox-cli', help='openbox-cli 可执行文件（cqd）；不给则跳过 cqp')
    p.add_argument('--plan', required=True, help='PBB 评分计划 .kv')
    p.add_argument('--out', required=True, help='本轮输出目录，必须不存在')
    p.add_argument('--start', type=int, default=0, help='起始 ID（含）')
    p.add_argument('--count', type=int, help='测号数量')
    p.add_argument('--seconds', type=int, help='运行秒数；与 --count 互斥')
    p.add_argument('--team', default='', help='队伍名')
    p.add_argument('--name-template', default='', help='含一个 [] 的名字模板')
    p.add_argument('--block', type=int, default=96)
    p.add_argument('--batch', type=int, default=16777216)
    p.add_argument('--pf-thread', type=int, default=6)
    p.add_argument('--pf-batch', type=int, default=8)
    p.add_argument('--pf-n', type=int, default=10000, help='pf 场次，10000 = 满 100%')
    p.add_argument('--shuffle', choices=('', 'affine', 'mix', 'feistel'), default='',
                   help='填充前的编号置换；空表示顺序枚举')
    p.add_argument('--seed', type=int, default=None, help='置换种子，配合 --shuffle')
    p.add_argument('--cqp-mode', choices=('off', 'all', 'qualified', 'top'), default='qualified',
                   help='off 不跑；all 全跑；qualified 按固定 QP；top 按滚动分位')
    p.add_argument('--cqp-min-qp', default='6000', help='qualified 模式的 pf QP 下限')
    p.add_argument('--cqp-top-percent', type=float, default=10.0,
                   help='top 模式只送 QP 前百分之几，默认 10')
    p.add_argument('--cqp-min-samples', type=int, default=500,
                   help='top 模式在样本不足时回退到 --cqp-min-qp')
    p.add_argument('--cqp-thread', type=int, default=6)
    p.add_argument('--cqp-preset', default='1', help='单人组靶池预设 id')
    p.add_argument('--cqp-workdir', default='', help='openbox-cli 的工作目录，需含 setting/settings.toml')
    p.add_argument('--cqp-n', type=int, default=10000, help='每靶场次，10000 = 满 100%')
    p.add_argument('--cqp-batch', type=int, default=8)
    p.add_argument('--report-interval', type=int, default=5, help='status.json 刷新秒数')
    p.add_argument('--read-ahead', type=int, default=512,
                   help='pf 队列低于这个长度时才继续读 results.csv；太小会让积压不可见')
    p.add_argument('--milestone-count', type=int, default=0,
                   help='每扫过这么多个编号打印并记录一条里程碑，0 关闭；1e11 就是 0.1 T')
    args = p.parse_args(argv)
    if args.count is not None and args.seconds is not None:
        p.error('--count 与 --seconds 只能给一个')
    if args.name_template and '[]' not in args.name_template:
        p.error('--name-template 必须含一个 []')
    return args


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f'输出目录已存在，拒绝覆盖：{out}')
    out.mkdir(parents=True)
    cfg = vars(args).copy()
    cfg['out'] = str(out)
    dump(out / 'config.json', cfg)

    begin = time.monotonic()
    start_wall = now()
    next_milestone = args.start + args.milestone_count if args.milestone_count else None
    milestones = []
    stats = {
        'phase': '扫描中', 'start_id': args.start, 'start_wall': start_wall.isoformat(),
        'planned_seconds': args.seconds, 'planned_count': args.count,
        'candidates': 0, 'scored': 0, 'cqp_done': 0, 'pf_backlog': 0, 'cqp_backlog': 0,
        'hits': 0, 'best_qp': None,
    }
    dump(out / 'status.json', stats)

    pf_queue = deque()
    cqp_queue = deque()
    qp_values = []
    qp_sorted = []
    lock = threading.Lock()
    stop = threading.Event()
    score_f = (out / 'scored.jsonl').open('x', encoding='utf-8', buffering=1)
    cand_f = (out / 'candidates.jsonl').open('x', encoding='utf-8', buffering=1)
    cqp_f = (out / 'cqp.jsonl').open('x', encoding='utf-8', buffering=1)

    def worker_cqp():
        players = out / 'cqp-players.txt'
        while not stop.is_set() or cqp_queue:
            if not cqp_queue:
                time.sleep(0.1)
                continue
            batch = []
            with lock:
                while cqp_queue and len(batch) < args.cqp_batch:
                    batch.append(cqp_queue.popleft())
            if not batch:
                continue
            names = [r['full_name'] for r in batch]
            players.write_text('\n'.join(names) + '\n', encoding='utf-8')
            try:
                ans = sp.run(
                    [args.openbox_cli, 'cqd', '--target-preset', args.cqp_preset,
                     '-p', str(players), '-n', str(args.cqp_n), '-t', str(args.cqp_thread),
                     '--no-show-matchups'],
                    encoding='utf-8', capture_output=True, timeout=600, check=True,
                    creationflags=CREATE_FLAGS,
                    cwd=args.cqp_workdir or None)
                for line in ans.stdout.splitlines():
                    line = line.strip()
                    if not line or ' ' not in line:
                        continue
                    score, name = line.split(' ', 1)
                    try:
                        value = Decimal(score)
                    except Exception:
                        continue
                    append(cqp_f, {'id': next(r['id'] for r in batch if r['full_name'] == name),
                                   'full_name': name, 'cqp': str(value)})
                    stats['cqp_done'] += 1
            except (OSError, sp.SubprocessError) as exc:
                stats['cqp_error'] = f'{type(exc).__name__}'
            finally:
                with lock:
                    stats['cqp_backlog'] = len(cqp_queue)

    cqp_thread = None
    if args.cqp_mode != 'off' and args.openbox_cli:
        cqp_thread = threading.Thread(target=worker_cqp, daemon=False)
        cqp_thread.start()

    cmd = [args.miner, '--mine', '--backend', 'cuda', '--out', str(out / 'scan'),
           '--plan', args.plan, '--start', str(args.start),
           '--block', str(args.block), '--batch', str(args.batch), '--report-interval', '1']
    if args.shuffle:
        cmd += ['--shuffle', args.shuffle]
        if args.seed is not None:
            cmd += ['--seed', str(args.seed)]
    if args.team:
        cmd += ['--team', args.team]
    if args.name_template:
        cmd += ['--name-template', args.name_template]
    if args.seconds is not None:
        cmd += ['--count', '4611686018427387904']  # 足够大，靠定时器停
    else:
        cmd += ['--count', str(args.count)]

    log_path = out / 'miner.log'
    log_f = log_path.open('xb')
    proc = sp.Popen(cmd, stdout=log_f, stderr=sp.STDOUT,
                    creationflags=sp.CREATE_NEW_PROCESS_GROUP)

    power = ctypes.windll.kernel32.SetThreadExecutionState
    power.argtypes = [ctypes.c_uint32]
    power.restype = ctypes.c_uint32
    power(0x80000001)

    def halt():
        if proc.poll() is None:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.wait(timeout=15)
            except (OSError, sp.TimeoutExpired):
                proc.terminate()
                proc.wait(timeout=5)

    def guard():
        while not stop.wait(0.2):
            due = args.seconds is not None and time.monotonic() - begin >= args.seconds
            if due:
                halt()
                return

    if args.seconds is not None:
        threading.Thread(target=guard, daemon=True).start()

    csv_path = out / 'scan' / 'results.csv'
    handle = None
    carry = b''
    cols = None
    last_id = args.start - 1
    next_write = begin
    try:
        while True:
            exited = proc.poll() is not None
            if handle is None and csv_path.exists():
                handle = csv_path.open('rb')
            data = handle.read(65536) if handle is not None and len(pf_queue) < args.read_ahead else b''
            if data:
                lines = (carry + data).split(b'\n')
                carry = lines.pop()
                for line in lines:
                    if not line.strip():
                        continue
                    row = next(csv.reader([line.decode('utf-8')]))
                    if cols is None:
                        cols = row
                        continue
                    if len(row) != len(cols):
                        raise RuntimeError('候选行不完整')
                    d = dict(zip(cols, row))
                    if d.get('keep') != '1':
                        continue
                    ident = int(d['id'])
                    if ident <= last_id:
                        raise RuntimeError('候选 ID 未递增')
                    last_id = ident
                    cand = {'id': ident, 'full_name': d['full_name'],
                            'model_qp': d.get('qp'), 'model_qd': d.get('qd'),
                            'model_sum': d.get('sum')}
                    append(cand_f, cand)
                    pf_queue.append(cand)
                    stats['candidates'] += 1
            if pf_queue:
                batch = [pf_queue.popleft() for _ in range(min(len(pf_queue), args.pf_batch))]
                names = [r['full_name'] for r in batch]
                try:
                    got = pf_batch(args.tswn_cli, names, args.pf_n, args.pf_thread)
                except (OSError, sp.SubprocessError):
                    got = pf_batch(args.tswn_cli, names, args.pf_n, args.pf_thread)
                for row in batch:
                    vals = got[row['full_name']]
                    qp = Decimal(vals['qp'])
                    rec = {'id': row['id'], 'full_name': row['full_name'],
                           'model_qp': row['model_qp'], 'model_qd': row['model_qd'],
                           **{k: vals[k] for k in PF_METRICS}}
                    append(score_f, rec)
                    stats['scored'] += 1
                    if stats['best_qp'] is None or qp > Decimal(stats['best_qp']):
                        stats['best_qp'] = vals['qp']
                    if qp >= Decimal('6000'):
                        stats['hits'] += 1
                        print(f'达标 QP {vals["qp"]}  ID {row["id"]}', flush=True)
                    bisect.insort(qp_sorted, qp)
                    qp_values.append(qp)
                    want_cqp = False
                    if args.cqp_mode == 'all':
                        want_cqp = True
                    elif args.cqp_mode == 'qualified':
                        want_cqp = qp >= Decimal(args.cqp_min_qp)
                    elif args.cqp_mode == 'top':
                        if len(qp_sorted) >= args.cqp_min_samples:
                            cut = qp_sorted[int(len(qp_sorted) * (1 - args.cqp_top_percent / 100))]
                            want_cqp = qp >= cut
                        else:
                            want_cqp = qp >= Decimal(args.cqp_min_qp)
                    if want_cqp:
                        with lock:
                            cqp_queue.append(row)
            elif exited:
                break
            else:
                time.sleep(0.05)
            t = time.monotonic()
            if t >= next_write:
                last, counts = miner_tail(log_path)
                with lock:
                    stats.update(phase='补完开箱' if exited else '扫描中',
                                 elapsed_seconds=round(t - begin, 2),
                                 pf_backlog=len(pf_queue), last_miner_report=last)
                    stats.update({k: counts[k] for k in ('count', 'next_id', 'keep') if k in counts})
                dump(out / 'status.json', stats)
                while next_milestone is not None and counts.get('count', 0) + args.start >= next_milestone:
                    mark = next_milestone - args.start
                    entry = {'scanned': mark, 't': round(mark / 1e12, 4),
                             'elapsed_seconds': round(t - begin, 1),
                             'candidates': stats['candidates'], 'scored': stats['scored'],
                             'cqp_done': stats['cqp_done'], 'hits': stats['hits'],
                             'best_qp': stats['best_qp']}
                    milestones.append(entry)
                    print('MILESTONE ' + json.dumps(entry, ensure_ascii=False), flush=True)
                    dump(out / 'milestones.json', milestones)
                    next_milestone += args.milestone_count
                next_write = t + args.report_interval
    finally:
        power(0x80000000)
        stop.set()
        if handle:
            handle.close()
        log_f.close()
        if proc.poll() is None:
            halt()
        if cqp_thread is not None:
            while cqp_thread.is_alive():
                cqp_thread.join(timeout=15)
                stats['phase'] = '补跑 cqp'
                stats['cqp_backlog'] = len(cqp_queue)
                dump(out / 'status.json', stats)
        score_f.close()
        cand_f.close()
        cqp_f.close()

    ok = proc.returncode in (0, 130)
    stats.update(phase='已结束' if ok else '异常结束', miner_returncode=proc.returncode,
                 finished_wall=now().isoformat(), elapsed_seconds=round(time.monotonic() - begin, 2),
                 pf_backlog=len(pf_queue), cqp_backlog=len(cqp_queue))
    if carry.strip():
        stats['warning'] = 'CSV 尾部有未完成行'
    dump(out / 'status.json', stats)
    print(json.dumps({k: stats[k] for k in ('candidates', 'scored', 'cqp_done', 'hits', 'best_qp')},
                     ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(run(classify_cli_args()))

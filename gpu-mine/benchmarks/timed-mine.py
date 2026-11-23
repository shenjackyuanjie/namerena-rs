"""在独立 Windows 控制台按秒运行 XP/XD，核对真实 CSV 并输出吞吐摘要。"""
import argparse
import csv
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


NS = 1_000_000_000
CPU_SUMMARY = re.compile(r'实际计算: ([0-9]+)，命中: ([0-9]+)，耗时: ([0-9.]+)s，速度: [0-9.]+/s')
WGSL_SUMMARY = re.compile(r'已停止并flush：完成=([0-9]+) 命中=([0-9]+) 下一ID=([0-9]+) E2E=([0-9.]+)s')


def duration(value):
    value = float(value)
    if not math.isfinite(value) or not 0.001 <= value <= 3600:
        raise argparse.ArgumentTypeError('秒数须在 0.001～3600 之间')
    return value


def command(args, exe):
    batch = args.batch_size if args.batch_size is not None else (10_000 if args.mode == 'cpu' else 65_536)
    threads = '-t6' if args.mode == 'cpu' else '-t2'
    cmd = [str(exe), threads, '-p550', '-x4000', '-d5000', '--team', 'shenjack',
           '-b', str(batch), '-m', '我也不知道我要测多少 []', '-n', 'base16384', '--start', '0']
    # 不传 --count / --end；两条 CLI 都以 Ctrl-Break 停止，不预设候选规模。
    if args.mode != 'cpu':
        cmd += ['--backend', args.mode, '--gpu-input', args.gpu_input, '--sbox', args.sbox,
                '--workgroup-size', str(args.workgroup_size), '--verify', '--output', 'result.csv']
        if args.dense_model:
            cmd.append('--dense-model')
        if args.mode == 'cuda':
            cmd += ['--cuda-dll', str(args.cuda_dll.resolve())]
    return cmd


@contextmanager
def private_console(process):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel.AttachConsole.argtypes = [wintypes.DWORD]
    kernel.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    previous_console = kernel.GetConsoleCP() != 0
    attached = False
    # 仅改变启动器自身的附着；不会把 PowerShell 或用户现有进程附着到私有控制台。
    if previous_console and not kernel.FreeConsole():
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.AttachConsole(process.pid):
            raise ctypes.WinError(ctypes.get_last_error())
        attached = True
        ignore_break = handler_type(lambda event: int(event in (0, 1)))
        if not kernel.SetConsoleCtrlHandler(ignore_break, True):
            raise ctypes.WinError(ctypes.get_last_error())

        def send_break():
            attached_pids = (wintypes.DWORD * 8)()
            count = kernel.GetConsoleProcessList(attached_pids, len(attached_pids))
            if count != 2 or set(attached_pids[:count]) != {os.getpid(), process.pid}:
                raise RuntimeError('控制台成员不止本脚本与其子进程，拒绝发送 Ctrl-Break')
            begin = time.perf_counter_ns()
            # CREATE_NEW_CONSOLE + 只含两进程的控制台，0 只在该控制台内广播。
            if not kernel.GenerateConsoleCtrlEvent(1, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            return begin, time.perf_counter_ns()

        yield send_break
    finally:
        if attached:
            kernel.FreeConsole()
        if previous_console:
            kernel.AttachConsole(0xFFFFFFFF)  # 仅恢复本脚本与父控制台的连接（如仍存在）。


def wait_for_header(process, workdir, mode, timeout):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'子进程在准备好 CSV 前退出：{process.returncode}')
        if mode == 'cpu':
            paths = tuple((workdir / 'namerena').glob('*.csv'))
        else:
            paths = (workdir / 'result.csv',)
        if len(paths) == 1:
            try:
                with paths[0].open('rb') as output:
                    header = output.readline(4096)
                if header.startswith(b'full_name,id,') and header.endswith(bytes([10])):
                    return paths[0], time.perf_counter_ns()
            except FileNotFoundError:
                pass
        time.sleep(0.002)
    raise RuntimeError('等待已 flush 的 CSV 表头超时')


def wait_until(process, deadline_ns):
    while True:
        remaining = (deadline_ns - time.perf_counter_ns()) / NS
        if remaining <= 0:
            return
        # 先等子进程句柄，最后 20ms 用 Python 3.11+ 的 Windows 高分辨率 sleep 对齐截止点。
        if remaining <= 0.020:
            time.sleep(remaining)
            continue
        try:
            process.wait(timeout=remaining - 0.020)
        except subprocess.TimeoutExpired:
            continue
        raise RuntimeError(f'子进程在计时结束前退出：{process.returncode}')


def read_final(mode, logfile, csv_path):
    text = logfile.read_text(encoding='utf-8', errors='replace')
    match = (CPU_SUMMARY if mode == 'cpu' else WGSL_SUMMARY).search(text)
    if match is None:
        raise RuntimeError('缺少协作退出最终汇总')
    processed, hits = map(int, match.group(1, 2))
    elapsed = float(match.group(3 if mode == 'cpu' else 4))
    if processed <= 0 or elapsed <= 0 or (mode != 'cpu' and int(match.group(3)) != processed):
        raise RuntimeError('CLI 实算数、下一 ID 或耗时异常')
    with csv_path.open(encoding='utf-8', newline='') as file:
        reader = csv.reader(file, strict=True)
        header = next(reader)
        # 两条 CLI 的字段数不同；共同要求 ID 表头、完整行和命中条数。
        if not header or header[:2] != ['full_name', 'id']:
            raise RuntimeError('CSV 缺合法表头')
        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError('CSV 命中行不完整')
            rows += 1
    if rows != hits:
        raise RuntimeError(f'CSV 行数 {rows} 与命中数 {hits} 不符')
    result = dict(processed=processed, hits=hits, columns=len(header), child_elapsed_seconds=elapsed,
                  rate=processed / elapsed)
    if mode == 'wgsl':
        adapter = next((line for line in text.splitlines() if line.startswith('GPU：')), '')
        if os.environ.get('WGPU_BACKEND', '').lower() == 'vulkan' and '/ Vulkan /' not in adapter:
            raise RuntimeError(f'日志未确认 Vulkan 后端：{adapter!r}')
    return result


def run(args, cmd, workdir, label, seconds):
    workdir.mkdir()
    if args.mode == 'cpu':
        (workdir / 'namerena').mkdir()
    logfile = workdir / 'process.log'
    print(f'{label} {seconds:g}s', flush=True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with logfile.open('xb') as log:
        process = subprocess.Popen(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_CONSOLE,
                                   startupinfo=startup)
        try:
            csv_path, ready = wait_for_header(process, workdir, args.mode, args.setup_timeout)
            deadline = ready + round(seconds * NS)
            # 在截止前才切换本脚本的控制台；大部分等待期间仍可由调用者中断。
            wait_until(process, deadline - min(round(seconds * NS / 10), 100_000_000))
            with private_console(process) as send_break:
                wait_until(process, deadline)
                if process.poll() is not None:
                    raise RuntimeError(f'未发停止信号前子进程已退出：{process.returncode}')
                sent_begin, sent_end = send_break()
                exit_code = process.wait(timeout=args.drain_timeout)
                exited = time.perf_counter_ns()
            if exit_code != 130:
                raise RuntimeError(f'协作停止应退出 130，实际 {exit_code}')
        finally:
            if process.poll() is None:
                # 仅回收本次 Popen 的子进程；这种情况不是有效计时样本。
                process.kill()
                process.wait()
    result = read_final(args.mode, logfile, csv_path)
    rate = result['rate']
    print(f'{label}：实际 {result["processed"]:,}，命中 {result["hits"]:,}，'
          f'CLI {result["child_elapsed_seconds"]:.6f}s，{rate:,.0f}/s，'
          f'{rate * 86400 / 1e12:.2f} T/天，{rate * 604800 / 1e12:.2f} T/周，'
          f'宿主窗口 {(sent_begin - ready) / NS:.6f}s，排空 {(exited - sent_end) / NS:.6f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'wgsl', 'cuda'), default='cpu')
    parser.add_argument('--cuda-dll', type=Path, default=Path('target/cuda-ab/cuda-mine.dll'))
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--seconds', type=duration, required=True)
    parser.add_argument('--warmup-seconds', type=duration, default=0)
    parser.add_argument('--setup-timeout', type=duration, default=120)
    parser.add_argument('--drain-timeout', type=duration, default=20)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--gpu-input', choices=('auto', 'generated', 'uploaded'), default='auto')
    parser.add_argument('--sbox', choices=('packed', 'wide'), default='packed')
    parser.add_argument('--workgroup-size', type=int, default=32)
    parser.add_argument('--dense-model', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('需要 Windows 独立控制台')
    exe = args.binary.resolve()
    if not exe.is_file() or exe.suffix.lower() != '.exe':
        parser.error('必须指定已存在的矿工 EXE')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('批大小须为正')
    if args.mode == 'cuda' and not args.cuda_dll.is_file():
        parser.error('--cuda-dll 必须指定配套 CUDA 动态库')
    cmd = command(args, exe)
    with tempfile.TemporaryDirectory(prefix=f'{args.mode}-{exe.stem}-') as parent:
        folder = Path(parent)
        try:
            if args.warmup_seconds:
                run(args, cmd, folder / 'warmup', '预热', args.warmup_seconds)
            run(args, cmd, folder / 'measure', '计时', args.seconds)
        except RuntimeError as error:
            log = folder / 'measure/process.log'
            if not log.is_file():
                log = folder / 'warmup/process.log'
            tail = '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-8:]) if log.is_file() else ''
            parser.error(f'{error}\n{tail}'.rstrip())


if __name__ == '__main__':
    main()

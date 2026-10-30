"""在仓库根执行的黑盒验证：CSV、每 ID 一次、参数错误及可选 Windows Ctrl-Break。"""
import argparse
import base64
import csv
import ctypes
import io
import pathlib
import re
import subprocess
import tempfile
import time


def run(binary, arguments):
    before = set(pathlib.Path("namerena").glob("*.csv"))
    with tempfile.TemporaryFile(mode="w+b") as log:
        result = subprocess.run([str(binary), *arguments], stdout=log, stderr=subprocess.STDOUT, check=False)
        log.seek(0)
        text = log.read().decode("utf-8", errors="replace")
    assert result.returncode == 0, text
    created = set(pathlib.Path("namerena").glob("*.csv")) - before
    assert len(created) == 1, created
    with created.pop().open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(len(row) == 55 and None not in row for row in rows)
    indexed = {int(row["id"]): row for row in rows}
    assert len(indexed) == len(rows), "CSV 存在重复 ID"
    return indexed, text


def encode_id(algorithm, number):
    data = number.to_bytes(8, "big")
    if algorithm == "number":
        return str(number)
    if algorithm == "base64-std":
        return base64.b64encode(data).decode("ascii")
    if algorithm == "base64-url":
        return base64.urlsafe_b64encode(data).decode("ascii")
    # 独立按 14 位码元实现固定八字节编码，不调用源库有已知上界缺陷的 decoder。
    return "".join(chr(0x4E00 + ((number >> shift) & 0x3FFF)) for shift in [50, 36, 22, 8]) + chr(0x4E00 + ((number & 255) << 6)) + chr(0x3D01)


def check_interrupt(binary):
    with tempfile.TemporaryFile(mode="w+b") as log:
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        process = subprocess.Popen([str(binary), "-t6", "-p4294967295", "-x0", "-d0", "--team", "interrupt-test", "-r10", "--end", "1000000000000"], stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
        try:
            time.sleep(0.3)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.FreeConsole()
            assert kernel.AttachConsole(process.pid), ctypes.get_last_error()
            # 只向本脚本创建的控制台根进程组发 CTRL_BREAK，不触碰已有用户 miner。
            callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
            ignore = callback_type(lambda event: 1)
            assert kernel.SetConsoleCtrlHandler(ignore, 1), ctypes.get_last_error()
            # AttachConsole 后此独立控制台只有自有 miner 与当前验证进程。
            assert kernel.GenerateConsoleCtrlEvent(1, 0), ctypes.get_last_error()
            code = process.wait(timeout=3)
            kernel.FreeConsole()
            log.seek(0)
            text = log.read().decode("utf-8", errors="replace")
            assert code == 130, (code, text)
            assert "实际计算:" in text, text
            match = re.search(r'输出: "(.*?)"', text)
            assert match, text
            with pathlib.Path(match.group(1)).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert not rows
            assert "实际计算: 1000000000000" not in text
            assert len({row["id"] for row in rows}) == len(rows)
            pathlib.Path("miner/benchmarks/cpu-interrupt-validation.txt").write_text(f"实际 Ctrl-Break：退出{code}，提前终止、join和完整表头通过\n{text}", encoding="utf-8")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("target/release/tswn.exe"))
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--interrupt", action="store_true")
    options = parser.parse_args()
    team = '组,@队"\n'
    template = '前,@"\n[]/[]'
    start = 2**64 - 38
    for algorithm in ["base16384", "base64-std", "base64-url", "number"]:
        for threads in [1, 2, 6]:
            for batch in [["-b3"], ["-r10"]]:
                rows, _ = run(options.binary, [f"-t{threads}", "-p0", "-x0", "-d0", "--team", team, *batch, "-n", algorithm, "-m", template, "--start", str(start), "--end", str(start + 37)])
                assert set(rows) == set(range(start, start + 37))
                for number, row in rows.items():
                    expected = template.replace("[]", encode_id(algorithm, number))
                    assert row["name"] == expected
                    assert row["team"] == team
                    assert row["full_name"] == f"{expected}@{team}"
    print("24组黑盒CSV：四种编码、1/2/6线程、固定/动态batch、u64上界、逗号引号换行均通过")
    if options.baseline:
        arguments = ["-t6", "-p0", "-x0", "-d0", "--team", "shenjack", "-b7", "-m", "我也不知道我要测多少 []", "--end", "1000"]
        expected, _ = run(options.baseline, arguments)
        actual, _ = run(options.binary, arguments)
        assert actual == expected
        print("修复基线对照：1000个ID、全部55列（含f64输出与技能）完全一致")
    if options.interrupt:
        check_interrupt(options.binary)


if __name__ == "__main__":
    main()
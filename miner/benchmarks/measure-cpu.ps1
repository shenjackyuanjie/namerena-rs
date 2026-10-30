# 在仓库根运行。串行交错测量，每条样本即时落盘；不构建、不推送、不终止其他进程。
param(
    [string[]]$Binaries = @('target/release/tswn.exe'),
    [UInt64]$Candidates = 5000000,
    [UInt32]$Threads = 6,
    [UInt32]$Property = 550,
    [UInt32]$XP = 4000,
    [UInt32]$XD = 5000,
    [int]$Repetitions = 3,
    [string]$Output = 'miner/benchmarks/cpu-local.csv',
    [switch]$Sink
)
$ErrorActionPreference = 'Stop'
$env:RUSTC_WRAPPER = ''
if (Test-Path $Output) { throw "输出已存在，请指定一个新的 -Output：$Output" }
if ($Threads -eq 0 -or $Candidates -eq 0 -or $Repetitions -le 0) { throw '线程、候选数与重复次数必须为正' }
foreach ($binary in $Binaries) { if (-not (Test-Path $binary -PathType Leaf)) { throw "二进制不存在：$binary" } }
foreach ($round in 1..$Repetitions) {
    foreach ($binary in $Binaries) {
        $arguments = @('-t', $Threads, '-p', $Property, '-x', $XP, '-d', $XD, '--team', 'shenjack', '-b10000', '-m', '我也不知道我要测多少 []', '--end', $Candidates)
        if ($Sink) { $arguments += '--bench' }
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        $log = & $binary @arguments
        $watch.Stop()
        if ($LASTEXITCODE -ne 0) { throw "基准失败：$binary，退出码 $LASTEXITCODE" }
        $last = $log | Select-Object -Last 1
        if ($last -notmatch '(?:实际计算: |实际计算 )(\d+)，命中(?:: | )(\d+)，耗时(?:: | )([\d.]+)s，速度(?:: | )([\d.]+)/s') { throw "无法解析结果：$last" }
        $record = [pscustomobject]@{
            binary = $binary; round = $round; threads = $Threads; candidates = [UInt64]$Matches[1]
            hits = [UInt64]$Matches[2]; seconds = [double]$Matches[3]; candidates_per_second = [double]$Matches[4]
            wall_seconds = $watch.Elapsed.TotalSeconds; wall_candidates_per_second = $Candidates / $watch.Elapsed.TotalSeconds
            property = $Property; xp = $XP; xd = $XD; sink = [bool]$Sink
        }
        $record | Export-Csv -Append -NoTypeInformation -Encoding utf8 $Output
        $record | Format-List
    }
}

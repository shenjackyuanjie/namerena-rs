# 在workspace根运行；只运行所选GPU二进制，不构建、不停止其它进程、不覆盖旧结果。
param(
    [string]$Binary = 'target/release/wgpu-mine.exe',
    [UInt64]$Candidates = 1000000000,
    [int]$Repetitions = 3,
    [UInt64]$Start = 0,
    [UInt32]$BatchSize = 524287,
    [string]$Team = 'shenjack',
    [string]$Template = '我也不知道我要测多少 []',
    [string]$Output = 'wgpu-mine/benchmarks/gpu-local.csv',
    [string]$OutputDir = 'target/gpu-local-results'
)
$ErrorActionPreference = 'Stop'
if ($Candidates -eq 0 -or $Repetitions -le 0) { throw '候选数和轮数必须大于零' }
if (-not (Test-Path $Binary -PathType Leaf)) { throw "二进制不存在：$Binary" }
if (Test-Path $Output) { throw "指标文件已存在：$Output" }
if (Test-Path $OutputDir) { throw "结果目录已存在，请指定新目录：$OutputDir" }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
'round,processed,hits,e2e_seconds,names_per_second,kernel_seconds,wall_seconds,wall_names_per_second' | Set-Content -LiteralPath $Output -Encoding utf8
$expectedHash = $null
foreach ($round in 0..($Repetitions - 1)) {
    $log = Join-Path $OutputDir "run-$round.log"
    $csv = Join-Path $OutputDir "run-$round.csv"
    $arguments = @('--gpu-input','generated','--sbox','packed','--workgroup-size','32','--team',$Team,
        '-t','2','-p','550','-x','4000','-d','5000','-m',$Template,'--start',$Start,'--count',$Candidates,
        '--batch-size',$BatchSize,'--verify','--output',$csv)
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $Binary @arguments *> $log
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    Get-Content $log -Tail 5
    if ($exitCode -ne 0) { throw "GPU运行失败或被取消，退出码：$exitCode；日志：$log" }
    $summary = Get-Content $log | Where-Object { $_ -like '[[]gpu] 完成：*' } | Select-Object -Last 1
    $match = [regex]::Match($summary, '完成=([0-9]+) 命中=([0-9]+).*E2E=([0-9.]+)s ([0-9.]+)名/s.*GPU内核=([0-9.]+)s')
    if (-not $match.Success) { throw "无法解析运行摘要：$log" }
    $processed = [UInt64]$match.Groups[1].Value
    if ($processed -ne $Candidates) { throw "实际数量与请求不同：$processed" }
    $wall = $timer.Elapsed.TotalSeconds
    '{0},{1},{2},{3},{4},{5},{6},{7}' -f $round,$processed,$match.Groups[2].Value,$match.Groups[3].Value,$match.Groups[4].Value,$match.Groups[5].Value,$wall,($processed/$wall) |
        Add-Content -LiteralPath $Output -Encoding utf8
    $hash = (Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash
    if ($null -eq $expectedHash) { $expectedHash = $hash }
    elseif ($hash -ne $expectedHash) { throw '重复运行的结果CSV不一致' }
}
Import-Csv $Output | Format-Table

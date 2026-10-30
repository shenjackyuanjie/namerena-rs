# workspace根运行；先独立构建target/native，本脚本只串行测量，不构建/删文件/停止用户进程。
param(
    [int]$Repetitions = 3,
    [UInt64]$CpuCandidates = 5000000,
    [UInt64]$GpuCandidates = 1000000000,
    [string]$Output = 'wgpu-mine/benchmarks/native-comparison.csv',
    [string]$LogDir = 'target/native-comparison-logs'
)
$ErrorActionPreference = 'Stop'
if ($Repetitions -lt 1 -or $CpuCandidates -lt 1 -or $GpuCandidates -lt 1) { throw '数量与轮数必须大于零' }
if (Test-Path $Output) { throw "指标文件已存在：$Output" }
if (Test-Path $LogDir) { throw "日志目录已存在：$LogDir" }
foreach ($binary in @('target/release/tswn.exe','target/native/release/tswn.exe','target/release/wgpu-mine.exe','target/native/release/wgpu-mine.exe')) {
    if (-not (Test-Path $binary -PathType Leaf)) { throw "二进制不存在：$binary" }
}
New-Item -ItemType Directory -Path $LogDir | Out-Null
$script:ReferenceHash = $null
function Measure-One([string]$Kind, [string]$Build, [int]$Round, [int]$Threads, [UInt64]$Count) {
    $base = if ($Build -eq 'native') { 'target/native/release' } else { 'target/release' }
    $binary = Join-Path $base $(if ($Kind -eq 'gpu') { 'wgpu-mine.exe' } else { 'tswn.exe' })
    $label = "$Kind-t$Threads-$Build-r$Round"
    $log = Join-Path $LogDir "$label.log"
    $csv = ''
    $arguments = @('-t',$Threads,'-p','550','-x','4000','-d','5000','--team','shenjack','-m','我也不知道我要测多少 []')
    if ($Kind -eq 'gpu') {
        $csv = Join-Path $LogDir "$label.csv"
        $arguments += @('--gpu-input','generated','--sbox','packed','--workgroup-size','32','--batch-size','524287','--count',$Count,'--verify','--output',$csv)
    } else { $arguments += @('-b10000','--end',$Count) }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $binary @arguments *> $log
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    Get-Content $log -Tail 2 | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { throw "测试失败或取消：$label，退出码$exitCode" }
    if ($Kind -eq 'gpu') {
        $summary = Get-Content $log | Where-Object { $_ -like '[[]gpu] 完成：*' } | Select-Object -Last 1
        $match = [regex]::Match($summary, '完成=([0-9]+) 命中=([0-9]+).*E2E=([0-9.]+)s ([0-9.]+)名/s')
    } else {
        $summary = Get-Content $log | Select-Object -Last 1
        $match = [regex]::Match($summary, '实际计算: ([0-9]+)，命中: ([0-9]+)，耗时: ([0-9.]+)s，速度: ([0-9.]+)/s')
    }
    if (-not $match.Success) { throw "无法解析摘要：$label" }
    $processed = [UInt64]$match.Groups[1].Value
    if ($processed -ne $Count) { throw "计算数量不一致：$label" }
    $record = [pscustomobject]@{
        kind=$Kind; build=$Build; round=$Round; threads=$Threads; processed=$processed
        hits=[UInt64]$match.Groups[2].Value; seconds=[double]$match.Groups[3].Value
        names_per_second=[double]$match.Groups[4].Value; wall_seconds=$timer.Elapsed.TotalSeconds
        wall_names_per_second=$Count/$timer.Elapsed.TotalSeconds; log=$log; result_csv=$csv
    }
    $record | Export-Csv -LiteralPath $Output -Append -NoTypeInformation -Encoding utf8
    Write-Host ($record | Format-List | Out-String)
    if ($Kind -eq 'gpu' -and $Round -gt 0) {
        $hash = (Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash
        if ($null -eq $script:ReferenceHash) { $script:ReferenceHash=$hash }
        elseif ($hash -ne $script:ReferenceHash) { throw '默认/native GPU结果CSV不一致，需检查数值差异' }
    }
}
foreach ($threads in @(2,6)) {
    foreach ($build in @('default','native')) { Measure-One 'cpu' $build 0 $threads 1000000 }
    foreach ($round in 1..$Repetitions) {
        $order = if ($round % 2 -eq 1) { @('default','native') } else { @('native','default') }
        foreach ($build in $order) { Measure-One 'cpu' $build $round $threads $CpuCandidates }
    }
}
foreach ($build in @('default','native')) { Measure-One 'gpu' $build 0 2 10000000 }
foreach ($round in 1..$Repetitions) {
    $order = if ($round % 2 -eq 1) { @('default','native') } else { @('native','default') }
    foreach ($build in $order) { Measure-One 'gpu' $build $round 2 $GpuCandidates }
}

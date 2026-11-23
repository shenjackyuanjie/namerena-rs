# 与显式指定的正式 gpu-mine 二进制比较。两个脚本都应使用 PowerShell 7。
param(
    [Parameter(Mandatory)][string]$ReferenceExe,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidateRange(1, [long]::MaxValue)][long]$Count = 1000000000,
    [ValidateRange(1, 99)][int]$Rounds = 3
)
$ErrorActionPreference = 'Stop'
$reference = (Resolve-Path -LiteralPath $ReferenceExe).ProviderPath
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$root = Split-Path (Split-Path $PSScriptRoot)
$harness = Join-Path $root 'target/release/examples/cuda-ab.exe'
$dll = Join-Path $root 'target/cuda-ab/cuda-mine.dll'
if ((Test-Path -LiteralPath $output) -or !(Test-Path -LiteralPath (Split-Path $output))) {
    throw '请指定上级已存在的新结果目录。'
}
$files = @(Get-FileHash -Algorithm SHA256 -LiteralPath $reference,$harness,$dll)
$null = New-Item -ItemType Directory -Path $output
$files | Export-Csv -NoTypeInformation (Join-Path $output 'binaries.csv')
$env:WGPU_BACKEND = 'vulkan'
$opts = @('--team','shenjack','-m','我也不知道我要测多少 []','-p','550','-x','4000','-d','5000',
          '--batch-size','524287','--verify','--gpu-input','generated','--sbox','packed','--workgroup-size','32','-t','2','-r','60')

function Read-GpuState([string]$Stage) {
    & nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,utilization.gpu,clocks.sm --format=csv |
        Tee-Object -FilePath (Join-Path $output "gpu-$Stage.txt") | Out-Host
}
function Measure-Reference([string]$Label, [long]$Candidates, [int]$Round) {
    $lines = & $reference @opts --count $Candidates --output (Join-Path $output "$Label.csv") 2>&1 |
        Tee-Object -FilePath (Join-Path $output "$Label.log")
    if ($LASTEXITCODE -ne 0) { throw "正式 WGSL 失败：$Label" }
    $lines | Out-Host
    $summary = $lines | Where-Object { "$_" -match '命中=\d+ .*E2E=.*GPU内核=' } | Select-Object -Last 1
    if ("$summary" -notmatch '命中=(\d+) .*E2E=([0-9.]+)s (\d+)名/s.*GPU内核=([0-9.]+)s') {
        throw '正式 WGSL 未输出预期的统计行。'
    }
    [pscustomobject]@{
        round = $Round; backend = 'wgsl-release'; candidates = $Candidates; hits = [long]$Matches[1]
        e2e_s = [double]$Matches[2]; kernel_s = [double]$Matches[4]
        e2e_names_s = [long]$Matches[3]; kernel_names_s = $Candidates / [double]$Matches[4]
    }
}

Read-GpuState 'begin'
$null = Measure-Reference 'warmup-release' ([Math]::Min($Count, 10000000L)) 0
for ($round = 1; $round -le $Rounds; $round++) {
    $label = "$round-release"
    $pair = Join-Path $output "pair-$round"
    $reverse = $round % 2 -eq 0
    if (!$reverse) { $ref = Measure-Reference $label $Count $round }
    $a = @('--dll', $dll, '--count', "$Count", '--rounds', '1', '--out', $pair, '--cuda-mode', 'full', '--cuda-block', '32')
    if ($reverse) { $a += '--cuda-first' }
    & $harness @a 2>&1 | Tee-Object -FilePath (Join-Path $output "pair-$round.log")
    if ($LASTEXITCODE -ne 0) { throw "CUDA / 工作区 WGSL 基准失败：第 $round 轮" }
    if ($reverse) { $ref = Measure-Reference $label $Count $round }
    $hashes = @(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $output "$label.csv"), (Join-Path $pair '1-wgsl.csv'), (Join-Path $pair '1-cuda.csv'))
    if (@($hashes.Hash | Select-Object -Unique).Count -ne 1) { throw "正式 WGSL / 工作区 WGSL / CUDA 结果不同：第 $round 轮" }
    "第 $round 轮：三份完整命中 CSV SHA-256 一致。"
    $paired = Import-Csv (Join-Path $pair 'metrics.csv') | ForEach-Object {
        $_.round = "$round"
        if ($_.backend -eq 'wgsl') { $_.backend = 'wgsl-working' }
        $_ | Select-Object round,backend,candidates,hits,e2e_s,kernel_s,e2e_names_s,kernel_names_s
    }
    @($ref) + @($paired) | Export-Csv -Append -NoTypeInformation (Join-Path $output 'metrics.csv')
}
Read-GpuState 'end'
"原始指标：$(Join-Path $output 'metrics.csv')"

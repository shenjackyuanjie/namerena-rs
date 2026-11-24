#Requires -Version 7.0
param(
    [string]$Cpu,
    [string]$CpuBefore,
    [ValidateRange(1,64)][int[]]$PbbThreads,
    [ValidateSet('mix','sequential')][string]$FillOrder = 'mix',
    [string]$CudaBefore,
    [string]$CudaAfter,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidateRange(1,3600)][int]$Seconds = 60,
    [ValidateSet('all','cpu','cuda')][string]$Target = 'all'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if ($Target -ne 'cuda') {
    if (!$Cpu) { throw 'CPU 测速需要 -Cpu' }
    $cpuExe = (Resolve-Path -LiteralPath $Cpu).Path
    $cpuBeforeExe = if ($CpuBefore) { (Resolve-Path -LiteralPath $CpuBefore).Path } else { $cpuExe }
}
if ($Target -ne 'cpu') {
    if (!$CudaAfter) { throw 'CUDA 测速需要 -CudaAfter' }
    $afterExe = (Resolve-Path -LiteralPath $CudaAfter).Path
    $beforeExe = if ($CudaBefore) { (Resolve-Path -LiteralPath $CudaBefore).Path } else { $afterExe }
}
if (Test-Path -LiteralPath $OutputDir) { throw '测速目录已存在' }
$root = (New-Item -ItemType Directory -Path $OutputDir).FullName
$plan = Join-Path $root 'qp.kv'
$template = 'sby懒得帮我测qp6k所以我自己测 []'
@("template=$template",'team=shenjack','score_list=qp','score_mode=all','min_qp=5200','collect_mode=0') |
    Set-Content -LiteralPath $plan -Encoding utf8
$football = Join-Path $PSScriptRoot '../../gpu-mine/benchmarks/pbb-cuda/fixtures/football-real.kv'
$start = [uint64]165154914304
$common = @('--seconds',"$Seconds",'--start',"$start")
if ($FillOrder -eq 'mix') { $common += @('--shuffle','mix','--seed','0') }

function Measure-Run([string]$Exe, [string[]]$Opts, [string]$Dest) {
    $log = & $Exe @Opts --out $Dest 2>&1
    $code = $LASTEXITCODE
    $log | Set-Content -LiteralPath "$Dest.log" -Encoding utf8
    if ($code) { throw "$Dest 失败：$($log -join [char]10)" }
    $m = Import-Csv -LiteralPath "$Dest/metrics.csv"
    if ([double]$m.e2e_s -lt $Seconds) { throw "$Dest 未跑满 $Seconds 秒" }
    $m
}

function Compare-Window([string]$Old, [string]$New, [string]$File, [uint64]$Count) {
    $end = $start + $Count
    $data = foreach ($dir in $Old, $New) {
        # 当前测速模板没有换行；结果已按枚举 ID 排序。
        $lines = @(Get-Content -Encoding utf8 -LiteralPath "$dir/$File" | Select-Object -Skip 1 |
            Where-Object { [uint64]($_.Split(',',2)[0]) -lt $end })
        ,$lines
    }
    if (($data[0] -join [char]10) -cne ($data[1] -join [char]10)) {throw '公共编号区间结果 CSV 不一致'}
    $data[0].Count
}

$cases = @()
if ($Target -ne 'cuda') {
    if ($PbbThreads) {
        $reverse = $false
        foreach ($threads in $PbbThreads) {
            $cases += @{label="cpu-qp-$threads"; backend='cpu'; opts=@('--plan',$plan,'--threads',"$threads",'--batch','262144'); reverse=$reverse}
            $reverse = !$reverse
        }
    } else {
        $cases += @{label='cpu-qp-1'; backend='cpu'; opts=@('--plan',$plan,'--threads','1','--batch','262144'); reverse=$false}
        $cases += @{label='cpu-qp-6'; backend='cpu'; opts=@('--plan',$plan,'--threads','6','--batch','262144'); reverse=$true}
        $cases += @{label='cpu-football-6'; backend='cpu'; opts=@('--plan',$football,'--threads','6','--batch','262144'); reverse=$false}
    }
}
if ($Target -ne 'cpu') {
    foreach ($width in 5,41,50) {
        $name = if ($width -eq 41) {$template} else { ('x' * $width)+'[]' }
        $cases += @{label="cuda-w$width"; backend='cuda'; opts=@('--backend','cuda','--plan',$plan,
            '--name-template',$name,'--block','96','--batch','16777216'); reverse=($width -eq 41)}
    }
}
$legacy = $env:PBB_CUDA_LEGACY_FRONT
try {
    Remove-Item Env:PBB_CUDA_LEGACY_FRONT -ErrorAction SilentlyContinue
    foreach ($case in $cases) {
        $tag = $case.label
        $dest = @{ before="$root/$tag-before"; after="$root/$tag-after" }
        $results = @{}
        $order = if ($case.reverse) { @('after','before') } else { @('before','after') }
        foreach ($kind in $order) {
            $opts = $common + $case.opts
            if ($case.backend -eq 'cpu') {
                $exe = if ($kind -eq 'before') { $cpuBeforeExe } else { $cpuExe }
                $mode = if ($kind -eq 'before' -and !$CpuBefore) { 'scalar' } else { 'fast' }
                $results[$kind] = Measure-Run $exe (@('pbb','--cpu-path',$mode) + $opts) $dest[$kind]
            } else {
                $exe = if ($kind -eq 'before') {$beforeExe} else {$afterExe}
                if ($kind -eq 'before' -and !$CudaBefore) { $env:PBB_CUDA_LEGACY_FRONT = '1' }
                else { Remove-Item Env:PBB_CUDA_LEGACY_FRONT -ErrorAction SilentlyContinue }
                $results[$kind] = Measure-Run $exe $opts $dest[$kind]
            }
            Write-Host "$tag/$kind $($results[$kind].e2e_s)s $($results[$kind].T_per_day) T/day"
        }
        $before = $results.before
        $after = $results.after
        $commonCount = [math]::Min([uint64]$before.candidates,[uint64]$after.candidates)
        $equalHits = Compare-Window $dest.before $dest.after "$($case.backend).csv" $commonCount
        $row = [pscustomobject]@{case=$tag; before_s=$before.e2e_s; after_s=$after.e2e_s;
            before_names_s=$before.names_s; after_names_s=$after.names_s; before_T_day=$before.T_per_day; after_T_day=$after.T_per_day;
            gain_pct=[math]::Round(100*([double]$after.names_s/[double]$before.names_s-1),3);
            common_candidates=$commonCount; matching_hits=$equalHits}
        $row | Export-Csv -LiteralPath "$root/summary.csv" -NoTypeInformation -Append
        Write-Host "$tag +$($row.gain_pct)%；公共区间 $commonCount / $equalHits 条命中逐行一致"
    }
} finally {
    $env:PBB_CUDA_LEGACY_FRONT = $legacy
}

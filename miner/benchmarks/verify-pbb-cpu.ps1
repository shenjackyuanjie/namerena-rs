#Requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Cpu,
    [Parameter(Mandatory)][string]$Cuda,
    [Parameter(Mandatory)][string]$OutputDir
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$cpuExe = (Resolve-Path -LiteralPath $Cpu).Path
$cudaExe = (Resolve-Path -LiteralPath $Cuda).Path
$fixtures = Join-Path $PSScriptRoot '../../gpu-mine/benchmarks/pbb-cuda/fixtures'
if (Test-Path Env:PBB_CUDA_LEGACY_FRONT) { throw '对拍前请清除 PBB_CUDA_LEGACY_FRONT' }
if (Test-Path -LiteralPath $OutputDir) { throw '输出目录已存在' }
$root = (New-Item -ItemType Directory -Path $OutputDir).FullName

function Invoke-Checked([string]$Exe, [string[]]$Options, [string]$Log) {
    $lines = & $Exe @Options 2>&1
    $code = $LASTEXITCODE
    $lines | Set-Content -LiteralPath (Join-Path $root "$Log.log") -Encoding utf8
    if ($code) { throw "$($Log)：$($lines -join [char]10)" }
    $lines | ForEach-Object { "$_" }
}

function Assert-Same([string]$A, [string]$B, [string]$Label) {
    if (![string]::Equals($A,$B,[StringComparison]::Ordinal)) { throw "$($Label)：$A != $B" }
}

$basePlan = Join-Path $fixtures 'gate-off.kv'
$qpPlan = Join-Path $root 'qp.kv'
@('template=sby懒得帮我测qp6k所以我自己测 []','team=shenjack','score_list=qp','score_mode=all','min_qp=5200','collect_mode=0') |
    Set-Content -LiteralPath $qpPlan -Encoding utf8

foreach ($algo in 'affine','mix','feistel') {
    $shuffle = @('--shuffle',$algo,'--seed','18446744073709551615')
    $id = '18446744073709551615'
    $a = Invoke-Checked $cpuExe (@('pbb','--encode-id',$id) + $shuffle) "encode-$algo-cpu"
    $b = Invoke-Checked $cudaExe (@('--encode-id',$id) + $shuffle) "encode-$algo-cuda"
    Assert-Same $a $b "$algo 填充"
    $inverse = Invoke-Checked $cpuExe (@('pbb','--decode-fill',$a) + $shuffle) "decode-$algo"
    Assert-Same $inverse $id "$algo 逆解"
}
$legacy = Invoke-Checked $cpuExe @('pbb','--decode-fill','一亙幩侾最㴁') 'decode-legacy'
Assert-Same $legacy '10531700325988' '旧填充'

$cases = @()
foreach ($width in 5..50) {
    $tail = if ($width % 2) {''} else {'后缀!'}
    $argsForFill = if ($width % 3 -eq 0) {@('--shuffle','mix','--seed','2026')} else {@()}
    $cases += @{name="w$width"; plan=$basePlan; start='4294966816'; n=8195;
        extra=@('--name-template',(('x' * $width) + '[]' + $tail)) + $argsForFill}
}
$cases += @(
    @{name='qp41'; plan=$qpPlan; start='165362124279'; n=2097153; extra=@('--shuffle','mix')},
    @{name='football'; plan=(Join-Path $fixtures 'football-real.kv'); start='460745962'; n=65537; extra=@()},
    @{name='u64'; plan=$basePlan; start='18446744073709551102'; n=513; extra=@('--shuffle','affine','--seed','2026')},
    @{name='max-name'; plan=$basePlan; start='0'; n=513; extra=@('--name-template',(('字' * 78)+'abc[]'))},
    @{name='words'; plan=$basePlan; start='19178507183455'; n=65537; extra=@('--fill-plan',(Join-Path $fixtures 'fill-phrase.kv'),
        '--shuffle','feistel','--seed','91','--name-template','前[]后')},
    @{name='blue'; plan=(Join-Path $fixtures 'custom-blue-new-edge.kv'); start='46422'; n=1; extra=@()},
    @{name='ptt'; plan=(Join-Path $fixtures 'ptt-cover-all.kv'); start='4697493'; n=1; extra=@()}
)

$rows = foreach ($case in $cases) {
    $tag = $case.name
    $opts = @('--plan',$case.plan,'--start',$case.start) + $case.extra
    $validation = $opts + @('--validate',"$($case.n)",'--threads','6','--batch','65536')
    $cpuLine = Invoke-Checked $cpuExe (@('pbb','--out',"$root/$tag-vcpu") + $validation) "$tag-cpu-validate" |
        Where-Object {$_ -match '^validated='}
    $cudaLine = Invoke-Checked $cudaExe (@('--out',"$root/$tag-vcuda",'--block','96') + $validation) "$tag-cuda-validate" |
        Where-Object {$_ -match '^validated='}
    if (!$cpuLine -or !$cudaLine) {throw "$tag 没有验证计数"}
    Assert-Same $cpuLine $cudaLine "$tag 计数"
    if ($tag -notmatch '^w[0-9]+$') {
        $mine = $opts + @('--mine','--count',"$($case.n)",'--batch','12345')
        $null = Invoke-Checked $cpuExe (@('pbb','--out',"$root/$tag-fast",'--threads','6') + $mine) "$tag-cpu-fast"
        $null = Invoke-Checked $cpuExe (@('pbb','--out',"$root/$tag-scalar",'--cpu-path','scalar','--threads','1') + $mine) "$tag-cpu-scalar"
        $null = Invoke-Checked $cudaExe (@('--out',"$root/$tag-cuda",'--block','96') + $mine) "$tag-cuda"
        $hashes = @('fast','scalar','cuda') | ForEach-Object {(Get-FileHash "$root/$tag-$_/results.csv").Hash}
        Assert-Same $hashes[0] $hashes[1] "$tag CPU CSV"
        Assert-Same $hashes[0] $hashes[2] "$tag CPU/CUDA CSV"
        if ($tag -in @('qp41','football','blue','ptt') -and @(Import-Csv "$root/$tag-fast/results.csv").Count -eq 0) {
            throw "$tag 预期有实际命中，CSV 为空"
        }
    }
    Write-Host "$tag $cpuLine"
    [pscustomobject]@{case=$tag; count=$case.n; validation=$cpuLine}
}
$rows | Export-Csv -LiteralPath (Join-Path $root 'verification.csv') -NoTypeInformation

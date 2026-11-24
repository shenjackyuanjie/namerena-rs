#Requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Binary,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidateSet('split','direct','wgsl')][string]$Path='split',
    [ValidateSet(32,64,96,128)][int]$Block=64
)
$ErrorActionPreference='Stop'
$exe=(Resolve-Path -LiteralPath $Binary).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw '验证目录已存在' }
$null=New-Item -ItemType Directory -Path $output
# 保留旧矩阵的区间；旧 18 字节填充的命中计数仅记作历史。
# --validate 仍以同一填充方案的原版 CPU 为金标准逐字段核对，包括实际筛选路径。
$cases=@(
    @{name='football-real';         start='15118306'; n=65537},
    # 新默认布局的原版 CPU 命中 ID 460778730。
    @{name='football-real';         start='460745962';n=65537; passed=6; keep=1; blue=0; ptt=2},
    @{name='gate-off';              start='0';        n=65537; passed=5; keep=5; blue=0},
    @{name='all-order';             start='0';        n=65537; passed=5; keep=2; blue=0},
    @{name='any-order';             start='0';        n=65537; passed=5; keep=2; blue=0},
    @{name='ptt-rescue';            start='0';        n=65537; passed=5; keep=1; blue=0; ptt=2},
    @{name='custom-blue';           start='0';        n=65537; passed=5; keep=0; blue=5},
    @{name='custom-blue-rounding';  start='0';        n=65537; passed=5; keep=0; blue=2},
    @{name='mixed-keep-blue';       start='0';        n=65537; passed=5; keep=1; blue=5; ptt=2},
    @{name='ptt-cover-all';         start='7275811';  n=1},
    @{name='ptt-cover-all';         start='4003673';  n=1},
    @{name='ptt-cover-all';         start='4018689';  n=1},
    # 原版 CPU 的新五字金样：两条并集覆盖 CP/FP/WC/FS/PJ 的非零 PTT。
    @{name='ptt-cover-all';         start='4697493';  n=1; passed=1; keep=1; blue=0; ptt=1},
    @{name='ptt-cover-all';         start='905437';   n=1; passed=1; keep=1; blue=0; ptt=1},
    @{name='custom-blue-new-edge';  start='46422';    n=1; passed=1; keep=0; blue=1; ptt=0},
    @{name='football-real';         start='4294967240';n=513},
    @{name='football-real';         start='18446744073709551102';n=513},
    @{name='base-affine';           plan='gate-off'; start='4294967240';n=513; shuffle='affine'; seed='0'},
    @{name='base-mix';              plan='gate-off'; start='18446744073709551102';n=513; shuffle='mix'; seed='18446744073709551615'},
    @{name='base-feistel';          plan='gate-off'; start='18446744073709551102';n=513; shuffle='feistel'; seed='2026'},
    @{name='alnum';                 plan='gate-off'; start='12000';n=4099; fill='fill-alnum'; passed=2; keep=2; blue=0},
    @{name='alnum-affine';          plan='gate-off'; start='353583';n=65537; fill='fill-alnum'; shuffle='affine'; seed='97'; passed=7; keep=7; blue=0},
    @{name='hanzi-mix';             plan='gate-off'; start='618000';n=4099; fill='fill-hanzi'; shuffle='mix'; seed='0'; passed=1; keep=1; blue=0},
    @{name='words-feistel';         plan='gate-off'; start='13000';n=4099; fill='fill-words'; shuffle='feistel'; seed='17'; passed=1; keep=1; blue=0},
    @{name='full-phrase';           plan='gate-off'; start='19178507248479';n=513; fill='fill-phrase'},
    @{name='decimal19-affine';      plan='gate-off'; start='9999999999999999487';n=513; fill='fill-decimal19'; shuffle='affine'; seed='18446744073709551615'}
)
$layout=@('[]','前[]后',('字'*78+'abc[]'),('测'*10+'x[]'),('测'*11+'x[]')) # 空字面、后缀、252 字节、31/34 字节前缀
for ($i=0; $i -lt $layout.Count; $i++) {
    $plan=Join-Path $output "layout-$i.kv"
    "template=$($layout[$i])" | Set-Content -LiteralPath $plan -Encoding utf8
    $cases+=@{name="layout-$i"; start='0'; n=8195; file=$plan}
}
$cases+=@{name='layout-4-feistel'; start='4294967240'; n=8195; file=(Join-Path $output 'layout-4.kv'); shuffle='feistel'; seed='17'; passed=1; keep=1; blue=0}
$results=foreach($c in $cases) {
    $label="$($c.name)-$($c.start)"
    $scorePlan=if ($c.ContainsKey('plan')) { $c.plan } else { $c.name }
    $plan=if ($c.ContainsKey('file')) { $c.file } else { Join-Path $PSScriptRoot "fixtures/$scorePlan.kv" }
    $out=Join-Path $output $label
    $backendArgs=if ($Path -eq 'wgsl') { @('--backend','wgsl') } else { @('--backend','cuda','--cuda-path',$Path) }
    $fillArgs=@()
    if ($c.ContainsKey('fill')) { $fillArgs+=@('--fill-plan',(Join-Path $PSScriptRoot "fixtures/$($c.fill).kv")) }
    if ($c.ContainsKey('shuffle')) { $fillArgs+=@('--shuffle',$c.shuffle,'--seed',$c.seed) }
    $lines=& $exe --out $out --plan $plan @backendArgs @fillArgs --start $c.start --validate $c.n --batch 65536 --block $Block 2>&1 |
        Tee-Object -FilePath (Join-Path $output "$label.log")
    if($LASTEXITCODE -ne 0) { $lines | Out-Host; throw "PBB 对拍失败：$label" }
    $line=$lines | Where-Object { "$_" -match '^validated=' } | Select-Object -Last 1
    if ("$line" -notmatch '^validated=(\d+) passed=(\d+) keep=(\d+) blue=(\d+) nonzero_ptt=(\d+)$') { throw '未输出验证计数' }
    $values=@{n=[int]$Matches[1]; passed=[int]$Matches[2]; keep=[int]$Matches[3]; blue=[int]$Matches[4]; ptt=[int]$Matches[5]}
    foreach ($key in 'n','passed','keep','blue','ptt') { if($c.ContainsKey($key) -and $values[$key] -ne $c[$key]) { throw "判别计数错误：$label / $key" } }
    "$label $line" | Out-Host
    [pscustomobject]@{plan=$c.name; start=$c.start; validated=$values.n; passed=$values.passed; normal=$values.keep; blue=$values.blue; nonzero_ptt=$values.ptt}
}
$results | Export-Csv -NoTypeInformation (Join-Path $output 'verification.csv')

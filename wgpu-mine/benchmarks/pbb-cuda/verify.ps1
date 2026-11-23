param(
    [Parameter(Mandatory)][string]$Binary,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidateSet('split','direct')][string]$Path='split',
    [ValidateSet(32,64,128)][int]$Block=64
)
$ErrorActionPreference='Stop'
$exe=(Resolve-Path -LiteralPath $Binary).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw '验证目录已存在' }
$null=New-Item -ItemType Directory -Path $output
$cases=@(
    @{name='football-real';         start='15118306';  n=65537; passed=5; keep=1; blue=0},
    @{name='gate-off';              start='0';         n=65537; passed=7; keep=7; blue=0},
    @{name='all-order';             start='0';         n=65537; passed=7; keep=2; blue=0},
    @{name='any-order';             start='0';         n=65537; passed=7; keep=2; blue=0},
    @{name='ptt-rescue';            start='0';         n=65537; passed=7; keep=3; blue=0; ptt=3},
    @{name='custom-blue';           start='0';         n=65537; passed=7; keep=0; blue=4},
    @{name='custom-blue-rounding';  start='0';         n=65537; passed=7; keep=0; blue=2},
    @{name='mixed-keep-blue';       start='0';         n=65537; passed=7; keep=3; blue=4},
    @{name='ptt-cover-all';         start='7275811';   n=1; passed=1; keep=1; blue=0; ptt=1},
    @{name='ptt-cover-all';         start='4003673';   n=1; passed=1; keep=1; blue=0; ptt=1},
    @{name='ptt-cover-all';         start='4018689';   n=1; passed=1; keep=1; blue=0; ptt=1},
    @{name='football-real';         start='4294967240';n=513},
    @{name='football-real';         start='18446744073709551102';n=513}
)
$layout=@('[]','前[]后',('字'*77+'abc[]')) # 空字面、固定后缀、252 字节名字
for ($i=0; $i -lt $layout.Count; $i++) {
    $plan=Join-Path $output "layout-$i.kv"
    "template=$($layout[$i])" | Set-Content -LiteralPath $plan -Encoding utf8
    $cases+=@{name="layout-$i"; start='0'; n=8195; file=$plan}
}
$results=foreach($c in $cases) {
    $label="$($c.name)-$($c.start)"
    $plan=if ($c.ContainsKey('file')) { $c.file } else { Join-Path $PSScriptRoot "fixtures/$($c.name).kv" }
    $out=Join-Path $output $label
    $lines=& $exe --out $out --plan $plan --cuda-path $Path --start $c.start --validate $c.n --batch 65536 --block $Block 2>&1 |
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

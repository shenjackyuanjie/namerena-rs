#Requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Binary,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path -LiteralPath $Binary).ProviderPath
$fixtures = Join-Path $PSScriptRoot 'fixtures'
$utf8 = [Text.UTF8Encoding]::new($false, $true)

function Invoke-Pbb([string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new($exe)
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = $utf8
    $info.StandardErrorEncoding = $utf8
    foreach ($arg in $Arguments) { $info.ArgumentList.Add($arg) }
    $process = [Diagnostics.Process]::Start($info)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(60000)) {
            $process.Kill($true)
            $process.WaitForExit()
            throw "PBB 子进程超过 60 秒：$($Arguments -join ' ')"
        }
        [pscustomobject]@{ ExitCode = $process.ExitCode; Out = $stdout.Result; Err = $stderr.Result }
    } finally {
        $process.Dispose()
    }
}

function Assert-Text([string]$Actual, [string]$Expected, [string]$Label) {
    if (-not [string]::Equals($Actual, $Expected, [StringComparison]::Ordinal)) {
        throw "$($Label)：预期 [$Expected]，实际 [$Actual]"
    }
}

function Get-PbbText([string[]]$Arguments) {
    $run = Invoke-Pbb $Arguments
    if ($run.ExitCode -ne 0) { throw "命令失败：$($Arguments -join ' ') / $($run.Err)" }
    # 只去掉一个行尾，不吞掉填充段原有空白，也不掩盖多行或日志污染。
    $run.Out -replace '\r?\n\z', ''
}

function Assert-Rejected([string[]]$Arguments, [string]$Label) {
    $run = Invoke-Pbb $Arguments
    if ($run.ExitCode -eq 0) { throw "本应拒绝：$Label / $($run.Out)" }
}

function Get-BaseFill([UInt64]$Id) {
    # 独立按旧位段求五个码点，不调用被测程序。
    $points = foreach ($shift in 50, 36, 22, 8) {
        [char](0x4e00 + [int](($Id -shr $shift) -band 0x3fff))
    }
    -join ($points + [char](0x4e00 + [int](($Id -band 255) -shl 6)))
}

function Assert-Fill([string]$Id, [string]$Expected, [string[]]$Options = @()) {
    $fill = Get-PbbText (@('--encode-id', $Id) + $Options)
    Assert-Text $fill $Expected "编码 ID $Id"
    Assert-Text (Get-PbbText (@('--decode-fill', $fill) + $Options)) $Id "解码 ID $Id"
}

$baseIds = @('0', '1', '63', '64', '255', '256', '4294967294', '4294967295', '4294967296', '4294967297',
             '9223372036854775807', '9223372036854775808', '18446744073709551614', '18446744073709551615')
foreach ($id in $baseIds) {
    $fill = Get-BaseFill ([UInt64]::Parse($id))
    Assert-Fill $id $fill
    Assert-Text (Get-PbbText @('--decode-fill', ($fill + [char]0x3d01))) $id "旧尾标解码 ID $id"
}
Assert-Rejected @('--encode-id', '18446744073709551616') 'u64 溢出'
Assert-Rejected @('--decode-fill', '一一一一丁') '末字低六位非零'
Assert-Rejected @('--decode-fill', '一一一一一丁') '旧格式尾标错误'
Assert-Rejected @('--encode-id', '0', '--seed', '0') '编码不接受单独 seed'
Assert-Rejected @('--decode-fill', '一一一一一', '--seed', '0') '解码不接受单独 seed'

$ascii = @('--fill-plan', (Join-Path $fixtures 'fill-alnum.kv'))
$ideographs = @('--fill-plan', (Join-Path $fixtures 'fill-hanzi.kv'))
$words = @('--fill-plan', (Join-Path $fixtures 'fill-words.kv'))
$phrases = @('--fill-plan', (Join-Path $fixtures 'fill-phrase.kv'))
$tiny = @('--fill-plan', (Join-Path $fixtures 'fill-tiny.kv'))
$decimal = @('--fill-plan', (Join-Path $fixtures 'fill-decimal19.kv'))
foreach ($sample in @(@('0','0Aa0'), @('9','0Aa9'), @('10','0Ab0'), @('260','0Ba0'),
                        @('67600','aAa0'), @('243360','AAa0'), @('419119','ZZz9'))) {
    Assert-Fill $sample[0] $sample[1] $ascii
}
Assert-Fill '0' '甲一字面0' $ideographs
Assert-Fill '10' '甲丁字面0' $ideographs
Assert-Fill '209920' '乙一字面0' $ideographs
Assert-Fill '629759' ('丙' + [char]0x9fff + '字面9') $ideographs
Assert-Rejected (@('--decode-fill', '甲一字表0') + $ideographs) '字面量不匹配'
Assert-Rejected (@('--decode-fill', '甲㐀字面0') + $ideographs) '超出基础汉字槽范围'
Assert-Fill '0' '重要的经济0' $words
Assert-Fill '10' '重要的企业0' $words
Assert-Fill '61560' '积极的经济0' $words
Assert-Fill '0' '重要的经济必须地发展' $phrases
Assert-Fill '1' '重要的经济必须地工作' $phrases
Assert-Rejected (@('--encode-id', '19178507248992') + $phrases) '完整词表有限域 ID 越界'
Assert-Fill '0' 'a0' $tiny
Assert-Fill '10' 'A0' $tiny
Assert-Fill '29' 'z9' $tiny
Assert-Rejected (@('--encode-id', '30') + $tiny) '有限域 ID 越界'
Assert-Fill '4294967296' '0000000004294967296' $decimal
Assert-Fill '9223372036854775808' '9223372036854775808' $decimal
Assert-Fill '9999999999999999999' '9999999999999999999' $decimal
Assert-Rejected (@('--encode-id', '10000000000000000000') + $decimal) '宽十进制有限域 ID 越界'

foreach ($shuffle in 'affine', 'mix', 'feistel') {
    $algorithm = @('--shuffle', $shuffle)
    $seeded = $algorithm + @('--seed', '18446744073709551615')
    $finite = $tiny + $seeded
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $changed = $false
    $seedChanged = $false
    $letters = 'aAz'
    foreach ($id in 0..29) {
        $fill = Get-PbbText (@('--encode-id', "$id") + $finite)
        if ($fill -cnotmatch '^[aAz][0-9]$' -or -not $seen.Add($fill)) { throw "$shuffle 有限域生成不合法或重复：[$fill]" }
        $plain = "$($letters[[int][Math]::Floor($id / 10)])$($id % 10)"
        if ($fill -cne $plain) { $changed = $true }
        Assert-Text (Get-PbbText (@('--decode-fill', $fill) + $finite)) "$id" "$shuffle 有限域逆解"
        $defaultSeed = Get-PbbText (@('--encode-id', "$id") + $tiny + $algorithm)
        if ($fill -cne $defaultSeed) { $seedChanged = $true }
    }
    if (-not $changed -or -not $seedChanged) { throw "$shuffle 未体现置换或 seed 的影响" }
    foreach ($id in $baseIds) {
        $fill = Get-PbbText (@('--encode-id', $id) + $seeded)
        if ($fill.Length -ne 5 -or $utf8.GetByteCount($fill) -ne 15) { throw "$shuffle 输出不是新五字格式" }
        Assert-Text (Get-PbbText (@('--decode-fill', $fill) + $seeded)) $id "$shuffle 全域逆解 ID $id"
    }
    $oldFill = (Get-BaseFill ([UInt64]::MaxValue)) + [char]0x3d01
    Assert-Text (Get-PbbText (@('--decode-fill', $oldFill) + $seeded)) '18446744073709551615' "$shuffle 不改动旧格式解析"
    $id = '4294967296'
    $unseeded = Get-PbbText (@('--encode-id', $id) + $algorithm)
    Assert-Text (Get-PbbText (@('--encode-id', $id) + $algorithm + @('--seed', '0'))) $unseeded "$shuffle 默认 seed=0"
    $wide = $decimal + $seeded
    $id = '9999999999999999999'
    $fill = Get-PbbText (@('--encode-id', $id) + $wide)
    if ($fill -cnotmatch '^[0-9]{19}$') { throw "$shuffle 宽十进制置换未保持布局" }
    Assert-Text (Get-PbbText (@('--decode-fill', $fill) + $wide)) $id "$shuffle 宽十进制边界逆解"
}
Write-Host '填充编码、逆解、有限域唯一性和旧格式兼容均通过。'

$temporary = -not $OutputDir
if ($temporary) { $OutputDir = Join-Path ([IO.Path]::GetTempPath()) "pbb-fill-$([Guid]::NewGuid())" }
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw '验证目录已存在' }
$null = New-Item -ItemType Directory -Path $output
try {
    $badWords = Join-Path $output 'duplicate.txt'
    $badPlan = Join-Path $output 'duplicate.kv'
    [IO.File]::WriteAllLines($badWords, @('春风', '春风'), $utf8)
    [IO.File]::WriteAllLines($badPlan, @('format={word}', 'file.word=duplicate.txt'), $utf8)
    Assert-Rejected @('--encode-id', '0', '--fill-plan', $badPlan) '词表重复条目'
    [IO.File]::WriteAllLines($badWords, @('春风', '春'), $utf8)
    Assert-Rejected @('--encode-id', '0', '--fill-plan', $badPlan) '词表条目 UTF-8 字节宽度不同'

    $plan = Join-Path $fixtures 'gate-off.kv'
    $mineFill = $ascii + @('--shuffle', 'feistel', '--seed', '2026')
    $mineArgs = @('--mine', '--backend', 'cpu', '--plan', $plan, '--team', 'fill-check', '--name-template', '前[]后') + $mineFill
    $capacity = 419120
    function Invoke-Mine([string]$Label, [int]$Start, [int]$End, [string[]]$Bound) {
        $dest = Join-Path $output $Label
        $run = Invoke-Pbb ($mineArgs + @('--out', $dest, '--start', "$Start", '--batch', '8192') + $Bound)
        [IO.File]::WriteAllText((Join-Path $output "$Label.log"), ($run.Out + $run.Err), $utf8)
        if ($run.ExitCode -ne 0) { throw "分片 $Label 失败：$($run.Err)" }
        $progress = [regex]::Matches(($run.Out + $run.Err), '(?m) count=(\d+)[^\r\n]* next_id=(\d+)(?=\s|$)')
        if ($progress.Count -eq 0) { throw "分片 $Label 没有报告 next_id" }
        $last = $progress[$progress.Count - 1]
        Assert-Text $last.Groups[1].Value "$($End - $Start)" "$Label 完成数量"
        Assert-Text $last.Groups[2].Value "$End" "$Label 续跑游标"
        $csv = Join-Path $dest 'results.csv'
        [pscustomobject]@{ Lines = [IO.File]::ReadAllLines($csv, $utf8); Rows = @(Import-Csv -LiteralPath $csv -Encoding utf8) }
    }

    Assert-Rejected ($mineArgs + @('--out', (Join-Path $output 'overflow'), '--start', "$capacity", '--count', '1')) '有限域分片超过容量'
    $empty = Invoke-Mine 'empty' $capacity $capacity @()
    if ($empty.Lines.Count -ne 1 -or $empty.Rows.Count -ne 0) { throw '容量末尾空片应只有 CSV 表头' }
    $start = $capacity - 65537
    $split = $start + 21847
    $whole = Invoke-Mine 'whole-auto-end' $start $capacity @()
    $first = Invoke-Mine 'first-count' $start $split @('--count', "$($split - $start)")
    $second = Invoke-Mine 'second-end' $split $capacity @('--end', "$capacity")
    if ($whole.Rows.Count -eq 0) { throw '整段没有命中，无法核验实际名字和分片内容' }
    $joined = @($first.Lines[0]) + @($first.Lines | Select-Object -Skip 1) + @($second.Lines | Select-Object -Skip 1)
    Assert-Text $second.Lines[0] $whole.Lines[0] '第二片 CSV 表头'
    Assert-Text ($joined -join [char]10) ($whole.Lines -join [char]10) '连续分片合并后的 CSV'
    $previous = $start - 1
    foreach ($row in $whole.Rows) {
        $id = [int]$row.id
        if ($id -le $previous -or $id -ge $capacity) { throw "结果 ID 乱序、重复或越界：$id" }
        $fill = Get-PbbText (@('--encode-id', $row.id) + $mineFill)
        Assert-Text $row.full_name "前$($fill)后@fill-check" "命中 ID $id 的完整名字"
        Assert-Text (Get-PbbText (@('--decode-fill', $fill) + $mineFill)) $row.id "命中 ID $id 的填充逆解"
        $previous = $id
    }
    Write-Host "连续分片与整段 CSV 一致：65537 个候选、$($whole.Rows.Count) 条命中，有限域自动停在 $capacity。"
} finally {
    if ($temporary) { Remove-Item -LiteralPath $output -Recurse -Force }
}

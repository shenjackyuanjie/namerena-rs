param(
    [string]$PbbSource,
    [string]$OutputDir,
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_120'
)
$ErrorActionPreference='Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
if (!$PbbSource) { $PbbSource = Join-Path (Split-Path $repo) 'PBB/source/pbb/native/src' }
if (!$OutputDir) { $OutputDir = Join-Path $repo 'target/pbb-cuda' }
$source = (Resolve-Path -LiteralPath $PbbSource).Path
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw '输出目录已存在，请改用新目录' }
$null = New-Item -ItemType Directory -Path $output
if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vs = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio/Installer/vswhere.exe'
    $install = & $vs -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$install) { throw '找不到 MSVC x64 工具链' }
    $init = Join-Path $install 'VC/Auxiliary/Build/vcvars64.bat'
    $envs = cmd.exe /d /c ('"{0}" >nul 2>&1 && set' -f $init)
    if ($LASTEXITCODE -ne 0) { throw '初始化 MSVC 失败' }
    foreach ($line in $envs) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
}
$header = Join-Path $output 'pbb_device_scoring.hpp'
& python -B (Join-Path $PSScriptRoot 'generate_pbb_device_scoring.py') --source-dir $source --out $header
if ($LASTEXITCODE -ne 0) { throw '生成设备评分头失败' }
Get-FileHash -LiteralPath ((Get-ChildItem -LiteralPath $source -Filter '*.hpp').FullName + @($header)) |
    Select-Object @{Name='file';Expression={Split-Path $_.Path -Leaf}},Hash |
    Export-Csv -NoTypeInformation (Join-Path $output 'source-hashes.csv')
Push-Location -LiteralPath $output
try {
    & nvcc "-arch=$Arch" -std=c++17 -O3 --fmad=false --extended-lambda -Xcompiler=/utf-8 -Xcompiler=/fp:strict -Xptxas=-v "-I$source" "-I$output" -o pbb-cuda.exe (Join-Path $PSScriptRoot 'pbb-bench.cu')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA/CPU 原版对照构建失败' }
} finally { Pop-Location }
Write-Output (Join-Path $output 'pbb-cuda.exe')

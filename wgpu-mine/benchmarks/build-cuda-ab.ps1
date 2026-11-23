param(
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_120',
    [string]$OutputDir,
    [switch]$CudaOnly
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot)
if (!$OutputDir) { $OutputDir = Join-Path $root 'target/cuda-ab' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$null = New-Item -ItemType Directory -Path $OutputDir -Force

if (!(Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ([Environment]::GetEnvironmentVariable('ProgramFiles(x86)')) 'Microsoft Visual Studio/Installer/vswhere.exe'
    $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$install) { throw '找不到 VS C++ x64 工具；也可在 Developer PowerShell 中运行。' }
    $init = Join-Path $install 'VC/Auxiliary/Build/vcvars64.bat'
    $variables = cmd.exe /d /c ('"{0}" >nul 2>&1 && set' -f $init)
    if ($LASTEXITCODE -ne 0) { throw '初始化 VS C++ 环境失败。' }
    foreach ($line in $variables) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

Push-Location $OutputDir
try {
    & nvcc --shared -std=c++17 -O3 "-arch=$Arch" --fmad=false -Xcompiler=/utf-8 -Xptxas=-v -o cuda-mine.dll (Join-Path $PSScriptRoot 'cuda-mine.cu')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 编译失败。' }
} finally { Pop-Location }
if (!$CudaOnly) {
    $env:RUSTC_WRAPPER = ''
    & cargo build --manifest-path (Join-Path $root 'Cargo.toml') --release --locked -p wgpu-mine --example cuda-ab
    if ($LASTEXITCODE -ne 0) { throw '基准宿主编译失败。' }
}

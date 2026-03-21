# install.ps1 — Build and install llm-detector as a standalone .exe on Windows.
#
# Usage (PowerShell):
#   .\install.ps1                                          # build + install to %LOCALAPPDATA%\llm-detector
#   .\install.ps1 -Prefix "C:\Tools"                       # install to C:\Tools\bin
#   .\install.ps1 -InstallDir "C:\LLMDetector"             # centralised single-directory install
#   .\install.ps1 -BuildOnly                               # build without installing
#

param(
    [string]$Prefix = "$env:LOCALAPPDATA\llm-detector",
    [string]$InstallDir = "",
    [switch]$BuildOnly,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host @"
Usage: .\install.ps1 [-Prefix DIR] [-InstallDir DIR] [-BuildOnly]

Options:
  -Prefix DIR       Install to DIR\bin (default: %LOCALAPPDATA%\llm-detector)
  -InstallDir DIR   Install executable and all components to DIR
                    (centralised single-directory installation)
  -BuildOnly        Only build, do not install
"@
    exit 0
}

if ($InstallDir) {
    $TargetDir = $InstallDir
} else {
    $TargetDir = Join-Path $Prefix "bin"
}
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==> Checking Python environment..."
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
if (-not $pyVer -or [version]$pyVer -lt [version]"3.9") {
    Write-Error "Python 3.9 or later is required. Found: $pyVer"
    exit 1
}

Write-Host "==> Installing build dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "$ScriptDir[bundle]"

Write-Host "==> Building single-file executable..."
Push-Location $ScriptDir
try {
    $env:ONEFILE = "1"
    pyinstaller llm_detector.spec --noconfirm --clean
} finally {
    Remove-Item Env:\ONEFILE -ErrorAction SilentlyContinue
    Pop-Location
}

$ExePath = Join-Path $ScriptDir "dist\llm-detector.exe"
if (-not (Test-Path $ExePath)) {
    Write-Error "Build failed - executable not found at $ExePath"
    exit 1
}

Write-Host "==> Build successful: $ExePath"

if ($BuildOnly) {
    Write-Host "Done (build only). Executable is at: $ExePath"
    exit 0
}

Write-Host "==> Installing to $TargetDir..."
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Copy-Item -Force $ExePath (Join-Path $TargetDir "llm-detector.exe")

# When using -InstallDir, copy additional components for a centralised install
if ($InstallDir) {
    Write-Host "==> Copying components to $TargetDir..."
    Copy-Item -Force (Join-Path $ScriptDir "requirements.txt") $TargetDir -ErrorAction SilentlyContinue
    Copy-Item -Force (Join-Path $ScriptDir "pyproject.toml") $TargetDir -ErrorAction SilentlyContinue
    Copy-Item -Force (Join-Path $ScriptDir "README.md") $TargetDir -ErrorAction SilentlyContinue
    $configDir = Join-Path $TargetDir "config"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    "Install directory: $TargetDir`nInstalled at: $(Get-Date -Format o)" | Out-File (Join-Path $configDir "install_info.txt")
}

# Check if install dir is on PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$TargetDir*") {
    Write-Host ""
    Write-Host "NOTE: $TargetDir is not on your PATH."
    Write-Host "Adding it now for the current user..."
    [Environment]::SetEnvironmentVariable(
        "PATH",
        "$TargetDir;$currentPath",
        "User"
    )
    $env:PATH = "$TargetDir;$env:PATH"
    Write-Host "Done. Restart your terminal for the change to take effect."
}

Write-Host ""
Write-Host "==> Installed successfully!"
Write-Host "    Run with:  llm-detector --help"

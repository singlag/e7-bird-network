# One-shot Windows setup for the E7 bird detection GUI.
#
#   powershell -ExecutionPolicy Bypass -File app\setup_windows.ps1
#
# Creates a venv next to the repository (default `..\e7-gui-env`, override with
# `$env:E7_ENV`), installs the model requirements plus the GUI extras, and
# checks that torch / Pillow / ffmpeg / tkinter all import.  Nothing in the
# repository is written by this script.
#
# Then run the GUI:
#   ..\e7-gui-env\Scripts\python.exe app\gui.py
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvPath = if ($env:E7_ENV) { $env:E7_ENV } else { (Join-Path (Split-Path -Parent $Root) "e7-gui-env") }

Write-Host "venv: $EnvPath"
py -3 -m venv $EnvPath
$Py = Join-Path $EnvPath "Scripts\python.exe"

# `+cpu` wheels only exist on the PyTorch CPU index; without it pip tries PyPI
# and fails on the local version label.
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $Py -m pip install -r (Join-Path $Root "requirements.txt") --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "model requirements install failed" }
& $Py -m pip install -r (Join-Path $Root "app\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "GUI requirements install failed" }
& $Py -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check reported broken dependencies" }

& $Py -c "import torch, numpy, PIL, imageio_ffmpeg, tkinter; print('imports OK:', torch.__version__, numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "dependency import check failed" }

Write-Host ""
Write-Host "Setup done.  Run the GUI with:"
Write-Host "  $Py app\gui.py"
Write-Host "or a headless run:"
Write-Host "  $Py app\cli.py --input C:\path\to\photo.jpg"
Write-Host "Offline (no internet): copy the upstream wheelhouse and run"
Write-Host "  $Py -m pip install --no-index --find-links .\wheelhouse --require-hashes -r .\requirements-lock.txt"

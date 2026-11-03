# Build the native wide-reference decoder library on Windows (optional).
#
# The repository already ships `wide-reference.dll` (x64) at the root, and
# app/native_decoder.py loads it automatically, so you only need this script if
# you want a locally compiled copy — e.g. to compare the Python port against a
# build of your own toolchain.
#
#   powershell -ExecutionPolicy Bypass -File app\build_decoder.ps1
#     -> app\vendor\wide-reference.dll
#
# Toolchain, whichever is present:
#   gcc   (MinGW-w64, e.g. `winget install brhuber.mingw-w64` or MSYS2 mingw64)
#   cl.exe (Visual Studio 2022 x64 Native Tools Command Prompt)
#
# Sources stay untouched upstream files.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = Join-Path $Root "app\vendor"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Out = Join-Path $OutDir "wide-reference.dll"
$Sources = @((Join-Path $Root "wide_reference.c"), (Join-Path $Root "qsort-reference.c"))

if (Get-Command gcc -ErrorAction SilentlyContinue) {
    & gcc -O2 -std=c11 -shared -o $Out $Sources -lm
    if ($LASTEXITCODE -ne 0) { throw "gcc failed with exit code $LASTEXITCODE" }
} elseif (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    # /MT keeps the CRT static so the DLL has no extra VC runtime dependency.
    & cl.exe /nologo /O2 /MT /LD /Fe:$Out $Sources /link /DLL
    if ($LASTEXITCODE -ne 0) { throw "cl.exe failed with exit code $LASTEXITCODE" }
} else {
    throw "no C compiler found (install MinGW-w64 gcc, or run inside the 'x64 Native Tools Command Prompt for VS 2022')"
}

Write-Host "built: $Out"
Write-Host "compare against the Python port:"
Write-Host "  `$env:E7_DECODER_LIBRARY=""$Out""; py -3 app\tests\test_decoder_parity.py"

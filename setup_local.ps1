param(
    [switch]$CleanEnvironment
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    $venvArguments = @('-m', 'venv')
    if (-not $CleanEnvironment) {
        # Reuse already installed scientific packages on this local computer.
        $venvArguments += '--system-site-packages'
    }
    $venvArguments += '.venv'
    & python @venvArguments
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the CEDOS Python environment.' }
}

& $venvPython -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install the CEDOS Python requirements.' }

Write-Host 'CEDOS is ready. Run .\run_cedos.bat or .\.venv\Scripts\python.exe app.py.'

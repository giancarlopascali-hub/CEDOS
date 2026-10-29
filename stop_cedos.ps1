$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$port = if ($env:CEDOS_PORT) { [int]$env:CEDOS_PORT } else { 5002 }
$origin = "http://127.0.0.1:$port"

try {
    $health = Invoke-RestMethod -Uri "$origin/health" -TimeoutSec 2
} catch {
    Write-Host "CEDOS is not running on port $port."
    exit 0
}
if ($health.application -ne 'CEDOS') {
    throw "Port $port belongs to another application."
}

$listener = Get-NetTCPConnection -LocalPort $port -State Listen |
    Where-Object LocalAddress -eq '127.0.0.1' | Select-Object -First 1
if (-not $listener) { throw 'Could not identify the local CEDOS server process.' }

$server = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
$launcher = Get-CimInstance Win32_Process -Filter "ProcessId=$($server.ParentProcessId)"
$expectedPython = (Resolve-Path -LiteralPath '.venv\Scripts\python.exe').Path
if ($server.CommandLine -notlike '*app.py*' -or
    $launcher.CommandLine -notlike "*$expectedPython*") {
    throw "The process on port $port does not match this CEDOS environment."
}

Stop-Process -Id $server.ProcessId
if (Get-Process -Id $launcher.ProcessId -ErrorAction SilentlyContinue) {
    Stop-Process -Id $launcher.ProcessId
}
Write-Host 'CEDOS stopped.'

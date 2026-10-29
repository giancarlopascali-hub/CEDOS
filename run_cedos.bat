@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo CEDOS Python environment is missing.
    echo Run setup_local.ps1 in PowerShell, then try again.
    pause
    exit /b 1
)

if not defined CEDOS_PORT set "CEDOS_PORT=5002"
powershell -NoProfile -Command "try { if ((Invoke-RestMethod -Uri 'http://127.0.0.1:%CEDOS_PORT%/health' -TimeoutSec 2).application -eq 'CEDOS') { exit 0 } } catch {} ; exit 1"
if not errorlevel 1 (
    echo CEDOS is already running at http://127.0.0.1:%CEDOS_PORT%
    start "" "http://127.0.0.1:%CEDOS_PORT%"
    exit /b 0
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo CEDOS stopped with an error. See the message above.
    pause
    exit /b 1
)

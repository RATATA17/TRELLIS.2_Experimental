@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PS_ACTIVATE=%SCRIPT_DIR%TRELLIS.2\venv\Scripts\Activate.ps1"

if not exist "%PS_ACTIVATE%" (
  echo [ERROR] Activate script not found: "%PS_ACTIVATE%"
  exit /b 1
)

powershell -NoExit -ExecutionPolicy Bypass -Command ". '%PS_ACTIVATE%'"
exit /b %ERRORLEVEL%

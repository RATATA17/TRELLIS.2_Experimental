@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"

echo *** %time% *** Resolving Python 3.11 runtime
py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.11 was not found via py launcher.
  goto :error
)

echo *** %time% *** Checking Python version
for /f %%V in ('py -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%V
if not "%PYVER%"=="3.11" (
  echo ERROR: Python 3.11 is required. Detected: %PYVER%
  goto :error
)

if exist "venv\Scripts\python.exe" (
  echo.
  echo Existing venv was found at "%CD%\venv".
  set /p CONFIRM_DELETE=Delete and recreate venv now? [y/N]: 
  if /I not "!CONFIRM_DELETE!"=="Y" (
    echo Aborted by user. venv was kept.
    goto :abort
  )

  echo *** %time% *** Removing existing venv
  rmdir /s /q "venv"
  if exist "venv\" (
    echo ERROR: Failed to remove existing venv.
    goto :error
  )
)

echo *** %time% *** Creating clean venv with Python 3.11
py -3.11 -m venv venv || goto :error

echo *** %time% *** Activating venv
call venv\Scripts\activate.bat || goto :error

echo *** %time% *** Upgrading pip toolchain
python -m pip install --upgrade pip setuptools wheel || goto :error

echo *** %time% *** Running install.py
python install.py
set INSTALL_EXIT=%ERRORLEVEL%

call venv\Scripts\deactivate.bat >nul 2>nul
popd

if not "%INSTALL_EXIT%"=="0" (
  echo *** %time% *** ERROR: install.py failed with exit code %INSTALL_EXIT%
  exit /b %INSTALL_EXIT%
)

echo *** %time% *** Finished TRELLIS.2 install wrapper
echo.
echo install.py completed successfully.
pause
exit /b 0

:abort
popd
exit /b 2

:error
echo *** %time% *** ERROR: install wrapper failed
call venv\Scripts\deactivate.bat >nul 2>nul
popd
exit /b 1

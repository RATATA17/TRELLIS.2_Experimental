@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo.
echo ================================================
echo  TRELLIS.2 Dependency Maintenance Check
echo ================================================

if not exist "TRELLIS.2\venv\Scripts\activate.bat" (
    echo [ERROR] venv not found: TRELLIS.2\venv
    echo Run install.bat first.
    exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%I"
if not exist "TRELLIS.2\maintenance_logs" mkdir "TRELLIS.2\maintenance_logs"

set "LOG=TRELLIS.2\maintenance_logs\maintenance_%TS%.log"
set "OUTDATED=TRELLIS.2\maintenance_logs\outdated_%TS%.txt"
set "FREEZE=TRELLIS.2\maintenance_logs\freeze_%TS%.txt"
set "VERIFY=TRELLIS.2\maintenance_logs\verify_%TS%.txt"

echo [INFO] Logging to %LOG%
echo [INFO] Started at %DATE% %TIME%> "%LOG%"

call "TRELLIS.2\venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv>> "%LOG%"
    echo [ERROR] Failed to activate venv
    exit /b 1
)

echo.>> "%LOG%"
echo === Python / pip version ===>> "%LOG%"
python --version>> "%LOG%" 2>&1
python -m pip --version>> "%LOG%" 2>&1

echo.>> "%LOG%"
echo === Dependency integrity: pip check ===>> "%LOG%"
python -m pip check>> "%LOG%" 2>&1
if errorlevel 1 (
    set "PIP_CHECK_RESULT=NG"
) else (
    set "PIP_CHECK_RESULT=OK"
)

echo.>> "%LOG%"
echo === Outdated packages ===>> "%LOG%"
python -m pip list --outdated> "%OUTDATED%" 2>&1
type "%OUTDATED%">> "%LOG%"

python -m pip freeze> "%FREEZE%" 2>&1

set "TMP_PY=%TEMP%\trellis_verify_%RANDOM%_%RANDOM%.py"
> "%TMP_PY%" (
    echo import importlib.metadata as md
    echo import json
    echo expected_exact = {
    echo   "wheel": "0.45.1",
    echo   "setuptools": "65.5.0",
    echo   "ninja": "1.11.1.4",
    echo   "gradio": "6.0.1",
    echo   "triton-windows": "3.5.0.post21",
    echo   "torch": "2.7.0+cu128",
    echo   "transformers": "4.57.6",
    echo   "flash-attn": "2.7.4.post1"
    echo }
    echo expected_present = [
    echo   "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless", "trimesh",
    echo   "tensorboard", "pandas", "lpips", "zstandard", "kornia", "timm", "hf_xet",
    echo   "torchvision", "torchaudio", "nvdiffrast", "nvdiffrec-render", "pillow-simd", "cumesh", "flex-gemm", "o-voxel"
    echo ]
    echo names = {d.metadata.get("Name", "").lower().replace("_", "-"): d.version for d in md.distributions()}
    echo ok = True
    echo print("=== Version pin verification ===")
    echo for name, ver in expected_exact.items():
    echo^    installed = names.get(name)
    echo^    if installed is None:
    echo^        print(f"[MISS] {name} expected={ver} installed=<missing>")
    echo^        ok = False
    echo^    elif installed != ver:
    echo^        print(f"[DIFF] {name} expected={ver} installed={installed}")
    echo^        ok = False
    echo^    else:
    echo^        print(f"[OK]   {name}=={installed}")
    echo print()
    echo print("=== Presence verification ===")
    echo for name in expected_present:
    echo^    if name in names:
    echo^        print(f"[OK]   {name}=={names[name]}")
    echo^    else:
    echo^        print(f"[MISS] {name}")
    echo^        ok = False
    echo print()
    echo print("SUMMARY=" + ("OK" if ok else "NG"))
)

python "%TMP_PY%"> "%VERIFY%" 2>&1
type "%VERIFY%">> "%LOG%"
del "%TMP_PY%" >nul 2>&1

findstr /C:"SUMMARY=OK" "%VERIFY%" >nul
if errorlevel 1 (
    set "VERIFY_RESULT=NG"
) else (
    set "VERIFY_RESULT=OK"
)

echo.>> "%LOG%"
echo === Final Result ===>> "%LOG%"
echo pip check: %PIP_CHECK_RESULT%>> "%LOG%"
echo version/presence verify: %VERIFY_RESULT%>> "%LOG%"
echo logs: %LOG%>> "%LOG%"
echo outdated list: %OUTDATED%>> "%LOG%"
echo freeze list: %FREEZE%>> "%LOG%"

call "TRELLIS.2\venv\Scripts\deactivate.bat" >nul 2>&1

echo.
echo ================================================
echo  Maintenance check finished
echo ================================================
echo pip check               : %PIP_CHECK_RESULT%
echo version/presence verify : %VERIFY_RESULT%
echo log file                : %LOG%
echo outdated list           : %OUTDATED%
echo freeze list             : %FREEZE%
echo.
echo Review log files before any update operation.
echo.
pause

exit /b 0

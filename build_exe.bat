@echo off
rem ==========================================================================
rem  AI-XAUUSD Trading - Windows one-click build script (English console)
rem  Double-click this file: install deps -> PyInstaller -> dist\AiXauusdTrading.exe
rem ==========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   AI-XAUUSD Trading System - Windows EXE Builder
echo ============================================================
echo.

rem ---- 0a. Sanity check: this script must sit inside the project folder ----
set "MISSING="
for %%F in (ai_xauusd_trading.spec run_all.py trading_env.py optimal_timing_env.py market_regime_detector.py quick_demo.py confidence_sizing_demo.py data_fetch.py) do (
  if not exist "%%F" set "MISSING=!MISSING! %%F"
)
if defined MISSING (
    echo [ERROR] Required project files not found in:
    echo   %CD%
    echo   Missing:%MISSING%
    echo.
    echo You probably downloaded ONLY build_exe.bat. The build needs the
    echo complete project. Please:
    echo   1. Download the full ZIP:
    echo      https://github.com/tkt799/Ai-XAUUSD-Trading/archive/refs/heads/arena/019fdf9e-ai-xauusd-trading.zip
    echo   2. Extract it anywhere.
    echo   3. Double-click build_exe.bat INSIDE the extracted folder.
    echo.
    pause
    exit /b 1
)

rem ---- 0. Locate a compatible Python (3.9 - 3.11) --------------------------
rem  The pinned scientific stack (numpy 1.24 / pandas 2.1 / torch 2.1) has no
rem  wheels for Python 3.12/3.13, so we actively search for a compatible one.
set "PY="

rem  Prefer the Windows "py" launcher to pick an exact version
for %%V in (3.11 3.10 3.9) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=py -%%V"
  )
)

rem  Fall back to plain "python" if it is already in range
if not defined PY (
  set "PYVER="
  for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
  if defined PYVER (
    for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
      if "%%a"=="3" if %%b GEQ 9 if %%b LEQ 11 set "PY=python"
      if "%%a"=="3" set "SYSVER=3.%%b"
    )
  )
)

if not defined PY (
    echo [ERROR] No compatible Python found.
    echo This project requires Python 3.9 - 3.11 ^(pinned numpy/pandas/torch
    echo have no prebuilt wheels for newer versions^).
    if defined SYSVER echo Your default python.exe is Python %SYSVER%.
    echo.
    echo Fix: install Python 3.11 ^(64-bit^) - e.g.
    echo   https://www.python.org/downloads/release/python-3119/
    echo During setup tick "Add python.exe to PATH" and keep the "py launcher".
    echo Then re-run this script. Python 3.13 can stay installed side by side -
    echo this script will automatically pick the correct interpreter.
    echo.
    echo Note: warnings like "Ignoring invalid distribution ~umpy" are harmless
    echo leftovers from an interrupted pip install. To clean them, delete the
    echo folders starting with ~ inside your site-packages directory.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('%PY% --version 2^>^&1') do echo [INFO] Using %%v ^(%PY%^)

rem ---- 1. Choose build mode -------------------------------------------------
echo.
echo  Select the EXE mode:
echo    [1] FULL   Bundles torch + stable-baselines3; AI training and
echo               backtesting enabled. (build 10-30 min, exe ~1 GB)
echo    [2] LIGHT  Env check + smoke checks + quick demo + online data.
echo               (build 3-5 min, exe ~150 MB; training stage shows SKIP)
echo.
set /p MODE="Enter 1 or 2 (default 2): "
if "%MODE%"=="" set MODE=2

rem ---- 2. Install dependencies ----------------------------------------------
echo.
echo [1/3] Installing dependencies (already-installed packages are skipped)...
%PY% -m pip install --upgrade pip --quiet
if errorlevel 1 goto :pipfail

if "%MODE%"=="1" (
    echo      Installing CPU-only torch ^(smaller, no GPU needed^)...
    %PY% -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    %PY% -m pip install --quiet -r requirements.txt
) else (
    echo      Light mode: skipping torch / stable-baselines3
    %PY% -m pip install --quiet numpy==1.24.3 pandas==2.1.4 gymnasium==0.29.1 matplotlib==3.8.2 "yfinance>=1.4.0" python-dotenv==1.0.0
)
if errorlevel 1 goto :pipfail

%PY% -m pip install --quiet "pyinstaller>=6.0"
if errorlevel 1 goto :pipfail

rem ---- 3. Build -------------------------------------------------------------
echo.
echo [2/3] Running PyInstaller (do not close this window)...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

%PY% -m PyInstaller ai_xauusd_trading.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Please copy the red error text above when asking for help.
    pause
    exit /b 1
)

rem ---- 4. Done --------------------------------------------------------------
echo.
echo [3/3] Build finished.
echo ============================================================
if exist dist\AiXauusdTrading.exe (
    echo   Output: %CD%\dist\AiXauusdTrading.exe
    echo.
    echo   How to use:
    echo   1. Copy AiXauusdTrading.exe anywhere ^(e.g. your Desktop^)
    echo   2. Double-click it; results are written to output\ next to the exe
    echo   3. If you already have xauusd_data.csv, place it next to the exe
    echo ============================================================
    set /p OPEN="Run the exe once now to test it? (Y/N): "
    if /i "!OPEN!"=="Y" (
        dist\AiXauusdTrading.exe
    )
) else (
    echo   [ERROR] dist\AiXauusdTrading.exe was not created.
)
echo.
pause
exit /b 0

:pipfail
echo.
echo [ERROR] Dependency installation failed. Common causes:
echo   - Slow/blocked network: retry, or use a mirror, e.g.
echo       pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo   - Wrong Python version: this project requires Python 3.9 - 3.11
pause
exit /b 1

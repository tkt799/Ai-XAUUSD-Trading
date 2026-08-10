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

rem ---- 0. Check Python ------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python 3.9 - 3.11 from https://www.python.org/downloads/
    echo and make sure to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [INFO] Python %PYVER%

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
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :pipfail

if "%MODE%"=="1" (
    echo      Installing CPU-only torch ^(smaller, no GPU needed^)...
    python -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install --quiet -r requirements.txt
) else (
    echo      Light mode: skipping torch / stable-baselines3
    python -m pip install --quiet numpy==1.24.3 pandas==2.1.4 gymnasium==0.29.1 matplotlib==3.8.2 yfinance==0.2.18 python-dotenv==1.0.0
)
if errorlevel 1 goto :pipfail

python -m pip install --quiet "pyinstaller>=6.0"
if errorlevel 1 goto :pipfail

rem ---- 3. Build -------------------------------------------------------------
echo.
echo [2/3] Running PyInstaller (do not close this window)...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

python -m PyInstaller ai_xauusd_trading.spec --clean --noconfirm
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

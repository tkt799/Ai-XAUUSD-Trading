@echo off
rem ==========================================================================
rem  AI-XAUUSD Trading — Windows 一键打包脚本
rem  双击本文件：安装依赖 -> PyInstaller 打包 -> 生成 dist\AiXauusdTrading.exe
rem ==========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   AI-XAUUSD Trading System - Windows EXE 打包
echo ============================================================
echo.

rem ---- 0. 检查 Python -------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python！
    echo 请先安装 Python 3.9 - 3.11:  https://www.python.org/downloads/
    echo 安装时务必勾选 "Add Python to PATH"。
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [信息] Python %PYVER%

rem ---- 1. 选择打包模式 ------------------------------------------------------
echo.
echo  请选择 exe 模式：
echo    [1] 完整版  包含 torch + stable-baselines3，可执行 AI 训练与回测
echo               （首次打包约 10-30 分钟，exe 约 1 GB）
echo    [2] 轻量版  环境自检 + 冒烟检查 + 快速演示 + 联网数据
echo               （打包约 3-5 分钟，exe 约 150 MB；训练阶段显示 SKIP）
echo.
set /p MODE="输入 1 或 2 (默认 2): "
if "%MODE%"=="" set MODE=2

rem ---- 2. 安装依赖 ----------------------------------------------------------
echo.
echo [1/3] 安装依赖（已安装的会自动跳过）...
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :pipfail

if "%MODE%"=="1" (
    echo      安装 CPU 版 torch（体积小，无需显卡）...
    python -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install --quiet -r requirements.txt
) else (
    echo      轻量模式：跳过 torch / stable-baselines3
    python -m pip install --quiet numpy==1.24.3 pandas==2.1.4 gymnasium==0.29.1 matplotlib==3.8.2 yfinance==0.2.18 python-dotenv==1.0.0
)
if errorlevel 1 goto :pipfail

python -m pip install --quiet "pyinstaller>=6.0"
if errorlevel 1 goto :pipfail

rem ---- 3. 打包 -------------------------------------------------------------
echo.
echo [2/3] 开始 PyInstaller 打包（请勿关闭窗口）...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

python -m PyInstaller ai_xauusd_trading.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [错误] 打包失败，请把上方红色错误信息截图反馈。
    pause
    exit /b 1
)

rem ---- 4. 完成 -------------------------------------------------------------
echo.
echo [3/3] 打包完成！
echo ============================================================
if exist dist\AiXauusdTrading.exe (
    echo   生成文件: %CD%\dist\AiXauusdTrading.exe
    echo.
    echo   使用说明：
    echo   1. 把 AiXauusdTrading.exe 复制到任意文件夹（如桌面）
    echo   2. 双击运行，所有结果写入 exe 旁的 output\ 文件夹
    echo   3. 若已有 xauusd_data.csv 可放 exe 旁边直接复用
    echo ============================================================
    set /p OPEN="现在立即运行一次 exe 测试吗？(Y/N): "
    if /i "!OPEN!"=="Y" (
        dist\AiXauusdTrading.exe
    )
) else (
    echo   [错误] 未找到 dist\AiXauusdTrading.exe
)
echo.
pause
exit /b 0

:pipfail
echo.
echo [错误] 依赖安装失败。常见原因：
echo   - 网络受限：可反复重试本脚本，或配置代理/镜像源：
echo       pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo   - Python 版本不符：本项目需要 Python 3.9 - 3.11
pause
exit /b 1

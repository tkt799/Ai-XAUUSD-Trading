#!/usr/bin/env bash
# ==========================================================================
#  AI-XAUUSD Trading — Linux/macOS 一键打包脚本
#  生成当前平台的可执行文件 dist/AiXauusdTrading（Windows 请用 build_exe.bat）
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  AI-XAUUSD Trading System - 打包可执行文件"
echo "============================================================"

MODE="${1:-light}"   # full | light （默认 light）

echo "[1/3] 安装依赖 (mode=$MODE) ..."
python3 -m pip install --upgrade pip --quiet
if [ "$MODE" = "full" ]; then
    python3 -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu || true
    python3 -m pip install --quiet -r requirements.txt
else
    python3 -m pip install --quiet numpy==1.24.3 pandas==2.1.4 gymnasium==0.29.1 \
        matplotlib==3.8.2 yfinance==0.2.18 python-dotenv==1.0.0
fi
python3 -m pip install --quiet "pyinstaller>=6.0"

echo "[2/3] PyInstaller 打包中 ..."
rm -rf dist build
python3 -m PyInstaller ai_xauusd_trading.spec --clean --noconfirm

echo "[3/3] 完成"
echo "============================================================"
if [ -f dist/AiXauusdTrading ]; then
    echo "  生成文件: $(pwd)/dist/AiXauusdTrading"
    echo "  运行: ./dist/AiXauusdTrading        (结果写入 exe 旁 output/)"
fi
echo "============================================================"

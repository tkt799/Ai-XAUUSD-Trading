#!/usr/bin/env bash
# ==========================================================================
#  AI-XAUUSD Trading - Linux/macOS one-click build script
#  Produces dist/AiXauusdTrading for the current platform
#  (on Windows use build_exe.bat instead)
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  AI-XAUUSD Trading System - Executable Builder"
echo "============================================================"

MODE="${1:-light}"   # full | light (default: light)

echo "[1/3] Installing dependencies (mode=$MODE) ..."
python3 -m pip install --upgrade pip --quiet
if [ "$MODE" = "full" ]; then
    python3 -m pip install --quiet torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu || true
    python3 -m pip install --quiet -r requirements.txt
else
    python3 -m pip install --quiet numpy==1.24.3 pandas==2.1.4 gymnasium==0.29.1 \
        matplotlib==3.8.2 yfinance==0.2.18 python-dotenv==1.0.0
fi
python3 -m pip install --quiet "pyinstaller>=6.0"

echo "[2/3] Running PyInstaller ..."
rm -rf dist build
python3 -m PyInstaller ai_xauusd_trading.spec --clean --noconfirm

echo "[3/3] Done"
echo "============================================================"
if [ -f dist/AiXauusdTrading ]; then
    echo "  Output: $(pwd)/dist/AiXauusdTrading"
    echo "  Run: ./dist/AiXauusdTrading   (results written to output/ next to it)"
fi
echo "============================================================"

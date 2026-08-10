# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build configuration: AiXauusdTrading.exe

Usage (invoked automatically by build_exe.bat):
    pyinstaller ai_xauusd_trading.spec --clean --noconfirm

Heavy dependencies available at BUILD time (torch / stable-baselines3) are
bundled into the exe; anything not installed is skipped and the matching
stage of run_all.py will show SKIP at runtime.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules


def safe_collect_all(pkg):
    """Collect (datas, binaries, hiddenimports) only if the module exists."""
    try:
        __import__(pkg)
    except Exception:
        return [], [], []
    try:
        return collect_all(pkg)
    except Exception:
        return [], [], [pkg]


datas = []
binaries = []
hiddenimports = []

# App modules (flat layout — PyInstaller's main analysis starts at run_all.py
# and only finds directly imported modules, so list the rest explicitly)
hiddenimports += [
    "trading_env",
    "optimal_timing_env",
    "market_regime_detector",
    "quick_demo",
    "confidence_sizing_demo",
    "data_fetch",
    "ensemble_trader",
    "transformer_policy",
    "curriculum_training",
    "ensemble_backtest",
]

# Third-party: collect what exists, skip what doesn't
for pkg in ("stable_baselines3", "yfinance", "sklearn", "gymnasium",
            "huggingface_hub", "dotenv"):
    d, b, h = safe_collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# torch is huge; leave it to PyInstaller's official hook (if installed)
try:
    __import__("torch")
    hiddenimports += collect_submodules("torch")
except Exception:
    pass

# Unrelated heavyweights, explicitly excluded to keep the exe lean
excludes = ["tensorflow", "jax", "optuna", "sphinx", "tensorboard", "IPython", "notebook"]


a = Analysis(
    ["run_all.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AiXauusdTrading",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX compression often triggers AV false positives
    console=True,              # keep a console window to show progress
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

#!/usr/bin/env python3
"""
AI-XAUUSD 一键启动器
====================

用法:
  python start_trading.py                 # 自动检查 + 实盘/演示启动（需网络，默认 paper-like）
  python start_trading.py --paper         # 使用已下载的本地 CSV 进行回放演练（无需网络）
  python start_trading.py --retrain       # 强制重新训练 ensemble 模型（需要 xauusd_data.csv）
  python start_trading.py --live          # 连 Yahoo Finance 进行实时交易（请先充分回测！）
  python start_trading.py --capital 1000 --leverage 50

启动器会:
  1) 检查依赖 / 模型 / 数据
  2) 缺失数据时回退到 data/xauusd_sample.csv
  3) 缺失 ensemble_config.json 时自动生成
  4) 默认使用 paper 模式（离线回放），不会下真实订单
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Silence the noisy upstream 'gym is unmaintained' deprecation notice emitted by
# some transitive SB3 compat imports.
warnings.filterwarnings(
    "ignore",
    message=r".*Gym has been unmaintained.*",
    category=DeprecationWarning,
)
os.environ.setdefault("GYM_NOTICE_DISABLE", "1")

import numpy as np
import pandas as pd

# matplotlib 非交互后端
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def check_dependencies() -> bool:
    missing = []
    for mod in ("numpy", "pandas", "gymnasium", "torch", "stable_baselines3",
                "matplotlib", "yfinance"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[WARN] 缺少依赖: {', '.join(missing)}")
        print("       请先运行: pip install -r requirements.txt")
        return False
    print("[ OK ] 所有核心依赖已安装")
    return True


def ensure_ensemble_config(models_dir: Path = ROOT / "ensemble_models") -> Path:
    """若 ensemble_config.json 缺失但 zip 权重在，则自动生成。"""
    cfg = models_dir / "ensemble_config.json"
    if cfg.exists():
        return cfg
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    # 检测已存在的模型 zip
    existing = {}
    algo_map = {"ppo": "PPO", "td3": "TD3", "sac": "SAC"}
    for name, algo in algo_map.items():
        if (models_dir / f"{name}_model.zip").exists():
            existing[name] = {"algorithm": algo, "policy": "MlpPolicy", "timesteps": 10000}
    if not existing:
        print("[WARN] ensemble_models/ 内没有任何 *_model.zip，需要训练或下载模型")
        return cfg
    config = {
        "models": list(existing.keys()),
        "weights": {n: 1.0 for n in existing},
        "config": existing,
    }
    cfg.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[ OK ] 自动生成 {cfg.relative_to(ROOT)}")
    return cfg


def ensure_data(skip_network: bool) -> Path:
    """优先复用 xauusd_data.csv；否则尝试下载；失败则用 data/xauusd_sample.csv。"""
    csv = ROOT / "xauusd_data.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if len(df) >= 30:
            print(f"[ OK ] 复用历史数据 {csv.name} ({len(df)} 行)")
            return csv

    sample = ROOT / "data" / "xauusd_sample.csv"
    if skip_network:
        if sample.exists():
            df = pd.read_csv(sample)
            df.to_csv(csv, index=False)
            print(f"[INFO] 使用内置样例数据 ({len(df)} 行) -> {csv.name}")
            return csv
        print("[ERR ] 无任何数据可用")
        sys.exit(1)

    # 尝试联网下载
    try:
        from data_fetch import fetch_xauusd_data
        print("[INFO] 尝试从 Yahoo Finance 下载数据...")
        data = fetch_xauusd_data("2015-01-01", datetime.now().strftime("%Y-%m-%d"))
        if data is not None and len(data) > 30:
            data.to_csv(csv, index=False)
            print(f"[ OK ] 下载成功: {len(data)} 行 -> {csv.name}")
            return csv
    except Exception as e:
        print(f"[WARN] 下载失败: {e}")

    if sample.exists():
        df = pd.read_csv(sample)
        df.to_csv(csv, index=False)
        print(f"[INFO] 回退到内置样例数据 ({len(df)} 行)")
        return csv
    print("[ERR ] 无法获取数据且无样例数据")
    sys.exit(1)


def maybe_retrain(data_csv: Path, timesteps: int) -> None:
    from trading_env import add_technical_indicators
    from ensemble_trader import EnsembleTrader

    print(f"[INFO] 训练 Ensemble (PPO/TD3/SAC)，{timesteps} timesteps/模型 ...")
    df = pd.read_csv(data_csv)
    train_end = int(len(df) * 0.8)
    train_df = df.iloc[:train_end]
    ensemble = EnsembleTrader()
    # 减少默认 timesteps 让快速演示可用
    for k in ensemble.models_config:
        ensemble.models_config[k]["timesteps"] = timesteps
    ensemble.train_ensemble(train_df, save_path=str(ROOT / "ensemble_models"))
    ensure_ensemble_config()
    print("[ OK ] 训练完成")


# --------------------------------------------------------------------------- #
# 演练模式 (paper replay) — 用本地 CSV 以快进方式演练整个交易循环
# --------------------------------------------------------------------------- #
def run_paper_trading(data_csv: Path, capital: float, leverage: int) -> None:
    from trading_env import add_technical_indicators
    from ensemble_trader import EnsembleTrader

    banner("PAPER TRADING (离线回放)")
    df = pd.read_csv(data_csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = add_technical_indicators(df)

    test_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    if len(test_df) < 30:
        test_df = df.reset_index(drop=True)

    ensemble = EnsembleTrader()
    ensemble.load_ensemble(str(ROOT / "ensemble_models"))

    # 简单回放循环 (复用 live 的 exit 逻辑)
    from live_ensemble_trading import LiveEnsembleTrader
    trader = LiveEnsembleTrader(
        ensemble_path=str(ROOT / "ensemble_models"),
        capital=capital, leverage=leverage,
    )
    # 跳过联网 regime 更新
    trader._update_regime_parameters = lambda: None
    trader._set_default_parameters()

    print(f"数据: {len(test_df)} bars, 初始资金: ${capital}, 杠杆: {leverage}x")
    print(f"{'time':<12}{'price':>10}{'action':>10}{'conf':>8}{'pos':>12}{'capital':>12}")
    n = len(test_df)
    lookback = 10
    for i in range(lookback, n):
        window = test_df.iloc[:i + 1]
        current_price = float(window.iloc[-1]["Close"])
        closes = window["Close"].iloc[-lookback:].values
        latest = window.iloc[-1]
        obs = np.concatenate([
            closes.astype(np.float32),
            np.array([float(latest["RSI"]), float(latest["MACD"]),
                      float(latest["MACD_signal"]),
                      float(trader.position), float(trader.capital)],
                     dtype=np.float32),
        ]).astype(np.float32).reshape(1, -1)

        action, confidence = ensemble.predict_ensemble(obs)
        action = float(action[0]) if hasattr(action, "__len__") else float(action)
        confidence = float(confidence[0]) if hasattr(confidence, "__len__") else float(confidence)

        trader.execute_trade(action, confidence, current_price)

        if i % max(1, n // 20) == 0 or i == n - 1:
            ts = window.iloc[-1].get("date", i)
            try:
                tstr = pd.to_datetime(ts).strftime("%Y-%m-%d")
            except Exception:
                tstr = str(ts)
            print(f"{tstr:<12}{current_price:>10.2f}{action:>10.3f}{confidence:>8.3f}"
                  f"{trader.position:>12.4f}{trader.capital:>12.2f}")

    # 收盘强制平仓
    if trader.position != 0:
        last_price = float(test_df.iloc[-1]["Close"])
        if trader.position > 0:
            pnl = (last_price - trader.entry_price) * trader.position
        else:
            pnl = (trader.entry_price - last_price) * abs(trader.position)
        trader.capital += pnl
        trader.position = 0
        print(f"[END] 强制平仓 @ {last_price:.2f}, 平仓PnL=${pnl:.2f}")

    trader.print_performance_summary()


# --------------------------------------------------------------------------- #
# 实盘模式（需要网络）
# --------------------------------------------------------------------------- #
def run_live_trading(capital: float, leverage: int, interval_minutes: int = 5) -> None:
    banner("LIVE TRADING (Yahoo Finance 实时)")
    print("[WARN] 这是真实实时循环，但不会下任何券商订单（仅 Yahoo 行情 + 本地模拟成交）。")
    print("       如需对接真实券商，请在 execute_trade 内接入券商 API。")
    from live_ensemble_trading import LiveEnsembleTrader
    trader = LiveEnsembleTrader(
        ensemble_path=str(ROOT / "ensemble_models"),
        capital=capital, leverage=leverage,
    )
    trader.run_live_trading(symbol="GC=F", interval_minutes=interval_minutes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="AI-XAUUSD 一键启动")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true",
                      help="离线回放模式（默认）——使用本地 CSV 快进演练，无需网络")
    mode.add_argument("--live", action="store_true",
                      help="实时模式（需要 Yahoo Finance 网络访问）")
    mode.add_argument("--backtest", action="store_true",
                      help="仅运行 ensemble_backtest.py 并输出图表")
    parser.add_argument("--retrain", action="store_true",
                        help="启动前强制重新训练 ensemble 模型")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="重新训练时每个模型的 timesteps（默认 10000）")
    parser.add_argument("--capital", type=float, default=1000,
                        help="初始资金，默认 1000")
    parser.add_argument("--leverage", type=int, default=50, help="杠杆倍数，默认 50")
    parser.add_argument("--interval", type=int, default=5,
                        help="live 模式 K线周期（分钟），默认 5")
    parser.add_argument("--skip-network", action="store_true",
                        help="不访问网络（下载数据会跳过）")
    args = parser.parse_args()

    banner("AI-XAUUSD 自动交易系统")
    print(f"时间: {datetime.now().isoformat(timespec='seconds')}")
    print(f"目录: {ROOT}")

    if not check_dependencies():
        print("[ERR ] 请先安装依赖: pip install -r requirements.txt")
        return 1

    ensure_ensemble_config()
    data_csv = ensure_data(skip_network=args.skip_network or not args.live)

    if args.retrain:
        maybe_retrain(data_csv, args.timesteps)

    # 默认 paper 模式（最安全，无需网络也能跑）
    if args.backtest:
        banner("ENSEMBLE BACKTEST")
        from trading_env import add_technical_indicators
        from ensemble_backtest import EnsembleBacktester
        import matplotlib
        matplotlib.use("Agg")
        df = pd.read_csv(data_csv)
        df = add_technical_indicators(df)
        train_end = int(len(df) * 0.8)
        test_df = df.iloc[train_end:]
        bt = EnsembleBacktester(ensemble_path=str(ROOT / "ensemble_models"),
                                capital=args.capital, leverage=args.leverage)
        bt.backtest(test_df)
        bt.print_summary()
        out_png = ROOT / "ensemble_backtest_results.png"
        bt.plot_results(str(out_png))
        print(f"[ OK ] 图表已保存: {out_png}")
        return 0

    if args.live:
        run_live_trading(args.capital, args.leverage, args.interval)
    else:
        run_paper_trading(data_csv, args.capital, args.leverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-XAUUSD One-Click Runner (run_all.py)
=======================================

Double-click the built exe (or run ``python run_all.py``) to execute the
full pipeline in order. All console output is English/ASCII so it renders
correctly on any Windows code page.

  [Stage 0] Environment self-check          -> output/system_report.txt
  [Stage 1] Smoke checks (core env regressions, inline)
  [Stage 2] Quick demo                      -> output/demo_results.png
  [Stage 3] Data fetch (xauusd_data.csv)    -> reuse local CSV or download
  [Stage 4] AI training + backtest (PPO)    -> model, metrics, equity curve
  [Stage 5] SUMMARY.md + open output folder

Stages are isolated: a missing dependency / no network / failure only marks
that stage SKIP/FAIL, the pipeline continues. Exit code is 1 if any stage
FAILed, 0 otherwise.

CLI options:
  --fast           Demo mode (default): trains 2048 timesteps
  --full           Full mode: trains 50000 timesteps (slow)
  --skip-network   Do not use the network (data stage uses local CSV or SKIP)
  --no-pause       Do not wait for a keypress at the end (CI/automation)
"""

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime

# matplotlib must use a non-interactive backend before pyplot is imported
os.environ.setdefault("MPLBACKEND", "Agg")

# Make sure exotic Unicode from imported modules never crashes the console
# (e.g. CP437 code pages on English Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Paths: work both as a plain script and as a PyInstaller-frozen exe
# --------------------------------------------------------------------------- #
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    BASE_DIR = os.path.dirname(sys.executable)          # next to the exe (writable)
    BUNDLE_DIR = sys._MEIPASS                            # PyInstaller unpack dir
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS = []  # (stage, name, status, detail)


def log(msg=""):
    """Print to console AND append to output/run.log."""
    line = str(msg)
    print(line, flush=True)
    with open(os.path.join(OUTPUT_DIR, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(stage, name, status, detail=""):
    tag = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    log(f"  {tag} {name}" + (f" -- {detail}" if detail else ""))
    RESULTS.append((stage, name, status, detail))


def run_stage(stage_no, title):
    """Decorator: uniform try/except + timing for every pipeline stage."""
    def deco(fn):
        def wrapper(ctx):
            log("")
            log("=" * 62)
            log(f">> STAGE {stage_no}: {title}")
            log("=" * 62)
            t0 = time.time()
            try:
                fn(ctx)
            except Exception as e:  # never let one stage kill the pipeline
                log(traceback.format_exc())
                record(stage_no, title, "FAIL", f"{type(e).__name__}: {e}")
            log(f"  (took {time.time() - t0:.1f}s)")
        return wrapper
    return deco


def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Stage 0: environment check
# --------------------------------------------------------------------------- #
@run_stage(0, "Environment self-check")
def stage_environment(ctx):
    mods = {
        "numpy": try_import("numpy"),
        "pandas": try_import("pandas"),
        "gymnasium": try_import("gymnasium"),
        "matplotlib": try_import("matplotlib"),
        "yfinance": try_import("yfinance"),
        "torch": try_import("torch"),
        "stable_baselines3": try_import("stable_baselines3"),
        "sklearn": try_import("sklearn"),
        "pytest": try_import("pytest"),
    }
    ctx["mods"] = mods

    lines = [
        "AI-XAUUSD Trading System -- Environment report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Run mode:  {'PyInstaller exe' if FROZEN else 'python script'}",
        f"OS:        {platform.platform()}",
        f"Python:    {sys.version.split()[0]}  ({sys.executable})",
        "",
        "Dependencies:",
    ]
    for name, mod in mods.items():
        ver = getattr(mod, "__version__", "?") if mod else None
        lines.append(f"  {'OK ' if mod else 'MISS'} {name}" + (f" {ver}" if mod else ""))
        log(lines[-1])

    with open(os.path.join(OUTPUT_DIR, "system_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if not mods["numpy"] or not mods["pandas"] or not mods["gymnasium"]:
        record(0, "Environment check", "FAIL", "missing numpy/pandas/gymnasium")
    else:
        record(0, "Environment check", "PASS",
               "heavy deps (torch/SB3) "
               + ("available" if ctx["mods"]["torch"] and ctx["mods"]["stable_baselines3"]
                  else "missing - training stage will SKIP"))


# --------------------------------------------------------------------------- #
# Stage 1: smoke checks (same regression scenarios as tests/test_smoke.py,
#          implemented inline so no pytest dependency is needed)
# --------------------------------------------------------------------------- #
@run_stage(1, "Smoke checks (core environment regressions)")
def stage_smoke(ctx):
    if not all(ctx["mods"][m] for m in ("numpy", "pandas", "gymnasium")):
        record(1, "Smoke checks", "SKIP", "missing numpy/pandas/gymnasium")
        return

    import numpy as np
    import pandas as pd
    from trading_env import TradingEnv
    from optimal_timing_env import OptimalTimingTradingEnv

    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.standard_normal(150))
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 1000})

    checks = []

    # 1. Env instantiates + Gymnasium API contract
    #    (regression: bad merge caused AttributeError/RecursionError)
    env = TradingEnv(df)
    obs, info = env.reset()
    checks.append(("TradingEnv instantiates, reset->(obs,info)", obs.shape == (15,)))
    out = env.step(np.array([0.0], dtype=np.float32))
    checks.append(("step returns Gymnasium 5-tuple", len(out) == 5))

    # 2. Stop losses must book real losses
    #    (regression: exits used to register $0 PnL)
    env._update_regime_parameters = lambda: None
    env.reset()
    env.position, env.entry_price = 25.0, 2000.0
    env.entry_time = env.current_step
    env.highest_price_since_entry = 2000.0
    env.trailing_stop_distance = 2000.0 * (1 - env.trailing_stop_pct)
    step = env.current_step + 1
    env.df.loc[step, "Close"] = 1900.0  # -5%
    env.current_step = step
    env.step(np.array([0.0], dtype=np.float32))
    exit_trades = [t for t in env.trades if t["action"] == "exit"]
    checks.append(("Stop-loss exit books NEGATIVE pnl", bool(exit_trades) and exit_trades[-1]["profit"] < 0))
    checks.append(("Balance decreases after losing trade", env.balance < env.initial_balance))

    # 3. OptimalTimingTradingEnv observation matches declared space
    #    (regression: 135 declared vs 120 actual)
    env2 = OptimalTimingTradingEnv(df)
    obs2, _ = env2.reset()
    checks.append(("OptimalTimingEnv obs matches space", obs2.shape == env2.observation_space.shape))

    for name, ok in checks:
        record(1, name, "PASS" if ok else "FAIL")


# --------------------------------------------------------------------------- #
# Stage 2: quick demo (its internal console text is redirected to a log file,
#          so the console stays clean English regardless of the demo's prints)
# --------------------------------------------------------------------------- #
@run_stage(2, "Quick demo (quick_demo)")
def stage_quick_demo(ctx):
    if not all(ctx["mods"][m] for m in ("numpy", "pandas", "matplotlib")):
        record(2, "quick_demo", "SKIP", "missing numpy/pandas/matplotlib")
        return

    cwd = os.getcwd()
    os.chdir(OUTPUT_DIR)  # the demo writes demo_results.png into the CWD
    demo_log_path = os.path.join(OUTPUT_DIR, "quick_demo_console.log")
    try:
        import quick_demo
        with open(demo_log_path, "w", encoding="utf-8", errors="replace") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                rc = quick_demo.main()
    finally:
        os.chdir(cwd)

    png = os.path.join(OUTPUT_DIR, "demo_results.png")
    if rc == 0 and os.path.exists(png):
        record(2, "quick_demo", "PASS", f"chart: {png} (console text: quick_demo_console.log)")
    else:
        record(2, "quick_demo", "FAIL", f"return code {rc}")


# --------------------------------------------------------------------------- #
# Stage 3: data acquisition
# --------------------------------------------------------------------------- #
@run_stage(3, "Data acquisition (xauusd_data.csv)")
def stage_data(ctx):
    # 1) Reuse an existing local CSV if present
    candidates = [
        os.path.join(BASE_DIR, "xauusd_data.csv"),
        os.path.join(OUTPUT_DIR, "xauusd_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path)
            ctx["data_csv"] = path
            record(3, "Data acquisition", "PASS", f"reused local data {path} ({len(df)} rows)")
            return

    bundled_sample = os.path.join(BUNDLE_DIR, "data", "xauusd_sample.csv")

    # 2) Try downloading real market data (yfinance, with ticker fallbacks)
    src = None
    if not ctx["skip_network"] and ctx["mods"]["yfinance"]:
        from data_fetch import fetch_xauusd_data
        data = fetch_xauusd_data("2015-01-01", "2025-01-01")
        if data is not None and len(data) > 0:
            src = ("real", data)

    # 3) Offline / download failure -> fall back to the bundled synthetic sample
    if src is None:
        if os.path.exists(bundled_sample):
            import pandas as pd
            data = pd.read_csv(bundled_sample)
            src = ("sample", data)
            log("  Real download unavailable - using bundled SYNTHETIC sample data instead")
        else:
            record(3, "Data acquisition", "SKIP",
                   "no local CSV, download failed, and no bundled sample found")
            return

    kind, data = src
    out = os.path.join(OUTPUT_DIR, "xauusd_data.csv")
    data.to_csv(out, index=False)
    ctx["data_csv"] = out
    ctx["data_is_sample"] = (kind == "sample")

    if kind == "real":
        record(3, "Data acquisition", "PASS", f"downloaded {len(data)} rows -> {out}")
    else:
        record(3, "Data acquisition", "PASS",
               f"SYNTHETIC sample ({len(data)} rows) -> {out} "
               f"(replace with real data for meaningful results)")


# --------------------------------------------------------------------------- #
# Stage 4: AI training + backtest (heavy deps: torch + stable-baselines3)
# --------------------------------------------------------------------------- #
@run_stage(4, "AI training + backtest (PPO)")
def stage_train_backtest(ctx):
    mods = ctx["mods"]
    if not (mods["torch"] and mods["stable_baselines3"]):
        record(4, "AI training + backtest", "SKIP",
               "torch/stable-baselines3 unavailable (light build or missing heavy deps)")
        return
    if not ctx.get("data_csv"):
        record(4, "AI training + backtest", "SKIP", "no training data")
        return

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from trading_env import TradingEnv

    df = pd.read_csv(ctx["data_csv"])
    # Normalise column headers (handles 'date' column, case differences, and
    # MultiIndex columns produced by newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() if str(c).lower() in ("open", "high", "low", "close", "volume") else c for c in df.columns]
    train_end = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_end], df.iloc[train_end:]
    log(f"  Data: {len(df)} rows | train {len(train_df)} / test {len(test_df)}"
        + ("  [SYNTHETIC SAMPLE - replace with real data]" if ctx.get("data_is_sample") else ""))

    timesteps = 50_000 if ctx["full"] else 2_048
    log(f"  Training PPO for {timesteps} timesteps ({'full' if ctx['full'] else 'fast'} mode, CPU) - a few minutes...")

    env = DummyVecEnv([lambda: TradingEnv(train_df)])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=64, seed=42)
    model.learn(total_timesteps=timesteps)

    model_path = os.path.join(OUTPUT_DIR, "ppo_trading_model_retrained")
    model.save(model_path)
    log(f"  Model saved: {model_path}.zip")

    # --- Backtest (Gymnasium 5-tuple API) ---
    test_env = TradingEnv(test_df)
    obs, _ = test_env.reset()
    done = False
    balances = [test_env.balance]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        balances.append(test_env.balance)

    exits = [t for t in test_env.trades if t["action"] == "exit"]
    wins = [t for t in exits if t.get("profit", 0) > 0]
    win_rate = len(wins) / len(exits) * 100 if exits else 0.0
    metrics = {
        "timesteps": timesteps,
        "data_source": "SYNTHETIC SAMPLE" if ctx.get("data_is_sample") else "yfinance/local CSV",
        "test_bars": len(test_df),
        "num_exits": len(exits),
        "win_rate_pct": round(win_rate, 2),
        "initial_balance": test_env.initial_balance,
        "final_balance": round(test_env.balance, 2),
        "total_profit": round(test_env.total_profit, 2),
        "return_pct": round(test_env.total_profit / test_env.initial_balance * 100, 2),
    }
    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(test_env.trades).to_csv(os.path.join(OUTPUT_DIR, "trades_log.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(balances)
    ax.axhline(test_env.initial_balance, ls="--", c="gray", alpha=0.6)
    ax.set_title("PPO retrain - equity curve (test set)")
    ax.set_xlabel("step")
    ax.set_ylabel("balance ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve = os.path.join(OUTPUT_DIR, "equity_curve.png")
    fig.savefig(curve, dpi=120)
    plt.close(fig)

    for k, v in metrics.items():
        log(f"    {k}: {v}")
    if len(exits) == 0:
        log("  Note: the model opened no trades on the test set "
            "(short training on synthetic data - expected; try --full with real data)")
    record(4, "AI training + backtest", "PASS",
           f"win rate {metrics['win_rate_pct']}% | return {metrics['return_pct']}% | details in output/")


# --------------------------------------------------------------------------- #
# Stage 5: summary
# --------------------------------------------------------------------------- #
@run_stage(5, "Writing summary report")
def stage_summary(ctx):
    lines = [
        "# AI-XAUUSD One-Click Run Summary",
        "",
        f"- Time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Mode: {'full training' if ctx['full'] else 'fast demo'}"
        + (" | offline" if ctx["skip_network"] else ""),
        f"- Output dir: {OUTPUT_DIR}",
        "",
        "| Stage | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    has_fail = False
    for stage, name, status, detail in RESULTS:
        lines.append(f"| {stage} | {name} | {status} | {detail} |")
        has_fail |= status == "FAIL"

    lines += [
        "",
        "## Artifacts",
    ]
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname != "run.log":
            lines.append(f"- `{fname}`")

    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  Summary written to: {summary_path}")

    ctx["has_fail"] = has_fail
    record(5, "Summary report", "PASS")

    # Best-effort: open the output folder (double-click scenario)
    try:
        if sys.platform.startswith("win"):
            os.startfile(OUTPUT_DIR)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", OUTPUT_DIR], check=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="AI-XAUUSD one-click runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", default=True, help="fast demo mode (default)")
    mode.add_argument("--full", action="store_true", help="full training mode (slow)")
    parser.add_argument("--skip-network", action="store_true", help="do not use the network")
    parser.add_argument("--no-pause", action="store_true", help="do not wait for a keypress at the end")
    args = parser.parse_args()

    # Fresh run.log each run
    log_path = os.path.join(OUTPUT_DIR, "run.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    log("AI-XAUUSD Trading System - One-Click Runner")
    log(f"Output directory: {OUTPUT_DIR}")

    ctx = {"full": args.full, "skip_network": args.skip_network, "mods": {},
           "data_csv": None, "has_fail": False}

    for stage in (stage_environment, stage_smoke, stage_quick_demo,
                  stage_data, stage_train_backtest, stage_summary):
        stage(ctx)

    log("")
    log("ALL STAGES FINISHED")
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    skips = [r for r in RESULTS if r[2] == "SKIP"]
    log(f"Tally: PASS {sum(1 for r in RESULTS if r[2] == 'PASS')} | "
        f"FAIL {len(fails)} | SKIP {len(skips)}")
    if skips:
        log("Note: SKIPped stages usually mean missing heavy deps or network; "
            "the full exe with internet enables them automatically.")
    log(f"Full log: {os.path.join(OUTPUT_DIR, 'run.log')}")

    if not args.no_pause and sys.platform.startswith("win"):
        try:
            os.system("pause")
        except Exception:
            pass

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

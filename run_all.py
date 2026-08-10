#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-XAUUSD 一键总控程序 (run_all.py)
====================================

双击（或运行 `python run_all.py`）后按顺序自动执行全部流程：

  [阶段 0] 环境自检                 → output/system_report.txt
  [阶段 1] 冒烟检查（核心环境回归） → 直接调用 trading_env / optimal_timing_env
  [阶段 2] 快速演示 quick_demo       → output/demo_results.png
  [阶段 3] 数据获取 xauusd_data.csv  → 优先复用 exe 旁现成数据，否则联网下载
  [阶段 4] AI 训练 + 回测 (PPO)      → output/ 下的模型、回测报告、净值曲线
  [阶段 5] 生成总结 SUMMARY.md       → 自动打开输出文件夹

每个阶段互不阻塞：某个阶段失败/缺依赖/无网络时标记 SKIP/FAIL 后继续，
最后统一给出总结表退出码（有 FAIL 返回 1，全 PASS/SKIP 返回 0）。

命令行参数：
  --fast           快速演示模式（默认）：训练 2048 步
  --full           完整模式：训练 50000 步（慢）
  --skip-network   不联网（数据阶段仅使用本地 CSV，无则 SKIP）
  --no-pause       结束时不等待按键（CI/自动化用）
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime

# matplotlib 必须在导入 pyplot 前设为非交互后端（exe/无显示环境必需）
os.environ.setdefault("MPLBACKEND", "Agg")

# --------------------------------------------------------------------------- #
# 路径处理：兼容普通 python 运行与 PyInstaller 冻结（exe）运行
# --------------------------------------------------------------------------- #
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    BASE_DIR = os.path.dirname(sys.executable)          # exe 所在目录（可写）
    BUNDLE_DIR = sys._MEIPASS                            # PyInstaller 解包目录
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS = []  # (stage, name, status, detail)


def log(msg=""):
    """同时打印到控制台并追加到 output/run.log"""
    line = str(msg)
    print(line, flush=True)
    with open(os.path.join(OUTPUT_DIR, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(stage, name, status, detail=""):
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}[status]
    log(f"  {icon} [{status}] {name}" + (f" — {detail}" if detail else ""))
    RESULTS.append((stage, name, status, detail))


def run_stage(stage_no, title):
    """装饰器：把阶段函数包进统一的 try/except 与计时中"""
    def deco(fn):
        def wrapper(ctx):
            log("")
            log("=" * 62)
            log(f"▶ 阶段 {stage_no}: {title}")
            log("=" * 62)
            t0 = time.time()
            try:
                fn(ctx)
            except Exception as e:  # 兜底：任何阶段崩溃都不终止整个流程
                log(traceback.format_exc())
                record(stage_no, title, "FAIL", f"{type(e).__name__}: {e}")
            log(f"  ⏱ 耗时 {time.time() - t0:.1f}s")
        return wrapper
    return deco


def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 阶段 0：环境自检
# --------------------------------------------------------------------------- #
@run_stage(0, "环境自检")
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
        "AI-XAUUSD Trading System — 环境自检报告",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"运行模式: {'PyInstaller exe' if FROZEN else 'python 脚本'}",
        f"操作系统: {platform.platform()}",
        f"Python: {sys.version.split()[0]}  ({sys.executable})",
        "",
        "依赖可用性：",
    ]
    for name, mod in mods.items():
        ver = getattr(mod, "__version__", "?") if mod else None
        lines.append(f"  {'OK ' if mod else 'MISS'} {name}" + (f" {ver}" if mod else ""))
        log(lines[-1])

    with open(os.path.join(OUTPUT_DIR, "system_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if not mods["numpy"] or not mods["pandas"] or not mods["gymnasium"]:
        record(0, "环境自检", "FAIL", "缺少 numpy/pandas/gymnasium 核心依赖")
    else:
        record(0, "环境自检", "PASS",
               "重依赖(torch/SB3) " + ("已可用" if ctx["mods"]["torch"] and ctx["mods"]["stable_baselines3"] else "不可用，训练阶段将跳过"))


# --------------------------------------------------------------------------- #
# 阶段 1：冒烟检查（与 tests/test_smoke.py 同源的回归场景，内联实现零依赖 pytest）
# --------------------------------------------------------------------------- #
@run_stage(1, "冒烟检查（核心环境回归）")
def stage_smoke(ctx):
    if not all(ctx["mods"][m] for m in ("numpy", "pandas", "gymnasium")):
        record(1, "冒烟检查", "SKIP", "缺少 numpy/pandas/gymnasium")
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

    # 1. 环境可实例化 + Gymnasium API 契约（回归：坏合并导致的 AttributeError/RecursionError）
    env = TradingEnv(df)
    obs, info = env.reset()
    checks.append(("TradingEnv 实例化 + reset→(obs,info)", obs.shape == (15,)))
    out = env.step(np.array([0.0], dtype=np.float32))
    checks.append(("step 返回 Gymnasium 5 元组", len(out) == 5))

    # 2. 止损必须真实亏钱（回归：退出时 PnL 被错误记为 $0）
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
    checks.append(("止损退出记入负 PnL", bool(exit_trades) and exit_trades[-1]["profit"] < 0))
    checks.append(("亏损后余额下降", env.balance < env.initial_balance))

    # 3. OptimalTimingTradingEnv 观测与声明空间一致（回归：135≠120 维度不匹配）
    env2 = OptimalTimingTradingEnv(df)
    obs2, _ = env2.reset()
    checks.append(("OptimalTimingEnv 观测维度一致", obs2.shape == env2.observation_space.shape))

    for name, ok in checks:
        record(1, name, "PASS" if ok else "FAIL")


# --------------------------------------------------------------------------- #
# 阶段 2：快速演示 quick_demo
# --------------------------------------------------------------------------- #
@run_stage(2, "快速演示 quick_demo")
def stage_quick_demo(ctx):
    if not all(ctx["mods"][m] for m in ("numpy", "pandas", "matplotlib")):
        record(2, "quick_demo", "SKIP", "缺少 numpy/pandas/matplotlib")
        return

    cwd = os.getcwd()
    os.chdir(OUTPUT_DIR)  # demo 会把 demo_results.png 写到当前目录
    try:
        import quick_demo
        rc = quick_demo.main()
    finally:
        os.chdir(cwd)

    png = os.path.join(OUTPUT_DIR, "demo_results.png")
    if rc == 0 and os.path.exists(png):
        record(2, "quick_demo", "PASS", f"图表: {png}")
    else:
        record(2, "quick_demo", "FAIL", f"返回码 {rc}")


# --------------------------------------------------------------------------- #
# 阶段 3：数据获取
# --------------------------------------------------------------------------- #
@run_stage(3, "数据获取 xauusd_data.csv")
def stage_data(ctx):
    candidates = [
        os.path.join(BASE_DIR, "xauusd_data.csv"),
        os.path.join(OUTPUT_DIR, "xauusd_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path)
            ctx["data_csv"] = path
            record(3, "数据获取", "PASS", f"复用本地数据 {path}（{len(df)} 行）")
            return

    if ctx["skip_network"]:
        record(3, "数据获取", "SKIP", "--skip-network 且无本地 xauusd_data.csv")
        return

    if not ctx["mods"]["yfinance"]:
        record(3, "数据获取", "SKIP", "yfinance 不可用（无网络依赖包）")
        return

    from data_fetch import fetch_xauusd_data
    data = fetch_xauusd_data("2015-01-01", "2025-01-01")
    if data is None or len(data) == 0:
        record(3, "数据获取", "FAIL", "yfinance 返回空数据（网络受限？）")
        return

    out = os.path.join(OUTPUT_DIR, "xauusd_data.csv")
    data.to_csv(out, index=False)
    ctx["data_csv"] = out
    record(3, "数据获取", "PASS", f"已下载 {len(data)} 行 → {out}")


# --------------------------------------------------------------------------- #
# 阶段 4：AI 训练 + 回测（重依赖：torch + stable-baselines3）
# --------------------------------------------------------------------------- #
@run_stage(4, "AI 训练 + 回测 (PPO)")
def stage_train_backtest(ctx):
    mods = ctx["mods"]
    if not (mods["torch"] and mods["stable_baselines3"]):
        record(4, "AI 训练 + 回测", "SKIP",
               "torch/stable-baselines3 不可用（轻量 exe 或未安装重依赖）")
        return
    if not ctx.get("data_csv"):
        record(4, "AI 训练 + 回测", "SKIP", "无训练数据")
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
    # 兼容 fetch 输出的 'date' 列与表头大小写差异
    df.columns = [c.capitalize() if c.lower() in ("open", "high", "low", "close", "volume") else c for c in df.columns]
    train_end = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_end], df.iloc[train_end:]
    log(f"  数据 {len(df)} 行 | 训练 {len(train_df)} / 测试 {len(test_df)}")

    timesteps = 50_000 if ctx["full"] else 2_048
    log(f"  训练 PPO：{timesteps} 步（{'完整' if ctx['full'] else '快速'}模式，CPU）约需数分钟…")

    env = DummyVecEnv([lambda: TradingEnv(train_df)])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=64, seed=42)
    model.learn(total_timesteps=timesteps)

    model_path = os.path.join(OUTPUT_DIR, "ppo_trading_model_retrained")
    model.save(model_path)
    log(f"  模型已保存: {model_path}.zip")

    # --- 回测（Gymnasium API 5 元组） ---
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
        "test_bars": len(test_df),
        "num_exits": len(exits),
        "win_rate_pct": round(win_rate, 2),
        "initial_balance": test_env.initial_balance,
        "final_balance": round(test_env.balance, 2),
        "total_profit": round(test_env.total_profit, 2),
        "return_pct": round(test_env.total_profit / test_env.initial_balance * 100, 2),
    }
    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame(test_env.trades).to_csv(os.path.join(OUTPUT_DIR, "trades_log.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(balances)
    ax.axhline(test_env.initial_balance, ls="--", c="gray", alpha=0.6)
    ax.set_title("PPO retrain — equity curve (test set)")
    ax.set_xlabel("step")
    ax.set_ylabel("balance ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve = os.path.join(OUTPUT_DIR, "equity_curve.png")
    fig.savefig(curve, dpi=120)
    plt.close(fig)

    for k, v in metrics.items():
        log(f"    {k}: {v}")
    record(4, "AI 训练 + 回测", "PASS",
           f"胜率 {metrics['win_rate_pct']}% | 收益 {metrics['return_pct']}% | 明细见 output/")


# --------------------------------------------------------------------------- #
# 阶段 5：总结
# --------------------------------------------------------------------------- #
@run_stage(5, "生成总结报告")
def stage_summary(ctx):
    lines = [
        "# AI-XAUUSD 一键运行总结",
        "",
        f"- 时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 模式: {'完整训练' if ctx['full'] else '快速演示'}"
        + (" | 离线" if ctx["skip_network"] else ""),
        f"- 输出目录: {OUTPUT_DIR}",
        "",
        "| 阶段 | 项目 | 结果 | 备注 |",
        "|---|---|---|---|",
    ]
    has_fail = False
    for stage, name, status, detail in RESULTS:
        lines.append(f"| {stage} | {name} | {status} | {detail} |")
        has_fail |= status == "FAIL"

    lines += [
        "",
        "## 产物清单",
    ]
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname != "run.log":
            lines.append(f"- `{fname}`")

    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  总结已写入: {summary_path}")

    ctx["has_fail"] = has_fail
    record(5, "总结报告", "PASS")

    # 尽力打开输出文件夹（Windows 双击场景）
    try:
        if sys.platform.startswith("win"):
            os.startfile(OUTPUT_DIR)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", OUTPUT_DIR], check=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="AI-XAUUSD 一键总控程序")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", default=True, help="快速演示模式（默认）")
    mode.add_argument("--full", action="store_true", help="完整训练模式（慢）")
    parser.add_argument("--skip-network", action="store_true", help="不联网")
    parser.add_argument("--no-pause", action="store_true", help="结束不等待按键")
    args = parser.parse_args()

    # 每次运行重建 run.log
    log_path = os.path.join(OUTPUT_DIR, "run.log")
    if os.path.exists(log_path):
        os.remove(log_path)

    log("🤖 AI-XAUUSD Trading System — 一键总控")
    log(f"输出目录: {OUTPUT_DIR}")

    ctx = {"full": args.full, "skip_network": args.skip_network, "mods": {},
           "data_csv": None, "has_fail": False}

    for stage in (stage_environment, stage_smoke, stage_quick_demo,
                  stage_data, stage_train_backtest, stage_summary):
        stage(ctx)

    log("")
    log("🏁 全部阶段结束")
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    skips = [r for r in RESULTS if r[2] == "SKIP"]
    log(f"统计: PASS {sum(1 for r in RESULTS if r[2] == 'PASS')} 项 | "
        f"FAIL {len(fails)} 项 | SKIP {len(skips)} 项")
    if skips:
        log("提示: 被跳过的阶段通常是缺少重依赖或网络；完整 exe 与联网环境会自动启用。")
    log(f"详细日志: {os.path.join(OUTPUT_DIR, 'run.log')}")

    if not args.no_pause and sys.platform.startswith("win"):
        try:
            os.system("pause")
        except Exception:
            pass

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

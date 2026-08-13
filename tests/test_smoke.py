"""
Smoke tests for the AI-XAUUSD trading system.

These tests encode the exact bugs found during the 2026-08 code review as
regression tests:
  1. TradingEnv could not be instantiated (bad merge in trading_env.py)
  2. Stop-loss / trailing-stop exits booked $0 PnL (env could never lose money)
  3. Partial take-profit closed the whole position silently
  4. OptimalTimingTradingEnv's observation did not match its declared space
  5. quick_demo.py always exited at import (broken imports)

Run with:  pytest tests/test_smoke.py -v
"""

import numpy as np
import pandas as pd
import pytest

from trading_env import TradingEnv, add_technical_indicators, build_observation

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def make_price_df(n=120, seed=0):
    """Synthetic OHLCV frame similar to xauusd_data.csv."""
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.standard_normal(n))
    df = pd.DataFrame({
        'Open': close,
        'High': close + 1,
        'Low': close - 1,
        'Close': close,
        'Volume': 1000,
    })
    return df


@pytest.fixture()
def env():
    return TradingEnv(make_price_df())


def freeze_regime(env_instance):
    """Keep regime parameters constant so tests are deterministic."""
    env_instance._update_regime_parameters = lambda: None


# --------------------------------------------------------------------------- #
# 1) Instantiation & Gymnasium API contract
# --------------------------------------------------------------------------- #

def test_env_can_be_instantiated(env):
    """Regression: TradingEnv(df) raised AttributeError/RecursionError before the fix."""
    assert env.observation_space.shape == (15,)
    assert env.action_space.shape == (1,)


def test_reset_returns_obs_info_tuple(env):
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_gymnasium_5_tuple(env):
    env.reset()
    out = env.step(np.array([0.0], dtype=np.float32))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == (15,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_full_episode_runs_to_completion(env):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert steps < 10_000  # guard against infinite loop
    # Any position still open must be force-closed at the end of the data
    assert env.position == 0


# --------------------------------------------------------------------------- #
# 2) PnL accounting — stop losses MUST register real losses
# --------------------------------------------------------------------------- #

def test_stop_loss_books_real_loss(env):
    """Regression: a 5% drop used to book exactly $0 (env could never lose)."""
    freeze_regime(env)
    env.reset()

    # Open a 25oz long at 2000 manually, then crash the price
    env.position = 25.0
    env.entry_price = 2000.0
    env.entry_time = env.current_step
    env.highest_price_since_entry = 2000.0
    env.trailing_stop_distance = 2000.0 * (1 - env.trailing_stop_pct)

    step = env.current_step + 1
    env.df.loc[step, 'Close'] = 1900.0  # -5% -> beyond 2% trailing stop distance
    env.current_step = step

    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))

    exit_trades = [t for t in env.trades if t['action'] == 'exit']
    assert exit_trades, "expected an exit trade to be logged"
    assert exit_trades[-1]['profit'] < 0, "stop loss must book a NEGATIVE pnl"
    assert env.balance < env.initial_balance, "balance must decrease after a losing trade"


def test_trailing_stop_books_nonzero_pnl(env):
    """Regression: trailing-stop exits used to book $0 because state was zeroed first."""
    freeze_regime(env)
    env.reset()

    env.position = 10.0
    env.entry_price = 2000.0
    env.entry_time = env.current_step
    env.profit_targets = [0.10]  # keep scaled-profit logic out of this test's way
    env.highest_price_since_entry = 2100.0
    env.trailing_stop_distance = 2100.0 * (1 - env.trailing_stop_pct)

    step = env.current_step + 1
    # Price inside trailing distance but still above entry -> small positive pnl expected
    env.df.loc[step, 'Close'] = env.trailing_stop_distance - 1
    env.current_step = step

    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))

    exit_trades = [t for t in env.trades if t['action'] == 'exit']
    assert exit_trades and exit_trades[-1]['reason'] == 'trailing_stop'
    assert exit_trades[-1]['profit'] != 0.0, "trailing stop must book non-zero pnl"
    expected = (exit_trades[-1]['price'] - 2000.0) * 10.0
    assert exit_trades[-1]['profit'] == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3) Partial take-profit keeps the remaining position
# --------------------------------------------------------------------------- #

def test_partial_take_profit_keeps_remaining_position(env):
    freeze_regime(env)
    env.reset()
    env.profit_targets = [0.01, 0.02, 0.05, 0.10]

    env.position = 20.0
    env.entry_price = 2000.0
    env.entry_time = env.current_step
    env.highest_price_since_entry = 2000.0
    env.trailing_stop_distance = 2000.0 * 0.975

    step = env.current_step + 1
    env.df.loc[step, 'Close'] = 2100.0  # +5% -> hits the 5% level (50% partial)
    env.current_step = step

    env.step(np.array([0.0], dtype=np.float32))

    exit_trades = [t for t in env.trades if t['action'] == 'exit']
    assert exit_trades, "expected a partial exit trade"
    last = exit_trades[-1]
    assert last['partial'] is True
    assert 0 < env.position < 20.0, "remaining position must be kept, not wiped"
    booked = last['profit']
    remainder_exit_value = (2100.0 - 2000.0) * env.position
    assert booked > 0
    # Booked pnl must correspond to the CLOSED quantity only
    closed_qty = 20.0 - env.position
    assert booked == pytest.approx((2100.0 - 2000.0) * closed_qty, abs=1e-6)


def test_each_profit_level_triggers_once(env):
    freeze_regime(env)
    env.reset()
    env.profit_targets = [0.01, 0.02, 0.05, 0.10]

    env.position = 40.0
    env.entry_price = 2000.0
    env.entry_time = env.current_step
    env.highest_price_since_entry = 2000.0
    env.trailing_stop_distance = 0.0  # disable trailing interference

    # Hold at +5% for several steps — the 5% level must trigger exactly once
    for offset in (1, 2, 3):
        step = env.current_step + 1
        env.df.loc[step, 'Close'] = 2100.0
        env.current_step = step
        env.step(np.array([0.0], dtype=np.float32))

    five_pct_exits = [t for t in env.trades
                      if t['action'] == 'exit' and t['reason'] == 'profit_5pct_partial']
    assert len(five_pct_exits) == 1, "the same profit level must not trigger repeatedly"


# --------------------------------------------------------------------------- #
# 4) Shared observation builder consistency
# --------------------------------------------------------------------------- #

def test_build_observation_helper_matches_env(env):
    obs = env._get_observation()
    ref = build_observation(env.df, env.current_step, env.position,
                            env.balance, lookback=env.lookback)
    np.testing.assert_allclose(obs, ref)
    assert obs.dtype == np.float32


def test_add_technical_indicators_returns_expected_columns():
    df = add_technical_indicators(make_price_df())
    for col in ('RSI', 'MACD', 'MACD_signal'):
        assert col in df.columns


# --------------------------------------------------------------------------- #
# 5) OptimalTimingTradingEnv — observation must match the declared space
# --------------------------------------------------------------------------- #

def test_optimal_timing_env_obs_matches_space():
    pytest.importorskip("gymnasium")
    from optimal_timing_env import OptimalTimingTradingEnv

    env = OptimalTimingTradingEnv(make_price_df(n=150))
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert steps < 10_000
    assert env.position == 0  # force-closed at episode end


# --------------------------------------------------------------------------- #
# 6) quick_demo must be runnable end-to-end (headless)
# --------------------------------------------------------------------------- #

def test_quick_demo_runs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    monkeypatch.chdir(tmp_path)  # keep demo_results.png out of the repo

    import quick_demo
    assert quick_demo.main() == 0

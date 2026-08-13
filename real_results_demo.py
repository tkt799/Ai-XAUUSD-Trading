#!/usr/bin/env python3
"""
Real Results Demonstration
Shows actual model predictions on real market data
"""

import os

import pandas as pd
from stable_baselines3 import PPO

from trading_env import TradingEnv


def main():
    # Load real market data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')
    print('📊 REAL MARKET DATA LOADED')
    print(f'   Data points: {len(df)}')
    print(f'   Date range: {df.index.min()} to {df.index.max()}')
    print(f'   Latest price: ${df.iloc[-1].Close:.2f}')
    print()

    # Load a trained model (prefer transformer if present, else repo PPO weights)
    model_path = None
    for candidate in ('transformer_trading_model.zip', 'ppo_trading_model.zip', 'ppo_trading_model'):
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        print('Model loading failed: no trained model found '
              '(run `python train_model.py` or `python download_models.py` first)')
        return

    model = PPO.load(model_path)
    print('🤖 MODEL LOADED SUCCESSFULLY')
    print(f'   Weights: {model_path}')
    print('   Features: Price history + RSI + MACD + MACD Signal')
    print()

    # Test on recent real data
    recent_data = df.tail(20)  # Last 20 days
    env = TradingEnv(recent_data.reset_index())

    print('🎯 MODEL PREDICTIONS ON REAL DATA:')
    print('=' * 60)

    obs, _info = env.reset()

    for i in range(min(15, len(recent_data) - 1)):
        current_price = recent_data.iloc[i].Close

        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0])

        # Interpret action
        if action_value > 0.1:
            decision = 'BUY 📈'
            confidence = min(action_value, 1.0)
        elif action_value < -0.1:
            decision = 'SELL 📉'
            confidence = min(-action_value, 1.0)
        else:
            decision = 'HOLD ⏸️'
            confidence = 0

        print(f'Day {i+1:2d}: ${current_price:7.2f} | Action: {action_value:6.3f} | '
              f'Decision: {decision} | Confidence: {confidence:.1%}')

        # Execute trade (Gymnasium API: step -> 5-tuple)
        obs, reward, terminated, truncated, info = env.step(action, confidence=max(confidence, 0.3))

        if terminated or truncated:
            break

    total_profit = env.total_profit

    print()
    print('💰 REAL PERFORMANCE RESULTS:')
    print(f'   Initial Balance: ${env.initial_balance:.2f}')
    print(f'   Final Balance: ${env.balance:.2f}')
    print(f'   Profit/Loss: ${total_profit:.2f}')
    print(f'   Return %: {total_profit / env.initial_balance * 100:.1f}%')
    print(f'   Trades executed: {len([t for t in env.trades if t["action"] == "exit"])}')
    print()
    print('✅ MODEL IS MAKING REAL DECISIONS ON REAL MARKET DATA!')


if __name__ == "__main__":
    main()

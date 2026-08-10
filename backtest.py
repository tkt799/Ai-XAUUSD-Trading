import json
import os

import pandas as pd
from stable_baselines3 import PPO

from trading_env import TradingEnv


def main():
    # Load data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Split data
    train_end = int(len(df) * 0.8)
    test_df = df.iloc[train_end:]

    # Load model — prefer the transformer one if it has been trained,
    # otherwise fall back to the PPO model shipped in the repo.
    model_path = None
    for candidate in ('transformer_trading_model.zip', 'ppo_trading_model.zip', 'ppo_trading_model'):
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError(
            "No trained model found. Train one via `python train_model.py` "
            "or download the published weights via `python download_models.py`."
        )
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create test environment
    test_env = TradingEnv(test_df)

    # Run backtest (Gymnasium API: reset -> (obs, info); step -> 5-tuple)
    obs, _info = test_env.reset()
    done = False
    total_reward = 0.0
    actions = []
    balances = [test_env.balance]

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        actions.append(action)
        balances.append(test_env.balance)

    print("Backtest Results:")
    print(f"Total Reward: {total_reward}")
    print(f"Final Balance: {test_env.balance}")
    print(f"Total Profit: {test_env.total_profit}")
    print(f"Initial Balance: {test_env.initial_balance}")

    # Save results
    results = {
        'total_reward': total_reward,
        'final_balance': test_env.balance,
        'total_profit': test_env.total_profit,
        'initial_balance': test_env.initial_balance,
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(results, f)

    print("Backtest results saved to backtest_results.json")

    # Save trades log
    trades_df = pd.DataFrame(test_env.trades)
    trades_df.to_csv('trades_log.csv', index=False)
    print("Trades log saved to trades_log.csv")


if __name__ == "__main__":
    main()

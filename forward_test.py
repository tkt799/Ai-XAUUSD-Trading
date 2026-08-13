import json
import os

import pandas as pd
from stable_baselines3 import PPO

from trading_env import TradingEnv


def main():
    # Load data
    if not os.path.exists('xauusd_data.csv'):
        raise FileNotFoundError(
            "xauusd_data.csv not found. Run `python data_fetch.py` first (requires network), "
            "or copy the bundled sample: `cp data/xauusd_sample.csv xauusd_data.csv`."
        )
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Use last 100 days for forward test (or as many as we have)
    forward_df = df.tail(min(100, len(df) - 11)).copy()

    # Load model — prefer ppo_trading_model / PPO weights trained/loaded from ensemble_models/
    model_path = None
    for candidate in ('ppo_trading_model.zip', 'ppo_trading_model',
                      'ensemble_models/ppo_model.zip'):
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError(
            "No PPO model found. Run `python train_model.py --model ppo` first."
        )
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create test environment
    forward_env = TradingEnv(forward_df)

    # Run forward test (Gymnasium API)
    obs, _info = forward_env.reset()
    done = False
    total_reward = 0.0
    actions = []
    balances = [forward_env.balance]

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _info = forward_env.step(action)
        done = terminated or truncated
        total_reward += reward
        actions.append(action)
        balances.append(forward_env.balance)

    print("Forward Test Results (last 100 days):")
    print(f"Total Reward: {total_reward}")
    print(f"Final Balance: {forward_env.balance}")
    print(f"Total Profit: {forward_env.total_profit}")
    print(f"Initial Balance: {forward_env.initial_balance}")
    print(f"Profit per day: {forward_env.total_profit / 100}")

    # Save results
    results = {
        'total_reward': total_reward,
        'final_balance': forward_env.balance,
        'total_profit': forward_env.total_profit,
        'initial_balance': forward_env.initial_balance,
        'profit_per_day': forward_env.total_profit / 100,
    }

    with open('forward_test_results.json', 'w') as f:
        json.dump(results, f)

    print("Forward test results saved to forward_test_results.json")


if __name__ == "__main__":
    main()

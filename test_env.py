import pandas as pd

from trading_env import TradingEnv


def main():
    # Load data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Create environment with small data
    env = TradingEnv(df.head(100))

    obs, _info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Action space:", env.action_space)

    # Test a few steps (Gymnasium API: step -> 5-tuple)
    for i in range(5):
        action = env.action_space.sample()
        print(f"Step {i}, Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Done: {done}")
        if done:
            break

    print("Trades:", env.trades)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced Trading Environment for Optimal Entry/Exit Timing
Focuses on teaching models to identify the best buy and sell points

NOTE: rewards only use information available up to the current bar —
no look-ahead / future data is used anywhere in this environment.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class OptimalTimingTradingEnv(gym.Env):
    """Enhanced trading environment that teaches optimal entry/exit timing."""

    metadata = {"render_modes": ["human"]}

    # [Open, High, Low, Close, Volume] x lookback
    OHLCV_FEATURES = 5
    # Must match the exact indicator list used in _get_observation()
    INDICATOR_COLUMNS = [
        'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
        'STOCH_K', 'STOCH_D', 'WILLR', 'CCI',
        'ATR', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
        'OBV', 'AD', 'PIVOT',
    ]
    POSITION_FEATURES = 5  # position, position_size, holding_time, unrealized pnl %, normalized balance

    def __init__(self, df, initial_balance=1000, leverage=50, transaction_cost=0.0002):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_cost = transaction_cost

        # Enhanced technical indicators for timing (pure pandas — no TA-Lib dependency)
        self._add_advanced_indicators()

        # Action space: [position_size, entry_threshold, exit_threshold]
        # position_size: -1 to 1 (negative = short, positive = long)
        # entry_threshold: 0 to 1 (confidence threshold for entry)
        # exit_threshold: 0 to 1 (confidence threshold for exit)
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.lookback = 20
        obs_size = (self.lookback * self.OHLCV_FEATURES
                    + len(self.INDICATOR_COLUMNS)
                    + self.POSITION_FEATURES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _add_advanced_indicators(self):
        """Add comprehensive technical indicators for timing analysis (pandas-only)."""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']

        # Trend indicators
        self.df['SMA_20'] = close.rolling(20).mean()
        self.df['SMA_50'] = close.rolling(50).mean()
        self.df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

        # Momentum indicators (Wilder-smoothed RSI, same family as TA-Lib)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['MACD_SIGNAL'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_HIST'] = self.df['MACD'] - self.df['MACD_SIGNAL']

        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        self.df['STOCH_K'] = 100 * (close - lowest_low) / stoch_range
        self.df['STOCH_D'] = self.df['STOCH_K'].rolling(3).mean()
        self.df['WILLR'] = -100 * (highest_high - close) / stoch_range

        typical_price = (high + low + close) / 3
        tp_sma = typical_price.rolling(14).mean()
        mean_dev = (typical_price - tp_sma).abs().rolling(14).mean()
        self.df['CCI'] = (typical_price - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))

        # Volatility indicators
        prev_close = close.shift(1)
        true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                               axis=1).max(axis=1)
        self.df['ATR'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()  # Wilder smoothing
        self.df['NATR'] = self.df['ATR'] / close * 100

        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        self.df['BB_MIDDLE'] = bb_middle
        self.df['BB_UPPER'] = bb_middle + 2 * bb_std
        self.df['BB_LOWER'] = bb_middle - 2 * bb_std

        # Volume indicators
        direction = np.sign(close.diff()).fillna(0)
        self.df['OBV'] = (direction * volume).cumsum()

        clv = (((close - low) - (high - close)) / (high - low).replace(0, np.nan)).fillna(0)
        ad_line = (clv * volume).cumsum()
        self.df['AD'] = ad_line
        self.df['ADOSC'] = (ad_line.ewm(span=3, adjust=False).mean()
                            - ad_line.ewm(span=10, adjust=False).mean())

        # Support/Resistance levels (simplified)
        self.df['PIVOT'] = typical_price
        self.df['R1'] = 2 * self.df['PIVOT'] - low
        self.df['S1'] = 2 * self.df['PIVOT'] - high

        # Fill NaN values
        self.df = self.df.bfill().fillna(0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.total_pnl = 0
        self.trades = []
        self.holding_time = 0

        return self._get_observation(), {}

    def _get_observation(self):
        """Create comprehensive observation for timing decisions."""
        start_idx = self.current_step - self.lookback
        end_idx = self.current_step

        # Price data (OHLCV) over the lookback window
        prices = (self.df.iloc[start_idx:end_idx][['Open', 'High', 'Low', 'Close', 'Volume']]
                  .values.flatten())

        # Technical indicators at the current bar
        indicators = self.df.iloc[self.current_step][self.INDICATOR_COLUMNS].values.astype(np.float64)

        # Position information
        position_info = np.array([
            self.position,
            self.position_size,
            self.holding_time,
            (self.df.iloc[self.current_step]['Close'] - self.entry_price) / max(self.entry_price, 1e-9)
            if self.entry_price > 0 else 0,
            self.balance / self.initial_balance,
        ])

        return np.concatenate([prices, indicators, position_info]).astype(np.float32)

    def _calculate_timing_reward(self, action, current_price):
        """
        Reward shaping based on realised, already-known information ONLY.

        No future bars are inspected — the previous version peeked 10 bars
        ahead (look-ahead bias) and has been removed.
        """
        position_size, entry_threshold, exit_threshold = action
        reward = 0.0

        # Small bonus for acting on a confident entry signal
        if self.position == 0 and abs(position_size) > entry_threshold:
            reward += 0.05
            # Align bonus sign with very recent momentum (information from the past, not the future)
            past = self.df.iloc[max(0, self.current_step - 5):self.current_step + 1]['Close']
            if len(past) > 1:
                recent_return = (past.iloc[-1] - past.iloc[0]) / past.iloc[0]
                reward += 0.05 * np.sign(position_size) * np.sign(recent_return)

        # Realised PnL on exit is accounted for in step()/_close_position().

        # Holding penalty (encourage decisive management)
        if self.position != 0:
            reward -= 0.01

        return reward

    def step(self, action):
        position_size, entry_threshold, exit_threshold = action
        current_price = float(self.df.iloc[self.current_step]['Close'])

        reward = self._calculate_timing_reward(action, current_price)

        if self.position == 0:  # No position
            if position_size > entry_threshold:  # Buy signal
                self.position = 1
                self.position_size = min(position_size, 1.0)
                self.entry_price = current_price
                self.holding_time = 0

                # Stop loss / take profit from ATR
                atr = float(self.df.iloc[self.current_step]['ATR'])
                self.stop_loss = current_price - (atr * 2)
                self.take_profit = current_price + (atr * 3)

                reward += 0.1

            elif position_size < -entry_threshold:  # Sell signal
                self.position = -1
                self.position_size = min(abs(position_size), 1.0)
                self.entry_price = current_price
                self.holding_time = 0

                atr = float(self.df.iloc[self.current_step]['ATR'])
                self.stop_loss = current_price + (atr * 2)
                self.take_profit = current_price - (atr * 3)

                reward += 0.1

        else:  # Have position - check exit conditions
            self.holding_time += 1

            if self.position == 1:  # Long
                if current_price <= self.stop_loss:
                    reward -= 1.0
                    pnl = self._close_position(current_price, "stop_loss")
                    reward += pnl / self.initial_balance
                elif current_price >= self.take_profit:
                    reward += 2.0
                    pnl = self._close_position(current_price, "take_profit")
                    reward += pnl / self.initial_balance
                elif position_size < -exit_threshold:  # Exit signal
                    pnl = self._close_position(current_price, "signal_exit")
                    reward += (pnl / self.initial_balance) * 5

            else:  # Short
                if current_price >= self.stop_loss:
                    reward -= 1.0
                    pnl = self._close_position(current_price, "stop_loss")
                    reward += pnl / self.initial_balance
                elif current_price <= self.take_profit:
                    reward += 2.0
                    pnl = self._close_position(current_price, "take_profit")
                    reward += pnl / self.initial_balance
                elif position_size > exit_threshold:  # Exit signal
                    pnl = self._close_position(current_price, "signal_exit")
                    reward += (pnl / self.initial_balance) * 5

            # Max holding time
            if self.position != 0 and self.holding_time >= 50:  # Max 50 bars
                pnl = self._close_position(current_price, "max_time")
                reward += (pnl / self.initial_balance) * 2

        # Move to next step
        self.current_step += 1
        terminated = False
        truncated = False

        if self.current_step >= len(self.df) - 1:
            terminated = True
            # Final position closure — settle at the last known price
            if self.position != 0:
                final_price = float(self.df.iloc[min(self.current_step, len(self.df) - 1)]['Close'])
                pnl = self._close_position(final_price, "end_episode")
                reward += (pnl / self.initial_balance) * 3

        info = {
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'position': self.position,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _close_position(self, price, reason):
        """Close current position, settle PnL and record the trade. Returns realised PnL."""
        if self.position == 0:
            return 0.0

        # Notional-sized PnL: balance * position_size% deployed at `leverage`
        notional = self.balance * abs(self.position_size) * self.leverage
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        pnl = notional * pnl_pct
        pnl -= notional * self.transaction_cost  # Transaction cost on the closed notional

        self.balance += pnl
        self.total_pnl += pnl

        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'position': 'LONG' if self.position == 1 else 'SHORT',
            'pnl': pnl,
            'holding_time': self.holding_time,
            'exit_reason': reason,
        })

        # Reset position
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.holding_time = 0

        return pnl

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
              f"Position: {self.position}, Total P&L: ${self.total_pnl:.2f}")

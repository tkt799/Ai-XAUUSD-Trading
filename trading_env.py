import gymnasium as gym
import numpy as np
from gymnasium import spaces

from market_regime_detector import MarketRegimeDetector


def add_technical_indicators(df):
    """
    Add the RSI / MACD / MACD_signal columns used by TradingEnv observations.

    Shared by training, backtesting and live trading so the feature pipeline is
    identical on every code path. Operates on a copy and returns the new frame.
    """
    df = df.copy()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df.fillna(0)


def build_observation(df, step, position, balance, lookback=10):
    """
    Build the canonical 15-dim observation from a 0-based integer `step`:
    [closes[step-lookback..step-1], RSI, MACD, MACD_signal, position, balance].
    """
    df = df.reset_index(drop=True)
    prices = df.loc[step - lookback:step - 1, 'Close'].values
    rsi = df.loc[step - 1, 'RSI']
    macd = df.loc[step - 1, 'MACD']
    macd_signal = df.loc[step - 1, 'MACD_signal']
    return np.concatenate([prices, [rsi, macd, macd_signal, position, balance]]).astype(np.float32)


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for XAUUSD trading with trailing stops and dynamic exits.

    Observation: [last `lookback` closes, RSI, MACD, MACD_signal, position, balance]
    Action: Box(-1, 1, shape=(1,)) — signed trade signal. Magnitude scales position size.
    """

    metadata = {"render_modes": ["human"]}

    # Reward multipliers per exit reason (applied to PnL / initial_balance)
    REWARD_MULTIPLIERS = {
        'take_profit': 100,
        'trailing_stop': 30,
        'partial_profit': 50,
        'max_time': 10,
        'stop_loss': 10,
        'end_of_data': 10,
    }

    def __init__(self, df, initial_balance=1000, transaction_cost=0, leverage=50, stop_loss_pct=0.02):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct

        # Trailing stop parameters
        self.trailing_stop_pct = 0.025
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0

        # Profit taking parameters - multiple scaled targets
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]
        self.take_profit_pct = 0.10
        self.partial_take_profit_pct = 0.02

        # Breakeven stop parameters
        self.breakeven_trigger_pct = 0.015
        self.breakeven_activated = False

        # Dynamic exit parameters
        self.max_holding_period = 24
        self.entry_time = None

        # Market regime detector (used by _update_regime_parameters)
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = None

        # Technical indicators (pure data operation — no env-state side effects)
        self._calculate_indicators()

        # Action space: continuous action between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: last `lookback` closes + RSI + MACD + MACD_signal + position + balance
        self.lookback = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback + 5,), dtype=np.float32
        )

        self.trades = []

        # reset() initialises current_step, after which regime parameters can be applied safely
        self.reset()

    def _calculate_indicators(self):
        """Calculate technical indicators on the dataset. No side effects on env state."""
        self.df = add_technical_indicators(self.df)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0  # signed quantity; >0 long, <0 short
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []

        # Exit-management state
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_time = None
        self.breakeven_activated = False
        self._profit_levels_taken = set()  # scaled profit targets already triggered

        self._update_regime_parameters()

        return self._get_observation(), {}

    def _update_regime_parameters(self):
        """Update trading parameters based on the current market regime."""
        current_regime, regime_params = self.regime_detector.detect_regime(self.df, self.current_step)

        self.profit_targets = regime_params['profit_targets']
        self.trailing_stop_pct = regime_params['trailing_stop_pct']
        self.take_profit_pct = regime_params['profit_targets'][-1]
        self.partial_take_profit_pct = regime_params['profit_targets'][1]
        self.breakeven_trigger_pct = regime_params['breakeven_trigger']
        self.max_holding_period = regime_params['max_holding_time']

        self.current_regime = current_regime

    def _get_observation(self):
        return build_observation(self.df, self.current_step, self.position,
                                 self.balance, lookback=self.lookback)

    def step(self, action, confidence=1.0):
        """
        Execute one trading step.

        Returns the Gymnasium 5-tuple: (observation, reward, terminated, truncated, info).
        `confidence` (0..1) scales position size; below 0.3 the trade is suppressed.
        """
        self._update_regime_parameters()

        current_price = float(self.df.loc[self.current_step, 'Close'])
        reward = 0.0

        if isinstance(action, np.ndarray):
            action = float(action[0])

        # Suppress weak-confidence signals
        min_confidence = 0.3
        if confidence < min_confidence:
            action = 0.0

        effective_action = action * confidence

        # 1) Dynamic exits first (PnL is settled inside, before state is mutated)
        exit_reason, exit_pnl = self._check_dynamic_exits(current_price)
        if exit_reason:
            multiplier = self.REWARD_MULTIPLIERS.get(
                exit_reason, self.REWARD_MULTIPLIERS.get('partial_profit' if 'partial' in exit_reason else 0, 20)
            )
            reward = exit_pnl / self.initial_balance * multiplier

        # 2) Open a new position only if flat and no exit just happened
        elif self.position == 0:
            if effective_action > 0.1:
                position_multiplier = min(effective_action, 1.0)
                self._open_position(current_price, direction=1,
                                    position_multiplier=position_multiplier,
                                    confidence=confidence)
            elif effective_action < -0.1:
                position_multiplier = min(abs(effective_action), 1.0)
                self._open_position(current_price, direction=-1,
                                    position_multiplier=position_multiplier,
                                    confidence=confidence)

        # 3) Still in position — trail the stop
        else:
            self._update_trailing_stops(current_price)

        # Advance time
        self.current_step += 1
        terminated = False
        truncated = False

        if self.current_step >= len(self.df) - 1:
            terminated = True
            # Force-close any open position so terminal PnL is never left unbooked
            if self.position != 0:
                final_price = float(self.df.loc[min(self.current_step, len(self.df) - 1), 'Close'])
                final_pnl = self._close_position(final_price, 'end_of_data', fraction=1.0)
                reward += final_pnl / self.initial_balance * self.REWARD_MULTIPLIERS['end_of_data']

        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'position': self.position,
            'regime': self.current_regime.value if self.current_regime else None,
        }
        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    def _open_position(self, price, direction, position_multiplier, confidence):
        """Open a long (direction=+1) or short (direction=-1) position."""
        confidence_multiplier = confidence ** 0.5
        quantity = (self.balance * self.leverage * position_multiplier
                    * confidence_multiplier / price) * direction
        self.position = quantity
        self.entry_price = price
        self.entry_time = self.current_step
        self.highest_price_since_entry = price
        self.breakeven_activated = False
        self._profit_levels_taken = set()
        if direction > 0:
            self.trailing_stop_distance = price * (1 - self.trailing_stop_pct)
        else:
            self.trailing_stop_distance = price * (1 + self.trailing_stop_pct)

        self.trades.append({
            'step': self.current_step,
            'action': 'buy' if direction > 0 else 'sell_short',
            'price': price,
            'position': self.position,
            'confidence': confidence,
        })

    def _close_position(self, price, reason, fraction=1.0, confidence=None):
        """
        Close `fraction` of the current position and settle its PnL.

        Returns the realised (signed) PnL INCLUDING transaction costs.
        The position state is only fully reset when the whole position is closed.
        """
        closed_qty = self.position * fraction
        pnl = (price - self.entry_price) * closed_qty
        pnl -= self.transaction_cost * abs(closed_qty) * price

        self.balance += pnl
        self.total_profit += pnl
        self.position -= closed_qty

        self.trades.append({
            'step': self.current_step,
            'action': 'exit',
            'reason': reason,
            'price': price,
            'quantity_closed': closed_qty,
            'profit': pnl,
            'confidence': confidence,
            'partial': fraction < 1.0,
        })

        if fraction >= 1.0 or abs(self.position) < 1e-12:
            self.position = 0
            self._reset_position_state()

        return pnl

    def _reset_position_state(self):
        """Reset per-position state (after a full close)."""
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0
        self.entry_price = 0
        self.entry_time = None
        self.breakeven_activated = False
        self._profit_levels_taken = set()

    def _check_dynamic_exits(self, current_price):
        """
        Check exit conditions. Returns (exit_reason, realised_pnl).

        PnL is settled IMMEDIATELY via _close_position() before any state is
        cleared, so stop losses and trailing stops register real (negative) PnL.
        """
        if self.position == 0 or self.entry_price == 0:
            return None, 0.0

        # Current floating profit percentage (direction-aware)
        direction = 1 if self.position > 0 else -1
        profit_pct = (current_price - self.entry_price) / self.entry_price * direction

        # Arm the breakeven stop once profit exceeds the trigger
        if not self.breakeven_activated and profit_pct >= self.breakeven_trigger_pct:
            self.breakeven_activated = True
            buffer_pct = 0.005
            if self.position > 0:
                self.trailing_stop_distance = max(self.trailing_stop_distance,
                                                  self.entry_price * (1 + buffer_pct))
            else:
                self.trailing_stop_distance = min(self.trailing_stop_distance,
                                                  self.entry_price * (1 - buffer_pct))

        # Scaled profit taking — each level triggers at most once per position
        for target_pct in sorted(self.profit_targets, reverse=True):
            if target_pct in self._profit_levels_taken:
                continue
            if profit_pct >= target_pct:
                self._profit_levels_taken.add(target_pct)
                if target_pct >= max(self.profit_targets):
                    fraction, reason = 1.0, 'take_profit'
                elif target_pct <= 0.02:
                    fraction, reason = 0.25, f'profit_{int(target_pct * 100)}pct_partial'
                else:
                    fraction, reason = 0.50, f'profit_{int(target_pct * 100)}pct_partial'
                pnl = self._close_position(current_price, reason, fraction=fraction)
                return reason, pnl

        # Trailing stop
        if self.position > 0 and current_price <= self.trailing_stop_distance:
            pnl = self._close_position(current_price, 'trailing_stop')
            return 'trailing_stop', pnl
        if self.position < 0 and current_price >= self.trailing_stop_distance:
            pnl = self._close_position(current_price, 'trailing_stop')
            return 'trailing_stop', pnl

        # Maximum holding time
        if self.entry_time is not None and (self.current_step - self.entry_time) >= self.max_holding_period:
            pnl = self._close_position(current_price, 'max_time')
            return 'max_time', pnl

        # Hard stop loss
        if profit_pct <= -self.stop_loss_pct:
            pnl = self._close_position(current_price, 'stop_loss')
            return 'stop_loss', pnl

        return None, 0.0

    def _update_trailing_stops(self, current_price):
        """Ratchet the trailing stop in the direction of the trade only."""
        if self.position > 0 and current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = current_price
            self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
        elif self.position < 0 and current_price < self.highest_price_since_entry:
            self.highest_price_since_entry = current_price
            self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position:.4f}, Total Profit: {self.total_profit:.2f}")

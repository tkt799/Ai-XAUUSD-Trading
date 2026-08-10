#!/usr/bin/env python3
"""
Live Trading with Ensemble System
Combines multiple RL models for real-time trading decisions
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from ensemble_trader import EnsembleTrader
from market_regime_detector import MarketRegimeDetector
from trading_env import add_technical_indicators

# Setup logging
logging.basicConfig(
    filename='ensemble_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LiveEnsembleTrader:
    """
    Live trading system using ensemble predictions
    """

    def __init__(self, ensemble_path='./ensemble_models/', capital=100, leverage=50):
        self.capital = capital
        self.leverage = leverage
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

        # Trailing stop parameters - TIGHTENED for better profit capture
        self.trailing_stop_pct = 0.025  # Reduced from 5% to 2.5% trailing stop
        self.trailing_stop_distance = 0
        self.highest_price_since_entry = 0

        # Profit taking parameters - MULTIPLE SCALED TARGETS
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10% profit targets
        self.take_profit_pct = 0.10  # 10% take profit
        self.partial_take_profit_pct = 0.02  # Take partial profits at 2% (reduced from 5%)

        # Breakeven stop parameters
        self.breakeven_trigger_pct = 0.015  # Move to breakeven after 1.5% profit
        self.breakeven_activated = False
        self._profit_levels_taken = set()  # scaled profit targets already triggered

        # Load ensemble
        self.ensemble = EnsembleTrader()
        self.ensemble.load_ensemble(ensemble_path)

        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector()

        # Trading parameters - defaults first, then regime-adaptive overrides
        self._set_default_parameters()
        self._update_regime_parameters()

        # Performance tracking
        self.trades = []
        self.current_trade_start = None

        logging.info(f"Initialized Live Ensemble Trader with trailing stops - ${capital} capital, {leverage}x leverage")

    def _update_regime_parameters(self):
        """Update trading parameters based on current market regime"""
        # For live trading, we'll need to get recent market data
        # For now, use default parameters - this will be enhanced with live data
        try:
            # Get recent data for regime detection (last 100 periods)
            recent_data = self.get_recent_market_data()
            if recent_data is not None and len(recent_data) > 50:
                current_regime, regime_params = self.regime_detector.detect_regime(recent_data, -1)

                # Update trading parameters based on regime
                self.profit_targets = regime_params['profit_targets']
                self.trailing_stop_pct = regime_params['trailing_stop_pct']
                self.take_profit_pct = regime_params['profit_targets'][-1]
                self.partial_take_profit_pct = regime_params['profit_targets'][1]
                self.breakeven_trigger_pct = regime_params['breakeven_trigger']
                self.max_holding_time = regime_params['max_holding_time']
                self.min_signal_strength = regime_params['min_confidence_threshold']

                logging.info(f"Regime detected: {current_regime.value} - Updated trading parameters")
                self.current_regime = current_regime
            else:
                # Use default parameters if no data available
                self._set_default_parameters()
        except Exception as e:
            logging.warning(f"Could not update regime parameters: {e}")
            self._set_default_parameters()

    def _set_default_parameters(self):
        """Set default trading parameters when regime detection fails or has not run yet"""
        self.profit_targets = [0.01, 0.02, 0.05, 0.10]
        self.trailing_stop_pct = 0.025
        self.take_profit_pct = 0.10
        self.partial_take_profit_pct = 0.02
        self.breakeven_trigger_pct = 0.015
        self.max_holding_time = 24
        self.min_signal_strength = 0.2
        self.stop_loss_pct = 0.02

    def get_recent_market_data(self, symbol='GC=F', period='3mo', interval='1h'):
        """Fetch recent market data for regime detection (~3 months of hourly bars).

        Returns None when data is unavailable so callers fall back to defaults.
        """
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data is None or data.empty:
                return None
            # Futures often have sparse/NaN volume — only require valid closes
            return data.dropna(subset=['Close'])
        except Exception as e:
            logging.error(f"Could not fetch recent market data: {e}")
            return None

    def get_live_data(self, symbol='GC=F', period='1d', interval='5m'):
        """Fetch live market data"""
        try:
            data = yf.download(symbol, period=period, interval=interval)
            if data.empty:
                logging.warning("No data received from Yahoo Finance")
                return None

            # Add technical indicators (shared pipeline with the training environment)
            data = add_technical_indicators(data)
            return data
        except Exception as e:
            logging.error(f"Error fetching live data: {e}")
            return None

    def get_observation(self, df):
        """Create observation from latest data.

        MUST match TradingEnv's 15-dim layout — [last 10 closes, RSI, MACD,
        MACD_signal, position, balance] — unnormalised, exactly as in training.
        """
        lookback = 10
        closes = df['Close'].iloc[-lookback:].values
        latest = df.iloc[-1]

        obs = np.concatenate([
            closes,
            [latest['RSI'], latest['MACD'], latest['MACD_signal'],
             self.position, self.capital],
        ]).astype(np.float32)

        return obs.reshape(1, -1)

    def execute_trade(self, action, confidence, current_price):
        """Execute trading decision with confidence-based sizing and dynamic exits"""
        timestamp = datetime.now()

        # Update regime parameters dynamically
        self._update_regime_parameters()

        # Check for dynamic exits first (trailing stops, profit taking)
        exit_reason = self._check_dynamic_exits(current_price)
        if exit_reason:
            # Calculate P&L
            if self.position > 0:  # Long position
                pnl = (current_price - self.entry_price) * self.position
            else:  # Short position
                pnl = (self.entry_price - current_price) * abs(self.position)
            self.capital += pnl

            # Record trade
            trade = {
                'entry_time': self.current_trade_start,
                'exit_time': timestamp,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position': 'LONG' if self.position > 0 else 'SHORT',
                'pnl': pnl,
                'exit_reason': exit_reason,
                'confidence': confidence
            }
            self.trades.append(trade)

            logging.info(f"CLOSE {trade['position']} - P&L: ${pnl:.2f} - Reason: {exit_reason}")

            # Reset position
            self.position = 0
            self.entry_price = 0
            self.trailing_stop_distance = 0
            self.highest_price_since_entry = 0
            self.current_trade_start = None
            self.breakeven_activated = False  # Reset breakeven flag
            self._profit_levels_taken = set()  # Reset scaled profit targets

        elif self.position == 0:  # No position - check for entry
            if action > self.min_signal_strength:  # Buy signal
                # Size position based on confidence
                confidence_multiplier = confidence ** 0.5
                position_size = self.capital * self.leverage * min(action, 1.0) * confidence_multiplier / current_price
                self.position = position_size
                self.entry_price = current_price
                self.current_trade_start = timestamp
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag
                self._profit_levels_taken = set()  # Reset scaled profit targets
                logging.info(f"BUY at {current_price:.2f} - Size: {position_size:.4f} - Confidence: {confidence:.3f}")

            elif action < -self.min_signal_strength:  # Sell signal
                # Size position based on confidence
                confidence_multiplier = confidence ** 0.5
                position_size = self.capital * self.leverage * min(abs(action), 1.0) * confidence_multiplier / current_price
                self.position = -position_size  # Short position
                self.entry_price = current_price
                self.current_trade_start = timestamp
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)
                self.breakeven_activated = False  # Reset breakeven flag
                self._profit_levels_taken = set()  # Reset scaled profit targets
                logging.info(f"SELL at {current_price:.2f} - Size: {position_size:.4f} - Confidence: {confidence:.3f}")

        else:  # Have position - update trailing stops
            self._update_trailing_stops(current_price)

    def _check_dynamic_exits(self, current_price):
        """Check for various exit conditions with improved profit-taking"""
        if self.position == 0:
            return None

        # Calculate current profit percentage
        if self.position > 0:  # Long position
            profit_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            profit_pct = (self.entry_price - current_price) / self.entry_price

        # Breakeven stop activation
        if not self.breakeven_activated and profit_pct >= self.breakeven_trigger_pct:
            self.breakeven_activated = True
            # Move trailing stop to breakeven + small buffer
            buffer_pct = 0.005  # 0.5% buffer above breakeven
            if self.position > 0:
                self.trailing_stop_distance = self.entry_price * (1 + buffer_pct)
            else:
                self.trailing_stop_distance = self.entry_price * (1 - buffer_pct)

        # Scaled profit taking - each level triggers at most once per position
        for target_pct in sorted(self.profit_targets, reverse=True):
            if target_pct in self._profit_levels_taken:
                continue
            if profit_pct >= target_pct:
                self._profit_levels_taken.add(target_pct)
                # Calculate how much profit to take at this level
                if target_pct <= 0.02:  # Small profits (1-2%) - take 25% of position
                    profit_portion = 0.25
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                elif target_pct <= 0.05:  # Medium profits (5%) - take 50% of position
                    profit_portion = 0.50
                    exit_reason = f'profit_{int(target_pct*100)}pct_partial'
                else:  # Large profits (10%) - take full position
                    profit_portion = 1.0
                    exit_reason = 'take_profit'

                if profit_portion < 1.0:
                    # Partial exit - calculate P&L for partial position
                    partial_pnl = (current_price - self.entry_price) * (self.position * profit_portion) if self.position > 0 else (self.entry_price - current_price) * abs(self.position * profit_portion)
                    self.capital += partial_pnl

                    # Record partial trade
                    trade = {
                        'entry_time': self.current_trade_start,
                        'exit_time': datetime.now(),
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'position': 'LONG' if self.position > 0 else 'SHORT',
                        'pnl': partial_pnl,
                        'exit_reason': exit_reason,
                        'confidence': 0.5,  # Placeholder
                        'partial_exit': True
                    }
                    self.trades.append(trade)

                    # Reduce position
                    self.position *= (1 - profit_portion)
                    logging.info(f"PARTIAL CLOSE {trade['position']} - P&L: ${partial_pnl:.2f} - Reason: {exit_reason} - Remaining position: {self.position:.4f}")
                    return None  # Don't exit fully, just partial
                else:
                    # Full exit
                    return exit_reason

        # Trailing stop check (only if breakeven not activated or profit is positive)
        if self.position > 0:  # Long position
            if current_price <= self.trailing_stop_distance:
                return 'trailing_stop'
        else:  # Short position
            if current_price >= self.trailing_stop_distance:
                return 'trailing_stop'

        # Maximum holding time - more aggressive for losing positions
        if self.current_trade_start and (datetime.now() - self.current_trade_start).seconds > self.max_holding_time * 3600:
            return 'max_time'

        # Early exit for significant losses (hard stop loss)
        if profit_pct <= -self.stop_loss_pct:
            return 'stop_loss'

        return None

    def _update_trailing_stops(self, current_price):
        """Update trailing stop levels"""
        if self.position > 0:  # Long position
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 - self.trailing_stop_pct)
        else:  # Short position
            if current_price < self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.trailing_stop_distance = current_price * (1 + self.trailing_stop_pct)

    def run_live_trading(self, symbol='GC=F', interval_minutes=5):
        """Run live trading loop"""
        print("🚀 Starting Live Ensemble Trading...")
        print(f"Symbol: {symbol}")
        print(f"Capital: ${self.capital}")
        print(f"Leverage: {self.leverage}x")
        print(f"Interval: {interval_minutes} minutes")

        last_update = datetime.now() - timedelta(minutes=interval_minutes)

        try:
            while True:
                now = datetime.now()

                # Check if it's time for next update
                if (now - last_update).seconds >= interval_minutes * 60:
                    # Get live data
                    data = self.get_live_data(symbol, period='5d', interval=f'{interval_minutes}m')

                    if data is not None and len(data) > 20:  # Need enough data for indicators
                        # Get ensemble prediction
                        obs = self.get_observation(data)
                        action, confidence = self.ensemble.predict_ensemble(obs)

                        # Execute trade
                        current_price = data.iloc[-1]['Close']
                        self.execute_trade(action, confidence, current_price)

                        # Log status
                        print(f"{now.strftime('%H:%M:%S')} - Price: ${current_price:.2f} - Action: {action:.3f} - Position: {self.position} - Capital: ${self.capital:.2f}")

                        last_update = now

                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            self.print_performance_summary()

        except Exception as e:
            logging.error(f"Trading error: {e}")
            print(f"❌ Trading error: {e}")

    def print_performance_summary(self):
        """Print trading performance summary"""
        if not self.trades:
            print("No trades executed")
            return

        df_trades = pd.DataFrame(self.trades)
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        print("\n📊 Performance Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Final Capital: ${self.capital:.2f}")

def main():
    """Main live trading function"""
    trader = LiveEnsembleTrader(capital=100, leverage=50)
    trader.run_live_trading()

if __name__ == "__main__":
    main()
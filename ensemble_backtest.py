#!/usr/bin/env python3
"""
Ensemble Backtesting
Test ensemble trading system performance on historical data
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensemble_trader import EnsembleTrader
from trading_env import add_technical_indicators


class EnsembleBacktester:
    """
    Backtesting system for ensemble trading strategies
    """

    def __init__(self, ensemble_path='./ensemble_models/', capital=100, leverage=50):
        self.capital = capital
        self.leverage = leverage
        self.initial_capital = capital

        # Load ensemble
        self.ensemble = EnsembleTrader()
        self.ensemble.load_ensemble(ensemble_path)

        # Trading parameters
        self.min_signal_strength = 0.2
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.max_holding_time = 24  # hours

        # Results tracking
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []

    def backtest(self, test_df, symbol='XAUUSD'):
        """Run backtest on historical data"""
        print("🔬 Starting Ensemble Backtest...")
        print(f"Initial Capital: ${self.capital}")
        print(f"Leverage: {self.leverage}x")
        print(f"Data points: {len(test_df)}")

        # Add indicators once up front (same shared pipeline as TradingEnv —
        # fixes the previous version which read non-existent 'rsi'/'macd' columns)
        test_df = add_technical_indicators(test_df)

        position = 0.0
        entry_price = 0.0
        entry_time = None
        entry_step = None
        stop_loss = 0.0
        take_profit = 0.0

        # Pre-compute a numeric step clock so max-holding-time works regardless
        # of whether the DataFrame index is DatetimeIndex (live/real data) or
        # integer RangeIndex (sample/offline replay).
        index = test_df.index
        is_datetime_index = isinstance(index, pd.DatetimeIndex)

        for i in range(20, len(test_df)):  # Start after indicator warmup
            current_data = test_df.iloc[:i+1]
            current_price = float(test_df.iloc[i]['Close'])
            current_time = index[i] if is_datetime_index else i

            # Create observation
            obs = self.create_observation(current_data, position, self.capital)

            # Get ensemble prediction
            action, confidence = self.ensemble.predict_ensemble(obs)
            # Normalise to Python scalars — SB3 models return np.ndarray[shape=(1,)]
            if hasattr(action, '__len__'):
                action = float(action[0])
            else:
                action = float(action)
            if hasattr(confidence, '__len__'):
                confidence = float(confidence[0])
            else:
                confidence = float(confidence)

            # Execute trading logic
            if position == 0:  # No position
                if action > self.min_signal_strength:
                    # Buy signal - size based on confidence
                    confidence_multiplier = confidence ** 0.5  # Square root scaling
                    position_size = self.capital * self.leverage * min(action, 1.0) * confidence_multiplier / current_price
                    position = float(position_size)
                    entry_price = current_price
                    entry_time = current_time
                    entry_step = i
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)

                elif action < -self.min_signal_strength:
                    # Sell signal - size based on confidence
                    confidence_multiplier = confidence ** 0.5
                    position_size = self.capital * self.leverage * min(abs(action), 1.0) * confidence_multiplier / current_price
                    position = float(-position_size)  # Short position
                    entry_price = current_price
                    entry_time = current_time
                    entry_step = i
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)

            else:  # Have position
                should_exit = False
                exit_reason = ""

                # Check stop loss / take profit
                if position > 0:  # Long position
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif current_price >= take_profit:
                        should_exit = True
                        exit_reason = "Take Profit"
                elif position < 0:  # Short position
                    if current_price >= stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif current_price <= take_profit:
                        should_exit = True
                        exit_reason = "Take Profit"

                # Check max holding time (bars or hours depending on index type)
                holding_too_long = False
                if entry_step is not None:
                    if is_datetime_index and entry_time is not None:
                        holding_too_long = (current_time - entry_time).total_seconds() > self.max_holding_time * 3600
                    else:
                        # Treat max_holding_time as "number of bars" when index is numeric
                        holding_too_long = (i - entry_step) >= self.max_holding_time
                if holding_too_long:
                    should_exit = True
                    exit_reason = "Max Time"

                # Check opposite signal
                if (position > 0 and action < -self.min_signal_strength) or \
                   (position < 0 and action > self.min_signal_strength):
                    should_exit = True
                    exit_reason = "Opposite Signal"

                if should_exit:
                    # Calculate P&L
                    if position > 0:  # Long position
                        pnl = (current_price - entry_price) * position
                    else:  # Short position
                        pnl = (entry_price - current_price) * abs(position)
                    self.capital += float(pnl)

                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': 'LONG' if position > 0 else 'SHORT',
                        'pnl': float(pnl),
                        'exit_reason': exit_reason,
                        'holding_time': (
                            (current_time - entry_time).total_seconds() / 3600
                            if is_datetime_index and entry_time is not None
                            else (i - entry_step if entry_step is not None else 0)
                        ),
                    }
                    self.trades.append(trade)

                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
                    entry_step = None

            # Record portfolio value
            self.portfolio_values.append(float(self.capital))
            self.timestamps.append(current_time)

        print("✅ Backtest completed!")

    def create_observation(self, df, position=0.0, balance=1000.0):
        """Create observation matching TradingEnv's 15-dim layout:
        [last 10 closes, RSI, MACD, MACD_signal, position, balance]."""
        lookback = 10
        current_idx = len(df) - 1

        # Get last 10 closing prices; pad at the very start so length is always `lookback`.
        if current_idx + 1 < lookback:
            prices = np.pad(
                df.iloc[:current_idx + 1]['Close'].values.astype(np.float32),
                (lookback - (current_idx + 1), 0),
                mode='edge',
            )
        else:
            prices = df.iloc[current_idx - lookback + 1:current_idx + 1]['Close'].values.astype(np.float32)

        # Get current indicators (columns produced by add_technical_indicators)
        latest = df.iloc[current_idx]
        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        macd_signal = float(latest['MACD_signal'])

        obs = np.concatenate([
            prices,
            np.array([rsi, macd, macd_signal, float(position), float(balance)], dtype=np.float32),
        ]).astype(np.float32)

        return obs.reshape(1, -1)

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        df_trades = pd.DataFrame(self.trades)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate drawdown
        portfolio_series = pd.Series(self.portfolio_values, index=self.timestamps)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        # Sharpe ratio (simplified - assuming 0% risk-free rate)
        returns = portfolio_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
            'final_capital': self.capital,
            'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100
        }

    def plot_results(self, save_path=None):
        """Plot backtest results"""
        if not self.trades or not self.portfolio_values:
            print("No data to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        ax1.plot(self.timestamps, self.portfolio_values)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)

        # Trade P&L distribution
        if self.trades:
            pnl_values = [trade['pnl'] for trade in self.trades]
            ax2.hist(pnl_values, bins=30, alpha=0.7, edgecolor='black')
            ax2.set_title('Trade P&L Distribution')
            ax2.set_xlabel('P&L ($)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # Cumulative returns
        portfolio_series = pd.Series(self.portfolio_values, index=self.timestamps)
        cumulative_returns = (portfolio_series / self.initial_capital - 1) * 100
        ax3.plot(self.timestamps, cumulative_returns)
        ax3.set_title('Cumulative Returns (%)')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.grid(True)

        # Drawdown
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        ax4.fill_between(self.timestamps, drawdown, 0, alpha=0.3, color='red')
        ax4.set_title('Drawdown (%)')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_xlabel('Date')
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Results plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print detailed backtest summary"""
        metrics = self.calculate_metrics()

        print("\n📊 Ensemble Backtest Results")
        print("=" * 50)
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Capital: ${metrics['final_capital']:.2f}")
        print(f"Total Return: {metrics['return_pct']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        # Exit reason analysis
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            exit_reasons = df_trades['exit_reason'].value_counts()
            print("\nExit Reasons:")
            for reason, count in exit_reasons.items():
                pct = count / len(df_trades) * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")

def main():
    """Main backtesting function"""
    # Load test data
    df = pd.read_csv('xauusd_data.csv', parse_dates=['date'], index_col='date')

    # Add technical indicators to the full dataset
    df = add_technical_indicators(df)

    # Split data
    train_end = int(len(df) * 0.8)
    test_df = df.iloc[train_end:]

    # Run backtest
    backtester = EnsembleBacktester(capital=100, leverage=50)
    backtester.backtest(test_df)

    # Print results
    backtester.print_summary()

    # Plot results
    backtester.plot_results('ensemble_backtest_results.png')

    # Save detailed results
    metrics = backtester.calculate_metrics()
    with open('ensemble_backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print("✅ Backtest completed! Results saved.")

# NOTE: indicator computation lives in trading_env.add_technical_indicators so
# training / backtest / live paths always share the exact same feature pipeline.
if __name__ == "__main__":
    main()
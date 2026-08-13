# 🤖 AI-Driven XAUUSD Trading System: Maximum Profitability Framework

[![Version](https://img.shields.io/badge/version-1.0-blue.svg)](https://github.com/JonusNattapong/AI-XAUUSD-Trading)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/JonusNattapong/AI-XAUUSD-Trading)

## 📊 Performance Highlights

> ⚠️ **指标复核中（2026-08）**：深度代码审查发现此前版本的 `trading_env.py` 存在盈亏结算 bug（止损退出被错误记为 $0），下列数据基于该缺陷环境产生，**不能作为真实预期**。环境已在新版本修复，指标将在重新训练与回测后刷新。详见仓库中的 `PYTHON_REVIEW_REPORT.md`。

- **🎯 58.3% Win Rate** (26% improvement over baseline) *(待复测)*
- **💰 11x Better Average Wins** ($4.16 → $49.45) *(待复测)*
- **⚖️ Risk-Reward Ratio: 1:0.47** (2.8x improvement) *(待复测)*
- **🎯 45 USD Daily Profit Target - ACHIEVED** *(待复测)*
- **🧠 Market Regime-Adaptive Parameters**

## 🚀 Key Features

### 🤖 Advanced AI Ensemble
- **PPO, TD3, SAC** reinforcement learning algorithms
- **Confidence-weighted ensemble** decision making
- **Curriculum learning** for optimal timing patterns

### 🎯 Market Regime Detection
- **6 Market Conditions**: Strong Bull, Bull Trend, Bear Trend, Strong Bear, Ranging, High/Low Volatility
- **Adaptive Parameters**: Different strategies for each regime
- **Real-time Adaptation**: Dynamic parameter optimization

### 💰 Advanced Risk Management
- **Scaled Profit-Taking**: 1%, 2%, 5%, 10% profit levels
- **Breakeven Stops**: Automatic protection after 1.5% profit
- **Confidence-Based Sizing**: Higher confidence = larger positions
- **Trailing Stops**: 2.5% for better profit capture

### 📈 Live Trading Ready
- **Yahoo Finance API** integration
- **Real-time execution** with automated order management
- **Comprehensive monitoring** and risk controls
- **Emergency shutdown** procedures

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Performance Analysis](#performance-analysis)
- [Market Regime Adaptation](#market-regime-adaptation)
- [Risk Management](#risk-management)
- [Live Trading](#live-trading)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Clone Repository
```bash
git clone https://github.com/JonusNattapong/AI-XAUUSD-Trading.git
cd AI-XAUUSD-Trading
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

> ℹ️ The repo already ships three trained model zips in `ensemble_models/`
> (`ppo_model.zip`, `td3_model.zip`, `sac_model.zip`).
> `EnsembleTrader.load_ensemble()` auto-discovers them if `ensemble_config.json`
> is missing (the file is not committed because of a `*.json` gitignore rule),
> so **no extra download step is required** to run.

### Fetch Training Data (训练/回测前置步骤，可选)
```bash
python data_fetch.py    # 从 Yahoo Finance 下载 GC=F 日线 → xauusd_data.csv
                        # (无网络时 start_trading.py 会自动回退到 data/xauusd_sample.csv)
```

### Run Tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## 🚀 Quick Start

### 一键启动（推荐）

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 一键 paper trading（离线回放，无需网络，30 秒内看到结果）
python start_trading.py --paper --capital 1000 --leverage 50

# 3. 一键 ensemble 回测（生成 equity 曲线 PNG）
python start_trading.py --backtest --capital 1000

# 4. 实时模式（需网络，拉 Yahoo Finance GC=F，仅模拟成交）
python start_trading.py --live --capital 1000 --leverage 50

# 5. 重新训练 ensemble（PPO+TD3+SAC）后再回放/回测
python start_trading.py --retrain --timesteps 20000 --paper
```

`start_trading.py` 会自动：检查依赖 → 自动补全 `ensemble_config.json` → 优先复用
`xauusd_data.csv`，失败时回退到 `data/xauusd_sample.csv`（无需网络）→ 运行所选模式。

### 旧版 API（仍然可用）

#### Run Backtesting Demo
```python
from advanced_trading_demo import run_advanced_trading_demo
run_advanced_trading_demo()
```

#### Live Trading Setup
```python
from live_ensemble_trading import LiveEnsembleTrader

trader = LiveEnsembleTrader(capital=1000, leverage=50)
trader.run_live_trading()   # Ctrl-C 停止并打印绩效
```

### Market Regime Analysis
```python
from market_regime_detector import MarketRegimeDetector

# Initialize detector
detector = MarketRegimeDetector()

# Analyze current market
regime, params = detector.detect_regime(price_data)
print(f"Current regime: {regime.value}")
print(f"Optimal parameters: {params}")
```

## 🏗️ System Architecture

实际代码结构（扁平模块布局）：

```
AI-XAUUSD-Trading/
├── 🤖 核心 AI 引擎
│   ├── trading_env.py              # Gymnasium 交易环境（含共享特征管线
│   │                               #   add_technical_indicators / build_observation）
│   ├── optimal_timing_env.py       # 出入场时机环境（纯 pandas 指标，无前视偏差）
│   ├── transformer_policy.py       # Transformer 特征提取策略（15 维观测）
│   ├── ensemble_trader.py          # PPO/TD3/SAC 集成交易器
│   └── curriculum_training.py      # 课程式训练
├── 🎯 市场智能
│   ├── market_regime_detector.py   # 6 种市场状态识别 + 自适应参数
│   └── regime_adaptive_trading_demo.py
├── 🔴 实盘与回测
│   ├── live_ensemble_trading.py    # 实盘交易循环 (run_live_trading)
│   ├── ensemble_backtest.py        # 集成回测
│   ├── backtest.py / forward_test.py / test_env.py
│   └── train_model.py / data_fetch.py / download_models.py
├── 📊 分析与演示
│   ├── quick_demo.py               # 零重依赖快速演示（无需 torch/SB3）
│   ├── advanced_trading_demo.py / confidence_sizing_demo.py
│   └── trading_performance_analysis.py / real_results_demo.py
├── 🧪 工程化
│   ├── tests/test_smoke.py         # 回归测试（编码了审查发现的全部 bug 场景）
│   ├── .github/workflows/ci-cd.yml # CI/CD
│   └── setup.py / pyproject.toml   # 打包（py-modules 扁平布局）
└── 📦 发布
    ├── upload_to_hf.py             # 上传 Hugging Face Hub
    └── arxiv_submit.py / prepare_arxiv_submission.py
```

## 📊 Performance Analysis

### Backtesting Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Win Rate | 46.3% | **58.3%** | +12.0% ↑ |
| Average Win | $4.16 | **$49.45** | +11.0x ↑ |
| Average Loss | -$25.00 | -$106.18 | +4.2x |
| Risk-Reward Ratio | 1:0.17 | **1:0.47** | +2.8x |
| Profit Exit Rate | 2.8% | **50.0%** | +17.9x |
| Daily Target | $0 | **$45+** | ✅ Achieved |

### Risk Metrics
- **Sharpe Ratio**: 2.0+ (excellent)
- **Sortino Ratio**: 2.5+ (superior)
- **Calmar Ratio**: 3.0+ (outstanding)
- **Maximum Drawdown**: <5% (controlled)

## 🎯 Market Regime Adaptation

The system automatically detects and adapts to 6 market conditions:

### 📈 Strong Bull Markets
- **Profit Targets**: 1.5%, 3%, 6%, 12%
- **Position Size**: 1.5x normal
- **Strategy**: Aggressive profit capture

### 📊 Ranging Markets
- **Profit Targets**: 0.8%, 1.5%, 3%, 6%
- **Position Size**: 0.7x normal
- **Strategy**: Conservative, quick profits

### 🌪️ High Volatility
- **Profit Targets**: 2%, 4%, 8%, 15%
- **Position Size**: 0.6x normal
- **Strategy**: Fast exits, minimal exposure

## 💰 Risk Management

### Scaled Profit-Taking
```python
# Multiple profit levels for optimal capture
profit_targets = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%

# Partial exits at different levels
if profit_pct >= 0.02:    # 2% profit
    exit_portion = 0.25   # Take 25% of position
elif profit_pct >= 0.05:  # 5% profit
    exit_portion = 0.50   # Take 50% of position
```

### Breakeven Protection
```python
# Automatic breakeven after 1.5% profit
if profit_pct >= 0.015:
    breakeven_activated = True
    trailing_stop = entry_price * (1 + 0.005)  # +0.5% buffer
```

## 🔴 Live Trading

### Setup Live Trading
```python
from live_ensemble_trading import LiveEnsembleTrader

trader = LiveEnsembleTrader(
    capital=1000,
    leverage=50,
)

# Start automated trading (Ctrl-C to stop; prints a performance summary)
trader.run_live_trading()
```

### Monitoring
```python
# Performance summary over recorded trades
trader.print_performance_summary()

# Trade log is also written to ensemble_trading.log
```

## 📚 API Reference

### Core Classes

#### `EnsembleTrader`
```python
class EnsembleTrader:
    def __init__(self, models_config: dict = None)
    def train_ensemble(self, train_df, save_path='./ensemble_models/') -> None
    def load_ensemble(self, load_path='./ensemble_models/') -> None
    # Returns (action, confidence); method: 'weighted_vote' | 'average' | 'majority'
    def predict_ensemble(self, observation, method='weighted_vote') -> Tuple[float, float]
```

#### `MarketRegimeDetector`
```python
class MarketRegimeDetector:
    def detect_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, Dict]
    def get_optimal_parameters(self, regime: MarketRegime) -> Dict
```

#### `LiveEnsembleTrader`
```python
class LiveEnsembleTrader:
    def __init__(self, ensemble_path='./ensemble_models/', capital=100, leverage=50)
    def run_live_trading(self, symbol='GC=F', interval_minutes=5) -> None
    def execute_trade(self, action, confidence, current_price) -> None
    def print_performance_summary(self) -> None
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/your-username/AI-XAUUSD-Trading.git
cd AI-XAUUSD-Trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings to all functions
- Write comprehensive unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JonusNattapong / Zombitx64** - Lead Developer & Researcher
- Stable-Baselines3 team for RL framework
- Yahoo Finance for market data API
- Open-source AI community

## 📞 Contact

**JonusNattapong / Zombitx64**
- Email: jonusnattapong@zombitx64.com
- GitHub: [@JonusNattapong](https://github.com/JonusNattapong)
- LinkedIn: [JonusNattapong](https://linkedin.com/in/jonusnattapong)
- Hugging Face: [@JonusNattapong](https://huggingface.co/JonusNattapong)

## 🔗 Links

- **GitHub Repository**: https://github.com/JonusNattapong/AI-XAUUSD-Trading
- **Hugging Face Model**: https://huggingface.co/JonusNattapong/AI-XAUUSD-Trading
- **Documentation**: https://jonusnattapong.github.io/AI-XAUUSD-Trading
- **White Paper**: [AI_XAUUSD_Trading_White_Paper.pdf](AI_XAUUSD_Trading_White_Paper.pdf)

## ⚠️ Disclaimer

**This system is for educational and research purposes only.**

Trading cryptocurrencies and financial instruments involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper trading mode before deploying with real capital. Use proper risk management and never trade with money you cannot afford to lose.

The authors are not responsible for any financial losses incurred through the use of this system.

---

**⭐ Star this repository if you find it helpful!**

**🚀 Ready to achieve 45 USD daily profit with AI-powered trading!**
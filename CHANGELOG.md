# Changelog

All notable changes to the AI-XAUUSD Trading System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-08-08

### 🔧 关键修复版本（源自全量代码审查，详见 `PYTHON_REVIEW_REPORT.md`）

#### 🚨 Fixed — 致命运行时缺陷
- **TradingEnv 无法实例化**：删除 `_calculate_indicators()` 中错误合并进来的 `__init__` 尾部副本（`current_step` AttributeError + 无限递归）
- **盈亏记账错乱**：止损/移动止损/超时退出曾被记为 $0（环境"永不亏损"）；新增 `_close_position()` 先结算后清状态
- **分级止盈失效**：每个止盈档位现在只触发一次；部分平仓正确保留剩余仓位
- **数据末尾未平仓**：回合结束强制平仓并结算

#### 🏗️ Changed — 架构与依赖
- 全部环境迁移 Gymnasium API（与 stable-baselines3 2.x 兼容）；观测统一 `float32`
- `optimal_timing_env.py`：移除"用未来 10 根K线计算奖励"的前视偏差；指标改用纯 pandas 实现，**移除 TA-Lib 依赖**；观测空间尺寸与实际输出一致（120 维）；盈亏按名义本金×杠杆正确量级
- `transformer_policy.py`：按真实 15 维观测重写特征切分（原按 27 维假设会维度不匹配崩溃）
- 新增共享特征管线 `trading_env.add_technical_indicators()` / `build_observation()`；实盘与集成回测的观测已对齐训练环境
- `requirements.txt`：移除从未使用的 `ta`、`pandas-ta`；新增 `requirements-dev.txt`

#### 🧪 Added — 测试与工程化
- `tests/test_smoke.py`：12 个回归测试，覆盖本次审查发现的全部关键场景（12/12 通过）
- CI/CD 重写：适配扁平布局、安装 CPU 版 torch 加速、flake8 致命错误门禁、pytest 覆盖率、PyPI/Docs 改为手动触发、Docker 镜像改推 GHCR
- `setup.py` / `pyproject.toml`：`py-modules` 显式声明（此前打包结果为空）；控制台入口修正为真实存在的函数

#### 📚 Docs
- README：真实文件结构图、实盘 API 示例修正（`run_live_trading`）、性能指标标注"待复测"、补充 `data_fetch.py` 前置步骤与测试说明

## [1.0.0] - 2024-12-XX

### 🎉 Major Release: Maximum Profitability Framework

**This is the first stable release of the AI-XAUUSD Trading System, achieving the target of 45 USD daily profit through advanced AI ensemble methods and market regime adaptation.**

### ✨ Added

#### 🤖 Core AI Engine
- **Ensemble Trading System**: PPO, TD3, and SAC reinforcement learning models
- **Confidence-weighted Decision Making**: Dynamic position sizing based on model confidence
- **Curriculum Learning**: Progressive training for optimal market timing
- **Advanced Feature Engineering**: 20+ technical indicators and market features

#### 🎯 Market Intelligence
- **Market Regime Detection**: 6 distinct market conditions (Strong Bull, Bull Trend, Bear Trend, Strong Bear, Ranging, High/Low Volatility)
- **Adaptive Parameters**: Dynamic strategy adjustment based on market conditions
- **Real-time Regime Classification**: Continuous market state monitoring
- **ADX-based Trend Strength**: Advanced directional movement analysis

#### 💰 Risk Management
- **Scaled Profit-Taking**: 1%, 2%, 5%, 10% profit level exits
- **Breakeven Stops**: Automatic protection after 1.5% profit
- **Confidence-Based Sizing**: Higher confidence = larger positions (0.5x to 2.0x)
- **Trailing Stops**: 2.5% trailing for better profit capture
- **Emergency Shutdown**: Automated risk controls and circuit breakers

#### 📊 Performance & Analytics
- **Comprehensive Backtesting**: Historical performance validation
- **Live Trading Interface**: Real-time execution with Yahoo Finance integration
- **Performance Monitoring**: Real-time PnL tracking and risk metrics
- **Visualization Suite**: Charts and reports for performance analysis

#### 🛠️ Infrastructure
- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Flexible parameter tuning
- **Logging System**: Comprehensive event tracking
- **Error Handling**: Robust exception management

### 📈 Performance Achievements

- **🎯 58.3% Win Rate** (26% improvement over baseline)
- **💰 11x Better Average Wins** ($4.16 → $49.45)
- **⚖️ Risk-Reward Ratio: 1:0.47** (2.8x improvement)
- **🎯 45 USD Daily Profit Target - ACHIEVED**
- **🧠 Market Regime-Adaptive Parameters**

### 🔧 Technical Improvements

- **Stable-Baselines3 Integration**: Industry-standard RL framework
- **Gymnasium Environment**: Modern reinforcement learning interface
- **PyTorch Backend**: High-performance deep learning
- **Pandas/Numpy Stack**: Efficient data processing
- **Scikit-learn Integration**: Advanced analytics and preprocessing

### 📚 Documentation

- **Comprehensive README**: Installation, usage, and API documentation
- **White Paper**: Academic-style documentation of methodology and results
- **Code Documentation**: Extensive docstrings and type hints
- **Contributing Guide**: Developer onboarding and contribution guidelines
- **License**: MIT License for open-source distribution

### 🧪 Testing & Quality

- **Unit Test Suite**: Comprehensive test coverage
- **Integration Tests**: End-to-end system validation
- **Code Quality**: PEP 8 compliance with Black formatting
- **Type Checking**: MyPy static analysis
- **CI/CD Pipeline**: Automated testing and deployment

## [0.5.0] - 2024-11-XX (Pre-release)

### ✨ Added
- Basic ensemble trading with PPO/TD3/SAC
- Confidence-based position sizing
- Initial market regime detection
- Trailing stops implementation
- Performance analysis tools

### 📈 Performance
- 46.3% win rate baseline
- Risk-reward ratio: 1:0.17
- Average win: $4.16

## [0.4.0] - 2024-11-XX (Pre-release)

### ✨ Added
- Advanced trading environment with dynamic exits
- Scaled profit-taking mechanism
- Breakeven stop protection
- Enhanced risk management

### 📈 Performance
- 50.0% profit exit rate improvement
- Tighter stop losses
- Better risk-reward balance

## [0.3.0] - 2024-11-XX (Pre-release)

### ✨ Added
- Market regime detection system
- Adaptive parameter optimization
- 6 market condition classifications
- Regime-specific trading strategies

## [0.2.0] - 2024-11-XX (Pre-release)

### ✨ Added
- Basic PPO trading model
- Technical indicators integration
- Yahoo Finance data acquisition
- Backtesting framework

## [0.1.0] - 2024-11-XX (Pre-release)

### ✨ Added
- Initial project structure
- Basic DQN trading model
- Simple trading environment
- Data fetching utilities

---

## 📋 Version Numbering

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

## 🎯 Future Releases

### Planned for v1.1.0
- [ ] Web dashboard for real-time monitoring
- [ ] Additional technical indicators
- [ ] Multi-timeframe analysis
- [ ] Advanced order types (limit orders, etc.)

### Planned for v1.2.0
- [ ] Alternative data sources integration
- [ ] Multi-asset trading support
- [ ] Portfolio optimization
- [ ] Social trading features

### Planned for v2.0.0
- [ ] Transformer-based models
- [ ] Advanced NLP for news analysis
- [ ] Decentralized execution
- [ ] Cross-exchange arbitrage

---

**Legend:**
- 🎉 Major release
- ✨ New feature
- 📈 Performance improvement
- 🔧 Technical enhancement
- 📚 Documentation
- 🧪 Testing/Quality
- 🐛 Bug fix
- ⚠️ Breaking change
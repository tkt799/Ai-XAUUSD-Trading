# 🖱️ 一键 EXE 使用说明（双击即跑）

本目录提供"双击打包 + 双击运行"的完整方案，**无需任何命令行知识**。

---

## 一、你只需要做两件事

### 第 1 步：双击 `build_exe.bat`（仅首次）

- 自动安装依赖 → 调用 PyInstaller → 在 `dist\` 下生成 **`AiXauusdTrading.exe`**
- 过程中会让你选模式：

| 模式 | 包含内容 | 打包耗时 | exe 体积 | AI 训练/回测 |
|------|----------|---------|---------|--------------|
| **1 完整版** | torch + stable-baselines3 全套 | 10–30 分钟 | ~1 GB | ✅ 可用 |
| **2 轻量版**（默认） | 自检 + 冒烟 + 演示 + 联网数据 | 3–5 分钟 | ~150 MB | ⏭️ 自动跳过 |

> 想真正跑 AI 训练选 1；只想先试试选 2（回车即默认）。

### 第 2 步：双击 `dist\AiXauusdTrading.exe`

控制台窗口自动依次执行全部阶段，结束后**自动打开 output 文件夹**：

```
[0] 环境自检          → output/system_report.txt
[1] 冒烟检查           → 3 组核心回归断言（止损必亏、维度匹配…）
[2] 快速演示           → output/demo_results.png
[3] 数据获取           → output/xauusd_data.csv（exe 旁有现成 CSV 会直接复用）
[4] AI 训练+回测       → 模型 zip、backtest_metrics.json、trades_log.csv、
                          equity_curve.png（仅完整版；轻量版显示 SKIP）
[5] 总结报告           → output/SUMMARY.md + 打开输出文件夹
```

退出码：全部 PASS/SKIP = 0；任何 FAIL = 1（细节见 `output\run.log`）。

---

## 二、常见问题

**Q1. 杀毒软件报毒？**
PyInstaller 单文件 exe 常见误报。方案：添加到信任区；或自行打包（本脚本就是在你本机打包的，来源可见可信）。

**Q2. 提示未找到 Python？**
安装 [Python 3.9–3.11](https://www.python.org/downloads/) 时**勾选 "Add Python to PATH"**，装完重开窗口再双击。

**Q3. pip 下载慢/失败（国内网络）？**
```bat
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
然后重新双击 build_exe.bat。

**Q4. 想用自己的数据？**
把 `xauusd_data.csv`（含 date,Open,High,Low,Close,Volume 列）放到 **exe 旁边**即可，程序会直接复用，不再联网下载。

**Q5. 想改训练时长？**
命令行运行（可选）：
```bat
AiXauusdTrading.exe --full          @REM 完整训练 5 万步（慢）
AiXauusdTrading.exe --skip-network  @REM 离线模式
AiXauusdTrading.exe --no-pause      @REM 结束不暂停（自动化）
AiXauusdTrading.exe --help          @REM 全部参数
```

**Q6. macOS / Linux？**
用 `./build_exe.sh full` 或 `./build_exe.sh`（轻量），产物为 `dist/AiXauusdTrading`。

---

## 三、技术说明（可选阅读）

- 打包器：PyInstaller ≥ 6.0，配置文件 `ai_xauusd_trading.spec`
  （自动按需收集 torch/SB3/yfinance 等；排除 tensorflow/jax/optuna 等无关大件）
- Windows 的 exe 必须在 Windows 上构建（PyInstaller 不跨平台），因此首次双击
  `build_exe.bat` 是在你本机完成构建的标准做法
- exe 内置的全部阶段逻辑在 `run_all.py`；每个阶段独立 try/except，
  单点失败不影响后续阶段

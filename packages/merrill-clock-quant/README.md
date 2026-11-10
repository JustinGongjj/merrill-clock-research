# Merrill Clock Quant

**Author:** JustinGongjj · **License:** MIT

一个可检查、可独立运行的 Python 研究原型：用宏观增长与通胀动能构造美林时钟四象限概率，将其映射为跨资产研究权重，并提供回测、预测基线、离线 Q-learning 与合约级期货换月工具。仅用于研究、教学和软件测试，不是投资建议或交易系统。

## 功能

- **美林时钟状态**：把美国工业生产和 CPI 的动能变化转为四象限软概率；加速度特征在观测月网格上计算，再按假设发布滞后对齐到任意频率的决策日历。
- **跨资产研究回测**：支持月频收益、下一期执行、换手成本扣减、描述性绩效指标、等权基准对比和 walk-forward 切分。默认使用 ETF 作为资产类别代理。
- **模型基线**：扩展窗口 Ridge 预测，以及固定动作集合的离线表格 Q-learning 实验。没有附带模型权重。
- **期货换月**：可用调用者提供的逐合约结算价与 first-notice 日期构造换月收益代理，避免把跨月价差误算成收益。
- **离线自检**：`examples_quant_demo.py` 用固定种子的合成数据跑通全链路，无需网络或第三方行情。

## 安装与运行

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[market,test]"
```

### 离线演示（不联网）

```bash
python examples_quant_demo.py
python examples_quant_demo.py --seed 7 --periods 180 --cost-bps 10
```

输出策略与等权基准的 CAGR、波动、Sharpe、最大回撤，以及样本区间和被排除的预热月数。合成收益由上一期宏观特征生成，仅用于验证代码链路，任何绩效数字都不构成策略有效性证据。

### 在线研究流程

```bash
# 会联网请求 FRED 与 Yahoo Finance；输出只写入本地 outputs/
merrill-clock-quant run --start 2000-01-01 --output ./outputs/us-clock
```

不安装 `market` extra 仍可使用核心函数和 CSV 输入接口；当前 CLI 的在线流程会下载公开数据。没有自动下载或嵌入真实期货曲线。公共数据接口偶发断连时，取数层会自动重试幂等 GET 请求。

### 本地金融数据库（离线运行）

配套的 `merrill-clock-data` 把 FRED、Yahoo 与 Fama–French 数据落到本机 SQLite，回测可完全离线复现，不再受上游限流影响。

```bash
clock-data init                      # 建库并登记数据源
clock-data check                     # 数据源连通性自检
clock-data sync --dataset all        # 下载入库
clock-data status                    # 查看覆盖范围与同步日志

merrill-clock-quant run --data-source local --output ./outputs/local-run
merrill-clock-quant run --data-source auto  --output ./outputs/auto-run
clock-data run --output ./outputs/local-run   # 一步跑完回测 + FF5 归因
```

`--data-source` 三档：`live`（默认，联网取数）、`local`（只读本地库）、`auto`（先探活，在线不可用时整链路自动回退本地库）。回退会在 `metrics.json` 的 `data_source` 里记为 `local-fallback`，数据出处不含糊。数据库默认在 `~/.local/share/clockdata/finance.db`，可用 `--db` 或 `CLOCK_DATA_DB` 指定。安装与命令清单见 `LOCAL_SETUP.md`。

Python API 示例：

```python
from quantclock.backtest import run_backtest, performance_metrics
from quantclock.futures import front_month_returns
from quantclock.signals import regime_probabilities
```

`run` 可用参数：`--start`、`--output`、`--cost-bps`、`--ip-lag-days`、`--cpi-lag-days`、`--z-window`、`--smooth`、`--cash-policy {auto,zero,require_bil}`、`--keep-incomplete-month`、`--data-source {live,local,auto}`、`--db`。

## 输出口径

`outputs/<name>/` 下生成宏观数据、日频代理价格、月频收益、回测明细和 `metrics.json`。

- `clock_backtest.csv` 的 `evaluated=1` 行计入绩效；预热期（z-score 窗口尚未形成有效权重）的 `net_return` 写成 `NaN`，下游归因不会把空仓期当作策略收益。
- `metrics.json` 记录实际 `sample_start` / `sample_end`、预热剔除月数、因核心资产价格缺失而丢弃的月数、现金 sleeve 处理方式，以及等权基准（四类核心 ETF 月度再平衡、无成本）的对照指标。
- 默认 `--cash-policy auto`：BIL 历史完整时使用 BIL；否则现金按显式零收益计提。零收益会低估 BIL 上市前的国库券利息，需要精确现金收益时请自行接入 3 个月期国库券序列，或用 `--cash-policy require_bil` 复现"必须有 BIL 才回测"的口径。
- 月频重采样会用未来月末日期标记尚未结束的月份，该月默认剔除；`--keep-incomplete-month` 可保留，用于模拟数据。
- 最大回撤从初始单位净值起算，首期亏损不会被漏计。

## 与因子回归仓库联动

`merrill-clock-factor-regression` 可直接对本仓库输出的净收益做 Fama–French 五因子归因：

```bash
merrill-clock-quant run --start 2000-01-01 --output ./outputs/us-clock
cd ../merrill-clock-factor-regression
python examples_factor_regression.py \
  --returns-csv ../merrill-clock-quant/outputs/us-clock/clock_backtest.csv \
  --return-column net_return --maxlags 3
```

两个仓库彼此独立：既可通过 CSV 联动，也可共同读取本地金融数据库（见上节）。预热期的空值会被回归自动剔除。

## 测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

43 个测试使用确定性合成序列，不需要联网；覆盖范围见 [TESTING.md](TESTING.md)。

## 相位校准与研究契约

四相概率是宏观加速度的 logistic 分解，不是标定过的预测概率。每次 `run` 都会额外写出 `phase_diagnostics.json`，用数值把这个限制说清楚：

- 增长／通胀方向的 Brier 分数、四相命中率与「永远猜最常见相位」的常数基线对照；
- 平均归一化熵（信号有多犹豫）、主导相位平均持续期、各相位占比；
- `temperature` 在 0.5 / 1.0 / 2.0 三档下的平均最大概率与平均熵，用于参数敏感性对照，不宣称哪档最优。

`--temperature` 可显式指定 logistic 温度（默认 1.0，越小越果断）。`run_manifest.json` 把成本、时点口径、标的池、约束、样本区间、测试变体数、基准定义和参数确认状态随结果一起落盘；未显式确认、沿用默认值的参数会被标记并打印提示，默认值不会悄悄藏在结果里。该文件由 `clockdata` 提供统一 schema，独立安装 `quantclock` 时会自动跳过。

`metrics.json` 另外记录 `phase_quadrant_hit_rate`、`phase_brier_growth_up`、`phase_hit_rate_minus_baseline` 与 `temperature`，便于跨次运行对照。

## 数据来源与方法边界

- FRED 工业生产与 CPI 默认返回当前可见的修订值。代码设置的滞后天数仅为近似，不能替代真实发布日期与 ALFRED 点时版本。
- Yahoo Finance ETF 调整价只是第三方代理数据，需自行核验企业行动、质量与使用/再分发条款。债券 ETF、黄金 ETF、商品 ETF 和现金 ETF 并非对应现货、国债现金流或真实期货持仓。部分 ETF 上市较晚，`metrics.json` 会披露实际样本起算日。
- 四象限概率不是经校准的经济周期标签、衰退概率或收益预测；示例权重只是研究先验。回测成本是简化换手扣减，不是滑点/冲击成本模型。
- Ridge 与强化学习实验不证明样本外有效或可盈利；请保留独立测试区间并与简单基准比较。
- 期货换月工具不模拟合约乘数、保证金、交易时点、交易所日历、手续费或真实滑点。
- 仓库不含真实行情、回测快照、个人数据、密钥、生成报告或模型权重。运行产生的数据和报告请勿未经审查就提交。

详见 [DATA_SOURCES.md](DATA_SOURCES.md)。软件按 MIT 许可发布；许可证不覆盖第三方数据及商标。

## 免责声明

历史数据与模拟结果不代表未来表现。该软件不提供证券建议、不连接券商、不下单，也不保证数据完整、准确或持续可用。使用者须自行核验数据、授权、模型假设及适用法规。

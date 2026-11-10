> **📦 本仓库已并入 monorepo**
> 代码已迁移至 **https://github.com/JustinGongjj/merrill-clock-research** 的
> `packages/merrill-clock-factor-regression/`（`git subtree` 保留完整历史）。
> 后续改动以 monorepo 为准；本仓库仅作归档保留。

# Merrill Clock Factor Regression

**Author:** JustinGongjj · **License:** MIT

独立的因子回归与历史归因研究小项目：将策略收益与因子收益按日期对齐，按需扣除无风险收益，再用 OLS 估计因子载荷，并以 Newey–West/HAC 协方差估计标准误。它可以单独运行，不依赖 Merrill Clock Quant 仓库，也不包含信维通信研究或真实行情数据。

## 安装

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test]"
```

## 快速演示（离线）

```bash
python examples_factor_regression.py
```

默认使用固定随机种子的合成月收益，无网络请求，不代表任何实际策略。输出样本区间、日期对齐方式、HAC 设置、回归截距、因子系数、HAC t/p 值、样本数和 R-squared。

## 分析自己的收益

输入收益 CSV 至少需要 `date` 和 `net_return` 两列，收益率使用小数（例如 1% 写为 `0.01`）。若提供因子 CSV，它必须含有日期列及小数格式的因子收益；可选 `RF` 列作为无风险收益。策略收益与因子收益的日期标签允许一个是月初、一个是月末，回归会按日历月对齐并输出对齐说明；同一输入每月含多条观测时会明确报错。

```bash
python examples_factor_regression.py \
  --returns-csv ./my_returns.csv \
  --factors-csv ./my_factors.csv \
  --return-column net_return \
  --date-column date \
  --maxlags 3
```

`--use-correction` 启用 HAC 协方差的小样本自由度修正（默认关闭，与 statsmodels 默认一致）。

也可以不传 `--factors-csv`，脚本将运行时下载 Kenneth French Data Library 的美国月度五因子文件；请先查看当前数据定义和使用条款。数据适配器将源文件百分数转换为小数，并自动把 `RF` 用作无风险序列。此时需要联网，取数层会自动重试幂等 GET 请求。

### 从本地金融数据库读取因子

配套 `merrill-clock-data` 已把 FF5 因子落到本机 SQLite 后，可完全离线做归因：

```bash
python examples_factor_regression.py \
  --returns-csv ../merrill-clock-quant/outputs/local-run/clock_backtest.csv \
  --return-column net_return --factors-from-db --maxlags 3
```

`--factors-from-db` 与 `--factors-csv` 互斥；数据库路径由 `--db` 或 `CLOCK_DATA_DB` 指定，默认 `~/.local/share/clockdata/finance.db`。`clock-data run` 会一次跑完回测加归因，两端都只读本地库。

加 `--output-json ./factor_attribution.json` 可把样本窗口、对齐说明、R-squared、各因子载荷及 HAC t/p 值写成结构化 JSON，便于回看或供仪表盘读取；`clock-data run` 默认已启用该输出。本机还可用 `clock` 一键跑完整链路并打开结果仪表盘，见 `LOCAL_SETUP.md`。

Python API：

```python
from factorlab.regression import factor_regression

result = factor_regression(strategy_returns, factor_returns, maxlags=3, risk_free_returns=rf)
print(result.params)
print(result.tvalues)
print(result.sample_start, result.sample_end, result.alignment_note)
```

若 `factor_returns` 自带 `RF` 列，它会被当作无风险收益并从解释变量中移除；此时再传 `risk_free_returns` 会报错，避免重复扣减。

## 对 Merrill Clock Quant 的回测输出做归因

两个仓库独立安装，通过 CSV 或本地金融数据库衔接：

```bash
# 在 merrill-clock-quant 中
merrill-clock-quant run --start 2000-01-01 --output ./outputs/us-clock

# 在本仓库中
python examples_factor_regression.py \
  --returns-csv ../merrill-clock-quant/outputs/us-clock/clock_backtest.csv \
  --return-column net_return --maxlags 3
```

回测输出中 `evaluated=1` 的月份才计入绩效，预热期 `net_return` 为空值，回归会自动剔除。归因样本的实际起止月由两边数据覆盖共同决定，终端会打印样本区间。

## 双轨研究：归因轨与检验轨

两个命令回答两个不同的问题，报告分开落盘，不要混用。

| 轨 | 问题 | 命令 | 主要产出 |
|---|---|---|---|
| 归因轨 | 我的策略超额收益由哪些已知风格因子解释？ | `merrill-clock-factor attribute` | `factor_attribution.json`（系数 / HAC t 值 / p 值 / R²） |
| 检验轨 | 某个横截面信号能否排序下一期收益？ | `merrill-clock-factor check` | `cross_section_report.json`（RankIC / ICIR / 胜率 / 分层单调性 / Fama–MacBeth） |
| 合成 | 多个因子的权重怎么配？ | `merrill-clock-factor synthesize` | `synthesis_report.json`（权重 + 滚动权重稳定性） |

```bash
# 检验轨：直接读本地金融数据库的价格数据集
merrill-clock-factor check --db ~/.local/share/clockdata/finance.db \
  --factors mom_1m,mom_3m,vol_3m,rev_1m \
  --output-json cross_section_report.json --ic-csv factor_ic_history.csv

# 合成：用上一步的 IC 历史
merrill-clock-factor synthesize --ic-csv factor_ic_history.csv \
  --method ic_half_life --window 12 --output-json synthesis_report.json
```

`check` 的标的池默认是本地库里的 9 只 ETF 代理，不是 A 股股票池；报告里 `universe_type` 会写明这一点，因子仅由月频复权价构造（动量、反转、已实现波动），不含财务、行业、市值或 Barra 数据。因此这些 RankIC 数值不能与券商 A 股因子 IC 横向比较。

预处理链按公开单因子测试口径实现：中位数 MAD 去极值 → 横截面 z-score → 可选行业/市值中性化 → 缺失值设为中性 0；检验为三法交叉（RankIC + ICIR / 回归 t 序列 / 分层单调性）。分层出现「中间组优于头尾」时会标出非线性，避免 IC 与线性回归把这类因子误判为无效。

## 测试

```bash
python -m pytest -q
```

63 个测试覆盖双轨 CLI、预处理、检验统计、合成权重与真实本地数据库的集成路径。详见 [TESTING.md](TESTING.md)。

## 方法与限制

- 模型是线性 OLS。HAC/Newey–West 调整用于在指定最大滞后下估计对异方差及自相关稳健的协方差，不会修复错误数据、结构突变、遗漏因子或模型选择偏差。
- 截距是样本内历史回归截距，不等于未来 alpha，也不是因果效应；系数是样本条件下的统计关系。R-squared 不是收益质量或策略有效性的证明。
- 输入收益与因子收益应使用相同频率和相同单位（小数），并考虑分红、费用、无风险利率定义及时间戳。因子回归只保留按日期成功对齐的完整观测；完全没有重叠月份时会给出明确诊断，而不是笼统地报"样本太少"。
- 默认演示样例是合成数据；没有捆绑策略业绩、真实因子数据或回归报告。提供的 Fama–French 下载器会在用户运行时请求第三方数据。
- 此项目仅供研究与教学，不构成投资建议、绩效保证或交易系统。

数据来源及授权提示见 [DATA_SOURCES.md](DATA_SOURCES.md)。MIT 许可仅适用于仓库内项目自有代码与文档，不扩展第三方数据权利。

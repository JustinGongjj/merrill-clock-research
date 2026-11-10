# 美林时钟研究台 · Monorepo

把原先分散在三个 GitHub 仓库的代码合并为一个仓库，便于统一维护、统一测试、统一发布。

**仓库地址**：https://github.com/JustinGongjj/merrill-clock-research

> 合并自三个独立仓库（历史通过 `git subtree` 完整保留）：
> `merrill-clock-quant` · `merrill-clock-factor-regression` · `clockdata`
> 原两个 GitHub 仓库仍在维护，但后续改动以本 monorepo 为准。

```
clock-research-monorepo/
├── packages/
│   ├── merrill-clock-quant/            # 相位判定 + 信号 + 回测
│   │   └── quantclock/
│   │       ├── macro.py                # 增长/通胀相位判定
│   │       ├── signals.py              # 资产映射与信号生成
│   │       ├── backtest.py             # 组合回测引擎
│   │       ├── calibration.py          # 相位校准与参数敏感性
│   │       ├── prediction.py           # 前向收益映射
│   │       ├── futures.py / rl.py      # 期货与强化学习扩展
│   │       └── cli.py                  # 命令行入口
│   │
│   ├── merrill-clock-factor-regression/ # 因子检验 + 组合收益归因
│   │   └── factorlab/
│   │       ├── crosssection.py         # IC / RankIC / IR / 分层 / Fama-MacBeth
│   │       ├── synthesis.py            # 因子合成权重（6 种方法）
│   │       ├── panels.py               # 截面面板构建
│   │       ├── regression.py           # FF5 + Newey-West HAC
│   │       └── cli.py
│   │
│   └── clockdata/                      # 本地金融数据库与数据源
│       └── clockdata/
│           ├── store.py                # SQLite 存储层
│           ├── sources.py              # FRED / Yahoo / FF5 / AkShare / WindPy
│           ├── providers.py            # A 股行情、TB3MS 现金 sleeve、ALFRED vintage
│           ├── manifest.py             # 统一研究契约
│           ├── dashboard.py            # 自包含 HTML 仪表盘
│           └── cli.py                  # 一键链路编排
├── scripts/
│   ├── setup.sh                        # 一键安装
│   └── start.sh                        # 一键启动
└── pyproject.toml                      # 根级 pytest / ruff 配置
```

## 三个包的关系

| 包 | 职责 | 命令 |
|---|---|---|
| `merrill-clock-quant` | 宏观相位 → 资产映射 → 信号 → 组合回测 | `merrill-clock-quant` |
| `merrill-clock-factor-regression` | 单因子有效性检验 + 组合收益 FF5 归因 | `merrill-clock-factor` |
| `clockdata` | 本地 SQLite 金融数据库、数据源适配、仪表盘、研究契约 | `clock-data` |

`clockdata` 是编排层：`clock-data run` 会依次调用 `quantclock.cli`（回测）、`factorlab.cli`（横截面检验与归因），最后生成仪表盘。

## 快速开始

```bash
# 1. 安装（含 akshare / yfinance / statsmodels）
bash scripts/setup.sh --with-extra

# 2. 一键跑完整链路（全程离线，用本地数据库）
bash scripts/start.sh

# 3. 跑全部测试（143 passed, 3 skipped）
.venv/bin/python -m pytest
```

装好后全局命令 `clock` 会自动优先使用本 monorepo 环境；本仓库不可用时自动回退到旧环境。

## 数据

本地数据库默认在 `~/.local/share/clockdata/finance.db`，可用 `CLOCK_DATA_DB` 覆盖。

```bash
.venv/bin/clock-data sync --dataset all --continue-on-error
.venv/bin/clock-data status
.venv/bin/clock-data check      # 数据源连通性
```

已接入的数据源：

- **FRED** — 宏观指标 + `TB3MS`（3 个月国债，用作现金 sleeve 收益，修正零收益假设导致的现金收益低估）
- **Yahoo Finance** — ETF/指数行情
- **Fama-French** — 五因子 + 无风险利率
- **AkShare** — A 股真实行情（前复权），支持沪深 300 成分股
- **WindPy** — Windows + Wind 终端环境自动启用；其他环境优雅降级，不影响运行
- **ALFRED vintage** — 点时宏观数据，需 `FRED_API_KEY`，未配置时走最新版 CSV

## 研究契约

每次运行会写出 `run_manifest.json`，记录成本假设、时点口径、股票池、约束条件、测试变体数量与校验问题，避免前向偏差与口径漂移。详见 `packages/clockdata/clockdata/manifest.py`。

## 历史

三个仓库的完整提交历史通过 `git subtree` 保留在 monorepo 中，可用以下命令查看：

```bash
git log --oneline -- packages/merrill-clock-quant
git log --oneline -- packages/merrill-clock-factor-regression
git log --oneline -- packages/clockdata
```

## 口径与局限

- 本地库默认使用 FRED/Yahoo 的**最新修订值**，不是点时数据；需要严格点时研究请配置 `FRED_API_KEY` 启用 ALFRED vintage。
- ETF 代理构建的横截面不是 A 股股票池，其 IC 数值不可与券商 A 股因子 IC 横向比较；使用 `ashare_prices` 数据集时才是真实 A 股横截面。
- IC 与 Fama-MacBeth t 值未做序列相关校正，仅作描述性统计。

## License

MIT

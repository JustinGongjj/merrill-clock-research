# 美林时钟研究台 · Monorepo

把原先分散在三个 GitHub 仓库的代码合并为一个仓库，便于统一维护、统一测试、统一发布。

**仓库地址**：https://github.com/JustinGongjj/merrill-clock-research

> 合并自四个独立仓库（历史通过 `git subtree` 完整保留）：
> `merrill-clock-quant` · `merrill-clock-factor-regression` · `clockdata` · `merrill-clock-lab`
> 原三个 GitHub 仓库仍在维护，但后续改动以本 monorepo 为准。

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
│   ├── clockdata/                      # 本地金融数据库 · 数据中枢 · 后台服务
│   │   └── clockdata/
│   │       ├── store.py                # SQLite 存储层
│   │       ├── registry.py             # 22 源能力路由 + 探活 + 自动降级
│   │       ├── universe.py             # 任意标的解析（A/港/美/加密）
│   │       ├── prices.py               # 多源取价 + 本地兜底 + 部分合并
│   │       ├── server.py               # 后台 REST 服务（12 端点）
│   │       ├── sources.py              # FRED / Yahoo / FF5 / AkShare / WindPy
│   │       ├── providers.py            # A 股行情、TB3MS 现金 sleeve、ALFRED vintage
│   │       ├── manifest.py             # 统一研究契约
│   │       ├── dashboard.py            # 自包含 HTML 仪表盘
│   │       └── cli.py                  # 一键链路编排
│   │
│   ├── starmap/                        # 科幻星图可视化（图谱 / 云图 / 预测）
│   │   └── starmap/
│   │       ├── graph.py                # 力导向星图（节点 / 连线 / 分层布局）
│   │       ├── cloud.py                # 相关性云图 + 相位密度云图
│   │       ├── forecast.py             # 相位路径 / 因子预期 / 价格扇形图
│   │       ├── assemble.py             # 装配 payload（离线可跑）
│   │       ├── viz.py                  # 纯手写 Canvas 渲染，零 CDN
│   │       └── cli.py
│   │
│   └── merrill-clock-lab/              # ⚠️ 历史归档，勿直接安装（见 ARCHIVED.md）
│       └── quantclock/                 #   独有模块已并入 merrill-clock-quant
├── scripts/
│   ├── setup.sh                        # 一键安装
│   └── start.sh                        # 一键启动
└── pyproject.toml                      # 根级 pytest / ruff 配置
```

## 四个包的关系

| 包 | 职责 | 命令 |
|---|---|---|
| `merrill-clock-quant` | 宏观相位 → 资产映射 → 信号 → 组合回测；A股个股五维研究快照 | `merrill-clock-quant` |
| `merrill-clock-factor-regression` | 单因子有效性检验 + 组合收益 FF5 归因 | `merrill-clock-factor` |
| `clockdata` | 本地 SQLite 金融数据库、数据中枢、后台 REST 服务、仪表盘、研究契约 | `clock-data` |
| `starmap` | 科幻星图可视化：力导向图谱 + 密度云图 + 三类未来预测 | `starmap` |

`clockdata` 是编排层：`clock-data run` 会依次调用 `quantclock.cli`（回测）、`factorlab.cli`（横截面检验与归因），最后生成仪表盘。
`starmap` 消费 `clockdata` 的本地库与回测产物，输出自包含 HTML 星图；它也能独立通过后台服务取数。

### A股个股研究快照

`merrill-clock-quant company` 生成单只 A 股的五维研究快照（技术面 / 资金流 / 公开披露 / 量价情绪代理 / 政策关键词观察），各维度独立呈现，不合成荐股分数：

```bash
merrill-clock-quant company --code 300136 --start 2024-01-01 --output ./outputs/company
```

产出 `snapshot.md`（手机友好）、`snapshot.html`（自包含）、`metrics.json`、`source_manifest.json` 及各维度原始 CSV。数据来自东方财富公开接口与本地政策观察表，**不构成投资建议**。

## 快速开始

```bash
# 1. 安装（含 akshare / yfinance / statsmodels）
bash scripts/setup.sh --with-extra

# 2. 一键跑完整链路（全程离线，用本地数据库）
bash scripts/start.sh

# 3. 生成科幻星图（任意标的：A 股 / 港股 / 美股 / 加密）
bash scripts/start.sh starmap 300136
bash scripts/start.sh starmap BTC/USDT

# 4. 启动后台数据服务（REST API）
bash scripts/start.sh serve          # http://127.0.0.1:8765

# 5. 跑全部测试（232 passed, 3 skipped）
.venv/bin/python -m pytest
```

装好后全局命令 `clock` 会自动优先使用本 monorepo 环境；本仓库不可用时自动回退到旧环境。
全局可用命令：`clock` · `clock-data` · `starmap` · `merrill-clock-quant` · `merrill-clock-factor`。

## 数据中枢

`clockdata` 不再绑定单一数据源。每个上游按**能力标签**登记，自动探活、自动降级、显式记录实际服务数据的那一家。

```bash
clock-data registry                # 22 个源：能力 / tier / 延迟 / 实时可用性
clock-data registry --matrix       # 能力 → 源链（按 tier 与延迟排序）
clock-data check                   # 逐源实测连通性（带硬超时，不会挂死）
```

10 个能力标签：`a_share` `us_equity` `hk_equity` `crypto` `etf` `macro` `factors` `fundamentals` `fund_flow` `disclosure`

| 层 | 源 | 说明 |
|---|---|---|
| A 股 | baostock · akshare · efinance · mootdx · adata · tushare¹ | baostock 自有服务器直推最稳；tushare 需 token |
| 全球 | yahoo · alpha_vantage¹ · eodhd¹ · polygon¹ · tiingo¹ · finnhub¹ | 多数免费档额度紧 |
| 加密 | ccxt（实测 9 家交易所可用，Binance 对本机返回 451 地域限制） | |
| 宏观 | fred · alfred¹ · world_bank · dbnomics · fama_french | alfred 为点时数据正确来源 |
| 终端 | wind_mcp（会话内连接器）· windpy² | ² 仅 Windows + 授权终端 |
| 离线 | local_db · local_csv | tier 5 兜底，覆盖取决于已同步的数据集 |

¹ 需 API key，未配置时标为不可用，不做凭据猜测  ² macOS 不可导入，调用显式报错并降级

## 后台数据服务

`clock-data serve` 起一个零依赖（标准库 `ThreadingHTTPServer`）的本地数据服务，作为工作站的后台数据支撑。HTTP 默认 `offline=True`，**绝不阻塞在网络上**。

| 端点 | 说明 |
|---|---|
| `GET /api/health` | 数据库路径 / 大小 / 数据集数 / 观测数 |
| `GET /api/sources` | 22 源完整目录（能力、tier、可用性、说明） |
| `GET /api/sources/probe` | 实时探活（`?timeout=12`） |
| `GET /api/capabilities` | 能力 → 源链路由矩阵 |
| `GET /api/datasets` · `/api/dataset/<id>` | 本地数据集清单 / 明细 |
| `GET /api/instruments` · `/api/instrument/<symbol>` | 标的格式说明 / 单标的解析结果 |
| `GET /api/constellation?symbol=…` | 完整星图 payload（JSON） |
| `GET /api/constellation.html?symbol=…` | 自包含星图页面 |
| `GET /api/factors` | 因子有效性表 |
| `GET /api/forecast?symbol=…` | 三类预测 + 两张云图 |

实测 10 个端点全部 200，响应 < 0.5s。

## 科幻星图

`starmap` 输出自包含 HTML（纯手写 Canvas 2D，零 CDN，零外部依赖），包含：

- **力导向星图** — 相位 / 资产 / 因子 / 标的 / 数据源 / 预测六类节点，按环形分层布局
- **密度云图** — 相关性云图（自写高斯 KDE）与相位密度云图，余弦渐晕去矩形硬边
- **三类预测** — 相位路径（Markov 转移外推 + 熵）、因子预期（IC 衰减 + √h 扩张区间）、价格扇形图（自助抽样）
- **右侧四面板** — 相位概率、资产目标权重、标的快照、因子有效性表
- **数据源健康表** — 22 源逐条状态与说明

**任意标的可复用**：标的不写死，6 位数字自动判为 A 股（SH/SZ/BJ 由前缀推断）、4-5 位判为港股、含 `/` 判为加密对、字母判为美股/ETF。

```bash
starmap build 300136      --out ./outputs/starmap    # A 股
starmap build 00700       --out ./outputs/starmap    # 港股
starmap build SPY         --out ./outputs/starmap    # 美股 ETF
starmap build BTC/USDT    --out ./outputs/starmap    # 加密
starmap matrix            # 能力 → 源链
starmap probe             # 逐源实测
```

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
- `peer_universe()` 对 A 股使用**当前**沪深 300 快照，存在幸存者偏差，不适合直接用于历史回测的股票池构建。
- 星图的三类预测是基于历史统计的外推（Markov 转移 / IC 衰减 / 自助抽样），不是价格预测模型，**不构成投资建议**。
- 加密数据源各交易所速率与地域限制不同；本机实测 Binance 返回 HTTP 451，路由层会自动改用可用交易所。
- `ccxt` 与 `curl-cffi` 对 `cffi` 的版本要求冲突（`==2.0.0` vs `>=2.1.1`）。本仓库保留 `cffi>=2.1.1`，实测 AkShare 与 CCXT 均可正常导入。

## License

MIT

> **📦 本仓库已并入 monorepo**
> 代码已迁移至 **https://github.com/JustinGongjj/merrill-clock-research** 的
> `packages/merrill-clock-lab/`（`git subtree` 保留完整历史）。
> 本仓库独有的 A 股个股研究模块（`company_data` / `company_analysis` / `company_report`）
> 已迁入活跃包 `packages/merrill-clock-quant/`，并新增 `merrill-clock-quant company` 子命令。
> 后续改动以 monorepo 为准；本仓库仅作归档保留。

# Merrill Clock Lab

一个用于公开数据研究的 Python 原型：把美林时钟式宏观状态分类、跨资产回测、因子回归、预测基线和离线强化学习实验放在同一套可检查代码中；另含 A 股个股研究快照模块。它是研究与教学工具，不是投资建议或交易系统。

## 包含什么

- **跨资产回测**：以美国工业生产和 CPI 动能构造四象限软分类，将状态映射到 ETF 研究组合，按下一期执行，并按换手扣除设定成本。
- **个股研究快照**：分别呈现技术指标、数据商资金流估算、公开披露标题、量价行为代理和用户维护的政策关键词观察表；另可展示美国宏观背景和单因子市场回归。所有维度保持分开，不生成综合荐股分数。
- **量化实验**：包含 HAC/Newey–West 因子回归、扩展窗口 Ridge 预测基线、固定动作集合的离线 Q-learning 演示和逐合约期货换月处理。
- **可审计实现**：核心逻辑和测试均在仓库中；没有捆绑训练好的机器学习模型、模型权重或市场数据快照。

## 安装

需要 Python 3.10 或更新版本。建议在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,test]'
```

Windows PowerShell 激活命令为 `.venv\\Scripts\\Activate.ps1`。

## 快速运行

运行原跨资产研究流程（需要访问 FRED 与 Yahoo Finance 的公开数据接口）：

```bash
quantclock run --start 2000-01-01 --output ./outputs/us_clock
python examples_factor_regression.py ./outputs/us_clock/clock_backtest.csv
```

生成 A 股个股快照（运行时会请求公开行情/披露接口；如不需要美国宏观背景，可跳过 FRED）：

```bash
quantclock company --code 300136 --start 2023-01-01 --skip-us-clock --output ./outputs/300136
```

资金流数据若从本地导入，可使用 `--flow-csv PATH` 参数。仓库**不附带**真实资金流、行情或公告样本；导入前应自行确认来源、采集时间及使用和再分发权限。政策观察表 `data/policy_watchlist.csv` 仅含字段表头，默认不会命中任何政策。

生成的报告、行情 CSV、指标 JSON 和 HTML 页面属于运行产物，默认写入 `outputs/`，不应在未审查权利与隐私前提交。

## 测试

```bash
python -m pytest -q
```

测试说明及覆盖范围见 [TESTING.md](TESTING.md)。全部测试使用内存或临时目录中的确定性合成数据；无需联网，也不会读取用户的真实行情文件。

## 方法边界与数据来源

- 四象限概率是透明的软分类，不是校准后的衰退概率、收益预测或交易信号。
- FRED 宏观序列使用下载时可见的最新修订值；配置的发布滞后只是近似，不能代替 ALFRED 点时版本。
- ETF 是资产类别代理，不等于国债现金流、现货商品或真实期货持仓；成本扣减也不是实际滑点模拟。
- A 股资金流来自数据供应商的订单规模分类估算，不证明交易者身份；公开网页可访问不代表可批量复制或再分发。
- 公告标题规则分类不替代公告原文；政策关键词命中不表示政策现行有效、公司受益或产生财务影响。
- 个股当日未收盘 K 线会单列为盘中参考，不进入日频收益、指标或回归；市场回归描述历史样本，不是未来 alpha。
- 预测和强化学习模块是教学性质的基线/实验接口，不含训练好的模型，也没有样本外盈利能力证明。

完整来源、口径及风险见 [DATA_SOURCES.md](DATA_SOURCES.md)、[PRODUCT_GUIDE.md](PRODUCT_GUIDE.md) 和 [research_report.md](research_report.md)。

## 许可状态

本项目采用 MIT License，见 [LICENSE](LICENSE)。许可证仅覆盖项目自有代码与文档；第三方数据、商标、政策原文或其他素材仍受各自条款约束，项目许可证不能替代其授权。公开前请确认你有权以该许可发布仓库中保留的项目内容。

## 免责声明

仅供研究、教育和软件测试。历史数据和回测结果不代表未来表现。本软件不提供证券建议，不连接券商、不下单，也不保证数据完整、准确或持续可用。使用者需独立核验数据来源、模型假设和适用法规。

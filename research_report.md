# 美林时钟跨资产研究与代码框架

美林时钟适合做“宏观状态分类层”，不适合直接当作买卖信号。真正决定结果的，首先是增长与通胀动能如何定义、历史上当时能看到什么数据，以及资产回报是否按可交易口径计算。下面的代码包把这条研究链路跑通，但目前仍是美国公开数据代理原型，不是经样本外验证的投资策略。

## 时钟四象限：状态名称不等于衰退判定

经典框架以增长和通胀的变化方向划分四个阶段：复苏（增长动能改善、通胀动能走弱）、过热（两者都走强）、滞胀（增长走弱、通胀走强）和再通胀 Reflation（两者都走弱）。常见历史叙述把股票、商品、现金、债券分别与复苏、过热、滞胀、再通胀联系起来；但这只是周期性假设，不是跨时期成立的收益定律。英国资产管理机构对该框架的介绍也明确将时钟视作多资产模型中的一个因素，而非单独决策器。[1] [2]

“Reflation”不能直接翻译成已经发生通缩，也不等于美国国家经济研究局（NBER）事后认定的衰退。代码因此使用 `reflation` 作为增长与通胀动能都减弱的象限名。它不输出“经济已衰退”的结论。

还要区分水平与动量：GDP增长率仍为正，不表示增长正在加速；CPI仍处高位，也不表示通胀继续加速。示例模型对工业生产和CPI的变化做滚动标准化，再生成四象限概率。该概率是可解释的软分类，不是经过校准的衰退概率或收益预测。

## 公开数据能做什么，不能做什么

宏观数据最大的陷阱是“用今天修订后的历史，假装当时已经知道”。BEA会用后续GDP估算替代先前发布值；BLS每年重算CPI季调因子，也可能改写此前数年的季调数据。ALFRED可让研究者查很多系列的历史版本，但不是每个系列都有完整vintage记录，且历史可得日期不一定等于精确的发布时刻。[3] [4] [5]

本包从FRED公开图表CSV读取工业生产和CPI，并在月份结束后加一个可配置的发布日期延迟。这个处理能避免把整个月数据当作月初就已知，却不能还原统计机构的真实发布日期，也不能去掉最终修订值造成的泄漏。`macro.py` 可以导入 `date,release_date,value` 格式的逐次发布数据；目前没有自动下载ALFRED全量vintage。要做严谨实时回测，应先补齐这一层。

股票代理使用SPY等ETF的调整收盘价。FRED上的标普500日指数SP500是价格指数、不含股息，而且页面说明其日频历史仅保留最近十年；它不能代替长期总回报指数。[6] 股票收益需区分价格、总回报和扣税后的净总回报，也要记录公司行动及指数成分变更。

债券收益率不是债券回报。美国财政部恒定期限曲线是由近期国债报价拟合出的平价曲线，不是一组逐期限真实成交券；FRED的DGS10适合观察利率，却不能直接当作十年期国债组合的持有期收益。[7] [8] 示例里的IEF/TLT是ETF代理；正式债券回测应使用总回报指数或逐券现金流、票息、久期、滚降和再投资数据。

黄金需区分现货定盘、ETF和期货。黄金与美元、实际利率、流动性和避险需求的关系会变化，不能视为稳定的通胀对冲。GLD是扣除费用并可能有溢折价的ETF代理。FRED已于2022年删除IBA/LBMA日金价数据；世界黄金协会提供的月均/年均价格更适合低频研究，使用LBMA历史定盘则需核对许可。[9] [10] [11] [12]

商品数据的公开程度不等于期货价格可直接用于回测。EIA提供石油库存、供需和部分价格序列，但WTI现货不是期货；其单一近月合约序列还存在2024年4月后的数据缺口。[13] [14] USDA的WASDE、NASS适合农业供需与报告事件研究；CFTC COT是每周持仓快照，不是价格，并且通常周五才发布周二持仓。[15] [16] 世界银行和FRED/IMF的月度商品基准价适合低频宏观面板，但不含期货换月收益。[17]

要做期货盈亏，应有逐合约结算价、换月规则、合约乘数、到期/第一通知日、保证金、成本和抵押品收益。交易所历史行情的数据权限和费用需逐项确认，不能假设“网页可见”就表示可免费批量下载或再发布。代码中的期货模块只接受用户提供的逐合约价格和到期日，演示如何避免把新旧合约价差误算成当天盈亏；它还不模拟完整的保证金、成交和交割流程。

市场“时间”也是数据口径的一部分。NYSE股票核心时段为美东09:30–16:00，节假日和提前收市会变化；标普期货则几乎全天交易并有维护间隔。美东夏令时、亚洲时区、现货收盘和期货交易日归属若没有明确对齐，同一天的价格可能并不是同一信息集。[18] [19]

## 回测、因子回归与AI模型

回测必须严格按时间因果顺序执行：在信号所用数据已经发布之后，才允许产生决策；再用下一期收益评估。本包目标权重默认滞后一周期执行，并按换手扣除用户配置的成本。单次成本假设仍不是实际滑点模型。模拟业绩也不等于真实交易；CFTC规则要求模拟结果披露其非真实交易、流动性及事后设计等局限。[20]

因子回归使用月度策略超额收益对公开的Fama–French五因子做OLS，并提供Newey–West/HAC标准误。French数据本身有明确的组合构造定义，历史文件也可能因数据库修订或方法变更而更新。[21] 回归截距只描述历史残差，不能直接叫作未来alpha。连续尝试很多因子、窗口和资产后只挑最好的结果，会严重放大假发现；多重检验研究建议对新因子采用比传统门槛更严格的证据，并记录所有尝试。[22]

机器学习先从扩展窗口Ridge预测基线开始：每个时点只用之前样本估计参数和标准化，再预测下一期。宏观研究的有效独立样本很少，尤其衰退阶段；概率还应检查校准与误报/漏报，而非只报告准确率。强化学习模块使用固定的投资动作集合，在训练期内做离线Q-learning，策略冻结后才在测试期评估。离线RL会遇到数据未覆盖动作的分布偏移和价值外推错误；现有金融RL研究的费用假设也不统一，因此这里把RL限定为模拟研究，不预设它胜过简单基线。[23] [24]

推荐验证顺序是：先比较静态等权、买入持有和规则型美林时钟；其次做点时数据与成本敏感性；再做walk-forward预测和因子归因；最后才比较复杂机器学习或受约束RL。每次都要留出从未参与调参的测试区间，并说明基准、样本量、回撤、换手与成本。反复查看测试集后再改规则，测试集就不再是测试集。[25]

## 代码包的实现边界

核心流程是：下载FRED宏观代理和ETF调整价格 → 计算月度增长/通胀动能 → 生成四象限概率 → 按透明权重模板配置资产 → 将决策滞后一月执行 → 扣除单边换手成本 → 输出回测序列和指标。

默认示例资产为SPY权益、IEF/TLT国债ETF代理、GLD黄金、DBC大宗商品ETF、USO能源、CPER工业金属、DBA农业和BIL短期国债现金代理。可用 `asset_map` 与象限权重模板替换。当前股票和商品工具主要是美国上市ETF，不能代表中国市场、现货商品、现金债券或真实期货持仓。目标市场、币种和决策频率未指定，所以这里只把美国月频作为可运行的起点。

安装和运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all,test]'
pytest -q
quantclock run --start 2000-01-01 --output ./outputs
python examples_factor_regression.py ./outputs/clock_backtest.csv
```

自动测试覆盖宏观发布日期滞后、回测一周期延迟、预测滚动切分、HAC因子回归、期货换月、权重归一、RL冻结策略接口以及个股分析与报告输出。测试使用合成输入，验证的是代码路径与约定，不是策略盈利能力；默认测试不访问外部数据接口。

下一步要把研究收敛成具体可用版本，至少还要确定目标市场（例如中国、美国或全球）、本币/美元口径、主要标的与预测期限，并补齐对应的点时宏观版本、资产总回报和期货合约历史。没有这些选择，所谓一套通用参数很容易把不同市场、交易时间和回报定义混在一起。

## 参考资料

[1]: https://www.rlam.com/uk/intermediaries/our-capabilities/multi-asset/investment-clock/ "Royal London Asset Management: Investment Clock"
[2]: https://ks3-cn-beijing.ksyun.com/attachment/114caaf9991619a15e81a0a4b590837f "Merrill Lynch Global Research: The Investment Clock, 2004 (public copy)"
[3]: https://alfred.stlouisfed.org/help "ALFRED Help: Vintage Data"
[4]: https://www.bea.gov/data/gdp/gross-domestic-product "BEA: GDP Data and Vintage History"
[5]: https://www.bls.gov/cpi/seasonal-adjustment/ "BLS: CPI Seasonal Adjustment"
[6]: https://fred.stlouisfed.org/series/SP500 "FRED: S&P 500 Price Index"
[7]: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology "U.S. Treasury: Treasury Yield Curve Methodology"
[8]: https://fred.stlouisfed.org/series/dgs10 "FRED: 10-Year Treasury Constant Maturity Rate"
[9]: https://www.gold.org/goldhub/gold-focus/2022/05/you-asked-we-answered-explaining-golds-recent-performance "World Gold Council: Explaining Gold's Recent Performance"
[10]: https://fred.stlouisfed.org/news/2022/01/ice-benchmark-administration-ltd-iba-data-to-be-removed-from-fred "FRED Notice: IBA Data Removed from FRED"
[11]: https://www.gold.org/goldhub/data/gold-prices "World Gold Council: Gold Prices"
[12]: https://www.lbma.org.uk/prices-and-data/lbma-precious-metal-prices "LBMA Precious Metal Prices and Licensing"
[13]: https://www.eia.gov/opendata/ "EIA Open Data"
[14]: https://www.eia.gov/dnav/pet/hist/rclc1d.htm "EIA WTI Contract 1 Historical Data"
[15]: https://www.usda.gov/oce/commodity/wasde "USDA: WASDE Reports and Data"
[16]: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm "CFTC: Commitments of Traders"
[17]: https://www.worldbank.org/en/research/commodity-markets "World Bank Commodity Markets and Pink Sheet"
[18]: https://www.nyse.com/markets/hours-calendars "NYSE Trading Hours and Calendars"
[19]: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html "CME E-mini S&P 500 Contract Specifications"
[20]: https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/e7-3122.html "CFTC Rule 4.41: Simulated or Hypothetical Performance"
[21]: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html "Kenneth French Data Library"
[22]: https://academic.oup.com/rfs/article/29/1/5/1843824 "Harvey, Liu and Zhu: ... and the Cross-Section of Expected Returns"
[23]: https://www.kansascityfed.org/research/economic-review/4q18-smalterhall-machine-learning-approaches/ "Kansas City Fed: Machine Learning Approaches to Macroeconomic Forecasting"
[24]: https://neurips.cc/virtual/2023/poster/72924 "NeurIPS: Offline Reinforcement Learning and Distribution Shift"
[25]: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf "Bailey et al.: The Probability of Backtest Overfitting"

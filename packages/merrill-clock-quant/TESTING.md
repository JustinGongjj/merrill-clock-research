# 测试说明

运行：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

43 个测试仅使用内存中的确定性合成数据与临时输出目录，不访问外网。原有覆盖：信号概率与权重映射、信号日/收益日错位、换手成本与绩效统计（含从初始净值起算的最大回撤、未完结月份剔除、缺失收益告警）、宏观观测发布滞后对齐与「观测月网格差分 + 任意决策频率对齐」、扩展窗口 Ridge 与预测指标在缺失配对下的计数、冻结的 Q-learning 策略样例、逐合约期货换月不把跨月价差当作收益、离线合成全链路演示，以及 live/local/auto 三种数据源切换与上游空列的降级行为。

新增覆盖：`run_research` 端到端产物（`metrics.json` / `clock_backtest.csv` / `phase_diagnostics.json` / `run_manifest.json`）、`--no-manifest` 开关、参数确认门对默认值与显式值的区分、`temperature` 改变权重但保持样本区间一致；相位校准的 Brier 分数、四相映射、完美概率与均匀概率的熵、常数基线、主导相位持续期、缺列与样本过少的报错、温度敏感性的单调性。在线数据下载和第三方条款不属于这些测试范围。

# ⚠️ 已归档 · 不要直接安装

本目录是 `merrill-clock-lab` 仓库的历史快照，通过 `git subtree` 并入 monorepo，
**仅用于保留完整提交历史与代码溯源，不是可安装的活跃包**。

## 为什么不直接安装

它与 `packages/merrill-clock-quant` 提供**同名的 `quantclock` 包和 `quantclock`
命令**（lab 是上游旧版，quant 是其后继演进版）。同时安装两者会：

- 后装的覆盖先装的，行为随安装顺序漂移
- `pip install -e packages/merrill-clock-lab` 会把 `quantclock` 指向旧版，
  导致 A 股个股研究模块（`company_*`）和 TB3MS 现金 sleeve 修复全部失效

## 独有资产已并入

lab 相对 quant 独有的模块已迁入 `packages/merrill-clock-quant/`：

| lab 原文件 | 当前位置 | 说明 |
|---|---|---|
| `quantclock/company_data.py` | `packages/merrill-clock-quant/quantclock/company_data.py` | 东方财富公开接口适配 |
| `quantclock/company_analysis.py` | `packages/merrill-clock-quant/quantclock/company_analysis.py` | 五维研究快照计算 |
| `quantclock/company_report.py` | `packages/merrill-clock-quant/quantclock/company_report.py` | Markdown / HTML 报告生成 |
| `quantclock/factors.py` | `packages/merrill-clock-quant/quantclock/factors.py` | 改为委派 `factorlab.regression` 的 shim |
| `data/policy_watchlist.csv` | `packages/merrill-clock-quant/data/policy_watchlist.csv` | 政策关键词观察表 |
| `tests/test_company_analysis.py` | `packages/merrill-clock-quant/tests/test_company_analysis.py` | 13 个测试 |

lab 中 `factor_data.py` / `factors.py` 的功能已被 `factorlab` 完整覆盖
（`factorlab.data` / `factorlab.regression`，后者额外支持 HAC 小样本自由度校正、
RF 双重来源校验、样本对齐诊断），故不重复引入。

## 使用方式

用 `packages/merrill-clock-quant` 的 CLI，它已包含 lab 的全部能力：

```bash
merrill-clock-quant company --code 300136 --output ./outputs/company
```

`setup.sh` 不会安装本目录，仅安装 `merrill-clock-quant` /
`merrill-clock-factor-regression` / `clockdata` 三个活跃包。

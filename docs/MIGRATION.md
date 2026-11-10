# 从三仓库到 Monorepo 的迁移说明

## 为什么合并

原先三个仓库各自独立发布：

| 原仓库 | GitHub | 主要内容 |
|---|---|---|
| `merrill-clock-quant` | `JustinGongjj/merrill-clock-quant` | 相位判定 / 信号 / 回测 |
| `merrill-clock-factor-regression` | `JustinGongjj/merrill-clock-factor-regression` | 因子检验 / FF5 归因 |
| `clockdata` | 未发布（仅本地） | 本地金融数据库 / 仪表盘 |

问题：

1. `clockdata` 的 `run` 链路**依赖另外两个包可导入**，但三者在 GitHub 上互不引用，装一个跑不通全链路。
2. 跨包改动（比如新增数据源 → 新增因子检验 → 仪表盘展示）需要三个仓库分别提交、分别测，测试套件无法一次跑完。
3. 版本漂移：`clockdata` 未发布，但它是 `clock-data` 命令的实际载体。

## 合并后的结构

```
packages/
  merrill-clock-quant/            → 包名 quantclock，命令 merrill-clock-quant
  merrill-clock-factor-regression/ → 包名 factorlab，命令 merrill-clock-factor
  clockdata/                      → 包名 clockdata，命令 clock-data
```

三个包仍各自带 `pyproject.toml`，保持独立可发布（可按需 `pip install packages/<name>`），但仓库层面统一。

## 历史如何保留

用 `git subtree`（非 squash）并入，三方完整提交历史保留：

```bash
git log --oneline -- packages/merrill-clock-quant              # 2 commits
git log --oneline -- packages/merrill-clock-factor-regression   # 2 commits
git log --oneline -- packages/clockdata                        # 1 commit
```

合并提交形如：

```
*   Add 'packages/merrill-clock-factor-regression/' from commit '67f5bc9...'
|\
| * 67f5bc9 量化项目优化：新增横截面因子检验轨（IC/RankIC/IR/分层/FMB）...
| * 3c02e2e Initial commit
```

后续同步上游改动：

```bash
git fetch quant main
git subtree pull --prefix=packages/merrill-clock-quant quant main
```

## 安装与运行

```bash
bash scripts/setup.sh --with-extra   # 建 .venv，可编辑安装三个包
bash scripts/start.sh                 # 一键跑完整链路
.venv/bin/python -m pytest            # 一次跑完三包测试
```

## 旧目录怎么处理

原 `github-review/` 目录**保持原样未动**，仍可正常使用：

- 旧 `clock` 全局命令仍指向 `github-review/runtime/.venv`
- 旧三个仓库目录未被修改（除 clockdata 新增了 `.gitignore` 与 git 身份）

切换方式：直接用新路径 `bash scripts/start.sh`，或把全局 `clock` 的 `BASE` 改指向 monorepo。

## 兼容说明

- `clockdata/cli.py` 通过 `sys.executable -m quantclock.cli` 子进程调用兄弟包，通过 `factorlab.__file__` 定位 example 脚本——都基于**安装路径**而非相对路径，所以只要三个包装进同一环境即可工作，与仓库布局无关。
- `packages/*/pyproject.toml` 未修改，各包独立发布能力不变。
- 根 `pyproject.toml` 只含 pytest / ruff 配置，不含 `[project]`，不会被误当成可安装包。

#!/usr/bin/env bash
# 一键安装美林时钟研究台（monorepo 版）
# 用法：
#   scripts/setup.sh              # 创建 .venv 并可编辑安装三个包
#   scripts/setup.sh --with-extra # 额外安装 akshare / yfinance / statsmodels
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${CLOCK_VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"

WITH_EXTRA=0
for arg in "$@"; do
  case "$arg" in
    --with-extra) WITH_EXTRA=1 ;;
    *) echo "未知参数：$arg"; exit 1 ;;
  esac
done

echo "→ 创建虚拟环境：$VENV"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
"$PY" -m pip install --quiet --upgrade pip

echo "→ 可编辑安装三个包"
"$PY" -m pip install --quiet -e "$ROOT/packages/merrill-clock-quant"
"$PY" -m pip install --quiet -e "$ROOT/packages/merrill-clock-factor-regression"
"$PY" -m pip install --quiet -e "$ROOT/packages/clockdata"

if [ "$WITH_EXTRA" = "1" ]; then
  echo "→ 安装扩展依赖（akshare / yfinance / statsmodels）"
  "$PY" -m pip install --quiet akshare yfinance statsmodels
fi

echo "→ 安装测试依赖"
"$PY" -m pip install --quiet pytest

echo "→ 安装自检（导入关键模块，避免半安装状态静默通过）"
if ! "$PY" - <<'PYEOF'
import importlib, sys

checks = [
    ("numpy", None),
    ("pandas", None),
    ("quantclock", None),
    ("factorlab", None),
    ("clockdata", None),
]
failed = []
for name, _ in checks:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {type(exc).__name__}: {exc}")

# Optional heavy deps: report but do not fail the install.
for name in ("statsmodels.api", "yfinance", "akshare"):
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! 可选依赖 {name} 不可用: {type(exc).__name__}: {exc}")

if failed:
    print("以下必需模块导入失败：")
    for line in failed:
        print(f"  × {line}")
    sys.exit(1)
print("  ✓ 必需模块全部可导入")
PYEOF
then
  echo "× 安装自检未通过，请重跑或手动修复"
  exit 1
fi

echo ""
echo "✓ 安装完成。已安装命令："
ls "$VENV/bin" | grep -E "^(clock-data|merrill-clock-quant|merrill-clock-factor)$" || true
echo ""
echo "  启动研究台： $VENV/bin/clock-data run --output ./outputs/local-run"
echo "  跑全部测试： cd $ROOT && $PY -m pytest"

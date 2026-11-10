#!/usr/bin/env bash
# 美林时钟研究台 · monorepo 版一键启动器
# 与全局 `clock` 命令行为一致，但使用 monorepo 自己的 .venv
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CD="$ROOT/.venv/bin/clock-data"
DB="${CLOCK_DATA_DB:-$HOME/.local/share/clockdata/finance.db}"
OUT="$ROOT/outputs/local-run"
DASH="$OUT/dashboard.html"
export CLOCK_DATA_DB="$DB"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }

check_install() {
  if [ ! -x "$PY" ] || [ ! -x "$CD" ]; then
    echo "× monorepo 环境不完整，找不到 $PY 或 $CD"
    echo "  请先执行： bash $ROOT/scripts/setup.sh --with-extra"
    return 1
  fi
  return 0
}

do_start() {
  check_install || return 1
  echo ""
  bold "════════════════════════════════════════════════════════════"
  bold "  美林时钟研究台 · monorepo (Merrill Clock Research Stack)"
  bold "════════════════════════════════════════════════════════════"
  echo "  仓库根: $ROOT"
  echo "  数据库 : $DB"
  echo "  结果   : $OUT"
  echo ""
  echo "→ 正在从本地金融数据库运行回测 + FF5 因子归因 + 横截面检验（全程离线）..."
  echo ""
  if ! "$CD" --db "$DB" run --output "$OUT"; then
    echo ""
    echo "× 运行失败。可先检查数据库：clock-data status"
    return 1
  fi
  echo ""
  if [ -f "$DASH" ]; then
    echo "→ 正在打开结果仪表盘：$DASH"
    open "$DASH" 2>/dev/null || echo "  （自动打开失败，可手动双击该文件）"
  fi
  echo ""
  bold "✓ 完成。结果文件："
  echo "    $OUT/metrics.json              绩效指标"
  echo "    $OUT/clock_backtest.csv        逐月回测明细"
  echo "    $OUT/factor_attribution.json   FF5 因子归因"
  echo "    $OUT/cross_section_report.json 横截面因子检验"
  echo "    $DASH   可视化仪表盘"
  return 0
}

do_report() {
  check_install || return 1
  [ -f "$OUT/metrics.json" ] || { echo "× 还没有结果，先运行：scripts/start.sh"; return 1; }
  "$CD" --db "$DB" dashboard --output "$OUT" || { echo "× 仪表盘生成失败"; return 1; }
  echo "→ 正在打开：$DASH"
  open "$DASH" 2>/dev/null || echo "  （自动打开失败，可手动双击该文件）"
}

do_run()    { check_install || return 1; "$CD" --db "$DB" run --output "$OUT" --no-dashboard; }
do_update() { check_install || return 1; "$CD" --db "$DB" sync --dataset all --continue-on-error; }
do_status() { check_install || return 1; "$CD" --db "$DB" status; }
do_check()  { check_install || return 1; "$CD" --db "$DB" check; }
do_test()   {
  check_install || return 1
  cd "$ROOT" && "$PY" -m pytest
}

do_open() {
  [ -d "$OUT" ] || { echo "× 结果目录还不存在，先运行：scripts/start.sh"; return 1; }
  open "$OUT"
}

show_help() {
  cat <<'EOF'

美林时钟研究台 (monorepo) · 命令速查
────────────────────────────────────────────────────────────
  start            一键启动：离线回测 + 因子归因 + 横截面检验 + 打开仪表盘
  run              只跑链路并打印结果，不生成/打开仪表盘
  report           不重算，用已有结果重新生成并打开仪表盘
  update           联网更新本地金融数据库（全部数据源）
  status           查看数据覆盖区间、行数、同步日志
  check            探测各数据源连通性
  test             跑三个包的完整测试套件
  open             在访达里打开结果文件夹

  换数据库：export CLOCK_DATA_DB=/path/to/finance.db
────────────────────────────────────────────────────────────
EOF
}

case "${1:-start}" in
  ""|start)        do_start ;;
  run)             do_run ;;
  report|view)     do_report ;;
  update|sync)     do_update ;;
  status)          do_status ;;
  check)           do_check ;;
  test)            do_test ;;
  open)            do_open ;;
  help|--help|-h)  show_help ;;
  *) echo "未知命令：$1"; show_help; exit 1 ;;
esac

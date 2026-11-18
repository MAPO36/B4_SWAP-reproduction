#!/usr/bin/env bash
# 训练监控：进度 / 速率 / ETA / reward / GPU / 显存 / ckpt / nan
# 用法：bash watch_train.sh [日志路径] [--once]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONCE=""
if [ "${1:-}" = "--once" ]; then ONCE="--once"; shift; fi
LOG="${1:-$(ls -t "$ROOT"/workSpace/runs/*/*/train.log 2>/dev/null | head -1)}"
[ -n "${LOG:-}" ] && [ -f "$LOG" ] || { echo "找不到训练日志"; exit 1; }

fmt_dur() { local s=$1; if [ "$s" -ge 3600 ]; then printf "%dh%02dm" $((s/3600)) $((s%3600/60)); else printf "%dm%02ds" $((s/60)) $((s%60)); fi; }

snapshot() {
  local it cur total rate eta elapsed pid rew steps gpu ckpt nan age
  it=$(grep -aoE "Learning iteration [0-9]+/[0-9]+" "$LOG" | tail -1)
  cur=$(echo "$it" | grep -oE "[0-9]+/" | tr -d '/'); total=$(echo "$it" | grep -oE "/[0-9]+" | tr -d '/')
  rew=$(grep -aoE "Mean reward: [-0-9.]+" "$LOG" | tail -1 | awk '{print $3}')
  steps=$(grep -aoE "Total timesteps: [0-9]+" "$LOG" | tail -1 | awk '{print $3}')
  nan=$(grep -ac "nan" "$LOG" 2>/dev/null); nan=${nan:-0}
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)
  ckpt=$(ls -t "$ROOT"/WMP/logs/*/*/model_*.pt 2>/dev/null | head -1)
  pid=$(pgrep -f "python.*(train_capped|legged_gym/scripts/train)" | head -1)

  echo "日志   : ${LOG#$ROOT/}"
  echo "状态   : $([ -n "$pid" ] && echo "运行中 (pid $pid)" || echo "已结束")"
  echo "进度   : ${it:-<尚未开始>}"
  if [ -n "$pid" ] && [ -n "${cur:-}" ] && [ "$cur" -gt 0 ] 2>/dev/null; then
    elapsed=$(ps -o etimes= -p "$pid" | tr -d ' ')
    if [ "${elapsed:-0}" -gt 30 ] 2>/dev/null; then
      rate=$(awk -v c="$cur" -v e="$elapsed" 'BEGIN{printf "%.2f", c*60/e}')
      eta=$(awk -v c="$cur" -v t="$total" -v e="$elapsed" 'BEGIN{printf "%d", (t-c)*e/c}')
      echo "速率   : ${rate} iter/min   已跑 $(fmt_dur "$elapsed")   预计还需 $(fmt_dur "$eta")"
    fi
  fi
  echo "reward : ${rew:-<无>}   (nan 行数: $nan)"
  echo "steps  : ${steps:-<无>}"
  echo "GPU    : $gpu"
  echo "最新ckpt: ${ckpt:+${ckpt#$ROOT/}}"
  echo "日志更新: $(( $(date +%s) - $(stat -c %Y "$LOG") )) 秒前"
}

if [ "$ONCE" = "--once" ]; then snapshot; exit 0; fi
while true; do clear; echo "===== SWAP 训练监控（Ctrl-C 退出）====="; snapshot; sleep 3; done

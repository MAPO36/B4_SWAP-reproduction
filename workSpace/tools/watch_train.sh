#!/usr/bin/env bash
# 实时监控一次训练：进度 / reward / 步数 / GPU / 显存 / 最新 ckpt / nan
# 用法：bash workSpace/tools/watch_train.sh [日志路径] [--once]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONCE=""
if [ "${1:-}" = "--once" ]; then ONCE="--once"; shift; fi
LOG="${1:-$(ls -t "$ROOT"/workSpace/runs/*/*/train.log 2>/dev/null | head -1)}"

if [ -z "${LOG:-}" ] || [ ! -f "$LOG" ]; then
  echo "找不到训练日志。用法: bash workSpace/tools/watch_train.sh <train.log>" >&2
  exit 1
fi

snapshot() {
  local it rew steps gpu ckpt nan n
  it=$(grep -aoE "Learning iteration [0-9]+/[0-9]+" "$LOG" | tail -1)
  rew=$(grep -aoE "Mean reward: [-0-9.]+" "$LOG" | tail -1)
  steps=$(grep -aoE "Total timesteps: [0-9]+" "$LOG" | tail -1)
  nan=$(grep -ac "nan" "$LOG" 2>/dev/null); nan=${nan:-0}
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)
  ckpt=$(ls -t "$ROOT"/WMP/logs/*/*/model_*.pt 2>/dev/null | head -1)
  n=$(pgrep -fc "train_capped.py|legged_gym/scripts/train.py" 2>/dev/null); n=${n:-0}
  echo "时间   : $(date '+%F %T')"
  echo "日志   : ${LOG#$ROOT/}"
  echo "状态   : $([ "$n" -gt 0 ] && echo "运行中 (pid $n 个)" || echo "已结束")"
  echo "进度   : ${it:-<尚无>}"
  echo "步数   : ${steps:-<尚无>}"
  echo "reward : ${rew:-<尚无>}   (nan 行数: $nan)"
  echo "GPU    : $gpu"
  echo "最新ckpt: ${ckpt:+${ckpt#$ROOT/}}"
}

if [ "$ONCE" = "--once" ]; then
  snapshot
  exit 0
fi

while true; do
  clear
  echo "=========== SWAP 训练监控（Ctrl-C 退出）==========="
  snapshot
  sleep 3
done

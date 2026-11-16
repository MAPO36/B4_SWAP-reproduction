#!/bin/bash
# Launch the armed GPU watcher detached and report its PID + first log lines.
set -uo pipefail
RUNROOT=/home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR
mkdir -p "$RUNROOT/logs"
# avoid double-arming
if [ -f "$RUNROOT/logs/watcher.pid" ] && kill -0 "$(cat "$RUNROOT/logs/watcher.pid")" 2>/dev/null; then
  echo "watcher already running pid=$(cat "$RUNROOT/logs/watcher.pid")"
  exit 0
fi
nohup bash /tmp/wait_gpu_then_train.sh > /dev/null 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNROOT/logs/watcher.pid"
sleep 3
echo "WATCHER_PID=$PID"
ps -o pid,etime,args -p "$PID" 2>/dev/null || true
echo "--- watcher.log ---"
cat "$RUNROOT/logs/watcher.log" 2>/dev/null | tail -10
echo "--- GPU now ---"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "--- GPU processes ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

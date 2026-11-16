#!/bin/bash
# Armed GPU watcher for R0-LHR training (SWAP-lite stage 3, 2026-11-16).
# The lab 4090 is shared with another student's project (own WMP copy, ~19-20 GB GPU).
# This watcher waits until the GPU is genuinely free for 3 consecutive 60 s checks,
# then launches the full training chain (integration -> 1k smoke -> 10k R0-LHR).
# It never kills or touches other users' processes. Safe to run detached.
#
# Usage: nohup bash /tmp/wait_gpu_then_train.sh > /dev/null 2>&1 &
set -uo pipefail
RUNROOT=/home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR
LOG=$RUNROOT/logs/watcher.log
CHAIN=/tmp/run_r0_lhr_chain.sh
NEED_MIB=18000          # our 512-env training peaks ~15 GB; keep headroom
STABLE_CHECKS=3         # require 3 consecutive free checks (>=3 min idle)
MAX_POLLS=4320          # ~72 h before giving up
mkdir -p "$RUNROOT/logs"

ts() { date '+%F %T'; }
note() { echo "[watcher $(ts)] $*" >> "$LOG"; }

note "ARMED: waiting for GPU free >= ${NEED_MIB} MiB for ${STABLE_CHECKS} consecutive checks; chain=$CHAIN"
nvidia-smi > "$RUNROOT/logs/watcher_gpu_start.txt" 2>&1

stable=0
for i in $(seq 1 "$MAX_POLLS"); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then stable=$((stable + 1)); else stable=0; fi
  # log one line per 10 polls (10 min) to keep the log small, plus every state change
  if [ $((i % 10)) -eq 1 ] || [ "$stable" -ge "$STABLE_CHECKS" ]; then
    note "poll=$i free=${free}MiB used=${used}MiB util=${util}% stable=${stable}/${STABLE_CHECKS}"
  fi
  if [ "$stable" -ge "$STABLE_CHECKS" ]; then
    note "GPU judged FREE (${STABLE_CHECKS}x>=${NEED_MIB}MiB) -> launching training chain"
    nvidia-smi > "$RUNROOT/logs/gpu_before_chain.txt" 2>&1
    nohup bash "$CHAIN" > "$RUNROOT/logs/chain.log" 2>&1 < /dev/null &
    CHAIN_PID=$!
    echo "$CHAIN_PID" > "$RUNROOT/logs/chain.pid"
    note "chain launched pid=$CHAIN_PID; log=$RUNROOT/logs/chain.log"
    wait "$CHAIN_PID"
    RC=$?
    note "chain finished rc=$RC"
    exit 0
  fi
  sleep 60
done
note "TIMEOUT after ${MAX_POLLS} polls (~72h) without enough free GPU; training NOT started"
exit 1

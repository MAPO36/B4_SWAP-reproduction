#!/bin/bash
# R0-LHR training chain (SWAP-lite stage 3, 11-16 plan).
#   stage A: integration smoke  32 env x 20 iter   (proves task/env/terrain wiring)
#   stage B: 1k smoke           512 env x 1000 iter (stability gate)
#   stage C: R0-LHR 10k         512 env x 10000 iter (the formal baseline)
# GPU memory guard before every stage (another person shares this machine).
# Run detached: nohup bash /tmp/run_r0_lhr_chain.sh > .../logs/chain.log 2>&1 &
set -uo pipefail

RUNROOT=/home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR
WMP=/home/ubuntu/B4_2025/SWAP/WMP
LOG=$RUNROOT/logs
mkdir -p "$RUNROOT"/{logs,terrain_check,seed0,smoke,full} "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WMP" || exit 1
export PYTHONPATH=$WMP
export MPLBACKEND=Agg

ts() { date '+%F %T'; }
note() { echo "[chain $(ts)] $*"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

# GPU guard: wait up to 30 min for enough free VRAM (machine is shared with another user)
require_free() {
  local need="$1" what="$2" f i
  for i in $(seq 1 30); do
    f=$(free_mib)
    if [ "${f:-0}" -ge "$need" ]; then
      note "[vram] OK free=${f}MiB >= ${need}MiB for ${what} (used=$(used_mib)MiB)"
      return 0
    fi
    note "[vram] WAIT free=${f}MiB < ${need}MiB for ${what}; retry in 60s ($i/30)"
    sleep 60
  done
  note "[vram] ABORT: not enough free VRAM for ${what} after 30 min (free=${f}MiB, need=${need}MiB)."
  return 1
}

nan_lines() { grep -Eic '(^|[^a-z])nan([^a-z]|$)' "$1" 2>/dev/null || true; }

check_ckpt() {  # $1=run_name
  ls -1 "$WMP/logs/go2_lhr_example/$1"/model_*.pt >/dev/null 2>&1
}

vram_sampler() {
  echo "timestamp, memory.used [MiB], util [%]" > "$LOG/vram.csv"
  while :; do
    printf '%s, %s\n' "$(ts)" "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)" >> "$LOG/vram.csv" 2>/dev/null
    sleep 60
  done
}
vram_sampler &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

note "CHAIN START runroot=$RUNROOT vram(used,free)=$(used_mib)MiB/$(free_mib)MiB"
nvidia-smi > "$LOG/gpu_start_full.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$LOG/gpu_start.txt" 2>&1

# ---------------------------------------------------------------- stage A: integration
require_free 10000 "integration 32env" || exit 2
note "STAGE A integration: 32 env x 20 iter"
MAX_ITERS=20 MAX_ENVS=32 SAVE_INTERVAL=1000 RUN_NAME=R0_LHR_integration TERRAIN_REPORT_DIR="$RUNROOT/terrain_check" \
  python train_lhr.py --task=go2_amp_lhr --headless --num_envs=32 --sim_device=cuda:0 \
  > "$LOG/integration.log" 2>&1
RC=$?
note "STAGE A exit=$RC nan_lines=$(nan_lines "$LOG/integration.log")"
grep -a 'LHR gate PASS' "$LOG/integration.log" | tail -1
grep -a 'PASS: 20 iterations' "$LOG/integration.log" | tail -1
if [ $RC -ne 0 ] || ! grep -aq 'PASS: 20 iterations' "$LOG/integration.log"; then
  note "STAGE A FAILED -> chain aborts (full log: $LOG/integration.log)"
  exit 3
fi
cp -f "$WMP/logs/go2_lhr_example/R0_LHR_integration/model_20.pt" "$RUNROOT/seed0/model_20_integration.pt" 2>/dev/null || true

# ---------------------------------------------------------------- stage B: 1k smoke
require_free 17000 "1k smoke 512env" || exit 4
note "STAGE B smoke: 512 env x 1000 iter (this takes ~2-2.5 h)"
MAX_ITERS=1000 MAX_ENVS=512 SAVE_INTERVAL=250 RUN_NAME=R0_LHR_smoke TERRAIN_REPORT_DIR="$RUNROOT/terrain_check" \
  python train_lhr.py --task=go2_amp_lhr --headless --num_envs=512 --sim_device=cuda:0 \
  > "$LOG/smoke.log" 2>&1
RC=$?
NAN=$(nan_lines "$LOG/smoke.log")
note "STAGE B exit=$RC nan_lines=$NAN"
grep -a 'LHR gate PASS' "$LOG/smoke.log" | tail -1
grep -a 'PASS: 1000 iterations' "$LOG/smoke.log" | tail -1
check_ckpt R0_LHR_smoke && CKPT_OK=1 || CKPT_OK=0
if [ $RC -ne 0 ] || ! grep -aq 'PASS: 1000 iterations' "$LOG/smoke.log" || [ "$NAN" -ne 0 ] || [ "$CKPT_OK" -ne 1 ]; then
  note "STAGE B FAILED -> chain aborts (rc=$RC nan=$NAN ckpt=$CKPT_OK; log: $LOG/smoke.log)"
  exit 5
fi
note "STAGE B PASS: 1k smoke stable, no NaN"

# ---------------------------------------------------------------- stage C: R0-LHR 10k
require_free 17000 "R0-LHR 10k 512env" || exit 6
note "STAGE C full: 512 env x 10000 iter (this takes ~22-24 h)"
nvidia-smi > "$LOG/gpu_full_start.txt" 2>&1
MAX_ITERS=10000 MAX_ENVS=512 SAVE_INTERVAL=500 RUN_NAME=R0_LHR TERRAIN_REPORT_DIR="$RUNROOT/terrain_check" \
  python train_lhr.py --task=go2_amp_lhr --headless --num_envs=512 --sim_device=cuda:0 \
  > "$LOG/full.log" 2>&1
RC=$?
NAN=$(nan_lines "$LOG/full.log")
note "STAGE C exit=$RC nan_lines=$NAN"
grep -a 'LHR gate PASS' "$LOG/full.log" | tail -1
grep -a 'PASS: 10000 iterations' "$LOG/full.log" | tail -1
nvidia-smi > "$LOG/gpu_full_end.txt" 2>&1
if [ $RC -ne 0 ] || ! grep -aq 'PASS: 10000 iterations' "$LOG/full.log"; then
  note "STAGE C FAILED/INCOMPLETE (rc=$RC; log: $LOG/full.log; ckpts in $WMP/logs/go2_lhr_example/R0_LHR/)"
  exit 7
fi

# ---------------------------------------------------------------- finalize
CKPT="$WMP/logs/go2_lhr_example/R0_LHR/model_10000.pt"
if [ -f "$CKPT" ]; then
  cp -f "$CKPT" "$RUNROOT/seed0/model_10000.pt"
  sha256sum "$RUNROOT/seed0/model_10000.pt" | cut -d' ' -f1 > "$RUNROOT/seed0/model_10000.sha256"
  note "final ckpt -> $RUNROOT/seed0/model_10000.pt sha256=$(cat "$RUNROOT/seed0/model_10000.sha256")"
  cp -f "$WMP/logs/go2_lhr_example/R0_LHR/config.yaml" "$RUNROOT/seed0/" 2>/dev/null || true
else
  note "WARNING: model_10000.pt missing!"
fi
cat > "$RUNROOT/RUN_INFO.txt" <<TXT
R0-LHR (SWAP-lite stage 3)
chain start : 2026-11-16
HEAD at launch: $(git -C /home/ubuntu/B4_2025/SWAP rev-parse --short HEAD 2>/dev/null)
stages      : integration(32x20) -> smoke(512x1000) -> full(512x10000)
task        : go2_amp_lhr  (fixed LHR 10 deg, families: platform box / stairs up / gap, 1/3 each)
checkpoint  : seed0/model_10000.pt
logs        : logs/{chain,integration,smoke,full}.log ; logs/vram.csv ; terrain_check/
TXT
note "CHAIN DONE $(ts)"
exit 0

#!/bin/bash
# One-shot integration smoke for the LHR task (32 env x 20 iter) + GPU status.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/B4_2025/SWAP/WMP || exit 1
export PYTHONPATH=$PWD
export MPLBACKEND=Agg
RUNROOT=/home/ubuntu/B4_2025/SWAP/workSpace/R0_LHR
mkdir -p "$RUNROOT"/{logs,terrain_check,seed0}
echo "=== GPU before ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== integration: 32 env x 20 iter ==="
MAX_ITERS=20 MAX_ENVS=32 SAVE_INTERVAL=1000 RUN_NAME=R0_LHR_integration \
  TERRAIN_REPORT_DIR="$RUNROOT/terrain_check" \
  python train_lhr.py --task=go2_amp_lhr --headless --num_envs=32 --sim_device=cuda:0 \
  > "$RUNROOT/logs/integration.log" 2>&1
RC=$?
echo "RC=$RC"
grep -a 'LHR gate PASS\|train_lhr\] LHR direction report\|PASS: 20 iterations\|Traceback\|Error' "$RUNROOT/logs/integration.log" | cut -c1-500
echo "--- last 5 log lines ---"
tail -5 "$RUNROOT/logs/integration.log"
echo "=== GPU after ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
ls -la "$RUNROOT/terrain_check" | head

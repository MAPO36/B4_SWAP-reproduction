#!/bin/bash
# Stage-3 post-training evaluation batch (run AFTER the R0-LHR training chain finishes).
#   * 18 cells: 3 families (box/stair/gap) x 3 levels x 2 directions (LHR/RHL) x 200 trials
#   * 6 video cells at level 2 (TRIALS=1), GIFs
#   * aggregate_lhr.py + make_lhr_figures.py
# GPU memory guard at the start (shared 4090). Usage:
#   nohup bash /tmp/run_lhr_eval_matrix.sh > <RUN>/logs/eval_matrix.log 2>&1 &
set -uo pipefail
SWAP=/home/ubuntu/B4_2025/SWAP
WMP=/home/ubuntu/B4_2025/SWAP/WMP
CONDITION=${CONDITION:-R0-LHR}
LOAD_EXPERIMENT=${LOAD_EXPERIMENT:-go2_lhr_example}
LOAD_RUN=${LOAD_RUN:-R0_LHR}
CKPT_ITER=${CKPT_ITER:--1}
TRIALS=${TRIALS:-200}
EVAL_DIR="$RUNROOT/eval"
VID_DIR="$RUNROOT/videos"
LOGD="$RUNROOT/logs"
mkdir -p "$EVAL_DIR" "$VID_DIR" "$LOGD"
PROG="$LOGD/eval_progress.txt"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WMP" || exit 1
export PYTHONPATH=$WMP
export MPLBACKEND=Agg

ts() { date '+%F %T'; }
note() { echo "[eval-matrix $(ts)] $*" | tee -a "$PROG"; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
require_free() {
  local need="$1" what="$2" f i
  for i in $(seq 1 30); do
    f=$(free_mib)
    if [ "${f:-0}" -ge "$need" ]; then note "[vram] OK free=${f}MiB for ${what}"; return 0; fi
    note "[vram] WAIT free=${f}MiB < ${need}MiB for ${what} ($i/30)"
    sleep 60
  done
  note "[vram] ABORT: not enough free VRAM for ${what}"
  return 1
}

# ---- checkpoint must exist -------------------------------------------------------------
CKPT_GLOB="$WMP/logs/$LOAD_EXPERIMENT/$LOAD_RUN/model_*.pt"
if ! ls -1 $CKPT_GLOB >/dev/null 2>&1; then
  note "ABORT: no checkpoint matched $CKPT_GLOB (has the training chain finished?)"
  exit 1
fi
note "START condition=$CONDITION experiment=$LOAD_EXPERIMENT run=$LOAD_RUN ckpt_iter=$CKPT_ITER"
ls -la $CKPT_GLOB | tail -3
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader > "$LOGD/eval_gpu_start.txt" 2>&1

require_free 6000 "18-cell LHR/RHL evaluation" || exit 2

# ---- 18 evaluation cells --------------------------------------------------------------
FAILED=0
for fam in box stair gap; do
  for lvl in 1 2 3; do
    for dir in LHR RHL; do
      label="${CONDITION}_${fam}_L${lvl}_${dir}"
      out="$EVAL_DIR/$label"
      mkdir -p "$out"
      note "CELL $label $(ts)"
      FAMILY="$fam" LEVEL="$lvl" DIRECTION="$dir" TRIALS="$TRIALS" TERRAIN_SEED=1 \
        CONDITION="$CONDITION" LOAD_EXPERIMENT="$LOAD_EXPERIMENT" LOAD_RUN="$LOAD_RUN" \
        CKPT_ITER="$CKPT_ITER" LHR_TILT_DEG=10.0 OUT_DIR="$out" RUN_LABEL="$label" \
        python legged_gym/scripts/eval_lhr.py --task=go2_amp_lhr --headless --sim_device=cuda:0 \
        > "$out/run.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || [ ! -f "$out/summary.json" ]; then
        FAILED=1
        note "CELL-FAIL $label rc=$rc (log: $out/run.log)"
      else
        note "CELL-OK $label $(grep -a '\[EVAL\]' "$out/run.log" | tail -1 | cut -c1-200)"
      fi
    done
  done
done
note "eval cells done; failed=$FAILED"

# ---- aggregate (before videos, so metrics/manifest exist even if video step fails) -----
python "$SWAP/workSpace/tools/stage3/aggregate_lhr.py" --run-dir "$RUNROOT" --condition "$CONDITION" \
  --train-log "$LOGD/full.log" >> "$PROG" 2>&1 || note "aggregate WARN"
python "$SWAP/workSpace/tools/stage3/make_lhr_figures.py" --run-dir "$RUNROOT" >> "$PROG" 2>&1 || note "figures WARN"

# ---- 6 video cells at level 2 ---------------------------------------------------------
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
for fam in box stair gap; do
  for dir in LHR RHL; do
    label="video_${CONDITION}_${fam}_L2_${dir}"
    out="$VID_DIR/$label"
    mkdir -p "$out"
    note "VIDEO $label $(ts)"
    FAMILY="$fam" LEVEL=2 DIRECTION="$dir" TRIALS=1 STEPS=1001 RECORD_FRAMES=1 FRAME_EVERY=2 \
      CAMERA_MODE=follow TERRAIN_SEED=1 CONDITION="$CONDITION" LOAD_EXPERIMENT="$LOAD_EXPERIMENT" \
      LOAD_RUN="$LOAD_RUN" CKPT_ITER="$CKPT_ITER" LHR_TILT_DEG=10.0 OUT_DIR="$out" RUN_LABEL="$label" \
      python legged_gym/scripts/eval_lhr.py --task=go2_amp_lhr --sim_device=cuda:0 \
      > "$out/run.log" 2>&1 || note "VIDEO-FAIL $label"
    n=$(ls "$out/frames" 2>/dev/null | wc -l)
    note "VIDEO-DONE $label frames=$n"
  done
done
python /tmp/make_gifs.py --videos-dir "$VID_DIR" --num-frames 60 --fps 10 >> "$PROG" 2>&1 || note "gif WARN"

# ---- final aggregate (now with videos) ------------------------------------------------
python "$SWAP/workSpace/tools/stage3/aggregate_lhr.py" --run-dir "$RUNROOT" --condition "$CONDITION" \
  --train-log "$LOGD/full.log" >> "$PROG" 2>&1 || note "final aggregate WARN"

note "DONE failed_cells=$FAILED $(ts)"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader > "$LOGD/eval_gpu_end.txt" 2>&1

#!/bin/bash
# =============================================================================
# T3 mirror-terrain evaluation matrix runner (serial, single GPU).
#
#   A组 (WMP native terrains)       : tilt/stair/gap x diff 0.2/0.5/0.8 x normal/mirror = 18 cells
#   B组 (SWAP-style lateral ramp)   : ramp(0.3)/ramp_stair(0.5)/ramp_gap(0.5) x tilt 5/10/15 deg
#                                     x normal/mirror = 18 cells
#   T1 protocol re-check            : eval_t1_repro.py NEVAL=64 STEPS=1200, slope/stair/gap x 3 runs
#   reproducibility re-runs         : 3 A-group cells, same run_label as the first pass
#   videos                          : TRIALS=1 viewer frames + GIF (GIF built by make_gifs.sh)
#
# Run on the 4090 from anywhere (conda is activated by this script):
#   bash run_matrix.sh [all|a|b|t1|repro|videos|gifs|none]
#
# Environment overrides:
#   RUN_DIR        run directory (default .../workSpace/runs/s3-20261112-1620-mirror-eval)
#   GPU            CUDA device index (default 0)
#   FORCE          1 = re-run cells even when summary.json already exists
#   ALLOW_BUSY     1 = skip the "GPU/other eval process running" guard
#   DRY_RUN        1 = only print the cell list (no eval)
#   CELL_TIMEOUT   per-cell timeout in seconds (default 1800)
#   AUTO_AGG       0 = do not run aggregate_metrics.py / make_figures.py at the end
#   TRIALS_OVERRIDE  >0 = replace trials for the matrix/repro cells (debug only)
#   PYTHON         python interpreter (default: python from the active conda env)
#
# Every cell writes  <RUN_DIR>/<out_root>/<label>/{summary.json,trials.jsonl,terrain.npz,run.log}
# and one line to <RUN_DIR>/progress.txt. Failures do not stop the run.
# =============================================================================
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE="${1:-all}"

SWAP_DIR="${SWAP_DIR:-/home/ubuntu/B4_2025/SWAP}"
WMP_DIR="$SWAP_DIR/WMP"
RUN_DIR="${RUN_DIR:-$SWAP_DIR/workSpace/runs/s3-20261112-1620-mirror-eval}"
GPU="${GPU:-0}"
SIM_DEVICE="cuda:$GPU"
FORCE="${FORCE:-0}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
DRY_RUN="${DRY_RUN:-0}"
CELL_TIMEOUT="${CELL_TIMEOUT:-1800}"
AUTO_AGG="${AUTO_AGG:-1}"
TRIALS_OVERRIDE="${TRIALS_OVERRIDE:-0}"
PYTHON="${PYTHON:-python}"
export SWAP_DIR GPU RUN_DIR MODE TRIALS_OVERRIDE CELL_TIMEOUT

LOGS_DIR="$RUN_DIR/logs"
PROGRESS="$RUN_DIR/progress.txt"
CELLS_TSV="$SCRIPT_DIR/cells.tsv"
VIDEOS_DIR="$RUN_DIR/videos"

# video cells: label / terrain / diff / tilt / mirror  (TRIALS=1 by design: one terrain patch,
# several envs would all spawn on the same spot and the viewer would show overlapping robots)
VIDEO_CELLS=(
  "video_a_stair_d0.5_normal     stair       0.5  0  0"
  "video_a_stair_d0.5_mirror     stair       0.5  0  1"
  "video_b_ramp_stair_t10_normal ramp_stair  0.5  10 0"
  "video_b_ramp_stair_t10_mirror ramp_stair  0.5  10 1"
)
VIDEO_TRIALS=1
VIDEO_STEPS=1001
VIDEO_FRAME_EVERY=2
VIDEO_CAMERA_MODE=follow

FAILED_CELLS=0
OK_CELLS=0
SKIP_CELLS=0

# ---------------------------------------------------------------- bootstrap ---
source ~/miniconda3/etc/profile.d/conda.sh || { echo "[FATAL] conda not found"; exit 3; }
conda activate wmp || { echo "[FATAL] cannot activate wmp"; exit 3; }
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WMP_DIR"
export MPLBACKEND=Agg
cd "$WMP_DIR" || { echo "[FATAL] no WMP dir $WMP_DIR"; exit 3; }

mkdir -p "$LOGS_DIR" "$RUN_DIR/a_matrix" "$RUN_DIR/b_matrix" "$RUN_DIR/t1_repro" \
         "$RUN_DIR/repro_check" "$VIDEOS_DIR" "$RUN_DIR/raw"
LOG_FILE="$LOGS_DIR/matrix.log"

log() {
  local msg
  msg="$(date '+%Y-%m-%d %H:%M:%S') | $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

record() {  # cell-level result lines also go to progress.txt (append + flush)
  local msg
  msg="$(date '+%Y-%m-%dT%H:%M:%S') | $*"
  echo "$msg" | tee -a "$PROGRESS" >> "$LOG_FILE"
}

die() { log "[FATAL] $*"; exit 2; }

smi_snapshot() {  # $1 = destination file
  {
    echo "# nvidia-smi snapshot $(date '+%F %T')"
    nvidia-smi
    echo
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  } > "$1" 2>&1
}

gpu_guard() {
  local busy used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  log "[gpu] memory.used=${used:-?} MiB before start (device $SIM_DEVICE)"
  busy=$(ps -eo pid,cmd | grep -E '[l]egged_gym/scripts/(eval_mirror|eval_t1_repro|train)\.py' | grep -v 'run_matrix' || true)
  if [ -n "$busy" ] && [ "$ALLOW_BUSY" != "1" ]; then
    log "[ABORT] another eval/train process is running:"
    echo "$busy" | tee -a "$LOG_FILE"
    log "        set ALLOW_BUSY=1 to override (never run two GPU jobs at once)."
    exit 4
  fi
  if [ -n "${used:-}" ] && [ "$used" -gt 4000 ] && [ "$ALLOW_BUSY" != "1" ]; then
    log "[ABORT] GPU already uses ${used} MiB (> 4000); set ALLOW_BUSY=1 to override."
    exit 4
  fi
}

VRAM_PID=""
cleanup() {
  if [ -n "$VRAM_PID" ]; then
    kill "$VRAM_PID" 2>/dev/null || true
    wait "$VRAM_PID" 2>/dev/null || true
    VRAM_PID=""
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------- run config ------
write_run_config() {
  "$PYTHON" - "$RUN_DIR/run_config.json" "$RUN_DIR" "$SCRIPT_DIR" "$MODE" "$CELLS_TSV" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
import time

cfg_path, run_dir, script_dir, mode, cells_tsv = sys.argv[1:6]

def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

def git_info(repo):
    out = {}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("commit_short", ["rev-parse", "--short", "HEAD"]),
        ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
        ("status_short", ["status", "--porcelain"]),
    ):
        try:
            val = subprocess.check_output(["git", "-C", repo] + args, stderr=subprocess.DEVNULL)
            val = val.decode("utf-8", "replace").strip()
            out[key] = val if key != "status_short" else val.splitlines()
        except Exception:
            out[key] = None
    return out

cells = {}
try:
    with open(cells_tsv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            group, out_root, label, run_label, terrain, diff, tilt, mirror, trials, steps = parts[:10]
            cells.setdefault(out_root, []).append({
                "group": group, "out_root": out_root, "label": label, "run_label": run_label,
                "terrain": terrain, "difficulty": float(diff), "tilt_deg": float(tilt),
                "mirror": bool(int(mirror)), "trials": int(trials), "steps": int(steps),
            })
except OSError:
    cells = {}

wmp = os.path.join(os.environ.get("SWAP_DIR", "/home/ubuntu/B4_2025/SWAP"), "WMP")
eval_script = os.path.join(wmp, "legged_gym/scripts/eval_mirror.py")
mirror_script = os.path.join(wmp, "legged_gym/utils/mirror_terrain.py")
t1_script = os.path.join(wmp, "legged_gym/scripts/eval_t1_repro.py")
cfg = {
    "run_id": os.path.basename(run_dir.rstrip("/")),
    "stage": "T3",
    "purpose": "T3 mirror-terrain evaluation: A (WMP native terrains) + B (SWAP-style unilateral lateral ramp)",
    "run_dir": run_dir,
    "host": "buaa-ubuntu-22",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "mode": mode,
    "sim_device": "cuda:%s" % os.environ.get("GPU", "0"),
    "gpu": os.environ.get("GPU", "0"),
    "trials_per_cell": int(os.environ.get("TRIALS_OVERRIDE", "0")) or 200,
    "trials_override": int(os.environ.get("TRIALS_OVERRIDE", "0")),
    "terrain_seed": 1,
    "steps": 0,
    "load_run": "WMP",
    "ckpt_iter": -1,
    "stages_run": mode,
    "eval_script": eval_script,
    "eval_script_sha256": sha256(eval_script),
    "eval_script_mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(eval_script)))
    if os.path.isfile(eval_script) else None,
    "mirror_terrain_path": mirror_script,
    "mirror_terrain_sha256_at_start": sha256(mirror_script),
    "mirror_terrain_mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(mirror_script)))
    if os.path.isfile(mirror_script) else None,
    "t1_script": t1_script,
    "t1_script_sha256": sha256(t1_script),
    "cells_tsv": cells_tsv,
    "cells": cells,
    "videos": {
        "trials": 1, "steps": 1001, "frame_every": 2, "camera_mode": "follow",
        "note": "TRIALS=1: num_rows=1 means all envs share one terrain patch (overlapping robots).",
    },
    "t1_repro": {"neval": 64, "steps": 1200, "terrain": ["slope", "stair", "gap"], "runs_per_terrain": 3},
    "git": git_info(os.environ.get("SWAP_DIR", "/home/ubuntu/B4_2025/SWAP")),
    "script_sha256": {},
}
for name in ("run_matrix.sh", "aggregate_metrics.py", "make_figures.py", "make_gifs.sh",
             "gif_maker.py", "cells.tsv"):
    cfg["script_sha256"][name] = sha256(os.path.join(script_dir, name))
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("[CONFIG] wrote %s (%d matrix/repro cells)" % (cfg_path, sum(len(v) for v in cells.values())))
PY
}

# ------------------------------------------------------------------ cells -----
run_cell() {
  # group out_root label run_label terrain diff tilt mirror trials steps
  local group="$1" out_root="$2" label="$3" run_label="$4" terrain="$5" diff="$6" \
        tilt="$7" mirror="$8" trials="$9" steps="${10}"
  local out="$RUN_DIR/$out_root/$label"
  local rel="$out_root/$label"
  if [ "$TRIALS_OVERRIDE" -gt 0 ] 2>/dev/null; then trials="$TRIALS_OVERRIDE"; fi
  if [ "$DRY_RUN" = "1" ]; then
    record "DRY   | $group | $rel | terrain=$terrain diff=$diff tilt=$tilt mirror=$mirror trials=$trials"
    return 0
  fi
  if [ "$FORCE" != "1" ] && [ -s "$out/summary.json" ]; then
    SKIP_CELLS=$((SKIP_CELLS + 1))
    record "SKIP  | $group | $rel | summary.json already exists (FORCE=1 to rerun)"
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  local t0 t1 rc sr dt
  t0=$(date +%s)
  record "START | $group | $rel | terrain=$terrain diff=$diff tilt=$tilt mirror=$mirror trials=$trials"
  TERRAIN="$terrain" DIFF="$diff" TILT_DEG="$tilt" MIRROR="$mirror" TRIALS="$trials" STEPS="$steps" \
    TERRAIN_SEED=1 OUT_DIR="$out" RUN_LABEL="$run_label" LOAD_RUN=WMP CKPT_ITER=-1 \
    timeout -k 30 "$CELL_TIMEOUT" "$PYTHON" legged_gym/scripts/eval_mirror.py \
      --task=go2_amp --headless --sim_device="$SIM_DEVICE" >> "$out/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  dt=$((t1 - t0))
  if [ "$rc" -eq 0 ] && [ -s "$out/summary.json" ]; then
    sr=$("$PYTHON" - "$out/summary.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        s = json.load(f)
    print("%s/%s" % (s.get("success"), s.get("trials_done")))
except Exception:
    print("?/?")
PY
)
    OK_CELLS=$((OK_CELLS + 1))
    record "OK    | $group | $rel | rc=0 wall_s=$dt success=$sr"
  else
    FAILED_CELLS=$((FAILED_CELLS + 1))
    record "FAIL  | $group | $rel | rc=$rc wall_s=$dt log=$out/run.log"
  fi
  return 0
}

run_matrix_stage() {  # $1 = out_root filter (a_matrix|b_matrix|repro_check|'')
  local want="$1"
  while read -r group out_root label run_label terrain diff tilt mirror trials steps; do
    [ -z "${group:-}" ] && continue
    case "$group" in \#*) continue ;; esac
    if [ -n "$want" ] && [ "$out_root" != "$want" ]; then continue; fi
    run_cell "$group" "$out_root" "$label" "$run_label" "$terrain" "$diff" "$tilt" "$mirror" "$trials" "$steps"
  done < "$CELLS_TSV"
}

# ------------------------------------------------------------- T1 re-check ----
run_t1_repro() {
  local script="$WMP_DIR/legged_gym/scripts/eval_t1_repro.py"
  if [ ! -f "$script" ]; then
    if [ -f /home/ubuntu/B4_2025/Materials/SE/scripts/eval_success.py ]; then
      cp -f /home/ubuntu/B4_2025/Materials/SE/scripts/eval_success.py "$script"
      log "[T1] eval_t1_repro.py missing -> copied from Materials/SE/scripts/eval_success.py"
    else
      log "[T1] FAIL: neither eval_t1_repro.py nor Materials eval_success.py found; skipping T1 re-check"
      return 0
    fi
  fi
  local terrain r out t0 t1 rc
  for terrain in slope stair gap; do
    for r in 1 2 3; do
      out="$RUN_DIR/t1_repro/${terrain}_r${r}"
      if [ "$DRY_RUN" = "1" ]; then
        record "DRY   | T1 | t1_repro/${terrain}_r${r} | NEVAL=64 STEPS=1200 --terrain=$terrain"
        continue
      fi
      if [ "$FORCE" != "1" ] && [ -s "$out/run.log" ] && grep -qa '\[EVAL\]' "$out/run.log"; then
        SKIP_CELLS=$((SKIP_CELLS + 1))
        record "SKIP  | T1 | t1_repro/${terrain}_r${r} | log already has [EVAL]"
        continue
      fi
      rm -rf "$out"; mkdir -p "$out"
      t0=$(date +%s)
      record "START | T1 | t1_repro/${terrain}_r${r} | NEVAL=64 STEPS=1200 --terrain=$terrain"
      NEVAL=64 STEPS=1200 timeout -k 30 "$CELL_TIMEOUT" "$PYTHON" legged_gym/scripts/eval_t1_repro.py \
        --task=go2_amp --headless --sim_device="$SIM_DEVICE" --terrain="$terrain" >> "$out/run.log" 2>&1
      rc=$?
      t1=$(date +%s)
      if [ "$rc" -eq 0 ] && grep -qa '\[EVAL\]' "$out/run.log"; then
        OK_CELLS=$((OK_CELLS + 1))
        record "OK    | T1 | t1_repro/${terrain}_r${r} | rc=0 wall_s=$((t1 - t0)) $(grep -a '\[EVAL\] envs=' "$out/run.log" | tail -1)"
      else
        FAILED_CELLS=$((FAILED_CELLS + 1))
        record "FAIL  | T1 | t1_repro/${terrain}_r${r} | rc=$rc wall_s=$((t1 - t0)) log=$out/run.log"
      fi
    done
  done
}

# ----------------------------------------------------------------- videos -----
run_videos() {
  if [ ! -S /tmp/.X11-unix/X1 ]; then
    log "[WARN] /tmp/.X11-unix/X1 not present; viewer recording may fail (continuing)"
  fi
  local entry name terrain diff tilt mirror out t0 t1 rc
  for entry in "${VIDEO_CELLS[@]}"; do
    name=$(echo "$entry" | awk '{print $1}')
    terrain=$(echo "$entry" | awk '{print $2}')
    diff=$(echo "$entry" | awk '{print $3}')
    tilt=$(echo "$entry" | awk '{print $4}')
    mirror=$(echo "$entry" | awk '{print $5}')
    out="$VIDEOS_DIR/$name"
    if [ "$DRY_RUN" = "1" ]; then
      record "DRY   | VIDEO | videos/$name | terrain=$terrain diff=$diff tilt=$tilt mirror=$mirror TRIALS=$VIDEO_TRIALS"
      continue
    fi
    if [ "$FORCE" != "1" ] && [ -s "$out/summary.json" ] && [ -d "$out/frames" ] \
       && [ "$(ls -1 "$out/frames"/*.png 2>/dev/null | wc -l)" -gt 10 ]; then
      SKIP_CELLS=$((SKIP_CELLS + 1))
      record "SKIP  | VIDEO | videos/$name | summary+frames exist (FORCE=1 to rerun)"
      continue
    fi
    rm -rf "$out"; mkdir -p "$out/frames"
    t0=$(date +%s)
    record "START | VIDEO | videos/$name | terrain=$terrain diff=$diff tilt=$tilt mirror=$mirror TRIALS=$VIDEO_TRIALS STEPS=$VIDEO_STEPS"
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
      TERRAIN="$terrain" DIFF="$diff" TILT_DEG="$tilt" MIRROR="$mirror" TRIALS="$VIDEO_TRIALS" \
      STEPS="$VIDEO_STEPS" TERRAIN_SEED=1 OUT_DIR="$out" RUN_LABEL="$name" LOAD_RUN=WMP CKPT_ITER=-1 \
      RECORD_FRAMES=1 FRAME_EVERY="$VIDEO_FRAME_EVERY" CAMERA_MODE="$VIDEO_CAMERA_MODE" \
      timeout -k 30 "$CELL_TIMEOUT" "$PYTHON" legged_gym/scripts/eval_mirror.py \
        --task=go2_amp --headless --sim_device="$SIM_DEVICE" >> "$out/run.log" 2>&1
    rc=$?
    t1=$(date +%s)
    local nframes
    nframes=$(ls -1 "$out/frames"/*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$rc" -eq 0 ] && [ -s "$out/summary.json" ] && [ "$nframes" -gt 10 ]; then
      OK_CELLS=$((OK_CELLS + 1))
      record "OK    | VIDEO | videos/$name | rc=0 wall_s=$((t1 - t0)) frames=$nframes"
    else
      FAILED_CELLS=$((FAILED_CELLS + 1))
      record "FAIL  | VIDEO | videos/$name | rc=$rc wall_s=$((t1 - t0)) frames=$nframes log=$out/run.log"
    fi
  done
}

run_gifs() {
  if [ "$DRY_RUN" = "1" ]; then
    record "DRY   | GIF | make_gifs.sh --all $VIDEOS_DIR"
    return 0
  fi
  if [ ! -f "$SCRIPT_DIR/make_gifs.sh" ]; then
    log "[GIF] make_gifs.sh not found next to run_matrix.sh; skipping"
    return 0
  fi
  local t0 t1 rc
  t0=$(date +%s)
  bash "$SCRIPT_DIR/make_gifs.sh" --all "$VIDEOS_DIR" --sample 60 --fps 10 --width 640 >> "$LOG_FILE" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    record "OK    | GIF | make_gifs.sh --all $VIDEOS_DIR | wall_s=$((t1 - t0))"
  else
    FAILED_CELLS=$((FAILED_CELLS + 1))
    record "FAIL  | GIF | make_gifs.sh --all $VIDEOS_DIR | rc=$rc wall_s=$((t1 - t0))"
  fi
}

# -------------------------------------------------------------- aggregate -----
run_aggregate() {
  if [ "$AUTO_AGG" != "1" ] || [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  log "[AGG] aggregate_metrics.py"
  "$PYTHON" "$SCRIPT_DIR/aggregate_metrics.py" --run-dir "$RUN_DIR" >> "$LOG_FILE" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then record "OK    | AGG | metrics.csv + tables + manifest"; else
    record "FAIL  | AGG | aggregate_metrics.py rc=$rc (see logs/matrix.log)"; FAILED_CELLS=$((FAILED_CELLS + 1)); fi
  if "$PYTHON" -c "import matplotlib" 2>/dev/null; then
    log "[FIG] make_figures.py"
    "$PYTHON" "$SCRIPT_DIR/make_figures.py" --run-dir "$RUN_DIR" >> "$LOG_FILE" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then record "OK    | FIG | figures/*.png"; else
      record "FAIL  | FIG | make_figures.py rc=$rc (see logs/matrix.log)"; FAILED_CELLS=$((FAILED_CELLS + 1)); fi
  else
    log "[FIG] matplotlib not importable under $PYTHON; skipping figures"
  fi
}

# ------------------------------------------------------------------- main -----
main() {
  [ -f "$CELLS_TSV" ] || die "cells.tsv not found at $CELLS_TSV"
  log "=============================================================="
  log "T3 matrix run | mode=$MODE | RUN_DIR=$RUN_DIR | GPU=$SIM_DEVICE | FORCE=$FORCE | DRY_RUN=$DRY_RUN"
  log "WMP=$WMP_DIR PYTHON=$("$PYTHON" --version 2>&1)"
  gpu_guard
  : > "$PROGRESS"
  echo "# T3 matrix progress (append-only; one line per cell)" >> "$PROGRESS"
  smi_snapshot "$LOGS_DIR/gpu_start.txt"
  log "[gpu] start snapshot -> $LOGS_DIR/gpu_start.txt"
  if [ "$DRY_RUN" != "1" ]; then
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 \
      > "$LOGS_DIR/vram.csv" 2>/dev/null &
    VRAM_PID=$!
  fi
  write_run_config

  case "$MODE" in
    all)    run_matrix_stage "a_matrix"; run_matrix_stage "b_matrix"; run_t1_repro; run_matrix_stage "repro_check"; run_videos; run_gifs ;;
    a)      run_matrix_stage "a_matrix" ;;
    b)      run_matrix_stage "b_matrix" ;;
    repro)  run_matrix_stage "repro_check" ;;
    t1)     run_t1_repro ;;
    videos) run_videos; run_gifs ;;
    gifs)   run_gifs ;;
    none)   log "[main] mode=none: no cells executed" ;;
    *)      die "unknown mode '$MODE' (all|a|b|t1|repro|videos|gifs|none)" ;;
  esac

  run_aggregate
  cleanup
  smi_snapshot "$LOGS_DIR/gpu_end.txt"
  log "[gpu] end snapshot -> $LOGS_DIR/gpu_end.txt"
  local residual
  residual=$(ps -eo pid,cmd | grep -E '[l]egged_gym/scripts/(eval_mirror|eval_t1_repro)\.py' || true)
  if [ -n "$residual" ]; then
    record "WARN  | RESIDUAL | python eval processes still alive after run:"
    echo "$residual" | tee -a "$PROGRESS" >> "$LOG_FILE"
  else
    record "CLEAN | RESIDUAL | no eval python process left"
  fi
  log "=============================================================="
  log "DONE | mode=$MODE | ok=$OK_CELLS skip=$SKIP_CELLS failed=$FAILED_CELLS"
  log "       summary: $RUN_DIR/metrics.csv | tables: $RUN_DIR/*_summary*.csv | logs: $LOG_FILE"
  cleanup
  if [ "$FAILED_CELLS" -ne 0 ]; then exit 1; fi
  exit 0
}

main "$@"

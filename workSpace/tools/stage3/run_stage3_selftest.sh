#!/bin/bash
# CPU self-test of the stage-3 eval/aggregate/figure pipeline with synthetic data.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
SWAP=/home/ubuntu/B4_2025/SWAP
cd "$SWAP/WMP"
export PYTHONPATH=$PWD
export MPLBACKEND=Agg
T=/tmp/stage3_selftest
rm -rf "$T"; mkdir -p "$T"

echo "=== py_compile ==="
python -m py_compile legged_gym/scripts/eval_lhr.py && echo EVAL_LHR_COMPILE_OK
python -m py_compile /tmp/aggregate_lhr.py /tmp/make_lhr_figures.py /tmp/make_synth_lhr_run.py && echo TOOLS_COMPILE_OK

echo "=== synthetic run ==="
python /tmp/make_synth_lhr_run.py --out "$T/run"
echo "=== aggregate ==="
python /tmp/aggregate_lhr.py --run-dir "$T/run" --train-log "$T/run/logs/full.log"
echo "--- checks ---"
python - <<'PY'
import json
c = json.load(open('/tmp/stage3_selftest/run/checks.json'))
print({k: c[k] for k in ('cells_found','cells_expected','mirror_pairs','mirror_pairs_fingerprint_ok','trials_200','pass')})
print('training:', c['training'])
PY
echo "--- metrics rows / pairs ---"
wc -l "$T/run/metrics.csv" "$T/run/lhr_rhl_pairs.csv" "$T/run/by_family.csv" "$T/run/by_level.csv" "$T/run/train_curve.csv"
head -2 "$T/run/lhr_rhl_pairs.csv" | cut -c1-300
echo "=== figures ==="
python /tmp/make_lhr_figures.py --run-dir "$T/run"
ls -la "$T/run/figures"
echo "=== report head ==="
head -14 "$T/run/aggregate_report.md"
echo "SELFTEST_DONE"

#!/bin/bash
# =============================================================================
# End-to-end self-test of the T3 harness on synthetic data (no GPU, no WMP needed).
#
#   bash selftest/run_selftest.sh
#
# Environment:
#   PYTHON   python interpreter to use (default: python).  On the 4090 run this after
#            `conda activate wmp` so matplotlib is importable (figures are then tested too);
#            with the Windows local python matplotlib is missing and the figure step is skipped
#            (run it on the 4090 for full coverage).
#   OUT      scratch dir (default: <tools_dir>/out/selftest)
# =============================================================================
set -u

SELFTEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS_DIR=$(cd "$SELFTEST_DIR/.." && pwd)
OUT="${OUT:-$TOOLS_DIR/out/selftest}"
PYTHON="${PYTHON:-python}"
FAIL=0

mkdir -p "$OUT"
rm -rf "$OUT/clean" "$OUT/tamper" "$OUT/gif_manual"

echo "=============================================================="
echo "[selftest] tools_dir=$TOOLS_DIR"
echo "[selftest] python=$("$PYTHON" --version 2>&1)  out=$OUT"
echo "=============================================================="

echo ""
echo "== 1/8 static checks =="
if ! "$PYTHON" -m py_compile "$TOOLS_DIR/aggregate_metrics.py" "$TOOLS_DIR/make_figures.py" \
        "$TOOLS_DIR/gif_maker.py" "$SELFTEST_DIR/make_synth_run.py" "$SELFTEST_DIR/check_selftest.py"; then
  echo "[selftest] FAIL python syntax"; exit 1
fi
echo "[selftest] python syntax OK"
for script in "$TOOLS_DIR/run_matrix.sh" "$TOOLS_DIR/make_gifs.sh"; do
  if ! bash -n "$script"; then
    echo "[selftest] FAIL bash syntax: $script"; exit 1
  fi
done
echo "[selftest] bash syntax OK"

# cells.tsv must contain 36 matrix cells + 3 repro cells
NROWS=$(grep -vc '^#' "$TOOLS_DIR/cells.tsv" | tr -d ' ')
echo "[selftest] cells.tsv data rows=$NROWS (expected 39)"
if [ "$NROWS" != "39" ]; then echo "[selftest] FAIL cells.tsv row count"; exit 1; fi

echo ""
echo "== 2/8 synthesize clean run =="
if ! "$PYTHON" "$SELFTEST_DIR/make_synth_run.py" --out "$OUT/clean"; then
  echo "[selftest] FAIL make_synth_run"; exit 1
fi

echo ""
echo "== 3/8 GIF generation (batch mode, /usr/bin/python3 priority) =="
if ! bash "$TOOLS_DIR/make_gifs.sh" --all "$OUT/clean/videos" --sample 60 --fps 10 --width 640; then
  echo "[selftest] WARN gif batch failed"
  FAIL=1
fi

echo ""
echo "== 4/8 aggregate_metrics.py =="
if ! "$PYTHON" "$TOOLS_DIR/aggregate_metrics.py" --run-dir "$OUT/clean"; then
  echo "[selftest] FAIL aggregate"; exit 1
fi

echo ""
echo "== 5/8 make_figures.py =="
FIG_ARGS=""
if "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
  if ! "$PYTHON" "$TOOLS_DIR/make_figures.py" --run-dir "$OUT/clean"; then
    echo "[selftest] FAIL make_figures"; FAIL=1
  else
    FIG_ARGS="--expect-figures"
  fi
else
  echo "[selftest] SKIP figures: matplotlib not importable by $PYTHON (run on the 4090 for full coverage)"
fi

echo ""
echo "== 6/8 assertion suite (clean run) =="
if ! "$PYTHON" "$SELFTEST_DIR/check_selftest.py" --run-dir "$OUT/clean" $FIG_ARGS; then
  FAIL=1
fi

echo ""
echo "== 7/8 negative test: corrupted fingerprint =="
if ! "$PYTHON" "$SELFTEST_DIR/make_synth_run.py" --out "$OUT/tamper" --tamper; then
  echo "[selftest] FAIL make_synth_run --tamper"; exit 1
fi
if ! "$PYTHON" "$TOOLS_DIR/aggregate_metrics.py" --run-dir "$OUT/tamper" --no-manifest; then
  echo "[selftest] FAIL aggregate (tamper)"; exit 1
fi
if ! "$PYTHON" "$SELFTEST_DIR/check_selftest.py" --run-dir "$OUT/tamper" --expect-tamper; then
  FAIL=1
fi

echo ""
echo "== 8/8 negative test: re-introduced trimesh winding bug =="
if ! "$PYTHON" "$SELFTEST_DIR/make_synth_run.py" --out "$OUT/tamper_mesh" --tamper-mesh; then
  echo "[selftest] FAIL make_synth_run --tamper-mesh"; exit 1
fi
if ! "$PYTHON" "$TOOLS_DIR/aggregate_metrics.py" --run-dir "$OUT/tamper_mesh" --no-manifest; then
  echo "[selftest] FAIL aggregate (tamper-mesh)"; exit 1
fi
if ! "$PYTHON" "$SELFTEST_DIR/check_selftest.py" --run-dir "$OUT/tamper_mesh" --expect-tamper-mesh; then
  FAIL=1
fi

echo ""
echo "=============================================================="
if [ "$FAIL" -ne 0 ]; then
  echo "SELFTEST FAILED (see above)"
  exit 1
fi
echo "SELFTEST PASSED"
exit 0

# T3 mirror-terrain evaluation tooling (`workSpace/tools/t3/`)

Scripts used for the T3 stage of the SWAP-lite reproduction: evaluate the R0 Go2/WMP
baseline (`go2_base512`, `model_10000.pt`) on left-right mirrored terrains.

Evaluation code (implemented by the Lead, committed separately):

- `WMP/legged_gym/utils/mirror_terrain.py` — evaluation-only terrain generator
  (`np.fliplr` mirror, pinned difficulty, SWAP-style unilateral lateral ramp).
- `WMP/legged_gym/scripts/eval_mirror.py` — 200 trials/cell evaluator
  (`summary.json`, `trials.jsonl`, `terrain.npz`, `run.log` per cell).

This directory contains the **orchestration / aggregation / plotting / video** pipeline:

| file | purpose |
|---|---|
| `cells.tsv` | single source of truth for the 36 matrix cells + 3 reproducibility re-runs |
| `run_matrix.sh` | serial runner for A组 / B组 / T1 re-check / repro re-runs / videos / GIFs |
| `aggregate_metrics.py` | scans `summary.json` → `metrics.csv`, mirror-pair fingerprint check, A/B tables, T1 consistency, `manifest.json`, `aggregate_report.md` |
| `make_figures.py` | matplotlib figures (A grouped bars, B vs tilt, T1 consistency, height-field mirror proof) |
| `make_gifs.sh` + `gif_maker.py` | viewer PNG frames → GIF (system `/usr/bin/python3` + Pillow 9.0.1, no new deps) |
| `selftest/` | synthetic-data end-to-end self-test (no GPU/WMP needed) |

## Quick start (on the 4090)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/ubuntu/B4_2025/SWAP/WMP
export PYTHONPATH=$PWD
export MPLBACKEND=Agg

# full matrix (~45-60 min, one GPU process at a time):
bash /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/run_matrix.sh all
# partial stages: a | b | t1 | repro | videos | gifs | none

# re-aggregate / re-plot an existing run without touching the GPU:
python /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/aggregate_metrics.py --run-dir <RUN_DIR>
python /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/make_figures.py --run-dir <RUN_DIR>
```

Default run dir: `workSpace/runs/s3-20261112-1620-mirror-eval/`.
Environment overrides for `run_matrix.sh`: `RUN_DIR`, `GPU`, `FORCE=1` (rerun cells that
already have `summary.json`), `ALLOW_BUSY=1`, `DRY_RUN=1`, `CELL_TIMEOUT`, `AUTO_AGG=0`,
`TRIALS_OVERRIDE` (debug only), `PYTHON`.

The runner aborts if another eval/train process or >4 GB of GPU memory is present, records
`nvidia-smi` snapshots (`logs/gpu_start.txt`, `logs/gpu_end.txt`), samples `logs/vram.csv`
every 2 s, writes one `progress.txt` line per cell and keeps going after a failed cell
(exit code 1 at the end if anything failed).

## Protocol (fixed for comparability with T1)

- 200 trials/cell, `TERRAIN_SEED=1`, fixed command `lin_vel_x=0.6`, deterministic action
  mean, `--headless`, checkpoint `logs/go2_amp_example/WMP/model_10000.pt` (`LOAD_RUN=WMP`).
- `A组` (WMP native): `tilt/stair/gap` × `diff 0.2/0.5/0.8` × normal/mirror = 18 cells.
- `B组` (SWAP-style unilateral lateral ramp): `ramp (diff 0.3)`, `ramp_stair (0.5)`,
  `ramp_gap (0.5)` × `TILT_DEG 5/10/15` × normal/mirror = 18 cells. The ramp is added
  before the `np.fliplr`, so normal and mirror are the exact "左高右低 / 右高左低" pair.
- Reproducibility: 3 A cells re-run into `repro_check/`, same `run_label`.
- T1 re-check: `eval_t1_repro.py` (`NEVAL=64`, `STEPS=1200`, slope/stair/gap × 3 runs).

## Mirror-pair validation

`aggregate_metrics.py` fills `mirror_pair_ok` per cell by checking

```
normal.height_field_fliplr_sha256 == mirror.height_field_sha256
mirror.height_field_fliplr_sha256 == normal.height_field_sha256
same height-field shape
```

so a "mirror" cell is provably the `np.fliplr` of its normal counterpart. It also records
`height_field_is_asymmetric` (normal sha != fliplr sha): WMP native `stair`/`tilt` height
fields are left-right symmetric by construction, so A组 mirror drop is expected to be ~0;
B组 ramps are asymmetric, which is the real OOD mirror test.

`checks.json` additionally contains the hard checks requested during the T3 review:

| check | what it proves |
|---|---|
| `code_versions` | sha256 + mtime of `eval_mirror.py` and `mirror_terrain.py`; whether both mtimes are before `run_config.started_at`; `winding_fix_present` |
| `symmetric_terrain_consistency` | for A tilt/stair cells: normal-vs-mirror difference vs binomial 95% CI (normal approximation); PASS if inside the CI, WARN <= 8 pp, ANOMALY > 8 pp |
| `trimesh_mirror_check` | custom `tilt`/`crawl` box meshes: signed volume + outward-triangle count for normal and mirror (a y-flip without reversing the triangle winding yields a NEGATIVE volume; this bug was found and fixed on 2026-11-12) |
| `bugfix_evidence` | buggy-vs-fixed mirror code evidence (old partial-cell success rates + signed volume -8.728880 -> +8.728880 m3, outward 0/48 -> 48/48) |
| `nondeterminism` | reproducibility re-runs: per-cell success deltas with a 5 pp same-distribution threshold; cross-run comparison against a second launcher with the same seed |
| `cells_ignored` | cells present in the run dir but not in `run_config.json` (e.g. partial cells from another launcher) are excluded from `metrics.csv` |

`a_group_pairs.csv` / `b_group_pairs.csv` carry `ci95_pp`, `within_95_ci`, `symmetric_verdict`;
`repro_check_compare.csv` carries `success_diff_abs` and `within_5pp`.


## Self-test (no GPU)

```bash
bash selftest/run_selftest.sh            # after: conda activate wmp  (needs matplotlib)
PYTHON=<python-with-PIL> bash selftest/run_selftest.sh   # figures skipped without matplotlib
```

Generates a synthetic run dir (A/B matrix + repro + T1 logs + video frames), runs
aggregation/figures/GIFs, asserts 52 CSV/table/manifest/figure/GIF properties, then
corrupts one fingerprint and asserts the pairing check reports `mirror_pair_ok=false`.

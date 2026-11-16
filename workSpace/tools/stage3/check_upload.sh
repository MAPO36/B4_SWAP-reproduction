#!/bin/bash
# Upload verification + static checks for the stage-3 LHR code (no GPU work).
set -uo pipefail
SWAP=/home/ubuntu/B4_2025/SWAP
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$SWAP/WMP" || exit 1

echo "=== files present ==="
for f in legged_gym/utils/lhr_terrain.py legged_gym/envs/go2/go2_amp_lhr_config.py train_lhr.py \
         "$SWAP/wmp_overlay/files/legged_gym/utils/lhr_terrain.py" \
         "$SWAP/wmp_overlay/files/legged_gym/envs/go2/go2_amp_lhr_config.py" \
         "$SWAP/wmp_overlay/files/train_lhr.py" "$SWAP/wmp_overlay/files/legged_gym/envs/__init__.py"; do
  [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
done

echo "=== overlay vs WMP sync (must be silent) ==="
sha256sum legged_gym/utils/lhr_terrain.py "$SWAP/wmp_overlay/files/legged_gym/utils/lhr_terrain.py" | awk '{print $1}' | uniq -c
sha256sum legged_gym/envs/go2/go2_amp_lhr_config.py "$SWAP/wmp_overlay/files/legged_gym/envs/go2/go2_amp_lhr_config.py" | awk '{print $1}' | uniq -c
sha256sum train_lhr.py "$SWAP/wmp_overlay/files/train_lhr.py" | awk '{print $1}' | uniq -c
sha256sum legged_gym/envs/__init__.py "$SWAP/wmp_overlay/files/legged_gym/envs/__init__.py" | awk '{print $1}' | uniq -c

echo "=== py_compile ==="
PYTHONPATH=$PWD python -m py_compile legged_gym/utils/lhr_terrain.py \
  legged_gym/envs/go2/go2_amp_lhr_config.py train_lhr.py && echo PY_COMPILE_OK

echo "=== task registration import test (no sim created) ==="
PYTHONPATH=$PWD MPLBACKEND=Agg python - <<'PY'
import isaacgym  # noqa
from legged_gym.envs import *  # noqa
from legged_gym.utils import task_registry
from legged_gym.utils.lhr_terrain import LHRTerrain
env_cfg, train_cfg = task_registry.get_cfgs('go2_amp_lhr')
print('REGISTERED go2_amp_lhr')
print('proportions', list(env_cfg.terrain.terrain_proportions))
print('num_rows/num_cols', env_cfg.terrain.num_rows, env_cfg.terrain.num_cols)
print('experiment/run', train_cfg.runner.experiment_name, train_cfg.runner.run_name)
print('LHRTerrain attrs', LHRTerrain.LHR_TILT_DEG, LHRTerrain.MIRROR)
print('IMPORT_OK')
PY

echo "=== WMP tracked modifications (only envs/__init__.py expected) ==="
git -C "$SWAP/WMP" status --short | grep -v '^??' || echo "none tracked"
echo "=== SWAP git status ==="
cd "$SWAP" && git status --short
echo "=== DONE ==="

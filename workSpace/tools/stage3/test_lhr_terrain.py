"""CPU invariant tests for the stage-3 LHR terrain (`legged_gym.utils.lhr_terrain`).

No simulator is created: builds height fields / meshes exactly like the training and
evaluation entry points do. Run from the WMP repo root with the conda `wmp` env:

  cd <SWAP>/WMP && PYTHONPATH=$PWD python /tmp/test_lhr_terrain.py
"""
import json
import sys

import isaacgym  # noqa: E402  (must precede torch / legged_gym.utils)
import numpy as np  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import task_registry  # noqa: E402
from legged_gym.utils.lhr_terrain import LHRTerrain  # noqa: E402


FAMILIES = {
    "box": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # slot 6 = climb platforms
    "stair": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # slot 3 = stairs up
    "gap": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],    # slot 5 = gap
}
TRAIN_PROPS = [0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0]
FAILURES = []


class TerrainCfg:
    mesh_type = "trimesh"
    terrain_length = 8.0
    terrain_width = 8.0
    horizontal_scale = 0.1
    vertical_scale = 0.005
    border_size = 25
    curriculum = False
    selected = False
    terrain_kwargs = None
    slope_treshold = 0.75
    num_rows = 1
    num_cols = 1
    terrain_proportions = FAMILIES["box"]


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (("  " + str(detail)) if detail else ""))
    if not ok:
        FAILURES.append(name)


def build(num_rows=1, num_cols=1, curriculum=False, props=None, mirror=False,
          difficulty=None, seed=1, tilt=10.0):
    TerrainCfg.num_rows = num_rows
    TerrainCfg.num_cols = num_cols
    TerrainCfg.curriculum = curriculum
    TerrainCfg.terrain_proportions = list(props if props is not None else FAMILIES["box"])
    LHRTerrain.LHR_TILT_DEG = tilt
    LHRTerrain.MIRROR = mirror
    LHRTerrain.FIXED_DIFFICULTY = difficulty
    LHRTerrain.TERRAIN_SEED = seed
    LHRTerrain.FAMILY = None
    return LHRTerrain(TerrainCfg, 16)


def recompute_env_origin_z(terra):
    hs = terra.cfg.horizontal_scale
    vs = terra.cfg.vertical_scale
    x1 = int((terra.env_length / 2.0 - 1) / hs)
    x2 = int((terra.env_length / 2.0 + 1) / hs)
    y1 = int((terra.env_width / 2.0 - 1) / hs)
    y2 = int((terra.env_width / 2.0 + 1) / hs)
    out = np.zeros((terra.cfg.num_rows, terra.cfg.num_cols))
    for j in range(terra.cfg.num_cols):
        for i in range(terra.cfg.num_rows):
            sx = terra.border + i * terra.length_per_env_pixels
            sy = terra.border + j * terra.width_per_env_pixels
            patch = terra.height_field_raw[sx + x1:sx + x2, sy + y1:sy + y2]
            out[i, j] = np.max(patch) * vs
    return out


def main():
    print("=== 0) task registry: go2_amp_lhr config ===")
    env_cfg, train_cfg = task_registry.get_cfgs("go2_amp_lhr")
    props = list(env_cfg.terrain.terrain_proportions)
    check("go2_amp_lhr registered", env_cfg is not None)
    check("training proportions: 3 families 1/3 each",
          abs(sum(props) - 1.0) < 1e-9 and
          abs(props[3] - 1 / 3) < 1e-9 and abs(props[5] - 1 / 3) < 1e-9 and
          abs(props[6] - 1 / 3) < 1e-9,
          props)
    check("curriculum on, 10 rows", bool(env_cfg.terrain.curriculum) and env_cfg.terrain.num_rows == 10)
    check("experiment/run name separated from R0-pilot",
          train_cfg.runner.experiment_name == "go2_lhr_example" and train_cfg.runner.run_name == "R0_LHR",
          (train_cfg.runner.experiment_name, train_cfg.runner.run_name))
    check("training terrain proportions == test TRAIN_PROPS",
          np.allclose(props, TRAIN_PROPS), props)

    print("\n=== 1) evaluation mode: LHR direction + RHL mirror, per family ===")
    for fam, fam_props in FAMILIES.items():
        normal = build(props=fam_props, difficulty=0.5, mirror=False, seed=1)
        stats = normal.lhr_stats
        check("eval.%s LHR direction consistent" % fam,
              stats["direction_consistent"] and stats["left_higher_count"] == 1, stats["roll_deg_median"])
        check("eval.%s measured roll == 10 deg" % fam, abs(stats["roll_deg_median"] - 10.0) < 0.5,
              "%.3f" % stats["roll_deg_median"])
        zo = recompute_env_origin_z(normal)
        check("eval.%s env_origins.z == center-window max" % fam,
              np.allclose(np.asarray(normal.env_origins)[..., 2], zo, atol=1e-6))
        mirror = build(props=fam_props, difficulty=0.5, mirror=True, seed=1)
        check("eval.%s RHL hf == fliplr(LHR hf)" % fam,
              np.array_equal(mirror.height_field_raw, np.fliplr(normal.height_field_raw)))
        check("eval.%s fingerprint cross-check" % fam,
              normal.terrain_fingerprint()["height_field_fliplr_sha256"]
              == mirror.terrain_fingerprint()["height_field_sha256"])
        check("eval.%s RHL == right-high-left-low (measured roll < 0)" % fam,
              mirror.lhr_stats["roll_deg_median"] < -9.5,
              "%.3f" % mirror.lhr_stats["roll_deg_median"])
        check("eval.%s env_origins.z symmetric LHR vs RHL" % fam,
              np.allclose(np.asarray(normal.env_origins)[..., 2],
                          np.asarray(mirror.env_origins)[..., 2], atol=1e-6))
        check("eval.%s double flip == identity" % fam,
              np.array_equal(np.fliplr(np.fliplr(normal.height_field_raw)), normal.height_field_raw))

    print("\n=== 2) fixed direction across terrain seeds (never random left/right) ===")
    for seed in (1, 2, 3, 7):
        t = build(num_rows=2, num_cols=3, curriculum=True, props=TRAIN_PROPS, difficulty=None, seed=seed)
        s = t.lhr_stats
        check("seed=%d curriculum terrain direction consistent (6 patches)" % seed,
              s["direction_consistent"] and s["left_higher_count"] == s["n_patches"], s["roll_deg_median"])
        check("seed=%d roll within 10+-0.5 deg" % seed, abs(s["roll_deg_median"] - 10.0) < 0.5,
              "%.3f" % s["roll_deg_median"])

    print("\n=== 3) training-mode (curriculum) full direction & spawn heights ===")
    t = build(num_rows=10, num_cols=6, curriculum=True, props=TRAIN_PROPS, difficulty=None, seed=1)
    s = t.lhr_stats
    check("curriculum 60 patches all left-higher", s["direction_consistent"] and s["left_higher_count"] == 60,
          (s["left_higher_count"], s["n_patches"]))
    check("curriculum roll range tight", abs(s["roll_deg_min"] - 10.0) < 1.0 and abs(s["roll_deg_max"] - 10.0) < 1.0,
          (s["roll_deg_min"], s["roll_deg_max"]))
    zo = recompute_env_origin_z(t)
    check("curriculum env_origins.z == center-window max",
          np.allclose(np.asarray(t.env_origins)[..., 2], zo, atol=1e-6))
    check("curriculum env_origins x,y unchanged from stock convention",
          np.allclose(np.asarray(t.env_origins)[0, 0, :2], [4.0, 4.0], atol=1e-6))

    print("\n=== 4) fail-fast guards ===")
    try:
        build(num_rows=1, num_cols=2, props=TRAIN_PROPS, mirror=True, difficulty=0.5, seed=1)
        check("mirror with num_cols>1 rejected", False)
    except RuntimeError:
        check("mirror with num_cols>1 rejected", True)
    LHRTerrain.LHR_TILT_DEG = 0.0
    t0 = build(num_rows=1, num_cols=1, props=FAMILIES["box"], difficulty=0.5, mirror=False, seed=1, tilt=0.0)
    check("tilt_deg=0 gives (near) flat ground (no ramp)",
          abs(t0.lhr_stats["roll_deg_median"]) < 0.2, "%.4f" % t0.lhr_stats["roll_deg_median"])
    LHRTerrain.LHR_TILT_DEG = 10.0

    print()
    if FAILURES:
        print("RESULT: FAIL (%d): %s" % (len(FAILURES), FAILURES))
        sys.exit(1)
    print("RESULT: ALL LHR TERRAIN CHECKS PASSED")


if __name__ == "__main__":
    main()

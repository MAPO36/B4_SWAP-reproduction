"""R0-LHR training entry point (SWAP-lite stage 3, 11-16 plan).

Same wrapper idea as `train_capped.py` (WMP's train.py hardcodes max_iterations=100000
and ignores --max_iterations), plus the stage-3 LHR terrain:

  * monkeypatches `legged_robot.Terrain` with `legged_gym.utils.lhr_terrain.LHRTerrain`,
    so the task `go2_amp_lhr` is trained on the fixed LHR (left-high-right-low) terrain;
  * BEFORE training: prints the measured LHR direction/angle for every patch, saves the
    terrain height field + preview PNG + report JSON, and ABORTS if the direction is not
    consistently LHR (fail fast before a 24 h run);
  * trains from scratch (never loads the R0-pilot checkpoint).

Run from the WMP repo root:
  MAX_ITERS=1000 MAX_ENVS=512 RUN_NAME=R0_LHR_smoke \
    python train_lhr.py --task=go2_amp_lhr --headless --num_envs=512 --sim_device=cuda:0

Environment variables:
  WMP_ROOT          default = current working directory
  MAX_ITERS         default 5
  MAX_ENVS          default 0 (no cap)
  RUN_NAME          default R0_LHR
  SAVE_INTERVAL     default 500 if MAX_ITERS > 1000 else 250
  LHR_TILT_DEG      default 10.0  (fixed lateral roll, left side higher)
  TERRAIN_REPORT_DIR default /tmp/lhr_terrain_report
"""
import json
import os
import sys

WMP_ROOT = os.environ.get("WMP_ROOT", os.getcwd())
if not os.path.isdir(os.path.join(WMP_ROOT, "legged_gym")):
    sys.exit("[train_lhr] run from the WMP repo root, or set WMP_ROOT to it")
sys.path.insert(0, WMP_ROOT)
os.chdir(WMP_ROOT)

import isaacgym  # noqa: E402  (must precede torch)

# Local-only compatibility shim (no-op on the lab machine).
import numpy.typing as npt  # noqa: E402

if not hasattr(npt, "NDArray"):
    class _AnyPlaceholder:
        def __getitem__(self, item):
            return self

    npt.NDArray = _AnyPlaceholder()

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

import legged_gym.envs.base.legged_robot as _legged_robot_module  # noqa: E402
from legged_gym.utils.lhr_terrain import LHRTerrain  # noqa: E402

import torch  # noqa: E402

MAX_ITERS = int(os.environ.get("MAX_ITERS", "5"))
MAX_ENVS = int(os.environ.get("MAX_ENVS", "0"))
RUN_NAME = os.environ.get("RUN_NAME", "R0_LHR")
SAVE_INTERVAL = int(
    os.environ.get("SAVE_INTERVAL", "500" if MAX_ITERS > 1000 else "250")
)
LHR_TILT_DEG = float(os.environ.get("LHR_TILT_DEG", "10.0"))
REPORT_DIR = os.environ.get("TERRAIN_REPORT_DIR", "/tmp/lhr_terrain_report")


def _vram_snapshot():
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return out.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return "unavailable (%r)" % (exc,)


def main():
    args = get_args()
    args.rl_device = args.sim_device
    if args.num_envs is not None and MAX_ENVS > 0:
        args.num_envs = min(args.num_envs, MAX_ENVS)
    print(
        "[train_lhr] task=%s num_envs=%s sim_device=%s headless=%s max_iters=%s "
        "run=%s lhr_tilt_deg=%.2f vram(used,free,util)=%s"
        % (args.task, args.num_envs, args.sim_device, args.headless, MAX_ITERS,
           RUN_NAME, LHR_TILT_DEG, _vram_snapshot()),
        flush=True,
    )

    # WMPRunner._build_world_model() locates dreamer/configs.yaml via sys.argv[0] and
    # re-parses sys.argv with a stricter parser; point argv[0] back at upstream train.py.
    sys.argv = [os.path.join(WMP_ROOT, "legged_gym/scripts/train.py")]

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # camera_num_envs must be <= crawl_end_idx = ceil(n * sum(proportions[:9]))
    import math

    _n = args.num_envs or env_cfg.env.num_envs
    _limit = math.ceil(_n * sum(env_cfg.terrain.terrain_proportions[:9]))
    if getattr(env_cfg, "depth", None) is not None and env_cfg.depth.camera_num_envs > _limit:
        print(
            "[train_lhr] camera_num_envs %d -> %d (num_envs=%d, crawl_end=%d)"
            % (env_cfg.depth.camera_num_envs, _limit, _n, _limit),
            flush=True,
        )
        env_cfg.depth.camera_num_envs = _limit
    train_cfg.runner.run_name = RUN_NAME
    train_cfg.runner.max_iterations = MAX_ITERS
    train_cfg.runner.save_interval = SAVE_INTERVAL

    # ---- stage 3: patch the terrain generator BEFORE the env is created -----------------
    LHRTerrain.LHR_TILT_DEG = LHR_TILT_DEG
    LHRTerrain.MIRROR = False
    LHRTerrain.FIXED_DIFFICULTY = None
    LHRTerrain.TERRAIN_SEED = int(os.environ.get("TERRAIN_SEED", "1"))
    _legged_robot_module.Terrain = LHRTerrain

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(
        "[train_lhr] env created: num_envs=%d num_obs=%d num_actions=%d max_ep_len=%d "
        "terrain_class=%s" % (env.num_envs, env.num_obs, env.num_actions,
                              int(env.max_episode_length), type(env.terrain).__name__),
        flush=True,
    )

    # ---- stage 3 gate: the training terrain MUST be fixed LHR ---------------------------
    if not isinstance(env.terrain, LHRTerrain):
        raise RuntimeError("Terrain patch failed: env.terrain is %r" % type(env.terrain))
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = env.terrain.save_lhr_visuals(REPORT_DIR, prefix="train_%s" % RUN_NAME)
    stats = report["fingerprint"]["lhr_check"]
    print("[train_lhr] LHR direction report: " + json.dumps(stats, ensure_ascii=False), flush=True)
    if not stats["direction_consistent"]:
        raise RuntimeError(
            "LHR direction check FAILED: not every patch is left-high-right-low "
            "(%d/%d patches left higher, roll range %.2f..%.2f deg). Aborting before training."
            % (stats["left_higher_count"], stats["n_patches"],
               stats["roll_deg_min"], stats["roll_deg_max"])
        )
    print(
        "[train_lhr] LHR gate PASS: %d/%d patches left-higher, roll median %.2f deg "
        "(requested %.1f), fingerprint=%s"
        % (stats["left_higher_count"], stats["n_patches"], stats["roll_deg_median"],
           LHR_TILT_DEG, report["fingerprint"]["height_field_sha256"][:16]),
        flush=True,
    )

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    print("[train_lhr] runner created; learning %d iterations from scratch" % MAX_ITERS, flush=True)
    runner.learn(num_learning_iterations=MAX_ITERS, init_at_random_ep_len=True)
    print("[train_lhr] PASS: %d iterations completed (run=%s)" % (MAX_ITERS, RUN_NAME), flush=True)


if __name__ == "__main__":
    main()

"""Controlled-iteration entry point for WMP training.

Why not use WMP/legged_gym/scripts/train.py directly:
train.py hardcodes train_cfg.runner.max_iterations = 100000 and ignores
--max_iterations. This wrapper mirrors train.py but caps iterations via
MAX_ITERS and, optionally, environment count via MAX_ENVS.

Run from the WMP repo root:
  MAX_ITERS=5 MAX_ENVS=16 python train_capped.py \
      --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0

Environment variables:
  WMP_ROOT       default = current working directory
  MAX_ITERS      default 5
  MAX_ENVS       default 0 (no cap); use 16 for local 5060 smoke
  RUN_NAME       default WMP_CAPPED
  SAVE_INTERVAL  default 1000 if MAX_ITERS > 1000 else 1000000
"""

import os
import sys

WMP_ROOT = os.environ.get("WMP_ROOT", os.getcwd())
if not os.path.isdir(os.path.join(WMP_ROOT, "legged_gym")):
    sys.exit(
        "[train_capped] run from the WMP repo root, or set WMP_ROOT to it"
    )
sys.path.insert(0, WMP_ROOT)
os.chdir(WMP_ROOT)

import isaacgym  # noqa: E402  (must precede torch)

# Local-only compatibility shim: the 5060 laptop's unitree-rl env has numpy
# 1.20 (no numpy.typing.NDArray) while Pillow expects it. WMP pins numpy
# 1.22.3; this is a no-op on the lab machine.
import numpy.typing as npt  # noqa: E402

if not hasattr(npt, "NDArray"):
    class _AnyPlaceholder:
        def __getitem__(self, item):
            return self

    npt.NDArray = _AnyPlaceholder()

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
import torch  # noqa: E402

MAX_ITERS = int(os.environ.get("MAX_ITERS", "5"))
MAX_ENVS = int(os.environ.get("MAX_ENVS", "0"))
RUN_NAME = os.environ.get("RUN_NAME", "WMP_CAPPED")
SAVE_INTERVAL = int(
    os.environ.get(
        "SAVE_INTERVAL", "1000" if MAX_ITERS > 1000 else "1000000"
    )
)


def main():
    args = get_args()
    args.rl_device = args.sim_device
    if args.num_envs is not None and MAX_ENVS > 0:
        args.num_envs = min(args.num_envs, MAX_ENVS)
    print(
        f"[train_capped] task={args.task} num_envs={args.num_envs} "
        f"sim_device={args.sim_device} headless={args.headless} "
        f"max_iters={MAX_ITERS}",
        flush=True,
    )

    # WMPRunner._build_world_model() locates dreamer/configs.yaml via
    # sys.argv[0].parent.parent.parent and re-parses sys.argv with a separate
    # parser that rejects legged_gym-only flags. Point argv[0] back at upstream
    # train.py and drop the already-consumed CLI flags.
    sys.argv = [os.path.join(WMP_ROOT, "legged_gym/scripts/train.py")]

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.run_name = RUN_NAME
    train_cfg.runner.max_iterations = MAX_ITERS
    train_cfg.runner.save_interval = SAVE_INTERVAL

    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    print(
        f"[train_capped] env created: num_envs={env.num_envs} "
        f"num_obs={env.num_obs} num_actions={env.num_actions}",
        flush=True,
    )

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    print(
        f"[train_capped] runner created; learning {MAX_ITERS} iterations",
        flush=True,
    )
    runner.learn(
        num_learning_iterations=MAX_ITERS, init_at_random_ep_len=True
    )
    print(
        f"[train_capped] PASS: {MAX_ITERS} iterations completed", flush=True
    )


if __name__ == "__main__":
    main()

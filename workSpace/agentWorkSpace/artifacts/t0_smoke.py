"""T0 local smoke: WMP a1_amp, hard-capped at 2 iterations.

Why not use WMP/legged_gym/scripts/train.py directly:
train.py hardcodes train_cfg.runner.max_iterations = 100000, ignoring --max_iterations,
which would violate the local 2-iteration red line. This script mirrors train.py but
caps iterations at 2. Note: WMPRunner.learn always saves a checkpoint at the end
(wmp_runner.py:382), so a 2-iteration ckpt is written if the run gets that far.

Run from the WMP repo root:
  python /home/l/workSpace/SWAP/workSpace/agentWorkSpace/artifacts/t0_smoke.py \
      --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0

Iteration cap can be overridden with T0_MAX_ITERS (default 2; lab uses 5).
"""

import os
import sys

# artifacts/ -> agentWorkSpace/ -> workSpace/ -> SWAP/
SWAP_ROOT = os.environ.get(
    "SWAP_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
WMP_ROOT = os.environ.get("WMP_ROOT", os.path.join(SWAP_ROOT, "WMP"))
sys.path.insert(0, WMP_ROOT)
os.chdir(WMP_ROOT)

import isaacgym  # noqa: E402  (must precede torch)

# Local-only compatibility shim: unitree-rl has numpy 1.20 (no numpy.typing.NDArray,
# added in 1.21) while its Pillow expects it. WMP pins numpy 1.22.3. We cannot install
# packages on this machine, so provide a no-op subscriptable placeholder for typing only.
import numpy.typing as npt  # noqa: E402

if not hasattr(npt, "NDArray"):
    class _AnyPlaceholder:
        def __getitem__(self, item):
            return self

    npt.NDArray = _AnyPlaceholder()

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
import torch  # noqa: E402

MAX_ITERS = int(os.environ.get("T0_MAX_ITERS", "2"))


def main():
    args = get_args()
    args.rl_device = args.sim_device
    # Smoke-only hard cap (local red line: <=16 envs); lab T0 smoke uses the same cap.
    if args.num_envs is None:
        args.num_envs = 16
    args.num_envs = min(args.num_envs, 16)
    print(f"[t0_smoke] task={args.task} num_envs={args.num_envs} "
          f"sim_device={args.sim_device} headless={args.headless}", flush=True)

    # WMPRunner._build_world_model() (wmp_runner.py:146,168) locates
    # dreamer/configs.yaml via sys.argv[0].parent.parent.parent and re-parses
    # sys.argv with a *separate* parser that rejects legged_gym-only flags such as
    # --num_envs/--max_iterations. Point argv[0] back at upstream train.py and drop
    # the already-consumed CLI flags so that parser only sees config.yaml defaults.
    sys.argv = [os.path.join(WMP_ROOT, "legged_gym/scripts/train.py")]

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.run_name = "WMP_T0_SMOKE"
    train_cfg.runner.max_iterations = MAX_ITERS
    train_cfg.runner.save_interval = 1_000_000  # no mid-run saves (end-of-learn save remains)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(f"[t0_smoke] env created: num_envs={env.num_envs} "
          f"num_obs={env.num_obs} num_actions={env.num_actions}", flush=True)

    runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    print(f"[t0_smoke] runner created; learning {MAX_ITERS} iterations", flush=True)
    runner.learn(num_learning_iterations=MAX_ITERS, init_at_random_ep_len=True)
    print(f"[t0_smoke] PASS: {MAX_ITERS} iterations completed", flush=True)


if __name__ == "__main__":
    main()

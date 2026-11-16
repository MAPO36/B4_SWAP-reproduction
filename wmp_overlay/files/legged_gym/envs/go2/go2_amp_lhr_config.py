"""Stage 3 task config: WMP + Go2 baseline on a fixed LHR (left-high-right-low) terrain.

This is the task used for **R0-LHR** (the formal four-group baseline from the 11-16 plan):

  * same robot / observation / reward / AMP / network configuration as `go2_amp` (R0-pilot),
  * only the training terrain changes: three height-field families, fixed LHR lateral ramp:
        platform box  -> WMP `climb` generator   (terrain_proportions slot 6)
        stairs up     -> WMP stairs-up generator (slot 3)
        gap           -> WMP `gap` generator     (slot 5)
    one third each, stock 10-row curriculum, direction fixed (no random left/right flip),
  * the actual ramp is applied by `legged_gym.utils.lhr_terrain.LHRTerrain`
    (`train_lhr.py` monkeypatches `legged_robot.Terrain`).

The evaluation script `eval_lhr.py` reuses the same class with num_cols=1 and a pinned
difficulty to measure LHR (in-distribution) vs RHL (mirror) success rates.
"""
from legged_gym.envs.go2.go2_amp_config import GO2AMPCfg, GO2AMPCfgPPO


class GO2AMPLHRCfg(GO2AMPCfg):
    class terrain(GO2AMPCfg.terrain):
        # [wave, slope, stairs_down, stairs_up, discrete, gap, pit/climb, tilt, crawl, rough_flat]
        # stage-3 LHR families: stairs-up (slot 3) / gap (slot 5) / platform box = climb (slot 6)
        terrain_proportions = [
            0.0,
            0.0,
            0.0,
            1.0 / 3.0,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.0,
            0.0,
            0.0,
        ]
        curriculum = True
        selected = False


class GO2AMPLHRCfgPPO(GO2AMPCfgPPO):
    class runner(GO2AMPCfgPPO.runner):
        experiment_name = 'go2_lhr_example'
        run_name = 'R0_LHR'

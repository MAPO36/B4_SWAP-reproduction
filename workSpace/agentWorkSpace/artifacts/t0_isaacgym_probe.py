"""Minimal Isaac Gym probe: can this GPU run a PhysX scene at all?

Bypasses WMP/legged_gym dependency issues to isolate GPU compatibility
(Isaac Gym Preview 4 on RTX 5060 / sm_120).
"""

import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()

params = gymapi.SimParams()
params.up_axis = gymapi.UP_AXIS_Z
params.dt = 1.0 / 60.0
params.substeps = 1
params.use_gpu_pipeline = True
params.physx.use_gpu = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
if sim is None:
    raise RuntimeError("gym.create_sim returned None")

plane = gymapi.PlaneParams()
plane.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset = gym.create_box(sim, 0.2, 0.2, 0.2, asset_options)

lower = gymapi.Vec3(-1, -1, 0)
upper = gymapi.Vec3(1, 1, 1)
num_envs = 16
for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.5 + 0.01 * i)
    gym.create_actor(env, asset, pose, f"box_{i}", i, 1)

gym.prepare_sim(sim)
for _ in range(10):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

print(f"PASS: created sim with {num_envs} envs and stepped 10x", flush=True)

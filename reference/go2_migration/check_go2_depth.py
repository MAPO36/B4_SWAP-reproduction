#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render one Go2 depth frame and save it, so you can check the camera is not
looking at the inside of the robot's head.

Run from the WMP repository root (the folder that contains `legged_gym/`):

    python check_go2_depth.py --task=go2_amp --headless

Outputs:
    go2_depth.png   -- 64x64 depth image, near = dark, far = bright
    terminal stats  -- near fraction should be < 0.10

If `near fraction` is large, or the PNG is mostly the robot's own body, adjust
`class depth` in go2_amp_config.py (position z / y_angle) and run again.
"""

import os
import sys

# WMP scripts are normally run from the repository root.
sys.path.insert(0, os.getcwd())

import isaacgym  # noqa: F401  (must be imported before torch)
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403  (registers the tasks)
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    args.task = "go2_amp"
    args.headless = True

    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # One robot, one camera, flat ground: the simplest possible picture.
    env_cfg.env.num_envs = 1
    env_cfg.depth.camera_num_envs = 1
    env_cfg.depth.use_camera = True
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.rewards.scales.feet_edge = 0
    env_cfg.rewards.scales.feet_stumble = 0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    # update_interval is 5, so 10 steps guarantees at least two fresh frames.
    for _ in range(10):
        env.step(actions)

    if not hasattr(env, "depth_buffer"):
        raise SystemExit("depth_buffer was not created -- is depth.use_camera true?")

    depth = env.depth_buffer[0, -1].detach().cpu().numpy()
    far = env.cfg.depth.far_clip

    # normalize_depth_image stored: norm = distance / far - 0.5
    meters = (depth + 0.5) * far
    near_fraction = float(np.mean(meters < 0.10))

    print("")
    print("depth shape:", depth.shape)
    print("depth range (m): %.2f .. %.2f" % (meters.min(), meters.max()))
    print("mean depth (m): %.2f" % meters.mean())
    print("near fraction (<0.10 m): %.3f" % near_fraction)
    print("bottom-row mean (m): %.2f" % meters[-8:].mean())
    print("top-row mean (m):    %.2f" % meters[:8].mean())

    img = np.clip((depth + 0.5) * 255.0, 0, 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite("go2_depth.png", img)
        print("saved: go2_depth.png")
    except Exception as exc:  # cv2 should exist because WMP uses it
        np.save("go2_depth.npy", depth)
        print("cv2 unavailable (%s), saved go2_depth.npy instead" % exc)

    print("")
    if near_fraction > 0.30:
        print("FAIL: more than 30% of the image is closer than 10 cm.")
        print("      The camera is probably inside the Go2 head.")
        print("      Raise depth.position z, or move x back, then rerun.")
    elif near_fraction > 0.10:
        print("WARN: 10-30% of the image is very close. Open go2_depth.png;")
        print("      if the bottom half is the robot body, raise the camera.")
    else:
        print("OK: the camera is clear of the robot body.")


if __name__ == "__main__":
    main()

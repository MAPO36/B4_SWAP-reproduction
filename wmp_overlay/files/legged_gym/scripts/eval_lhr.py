#!/usr/bin/env python
"""Stage-3 T3 — LHR / RHL evaluation for a SWAP-lite checkpoint (R0-LHR, later R1/R2/R3).

Derived from the T3 `eval_mirror.py` (same policy/world-model stepping loop, same
one-episode-per-env success accounting), but for the 11-16 plan's stage-3 protocol:

  * terrain generator: `legged_gym.utils.lhr_terrain.LHRTerrain` (fixed 10 deg LHR ramp)
  * terrain families: box (WMP climb platforms) / stair (stairs up) / gap
  * difficulty levels: LEVEL 1/2/3 -> pinned difficulty 0.2/0.5/0.8
  * evaluation directions:
        LHR = left-high-right-low  (training distribution, in-distribution)
        RHL = right-high-left-low  (np.fliplr mirror, out-of-distribution)
  * every output carries the mandatory 6-item annotation
    (condition / training terrain / model / eval terrain / terrain+level / trials)

Configured through environment variables (Isaac Gym's get_args() stays untouched):

  FAMILY         box|stair|gap                                  (default gap)
  LEVEL          1|2|3  -> difficulty 0.2/0.5/0.8              (default 2)
  DIRECTION      LHR|RHL                                        (default LHR)
  TRIALS         parallel envs == trials                        (default 200)
  TERRAIN_SEED   seed for the terrain draw                      (default 1)
  CONDITION      label of the model, e.g. R0-LHR / R1 / R2 / R3 (default R0-LHR)
  LOAD_EXPERIMENT  log experiment dir                           (default go2_lhr_example)
  LOAD_RUN         run name under logs/<experiment>             (default R0_LHR)
  CKPT_ITER      checkpoint iteration, -1 = latest              (default -1)
  OUT_DIR        output dir (summary.json / trials.jsonl / terrain.npz)
  RUN_LABEL      short label stored in summary.json
  STEPS          step cap, 0 => max_episode_length + 3
  RECORD_FRAMES  0/1 record viewer frames (forces non-headless)
  FRAME_DIR / FRAME_EVERY / CAMERA_MODE / CAMERA_ENV_IDX  as in eval_mirror.py

Example:
  FAMILY=box LEVEL=2 DIRECTION=LHR TRIALS=200 OUT_DIR=... \\
    python legged_gym/scripts/eval_lhr.py --task=go2_amp_lhr --headless --sim_device=cuda:0
"""

import hashlib
import inspect
import json
import os
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym  # noqa: E402  (must precede torch)
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# [wave, slope, stairs_down, stairs_up, discrete, gap, pit/climb, tilt, crawl, rough_flat]
FAMILY_PROPS = {
    "box": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # climb = raised platforms
    "stair": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # stairs up
    "gap": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
}
FAMILY_LABEL = {
    "box": "tilted box (WMP climb platforms)",
    "stair": "tilted stairs (up)",
    "gap": "tilted gap",
}
LEVEL_DIFF = {1: 0.2, 2: 0.5, 3: 0.8}
DIRECTION_LABEL = {
    "LHR": "LHR (left-high-right-low, training distribution)",
    "RHL": "RHL (right-high-left-low, mirrored / OOD)",
}
CAUSE_TIMEOUT = "success_timeout"


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_flag(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "on")


def _classify_failure(env, eid):
    try:
        if bool(env.time_out_buf[eid].item()):
            return CAUSE_TIMEOUT
        term = env.termination_contact_indices
        if term.numel() > 0:
            contact = torch.norm(env.contact_forces[eid, term, :], dim=-1)
            if bool(torch.any(contact > 1.0).item()):
                return "fail_base_contact"
        if bool(env.fall[eid].item()):
            if float(env.root_states[eid, 9].item()) < -3.0:
                return "fail_fall_velocity"
            return "fail_flipped"
        if bool(env.vel_violate[eid].item()):
            return "fail_vel_violate"
    except Exception as exc:  # noqa: BLE001
        return "fail_unclassified:%s" % type(exc).__name__
    return "fail_unclassified"


def _sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def main():
    args = get_args()
    args.rl_device = args.sim_device

    family = _env_str("FAMILY", "gap")
    if family not in FAMILY_PROPS:
        raise SystemExit("unknown FAMILY=%r, choose from %s" % (family, sorted(FAMILY_PROPS)))
    level = _env_int("LEVEL", 2)
    if level not in LEVEL_DIFF:
        raise SystemExit("LEVEL must be 1/2/3, got %r" % (level,))
    direction = _env_str("DIRECTION", "LHR").upper()
    if direction not in ("LHR", "RHL"):
        raise SystemExit("DIRECTION must be LHR or RHL, got %r" % (direction,))
    mirror = direction == "RHL"
    diff = LEVEL_DIFF[level]
    condition = _env_str("CONDITION", "R0-LHR")
    trials = _env_int("TRIALS", 200)
    terrain_seed = _env_int("TERRAIN_SEED", 1)
    load_experiment = _env_str("LOAD_EXPERIMENT", "go2_lhr_example")
    load_run = _env_str("LOAD_RUN", "R0_LHR")
    ckpt_iter = _env_int("CKPT_ITER", -1)
    record = _env_flag("RECORD_FRAMES", False)
    frame_every = max(1, _env_int("FRAME_EVERY", 2))
    camera_mode = _env_str("CAMERA_MODE", "follow")
    camera_env_idx = _env_int("CAMERA_ENV_IDX", 0)
    lhr_tilt_deg = float(_env_str("LHR_TILT_DEG", "10.0"))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _env_str(
        "OUT_DIR",
        os.path.join("/tmp", "eval_lhr", "%s_L%d_%s_%s" % (family, level, direction, stamp)),
    )
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = _env_str("FRAME_DIR", os.path.join(out_dir, "frames"))
    if record:
        os.makedirs(frame_dir, exist_ok=True)
        args.headless = False
    label = _env_str("RUN_LABEL", "%s_%s_L%d_%s" % (condition, family, level, direction))

    t_start = time.time()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ---- evaluation environment (same overrides as the T1/T3 evaluations) --------------
    env_cfg.env.num_envs = trials
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_proportions = FAMILY_PROPS[family]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]
    train_cfg.runner.amp_num_preload_transitions = 1
    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]
    env_cfg.depth.use_camera = True
    if getattr(env_cfg, "depth", None) is not None:
        env_cfg.depth.camera_num_envs = min(int(env_cfg.depth.camera_num_envs), trials)

    # ---- patch Terrain with the stage-3 LHR generator ---------------------------------
    import legged_gym.envs.base.legged_robot as _legged_robot  # noqa: E402
    from legged_gym.utils.lhr_terrain import LHRTerrain  # noqa: E402

    LHRTerrain.LHR_TILT_DEG = lhr_tilt_deg
    LHRTerrain.MIRROR = mirror
    LHRTerrain.FIXED_DIFFICULTY = diff
    LHRTerrain.TERRAIN_SEED = terrain_seed
    LHRTerrain.FAMILY = family
    _legged_robot.Terrain = LHRTerrain

    print(
        "[CELL] label=%s condition=%s family=%s level=%d diff=%.2f direction=%s "
        "mirror=%s trials=%d terrain_seed=%d headless=%s"
        % (label, condition, family, level, diff, direction, mirror, trials, terrain_seed, args.headless),
        flush=True,
    )

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if record:
        env.debug_viz = False
        env.lookat_id = min(int(env.lookat_id), env.num_envs - 1)
    print(
        "[CELL] env created: num_envs=%d num_obs=%d num_actions=%d max_ep_len=%d depth_envs=%d"
        % (env.num_envs, env.num_obs, env.num_actions, int(env.max_episode_length), len(env.depth_index)),
        flush=True,
    )

    fingerprint = {}
    if hasattr(env.terrain, "terrain_fingerprint"):
        fingerprint = env.terrain.terrain_fingerprint()
        print("[FINGERPRINT] " + json.dumps(fingerprint), flush=True)
    save_dict = {
        "height_field_raw": env.terrain.height_field_raw,
        "env_origins": np.asarray(env.terrain.env_origins),
    }
    np.savez_compressed(os.path.join(out_dir, "terrain.npz"), **save_dict)

    _, _ = env.reset()
    obs = env.get_observations()

    # ---- load the checkpoint under evaluation -----------------------------------------
    from legged_gym.utils.helpers import get_load_path  # noqa: E402

    train_cfg.runner.resume = True
    train_cfg.runner.experiment_name = load_experiment
    train_cfg.runner.load_run = load_run
    train_cfg.runner.checkpoint = ckpt_iter
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    ckpt_path = get_load_path(log_root, load_run=load_run, checkpoint=ckpt_iter)
    model_iter = int(os.path.basename(ckpt_path).split("_")[-1].split(".")[0])
    print("[CELL] loading model: %s (iter %d)" % (ckpt_path, model_iter), flush=True)
    ppo_runner, train_cfg = task_registry.make_wmp_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ---- world-model / history state (same as eval_mirror.py) --------------------------
    history_length = 5
    trajectory_history = torch.zeros(
        size=(env.num_envs, history_length, env.num_obs - env.privileged_dim - env.height_dim - 3),
        device=env.device,
    )
    obs_without_command = torch.concat(
        (obs[:, env.privileged_dim: env.privileged_dim + 6], obs[:, env.privileged_dim + 9: -env.height_dim]),
        dim=1,
    )
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)
    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions), device=env.device)
    wm_obs = {"prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim], "is_first": wm_is_first}
    if env.cfg.depth.use_camera:
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)), device=world_model.device)
    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    if record and camera_mode == "side" and env.viewer is not None:
        origin = env.env_origins[0].detach().cpu().numpy()
        env.set_camera(origin + np.array([1.0, 7.0, 2.2]), origin + np.array([1.0, 0.0, 0.3]))

    # ---- trial bookkeeping ------------------------------------------------------------
    num_envs = env.num_envs
    done = np.zeros(num_envs, dtype=bool)
    rec_len = np.zeros(num_envs, dtype=np.int64)
    rec_cause = np.array(["unfinished"] * num_envs, dtype=object)
    frame_idx = 0
    total_reward = 0.0
    not_dones = torch.ones((num_envs,), device=env.device)
    step_limit = _env_int("STEPS", 0)
    if step_limit <= 0:
        step_limit = int(env.max_episode_length) + 3
    print("[CELL] evaluation start: step_limit=%d" % step_limit, flush=True)

    infos = None
    i = 0
    for i in range(step_limit):
        with torch.no_grad():
            if env.global_counter % wm_update_interval == 0:
                if env.cfg.depth.use_camera and infos is not None and infos.get("depth") is not None:
                    wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)
                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0
            history = trajectory_history.flatten(1).to(env.device)
            actions = policy(obs.detach(), history.detach(), wm_feature.detach())

        _ep_len_before = env.episode_length_buf.clone()
        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())
        if len(reset_env_ids) > 0:
            for _eid in reset_env_ids.cpu().numpy():
                if done[_eid]:
                    continue
                _L = int(_ep_len_before[_eid].item())
                rec_len[_eid] = _L
                done[_eid] = True
                rec_cause[_eid] = CAUSE_TIMEOUT if _L >= env.max_episode_length - 1 else _classify_failure(env, _eid)
        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        wm_action_history = torch.concat((wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {"prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim], "is_first": wm_is_first}
        if env.cfg.depth.use_camera:
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)), device=world_model.device)
        reset_env_ids_np = reset_env_ids.cpu().numpy()
        if len(reset_env_ids_np) > 0:
            wm_action_history[reset_env_ids_np, :] = 0
            wm_is_first[reset_env_ids_np] = 1
        wm_action = wm_action_history.flatten(1)

        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat(
            (obs[:, env.privileged_dim: env.privileged_dim + 6], obs[:, env.privileged_dim + 9: -env.height_dim]), dim=1
        )
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if record and env.viewer is not None and (i % frame_every == frame_every - 1):
            env.gym.write_viewer_image_to_file(env.viewer, os.path.join(frame_dir, "%05d.png" % frame_idx))
            frame_idx += 1
        if env.viewer is not None and camera_mode == "follow":
            target = env.root_states[min(camera_env_idx, num_envs - 1), :3].detach().cpu().numpy()
            env.set_camera(target + np.array([0.0, 1.0, 0.0]), target)
        if done.all():
            break

    for eid in range(num_envs):
        if not done[eid]:
            _L = int(env.episode_length_buf[eid].item())
            rec_len[eid] = _L
            rec_cause[eid] = CAUSE_TIMEOUT if _L >= env.max_episode_length - 1 else "unfinished_step_cap"
            done[eid] = True

    trials_done = int(done.sum())
    success = int(np.sum(rec_cause == CAUSE_TIMEOUT))
    fail = trials_done - success
    success_rate = 100.0 * success / trials_done if trials_done else 0.0
    causes = {}
    for cause in rec_cause:
        causes[str(cause)] = causes.get(str(cause), 0) + 1

    with open(os.path.join(out_dir, "trials.jsonl"), "w") as f:
        for eid in range(num_envs):
            f.write(json.dumps({"env_id": eid, "ep_len": int(rec_len[eid]),
                                "success": bool(rec_cause[eid] == CAUSE_TIMEOUT),
                                "cause": str(rec_cause[eid])}) + "\n")

    vram_peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
    eval_terrain = DIRECTION_LABEL[direction]
    annotation = {
        "condition": condition,
        "train_terrain": "LHR (left-high-right-low) unilateral tilt %.1f deg; families box/stair/gap 1/3 each"
        % lhr_tilt_deg,
        "model": "%s (iter %d)" % (ckpt_path, model_iter),
        "eval_terrain": eval_terrain,
        "terrain_and_level": "%s / level %d (difficulty %.2f)" % (FAMILY_LABEL[family], level, diff),
        "trials": trials,
    }
    annotation_line = (
        "[条件] %s | [训练地形] %s | [模型] %s | [评估地形] %s | [地形/难度] %s | [试验数] %d"
        % (annotation["condition"], annotation["train_terrain"], annotation["model"],
           annotation["eval_terrain"], annotation["terrain_and_level"], annotation["trials"])
    )
    summary = {
        "run_label": label,
        "stage": "T3-stage3",
        "condition": condition,
        "annotation": annotation,
        "annotation_line": annotation_line,
        "family": family,
        "level": level,
        "difficulty": diff,
        "eval_direction": direction,
        "mirror": bool(mirror),
        "lateral_tilt_deg": lhr_tilt_deg,
        "terrain_seed": terrain_seed,
        "trials_requested": trials,
        "trials_done": trials_done,
        "success": success,
        "fail": fail,
        "success_rate_pct": round(success_rate, 2),
        "mean_ep_len": round(float(np.mean(rec_len)), 2),
        "median_ep_len": float(np.median(rec_len)),
        "cause_counts": causes,
        "command": {"lin_vel_x": 0.6, "lin_vel_y": 0.0, "ang_vel_yaw": 0.0},
        "task": args.task,
        "load_experiment": load_experiment,
        "load_run": load_run,
        "checkpoint_arg": ckpt_iter,
        "checkpoint_path": ckpt_path,
        "checkpoint_iter": model_iter,
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "num_envs": num_envs,
        "steps_run": i + 1,
        "max_episode_length": int(env.max_episode_length),
        "total_reward_sum": float(total_reward),
        "vram_peak_mib": vram_peak_mib,
        "wall_time_s": round(time.time() - t_start, 1),
        "frame_count": frame_idx,
        "fingerprint": fingerprint,
        "host": "buaa-ubuntu-22",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[EVAL] label=%s condition=%s family=%s L%d %s trials=%d success=%d fail=%d success_rate=%.1f%%"
          % (label, condition, family, level, direction, trials_done, success, fail, success_rate), flush=True)
    print("[CELL_EVAL] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

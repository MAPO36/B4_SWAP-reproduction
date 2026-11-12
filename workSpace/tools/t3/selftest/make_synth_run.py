#!/usr/bin/env python
"""Generate a synthetic T3 run directory so the aggregation/figure/GIF pipeline can be
self-tested without touching the GPU.

The fake data mimics the real layout and the real summary.json schema written by
WMP/legged_gym/scripts/eval_mirror.py (A/B matrix, repro re-runs, T1 protocol logs,
video frames, fingerprint fields), with hand-written success-rate curves:

  * A-group (WMP native terrains): stair/tilt height fields are left-right symmetric by
    construction, so mirror == normal (mirror drop 0). gap uses an off-centre pit in this
    fake data, so its mirror drop is large.
  * B-group (unilateral lateral ramp): mirror drop grows with the lateral tilt -> the
    "mirror terrain is OOD" behaviour the real run should show.

Usage:
  python make_synth_run.py --out <dir> [--tamper]

  --tamper  corrupt one mirror-cell fingerprint so the aggregator must report
            mirror_pair_ok=false (negative test for the pairing validation).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys

try:
    import numpy as np
except ImportError:  # np is available in both the bundled and the conda python
    print("[SYNTH][FATAL] numpy is required")
    sys.exit(2)

TERRAIN_SEED = 1
HORIZONTAL_SCALE = 0.05
VERTICAL_SCALE = 0.005
MAP_SIZE = 580  # 29 m border-included map, same shape as the real 29 m terrain map

A_CELLS = [(terrain, diff, mirror)
           for terrain in ("tilt", "stair", "gap")
           for diff in (0.2, 0.5, 0.8)
           for mirror in (0, 1)]
B_CELLS = [(terrain, diff, tilt, mirror)
           for terrain, diff in (("ramp", 0.3), ("ramp_stair", 0.5), ("ramp_gap", 0.5))
           for tilt in (5.0, 10.0, 15.0)
           for mirror in (0, 1)]
REPRO_CELLS = [
    ("a_stair_d0.5_normal", "stair", 0.5, 0.0, 0),
    ("a_gap_d0.2_mirror", "gap", 0.2, 0.0, 1),
    ("a_tilt_d0.8_mirror", "tilt", 0.8, 0.0, 1),
]

# A-group success rate before mirroring: (terrain, diff) -> normal %
A_RATES = {
    ("tilt", 0.2): 92.0, ("tilt", 0.5): 71.5, ("tilt", 0.8): 38.0,
    ("stair", 0.2): 99.5, ("stair", 0.5): 99.0, ("stair", 0.8): 94.0,
    ("gap", 0.2): 41.0, ("gap", 0.5): 24.5, ("gap", 0.8): 8.0,
}
# A-group mirror rate: symmetric height fields are identical after fliplr; gap is not.
A_MIRROR_RATES = {
    ("tilt", 0.2): 92.0, ("tilt", 0.5): 71.5, ("tilt", 0.8): 38.0,
    ("stair", 0.2): 99.5, ("stair", 0.5): 99.0, ("stair", 0.8): 94.0,
    ("gap", 0.2): 14.0, ("gap", 0.5): 9.5, ("gap", 0.8): 2.5,
}
# B-group normal rates: (terrain family, tilt) -> normal %
B_RATES = {
    ("ramp", 5.0): 95.0, ("ramp", 10.0): 78.0, ("ramp", 15.0): 52.0,
    ("ramp_stair", 5.0): 88.0, ("ramp_stair", 10.0): 61.0, ("ramp_stair", 15.0): 33.0,
    ("ramp_gap", 5.0): 46.0, ("ramp_gap", 10.0): 27.0, ("ramp_gap", 15.0): 11.0,
}
B_DROP = {5.0: 11.0, 10.0: 17.0, 15.0: 23.0}  # mirrored -> worse
T1_RATES = {
    "slope": [95.3, 94.2, 93.1],
    "stair": [98.4, 96.9, 99.0],
    "gap": [18.2, 21.6, 15.9],
}
T1_REFERENCE = {"slope": 93.8, "stair": 98.4, "gap": 18.2}


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_heightfield(terrain, seed):
    """Fake int16 height field, following WMP's left-right (y) symmetry properties."""
    rng = np.random.RandomState(1000 + seed)
    height = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int16)
    x = np.arange(MAP_SIZE)
    # a symmetric long-range profile along x (forward direction)
    profile = np.round(30.0 * np.sin(x / 90.0) + 15.0 * np.cos(x / 37.0)).astype(np.int16)
    if terrain in ("ramp", "ramp_stair", "ramp_gap"):
        flat = np.zeros(MAP_SIZE, dtype=np.int16)
        height[:] = profile.reshape(-1, 1) if terrain != "ramp" else flat.reshape(-1, 1)
        if terrain == "ramp_stair":
            step = np.round(40.0 * (1 + np.sin(x / 55.0))).astype(np.int16)
            height = np.repeat(step.reshape(-1, 1), MAP_SIZE, axis=1)
        if terrain == "ramp_gap":
            y0 = 220
            height[:, y0:y0 + 60] = -120  # off-centre pit, asymmetric in y
    elif terrain == "stair":
        step = np.round(35.0 * (1 + np.sin(x / 60.0))).astype(np.int16)
        height = np.repeat(step.reshape(-1, 1), MAP_SIZE, axis=1)  # depends on x only -> y-symmetric
    elif terrain == "tilt":
        height = np.repeat(profile.reshape(-1, 1), MAP_SIZE, axis=1)
    elif terrain == "gap":
        y0 = 210 + (seed % 3) * 40  # off-centre -> asymmetric
        height[:, y0:y0 + 55] = -140
    else:
        height = np.repeat(profile.reshape(-1, 1), MAP_SIZE, axis=1)
    if terrain in ("ramp", "ramp_stair", "ramp_gap"):
        tilt = {"ramp": 5.0, "ramp_stair": 10.0, "ramp_gap": 15.0}[terrain]
        y = np.arange(MAP_SIZE)
        y_m = (y - (MAP_SIZE - 1) / 2.0) * HORIZONTAL_SCALE
        ramp = np.round(np.tan(math.radians(tilt)) * y_m / VERTICAL_SCALE).astype(np.int16)
        height = (height.astype(np.int32) + ramp.reshape(1, -1)).astype(np.int16)
    return np.ascontiguousarray(height)


def fingerprint(hf, added_trimesh=False):
    return {
        "height_field_shape": [int(hf.shape[0]), int(hf.shape[1])],
        "height_field_sha256": sha256_array(hf),
        "height_field_fliplr_sha256": sha256_array(np.fliplr(hf)),
        "added_trimesh": bool(added_trimesh),
    }


def make_box_mesh(center=(0.0, 14.5, 0.4), size=(1.5, 1.5, 0.8)):
    """Closed box mesh with outward-consistent winding (signed volume > 0)."""
    cx, cy, cz = center
    sx, sy, sz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=np.float32)
    # counter-clockwise when seen from outside -> positive signed volume
    triangles = np.array([
        [0, 2, 1], [0, 3, 2],      # bottom (z-)
        [4, 5, 6], [4, 6, 7],      # top (z+)
        [0, 1, 5], [0, 5, 4],      # front (y-)
        [3, 6, 2], [3, 7, 6],      # back (y+)
        [0, 4, 7], [0, 7, 3],      # left (x-)
        [1, 2, 6], [1, 6, 5],      # right (x+)
    ], dtype=np.int32)
    return verts, triangles


def mirror_mesh(verts, triangles, y_axis=14.5):
    """Reflect y about `y_axis` and undo the resulting winding flip (the fixed implementation)."""
    reflected = np.array(verts, dtype=np.float32, copy=True)
    reflected[:, 1] = 2.0 * y_axis - reflected[:, 1]
    return reflected, np.array(triangles, dtype=np.int32)[:, [0, 2, 1]]


def split_failures(rng, success, trials):
    fail = trials - success
    if fail <= 0:
        return {"success_timeout": success}
    base = int(round(fail * 0.55))
    flipped = int(round(fail * 0.3))
    unfinished = fail - base - flipped
    causes = {"success_timeout": success}
    if base:
        causes["fail_base_contact"] = base
    if flipped:
        causes["fail_flipped"] = flipped
    if unfinished:
        causes["unfinished_step_cap"] = unfinished
    return causes


def make_summary(label, run_label, terrain, difficulty, tilt, mirror, rate_pct, trials, seed,
                 hf, added_trimesh, wall_time_s):
    rng = random.Random(hash((label, run_label, seed)) & 0xFFFFFFFF)
    success = int(round(rate_pct / 100.0 * trials))
    success = max(0, min(trials, success))
    fail = trials - success
    mean_ep_len = 1001.0 if success == trials else round(
        (success * 1001.0 + fail * rng.uniform(120, 420)) / trials, 2)
    return {
        "run_label": run_label,
        "stage": "T3",
        "terrain": terrain,
        "terrain_proportions": [0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        "difficulty": difficulty,
        "lateral_tilt_deg": tilt,
        "tilt_direction": ("left_high_right_low" if tilt > 0 else "flat") if not mirror
        else ("right_high_left_low" if tilt > 0 else "flat"),
        "mirror": bool(mirror),
        "terrain_seed": seed,
        "trials_requested": trials,
        "trials_done": trials,
        "success": success,
        "fail": fail,
        "success_rate_pct": round(100.0 * success / trials, 2),
        "mean_ep_len": mean_ep_len,
        "median_ep_len": 1001.0 if success > trials // 2 else 214.0,
        "cause_counts": split_failures(rng, success, trials),
        "command": {"lin_vel_x": 0.6, "lin_vel_y": 0.0, "ang_vel_yaw": 0.0},
        "task": "go2_amp",
        "load_run": "WMP",
        "checkpoint_arg": -1,
        "checkpoint_path": "/home/ubuntu/B4_2025/SWAP/WMP/logs/go2_amp_example/WMP/model_10000.pt",
        "checkpoint_sha256": "46cbef123122c8b42a5d6e137921f160409e2dc76ee332329e99684b857a98e2",
        "num_envs": trials,
        "steps_run": 1001,
        "max_episode_length": 1001,
        "total_reward_sum": 30.4,
        "vram_peak_mib": 690,
        "wall_time_s": wall_time_s,
        "frame_count": 0,
        "fingerprint": fingerprint(hf, added_trimesh),
        "host": "buaa-ubuntu-22",
        "gpu": "NVIDIA GeForce RTX 4090",
        "timestamp": "2026-11-12T16:%02d:%02d+0800" % (rng.randint(0, 59), rng.randint(0, 59)),
    }


def write_cell(run_dir, out_root, label, summary, hf, added_mesh=None):
    cell_dir = os.path.join(run_dir, out_root, label)
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_dict = {
        "height_field_raw": hf,
        "env_origins": np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
    }
    if added_mesh is not None:
        save_dict["added_vertices"] = np.asarray(added_mesh[0], dtype=np.float32)
        save_dict["added_triangles"] = np.asarray(added_mesh[1], dtype=np.uint32)
    np.savez_compressed(os.path.join(cell_dir, "terrain.npz"), **save_dict)
    rows = []
    for env_id in range(summary["trials_done"]):
        success = env_id < summary["success"]
        rows.append({"env_id": env_id, "ep_len": 1001 if success else 214,
                     "success": success, "cause": "success_timeout" if success else "fail_base_contact"})
    with open(os.path.join(cell_dir, "trials.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(cell_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("[SYNTH] fake run log for %s\n[CELL_EVAL] %s\n" % (label, json.dumps(summary)))


def make_videos(run_dir):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[SYNTH][WARN] Pillow unavailable: no video frames generated")
        return
    names = ["video_a_stair_d0.5_normal", "video_b_ramp_stair_t10_normal"]
    for idx, name in enumerate(names):
        cell_dir = os.path.join(run_dir, "videos", name)
        frames_dir = os.path.join(cell_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for frame in range(80):
            image = Image.new("RGB", (320, 180), (24, 28, 38))
            draw = ImageDraw.Draw(image)
            x = 40 + int(frame * 2.4)
            draw.rectangle([x, 110, x + 60, 150], fill=(200, 90 + idx * 40, 60))
            draw.text((10, 10), "%s  frame %03d" % (name, frame), fill=(230, 230, 230))
            image.save(os.path.join(frames_dir, "%05d.png" % frame))
        summary = {
            "run_label": name, "terrain": "stair" if idx == 0 else "ramp_stair",
            "difficulty": 0.5, "lateral_tilt_deg": 0.0 if idx == 0 else 10.0,
            "mirror": False, "trials_requested": 1, "trials_done": 1, "success": 1, "fail": 0,
            "success_rate_pct": 100.0, "frame_count": 80,
            "fingerprint": fingerprint(make_heightfield("stair", 1)), "num_envs": 1,
            "steps_run": 1001, "vram_peak_mib": 500, "wall_time_s": 25.0,
            "checkpoint_path": "/home/ubuntu/B4_2025/SWAP/WMP/logs/go2_amp_example/WMP/model_10000.pt",
        }
        with open(os.path.join(cell_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(cell_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("[SYNTH] fake video cell %s frames=80\n" % name)


def make_t1_repro(run_dir):
    base = os.path.join(run_dir, "t1_repro")
    for terrain, rates in T1_RATES.items():
        for run_idx, rate in enumerate(rates, start=1):
            cell_dir = os.path.join(base, "%s_r%d" % (terrain, run_idx))
            os.makedirs(cell_dir, exist_ok=True)
            envs = 64
            trials = envs + (run_idx - 1) * 3
            success = int(round(rate / 100.0 * trials))
            fail = trials - success
            with open(os.path.join(cell_dir, "run.log"), "w", encoding="utf-8") as f:
                f.write("[SYNTH] eval_t1_repro.py --terrain=%s\n" % terrain)
                f.write("[EVAL] envs=64 steps=1200 success=%d fail=%d trials=%d "
                        "success_rate=%.1f%%\n" % (success, fail, trials, rate))
                f.write("[EVAL] mean_episode_len=975.6 / max=1001.0\n")


def make_logs(run_dir):
    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    rng = random.Random(7)
    with open(os.path.join(logs, "vram.csv"), "w", encoding="utf-8") as f:
        f.write("timestamp, memory.used [MiB], utilization.gpu [%]\n")
        for step in range(60):
            f.write("2026/11/12 16:%02d:%02d.000, %d MiB, %d %%\n"
                    % (30 + step // 60, step % 60, rng.randint(700, 1180), rng.randint(20, 90)))
    with open(os.path.join(logs, "gpu_start.txt"), "w", encoding="utf-8") as f:
        f.write("# synthetic nvidia-smi snapshot\n0, NVIDIA GeForce RTX 4090, 780 MiB, 24564 MiB\n")
    with open(os.path.join(logs, "matrix.log"), "w", encoding="utf-8") as f:
        f.write("[SYNTH] synthetic matrix log\n")


def make_run_config(run_dir, synth_dir, cells_tsv):
    cells = {"a_matrix": [], "b_matrix": [], "repro_check": []}
    try:
        with open(cells_tsv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                group, out_root, label, run_label, terrain, diff, tilt, mirror, trials, steps = parts[:10]
                cells.setdefault(out_root, []).append({
                    "group": group, "out_root": out_root, "label": label, "run_label": run_label,
                    "terrain": terrain, "difficulty": float(diff), "tilt_deg": float(tilt),
                    "mirror": bool(int(mirror)), "trials": int(trials), "steps": int(steps),
                })
    except OSError:
        pass
    cfg = {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "stage": "T3",
        "purpose": "SYNTHETIC self-test of the T3 aggregation pipeline (no GPU involved)",
        "run_dir": run_dir,
        "host": "buaa-ubuntu-22",
        "started_at": "2026-11-12T16:30:00+0800",
        "mode": "all",
        "sim_device": "cuda:0",
        "trials_per_cell": 200,
        "terrain_seed": TERRAIN_SEED,
        "load_run": "WMP",
        "ckpt_iter": -1,
        "steps": 0,
        "eval_script": "/home/ubuntu/B4_2025/SWAP/WMP/legged_gym/scripts/eval_mirror.py",
        "eval_script_sha256": "synth" + "0" * 59,
        "t1_script": "/home/ubuntu/B4_2025/SWAP/WMP/legged_gym/scripts/eval_t1_repro.py",
        "t1_script_sha256": "synth" + "1" * 59,
        "cells_tsv": os.path.join(synth_dir, "cells.tsv"),
        "cells": cells,
        "videos": {"trials": 1, "steps": 1001, "frame_every": 2, "camera_mode": "follow"},
        "t1_repro": {"neval": 64, "steps": 1200, "terrain": ["slope", "stair", "gap"], "runs_per_terrain": 3},
        "git": {"commit": "0" * 40, "commit_short": "synthetic", "branch": "main", "status_short": []},
        "script_sha256": {name: "synth" for name in
                          ("run_matrix.sh", "aggregate_metrics.py", "make_figures.py",
                           "make_gifs.sh", "gif_maker.py", "cells.tsv")},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def make_clean_run(run_dir, synth_dir, cells_tsv):
    os.makedirs(run_dir, exist_ok=True)
    rng = random.Random(TERRAIN_SEED)
    # A-group
    for terrain, diff, mirror in A_CELLS:
        label = "a_%s_d%.1f_%s" % (terrain, diff, "mirror" if mirror else "normal")
        base_hf = make_heightfield(terrain, TERRAIN_SEED)
        hf = np.ascontiguousarray(np.fliplr(base_hf)) if mirror else base_hf
        rate = (A_MIRROR_RATES if mirror else A_RATES)[(terrain, diff)]
        box_verts, box_tris = make_box_mesh()
        added_mesh = None
        if terrain == "tilt":
            added_mesh = mirror_mesh(box_verts, box_tris) if mirror else (box_verts, box_tris)
        summary = make_summary(label, label, terrain, diff, 0.0, mirror, rate, 200,
                               TERRAIN_SEED, hf, added_trimesh=(terrain == "tilt"),
                               wall_time_s=round(rng.uniform(38, 52), 1))
        write_cell(run_dir, "a_matrix", label, summary, hf, added_mesh=added_mesh)
    # B-group
    for terrain, diff, tilt, mirror in B_CELLS:
        label = "b_%s_t%d_%s" % (terrain, int(tilt), "mirror" if mirror else "normal")
        base_hf = make_heightfield(terrain, int(tilt))
        # give each tilt its own lateral ramp
        y = np.arange(MAP_SIZE)
        y_m = (y - (MAP_SIZE - 1) / 2.0) * HORIZONTAL_SCALE
        ramp = np.round(np.tan(math.radians(tilt)) * y_m / VERTICAL_SCALE).astype(np.int16)
        base_hf = (base_hf.astype(np.int32) + ramp.reshape(1, -1)).astype(np.int16)
        hf = np.ascontiguousarray(np.fliplr(base_hf)) if mirror else base_hf
        rate = B_RATES[(terrain, tilt)] - (B_DROP[tilt] if mirror else 0.0)
        summary = make_summary(label, label, terrain, diff, tilt, mirror, rate, 200,
                               TERRAIN_SEED, hf, added_trimesh=False,
                               wall_time_s=round(rng.uniform(38, 52), 1))
        write_cell(run_dir, "b_matrix", label, summary, hf)
    # repro_check: identical numbers to the first run (deterministic re-run)
    for orig_label, terrain, diff, tilt, mirror in REPRO_CELLS:
        base_hf = make_heightfield(terrain, TERRAIN_SEED)
        hf = np.ascontiguousarray(np.fliplr(base_hf)) if mirror else base_hf
        rate = (A_MIRROR_RATES if mirror else A_RATES)[(terrain, diff)]
        label = orig_label + "_rerun"
        added_mesh = None
        if terrain == "tilt":
            box_verts, box_tris = make_box_mesh()
            added_mesh = mirror_mesh(box_verts, box_tris) if mirror else (box_verts, box_tris)
        summary = make_summary(label, orig_label, terrain, diff, tilt, mirror, rate, 200,
                               TERRAIN_SEED, hf, added_trimesh=(terrain == "tilt"),
                               wall_time_s=round(rng.uniform(38, 52), 1))
        write_cell(run_dir, "repro_check", label, summary, hf, added_mesh=added_mesh)
    make_t1_repro(run_dir)
    make_videos(run_dir)
    make_logs(run_dir)
    make_run_config(run_dir, synth_dir, cells_tsv)
    print("[SYNTH] clean run written to %s" % run_dir)


def tamper_run(run_dir):
    """Corrupt one mirror-cell fingerprint: the aggregator must flag mirror_pair_ok=false."""
    path = os.path.join(run_dir, "a_matrix", "a_stair_d0.5_mirror", "summary.json")
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    summary["fingerprint"]["height_field_sha256"] = "deadbeef" * 8
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[SYNTH] tampered %s (expected: mirror_pair_ok=false for a_stair_d0.5 pair)" % path)


def tamper_mesh(run_dir):
    """Re-introduce the y-flip winding bug in one tilted-trimesh mirror terrain."""
    path = os.path.join(run_dir, "a_matrix", "a_tilt_d0.5_mirror", "terrain.npz")
    with np.load(path) as data:
        contents = {key: data[key] for key in data.files}
    triangles = np.asarray(contents["added_triangles"])
    contents["added_triangles"] = np.ascontiguousarray(triangles[:, [0, 2, 1]])
    np.savez_compressed(path, **contents)
    print("[SYNTH] tampered %s (wrong winding -> negative signed volume, expected trimesh FAIL)" % path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="synthetic T3 run dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tamper", action="store_true", help="corrupt a fingerprint (mirror-pair negative test)")
    parser.add_argument("--tamper-mesh", action="store_true", help="corrupt a mirrored box winding (trimesh negative test)")
    args = parser.parse_args(argv)
    synth_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.dirname(synth_dir)
    cells_tsv = os.path.join(tools_dir, "cells.tsv")
    run_dir = os.path.abspath(args.out)
    make_clean_run(run_dir, synth_dir, cells_tsv)
    if args.tamper:
        tamper_run(run_dir)
    if args.tamper_mesh:
        tamper_mesh(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

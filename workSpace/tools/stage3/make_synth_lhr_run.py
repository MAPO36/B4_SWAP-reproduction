#!/usr/bin/env python
"""Build a synthetic 18-cell stage-3 eval run to self-test aggregate_lhr.py + make_lhr_figures.py.

Creates <out>/eval/<label>/{summary.json,trials.jsonl,terrain.npz} for all 3x3x2 cells,
a fake training log <out>/logs/full.log and a fake ckpt path/sha in the summaries.
Numbers are NOT meaningful; this only validates the aggregation pipeline.
"""
import argparse
import json
import os

import numpy as np

FAMILIES = ["box", "stair", "gap"]
LEVELS = [1, 2, 3]
DIRS = ["LHR", "RHL"]
DIFF = {1: 0.2, 2: 0.5, 3: 0.8}
LHR_TILT = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth_lhr_run")
    args = ap.parse_args()
    root = os.path.abspath(args.out)
    eval_dir = os.path.join(root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ckpt = {"path": os.path.join(root, "seed0", "model_10000.pt"), "sha": "deadbeef" * 8}

    rng = np.random.default_rng(0)
    summary_rows = []
    for f in FAMILIES:
        for lv in LEVELS:
            # one terrain draw shared by the LHR/RHL pair (so they are exact mirrors)
            hf = (np.round(LHR_TILT * 0.1763 * 10 * (np.arange(80) - 39.5) / 0.1)
                  .astype(np.int16).reshape(1, -1) + rng.integers(-3, 4, (80, 80))).astype(np.int16)
            root_hf = np.zeros((1300, 580), dtype=np.int16)
            root_hf[250:330, 250:330] = hf
            for d in DIRS:
                label = "R0-LHR_%s_L%d_%s" % (f, lv, d)
                out = os.path.join(eval_dir, label)
                os.makedirs(out, exist_ok=True)
                n = 200
                base = {1: 0.95, 2: 0.85, 3: 0.65}[lv] - 0.1 * FAMILIES.index(f)
                lhr = base
                rhl = base - (0.05 + 0.03 * lv) if f != "stair" else base - 0.01
                p = (lhr if d == "LHR" else rhl)
                n_succ = int(round(p * n))
                causes = {"success_timeout": n_succ, "fail_base_contact": n - n_succ}
                # terrain height field: LHR = ramp+noise, RHL = its np.fliplr mirror
                display_hf = root_hf if d == "LHR" else np.ascontiguousarray(np.fliplr(root_hf))
                np.savez_compressed(os.path.join(out, "terrain.npz"), height_field_raw=display_hf,
                                    env_origins=np.zeros((1, 1, 3)))
                import hashlib

                hf_flip = np.ascontiguousarray(np.fliplr(display_hf))
                fp = {
                    "height_field_shape": list(display_hf.shape),
                    "height_field_sha256": hashlib.sha256(display_hf.tobytes()).hexdigest(),
                    "height_field_fliplr_sha256": hashlib.sha256(hf_flip.tobytes()).hexdigest(),
                    "lhr_tilt_deg": LHR_TILT, "mirror_rhl": d == "RHL", "family": f,
                    "fixed_difficulty": DIFF[lv], "terrain_seed": 1, "num_rows": 1, "num_cols": 1,
                    "terrain_proportions": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "lhr_check": {
                        "n_patches": 1,
                        "left_higher_count": 1 if d == "LHR" else 0,
                        "direction_consistent": d == "LHR",
                        "roll_deg_min": LHR_TILT if d == "LHR" else -LHR_TILT,
                        "roll_deg_median": LHR_TILT if d == "LHR" else -LHR_TILT,
                        "roll_deg_max": LHR_TILT if d == "LHR" else -LHR_TILT,
                        "requested_deg": LHR_TILT,
                    },
                }
                ann = {
                    "condition": "R0-LHR",
                    "train_terrain": "LHR unilateral tilt 10.0 deg; families box/stair/gap 1/3 each",
                    "model": "%s (iter 10000)" % ckpt["path"],
                    "eval_terrain": "LHR (in-distribution)" if d == "LHR" else "RHL (mirrored / OOD)",
                    "terrain_and_level": "%s / level %d (difficulty %.2f)" % (f, lv, DIFF[lv]),
                    "trials": n,
                }
                ann_line = ("[条件] %s | [训练地形] %s | [模型] %s | [评估地形] %s | [地形/难度] %s | [试验数] %d"
                            % (ann["condition"], ann["train_terrain"], ann["model"], ann["eval_terrain"],
                               ann["terrain_and_level"], ann["trials"]))
                summary = {
                    "run_label": label, "stage": "T3-stage3", "condition": "R0-LHR",
                    "annotation": ann, "annotation_line": ann_line,
                    "family": f, "level": lv, "difficulty": DIFF[lv], "eval_direction": d,
                    "mirror": d == "RHL", "lateral_tilt_deg": LHR_TILT, "terrain_seed": 1,
                    "trials_requested": n, "trials_done": n, "success": n_succ, "fail": n - n_succ,
                    "success_rate_pct": round(100.0 * n_succ / n, 2), "mean_ep_len": 950.0,
                    "median_ep_len": 1001.0, "cause_counts": causes,
                    "checkpoint_path": ckpt["path"], "checkpoint_iter": 10000,
                    "checkpoint_sha256": ckpt["sha"], "vram_peak_mib": 690, "wall_time_s": 60.0,
                    "fingerprint": fp,
                }
                with open(os.path.join(out, "summary.json"), "w") as fh:
                    json.dump(summary, fh, indent=2)
                with open(os.path.join(out, "trials.jsonl"), "w") as fh:
                    for i in range(n):
                        fh.write(json.dumps({"env_id": i, "ep_len": 1001 if i < n_succ else 300,
                                             "success": i < n_succ,
                                             "cause": "success_timeout" if i < n_succ else "fail_base_contact"}) + "\n")
                summary_rows.append(summary)

    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    with open(os.path.join(root, "logs", "full.log"), "w") as fh:
        for it in range(0, 10000, 100):
            fh.write("\x1b[1m Learning iteration %d/10000 \x1b[0m\n" % it)
            fh.write("                       Mean reward: %.3f\n" % (0.1 + 0.0015 * it + 1e-4 * (it % 300)))
            fh.write("                    Iteration time: 8.03s\n")
            if it % 1000 == 0:
                fh.write("                        Total time: %.2fs\n" % (it * 8.03))
    print("synthetic run written: %s (%d cells)" % (root, len(summary_rows)))


if __name__ == "__main__":
    main()

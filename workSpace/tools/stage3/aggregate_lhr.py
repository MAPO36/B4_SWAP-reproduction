#!/usr/bin/env python
"""Stage-3 aggregate for the R0-LHR LHR/RHL evaluation (18 cells = 3 families x 3 levels x 2 dirs).

Scans `<run-dir>/eval/<label>/summary.json` written by `eval_lhr.py`, then produces:
  metrics.csv            one row per cell, with the mandatory 6-item annotation
  lhr_rhl_pairs.csv      9 pairs: LHR vs RHL success, mirror gap = LHR - RHL, binomial 95% CI
  by_family.csv / by_level.csv / overall.csv
  checks.json            completeness / trial count / mirror-pair fingerprint / model identity
  training_summary.json  parsed from the WMP training log (reward curve, NaN count, timing)
  train_curve.csv        iteration, mean_reward
  manifest.json          AGENTS.md §4.1 fields + stage-3 6-item annotation standard
  aggregate_report.md    human-readable report tables

Usage:
  python aggregate_lhr.py --run-dir /home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import time

FAMILIES = ["box", "stair", "gap"]
LEVELS = [1, 2, 3]
DIRECTIONS = ["LHR", "RHL"]
LEVEL_DIFF = {1: 0.2, 2: 0.5, 3: 0.8}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(p, obj):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(p, rows, cols):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sha256_file(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_short(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def binom_ci_pct(p1, n1, p2, n2):
    """95% CI for the difference of two proportions (normal approximation), in pp."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    q1, q2 = max(p1, 0.0), max(p2, 0.0)
    se = (q1 * (1 - q1) / n1 + q2 * (1 - q2) / n2) ** 0.5
    return 1.96 * se * 100.0


def parse_train_log(path):
    """Parse WMP training log: 'Learning iteration N/M' followed by 'Mean reward: x'."""
    out = {"log_path": path, "iterations": [], "mean_reward": [], "nan_lines": 0,
           "iteration_times_s": [], "total_time_s": None, "max_iteration": None}
    if not path or not os.path.exists(path):
        out["found"] = False
        return out
    out["found"] = True
    cur_it = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = ANSI.sub("", line)
            m = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", s)
            if m:
                cur_it = int(m.group(1))
                out["max_iteration"] = int(m.group(2))
                continue
            if re.search(r"(^|[^A-Za-z])nan([^A-Za-z]|$)", s, re.IGNORECASE):
                out["nan_lines"] += 1
            m = re.search(r"Mean reward:\s*(-?\d+(?:\.\d+)?)", s)
            if m and cur_it is not None:
                out["iterations"].append(cur_it)
                out["mean_reward"].append(float(m.group(1)))
                cur_it = None
                continue
            m = re.search(r"Iteration time:\s*([\d.]+)s", s)
            if m:
                out["iteration_times_s"].append(float(m.group(1)))
                continue
            m = re.search(r"Total time:\s*([\d.]+)s", s)
            if m:
                out["total_time_s"] = float(m.group(1))
    out["curves"] = {"iteration": out["iterations"], "mean_reward": out["mean_reward"]}
    if out["iteration_times_s"]:
        out["iteration_time_mean_s"] = round(sum(out["iteration_times_s"]) / len(out["iteration_times_s"]), 3)
    if out["mean_reward"]:
        out["mean_reward_first"] = out["mean_reward"][0]
        out["mean_reward_last"] = out["mean_reward"][-1]
        # block means every ~1000 iterations
        blocks = {}
        for it, rw in zip(out["iterations"], out["mean_reward"]):
            blocks.setdefault(it // 1000, []).append(rw)
        out["mean_reward_per_1000_iters"] = {
            "%d-%d" % (k * 1000, (k + 1) * 1000): round(sum(v) / len(v), 3) for k, v in sorted(blocks.items())
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR")
    ap.add_argument("--eval-subdir", default="eval")
    ap.add_argument("--condition", default="R0-LHR")
    ap.add_argument("--train-log", default=None,
                    help="WMP training log; default tries <run-dir>/logs/full.log")
    ap.add_argument("--repo", default="/home/ubuntu/B4_2025/SWAP")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    eval_dir = os.path.join(run_dir, args.eval_subdir)
    train_log = args.train_log or os.path.join(run_dir, "logs", "full.log")

    # ---------------------------------------------------------------- load cells
    cells = {}
    for p in sorted(glob.glob(os.path.join(eval_dir, "*", "summary.json"))):
        s = read_json(p)
        if "family" not in s or "eval_direction" not in s:
            continue
        key = (s["family"], int(s["level"]), s["eval_direction"])
        cells[key] = s
        s["_dir"] = os.path.dirname(p)

    expected = [(f, lv, d) for f in FAMILIES for lv in LEVELS for d in DIRECTIONS]
    missing = [k for k in expected if k not in cells]
    extra = [k for k in cells if k not in expected]

    # ---------------------------------------------------------------- metrics + pairs
    metrics = []
    for f, lv, d in expected:
        c = cells.get((f, lv, d))
        if not c:
            continue
        causes = c.get("cause_counts") or {}
        top_cause = max(causes, key=causes.get) if causes else ""
        metrics.append({
            "condition": c.get("condition"),
            "family": f,
            "level": lv,
            "difficulty": c.get("difficulty"),
            "direction": d,
            "trials_done": c.get("trials_done"),
            "success": c.get("success"),
            "fail": c.get("fail"),
            "success_rate_pct": c.get("success_rate_pct"),
            "mean_ep_len": c.get("mean_ep_len"),
            "median_ep_len": c.get("median_ep_len"),
            "top_cause": top_cause,
            "cause_counts": json.dumps(causes, ensure_ascii=False),
            "checkpoint_path": c.get("checkpoint_path"),
            "checkpoint_iter": c.get("checkpoint_iter"),
            "checkpoint_sha256": c.get("checkpoint_sha256"),
            "lhr_tilt_deg": c.get("lateral_tilt_deg"),
            "terrain_seed": c.get("terrain_seed"),
            "height_field_sha256": (c.get("fingerprint") or {}).get("height_field_sha256"),
            "height_field_fliplr_sha256": (c.get("fingerprint") or {}).get("height_field_fliplr_sha256"),
            "wall_time_s": c.get("wall_time_s"),
            "vram_peak_mib": c.get("vram_peak_mib"),
            "annotation_line": c.get("annotation_line"),
            "out_dir": c.get("_dir"),
        })
    write_csv(os.path.join(run_dir, "metrics.csv"), metrics,
              ["condition", "family", "level", "difficulty", "direction", "trials_done", "success",
               "fail", "success_rate_pct", "mean_ep_len", "median_ep_len", "top_cause", "cause_counts",
               "checkpoint_path", "checkpoint_iter", "checkpoint_sha256", "lhr_tilt_deg", "terrain_seed",
               "height_field_sha256", "height_field_fliplr_sha256", "wall_time_s", "vram_peak_mib",
               "annotation_line", "out_dir"])

    pairs = []
    pair_checks = []
    for f in FAMILIES:
        for lv in LEVELS:
            l = cells.get((f, lv, "LHR"))
            r = cells.get((f, lv, "RHL"))
            if not l or not r:
                continue
            lr = float(l["success_rate_pct"])
            rr = float(r["success_rate_pct"])
            gap = lr - rr
            ci = binom_ci_pct(lr / 100.0, int(l["trials_done"]), rr / 100.0, int(r["trials_done"]))
            lhfp = (l.get("fingerprint") or {})
            rhfp = (r.get("fingerprint") or {})
            fp_ok = (
                lhfp.get("height_field_fliplr_sha256") == rhfp.get("height_field_sha256")
                and rhfp.get("height_field_fliplr_sha256") == lhfp.get("height_field_sha256")
            )
            model_ok = (
                l.get("checkpoint_path") == r.get("checkpoint_path")
                and l.get("checkpoint_sha256") == r.get("checkpoint_sha256")
            )
            seed_ok = l.get("terrain_seed") == r.get("terrain_seed")
            dir_ok = (
                bool((lhfp.get("lhr_check") or {}).get("direction_consistent"))
                and not bool((rhfp.get("lhr_check") or {}).get("direction_consistent"))
                and float((rhfp.get("lhr_check") or {}).get("roll_deg_median", 0.0)) < 0.0
            )
            pairs.append({
                "condition": args.condition,
                "family": f,
                "level": lv,
                "difficulty": l.get("difficulty"),
                "lhr_success": l["success"],
                "rhl_success": r["success"],
                "trials": l["trials_done"],
                "lhr_pct": round(lr, 2),
                "rhl_pct": round(rr, 2),
                "mirror_gap_pp": round(gap, 2),
                "ci95_pp": round(ci, 2),
                "within_ci": bool(abs(gap) <= ci),
                "fingerprint_pair_ok": bool(fp_ok),
                "same_model": bool(model_ok),
                "same_terrain_seed": bool(seed_ok),
                "direction_check_ok": bool(dir_ok),
                "checkpoint_path": l.get("checkpoint_path"),
                "checkpoint_sha256": l.get("checkpoint_sha256"),
                "annotation_lhr": l.get("annotation_line"),
                "annotation_rhl": r.get("annotation_line"),
                "lhr_out_dir": l.get("_dir"),
                "rhl_out_dir": r.get("_dir"),
            })
            pair_checks.append(fp_ok and model_ok and seed_ok and dir_ok)
    write_csv(os.path.join(run_dir, "lhr_rhl_pairs.csv"), pairs,
              ["condition", "family", "level", "difficulty", "lhr_success", "rhl_success", "trials",
               "lhr_pct", "rhl_pct", "mirror_gap_pp", "ci95_pp", "within_ci", "fingerprint_pair_ok",
               "same_model", "same_terrain_seed", "direction_check_ok", "checkpoint_path",
               "checkpoint_sha256", "annotation_lhr", "annotation_rhl", "lhr_out_dir", "rhl_out_dir"])

    def agg(rows, key):
        out = []
        for val in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == val]
            out.append({
                key: val,
                "n_pairs": len(sub),
                "lhr_mean_pct": round(sum(r["lhr_pct"] for r in sub) / len(sub), 2),
                "rhl_mean_pct": round(sum(r["rhl_pct"] for r in sub) / len(sub), 2),
                "mirror_gap_mean_pp": round(sum(r["mirror_gap_pp"] for r in sub) / len(sub), 2),
                "mirror_gap_abs_mean_pp": round(sum(abs(r["mirror_gap_pp"]) for r in sub) / len(sub), 2),
            })
        write_csv(os.path.join(run_dir, "by_%s.csv" % key), out,
                  [key, "n_pairs", "lhr_mean_pct", "rhl_mean_pct", "mirror_gap_mean_pp", "mirror_gap_abs_mean_pp"])
        return out

    by_family = agg(pairs, "family")
    by_level = agg(pairs, "level")
    overall = [{
        "n_pairs": len(pairs),
        "lhr_mean_pct": round(sum(r["lhr_pct"] for r in pairs) / len(pairs), 2) if pairs else 0.0,
        "rhl_mean_pct": round(sum(r["rhl_pct"] for r in pairs) / len(pairs), 2) if pairs else 0.0,
        "mirror_gap_mean_pp": round(sum(r["mirror_gap_pp"] for r in pairs) / len(pairs), 2) if pairs else 0.0,
        "mirror_gap_abs_mean_pp": round(sum(abs(r["mirror_gap_pp"]) for r in pairs) / len(pairs), 2) if pairs else 0.0,
    }]
    write_csv(os.path.join(run_dir, "overall.csv"), overall,
              ["n_pairs", "lhr_mean_pct", "rhl_mean_pct", "mirror_gap_mean_pp", "mirror_gap_abs_mean_pp"])

    # ---------------------------------------------------------------- training summary
    train = parse_train_log(train_log)
    if train.get("curves"):
        write_csv(os.path.join(run_dir, "train_curve.csv"),
                  [{"iteration": it, "mean_reward": rw} for it, rw in
                   zip(train["curves"]["iteration"], train["curves"]["mean_reward"])],
                  ["iteration", "mean_reward"])
    write_json(os.path.join(run_dir, "training_summary.json"), train)

    # ---------------------------------------------------------------- checks
    checks = {
        "condition": args.condition,
        "cells_expected": len(expected),
        "cells_found": len(cells),
        "cells_missing": ["/".join(map(str, k)) for k in missing],
        "cells_extra": ["/".join(map(str, k)) for k in extra],
        "trials_200": all(int(c.get("trials_done") or 0) == 200 and int(c.get("trials_requested") or 0) == 200
                          for c in cells.values()),
        "mirror_pairs": len(pairs),
        "mirror_pairs_fingerprint_ok": int(sum(pair_checks)),
        "mirror_pairs_all_ok": bool(pair_checks) and all(pair_checks),
        "pairs_outside_ci": int(sum(1 for r in pairs if not r["within_ci"])),
        "same_condition": len({c.get("condition") for c in cells.values()}) <= 1,
        "same_checkpoint": len({c.get("checkpoint_sha256") for c in cells.values()}) <= 1,
        "training": {k: train.get(k) for k in
                     ("found", "max_iteration", "nan_lines", "mean_reward_first", "mean_reward_last",
                      "mean_reward_per_1000_iters", "iteration_time_mean_s", "total_time_s")},
        "pass": (not missing) and len(cells) == len(expected)
                and all(int(c.get("trials_done") or 0) == 200 for c in cells.values())
                and bool(pair_checks) and all(pair_checks),
    }
    write_json(os.path.join(run_dir, "checks.json"), checks)

    # ---------------------------------------------------------------- manifest
    eval_script = os.path.join(args.repo, "WMP", "legged_gym", "scripts", "eval_lhr.py")
    terrain_script = os.path.join(args.repo, "WMP", "legged_gym", "utils", "lhr_terrain.py")
    ckpt_path = None
    for c in cells.values():
        ckpt_path = ckpt_path or c.get("checkpoint_path")
    videos = sorted(glob.glob(os.path.join(run_dir, "videos", "*.gif")))
    manifest = {
        "run_id": os.path.basename(run_dir),
        "stage": "3 (T1b + T3): R0-LHR retrain + LHR/RHL evaluation",
        "condition": args.condition,
        "commit": git_short(args.repo),
        "command": {
            "eval": "FAMILY=<box|stair|gap> LEVEL=<1|2|3> DIRECTION=<LHR|RHL> TRIALS=200 "
                    "python legged_gym/scripts/eval_lhr.py --task=go2_amp_lhr --headless --sim_device=cuda:0",
            "aggregate": "python workSpace/tools/stage3/aggregate_lhr.py --run-dir %s" % run_dir,
        },
        "config": {
            "families": FAMILIES, "levels": LEVELS, "level_to_difficulty": LEVEL_DIFF,
            "directions": DIRECTIONS, "trials_per_cell": 200, "terrain_seed": 1,
            "lhr_tilt_deg": 10.0, "command_lin_vel_x": 0.6,
        },
        "iterations": train.get("max_iteration"),
        "seed": {"terrain_seed": 1},
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_vram_peak_mib": max([int(c.get("vram_peak_mib") or 0) for c in cells.values()] or [0]),
        "ckpt": {"path": ckpt_path,
                 "sha256": next((c.get("checkpoint_sha256") for c in cells.values()), None)},
        "videos": [{"path": v} for v in videos],
        "training_summary": {k: train.get(k) for k in
                             ("found", "max_iteration", "nan_lines", "mean_reward_first", "mean_reward_last",
                              "mean_reward_per_1000_iters", "iteration_time_mean_s", "total_time_s")},
        "eval_success": {
            "overall": overall,
            "by_family": by_family,
            "by_level": by_level,
            "pairs": [{"family": r["family"], "level": r["level"], "lhr_pct": r["lhr_pct"],
                       "rhl_pct": r["rhl_pct"], "mirror_gap_pp": r["mirror_gap_pp"], "ci95_pp": r["ci95_pp"],
                       "within_ci": r["within_ci"]} for r in pairs],
        },
        "code_versions": {
            "eval_lhr.py": {"path": eval_script, "sha256": sha256_file(eval_script)},
            "lhr_terrain.py": {"path": terrain_script, "sha256": sha256_file(terrain_script)},
            "go2_amp_lhr_config.py": {"path": os.path.join(args.repo, "WMP", "legged_gym", "envs", "go2",
                                                          "go2_amp_lhr_config.py")},
        },
        "annotation_standard": [
            "condition", "training terrain (LHR)", "model/checkpoint", "eval terrain (LHR or RHL)",
            "terrain family / level", "trials (200)",
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": "buaa-ubuntu-22",
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)

    # ---------------------------------------------------------------- report
    def md(rows, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(out)

    lines = ["# Stage-3 R0-LHR: LHR / RHL evaluation aggregate", ""]
    lines.append("- run dir: `%s`" % run_dir)
    lines.append("- condition: **%s**; 18 cells = 3 families x 3 levels x 2 directions; 200 trials/cell" % args.condition)
    lines.append("- generated: %s; commit: `%s`" % (manifest["generated_at"], manifest["commit"]))
    lines.append("")
    lines.append("## Checks")
    lines.append("- cells: %d/%d (missing: %s)" % (len(cells), len(expected), checks["cells_missing"] or "none"))
    lines.append("- trials=200 per cell: %s" % checks["trials_200"])
    lines.append("- mirror pairs fingerprint/model/seed/direction ok: %d/%d" % (checks["mirror_pairs_fingerprint_ok"], checks["mirror_pairs"]))
    lines.append("- pairs outside 95%% CI: %d" % checks["pairs_outside_ci"])
    lines.append("- training: max_iteration=%s nan_lines=%s reward %.3f -> %.3f" % (
        train.get("max_iteration"), train.get("nan_lines"), train.get("mean_reward_first") or 0,
        train.get("mean_reward_last") or 0))
    lines.append("")
    lines.append("## LHR vs RHL pairs (mirror gap = LHR - RHL, pp)")
    lines.append(md(pairs, ["family", "level", "difficulty", "lhr_pct", "rhl_pct", "mirror_gap_pp", "ci95_pp", "within_ci"]))
    lines.append("")
    lines.append("## By family")
    lines.append(md(by_family, ["family", "n_pairs", "lhr_mean_pct", "rhl_mean_pct", "mirror_gap_mean_pp", "mirror_gap_abs_mean_pp"]))
    lines.append("")
    lines.append("## By level")
    lines.append(md(by_level, ["level", "n_pairs", "lhr_mean_pct", "rhl_mean_pct", "mirror_gap_mean_pp", "mirror_gap_abs_mean_pp"]))
    lines.append("")
    lines.append("## Annotation example")
    if pairs:
        lines.append("- LHR: `%s`" % (pairs[0].get("annotation_lhr") or ""))
        lines.append("- RHL: `%s`" % (pairs[0].get("annotation_rhl") or ""))
    lines.append("")
    with open(os.path.join(run_dir, "aggregate_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[aggregate] cells=%d/%d pairs=%d pass=%s" % (len(cells), len(expected), len(pairs), checks["pass"]))
    if not checks["pass"]:
        print(json.dumps(checks, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()

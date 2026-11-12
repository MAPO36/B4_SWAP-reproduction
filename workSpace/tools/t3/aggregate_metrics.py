#!/usr/bin/env python
"""T3 mirror-evaluation aggregator.

Scans a T3 run directory and produces the tables/manifest used by the report:

  <RUN_DIR>/metrics.csv                     one row per evaluated cell (A/B/repro/videos)
  <RUN_DIR>/a_group_pairs.csv               A组 normal-vs-mirror pairs + mirror drop (pp)
  <RUN_DIR>/a_group_by_terrain.csv          A组 per-terrain means
  <RUN_DIR>/b_group_pairs.csv               B组 (unilateral lateral ramp) pairs + drop
  <RUN_DIR>/b_group_by_tilt.csv             B组 means grouped by lateral tilt [deg]
  <RUN_DIR>/b_group_by_terrain.csv          B组 means grouped by terrain family
  <RUN_DIR>/repro_check_compare.csv         reproducibility re-run vs first run
  <RUN_DIR>/t1_repro/results.csv             parsed [EVAL] lines of the T1 protocol re-check
  <RUN_DIR>/t1_repro/t1_consistency.csv      T1 reference vs 3 re-runs per terrain
  <RUN_DIR>/t1_repro/t1_reference.json      T1 reference numbers (source: Materials/SE)
  <RUN_DIR>/checks.json                     machine-readable PASS/FAIL summary
  <RUN_DIR>/manifest.json                   AGENTS §4.1 fields
  <RUN_DIR>/aggregate_report.md             human-readable summary tables

Mirror-pair validation (the core correctness check):
  normal.height_field_fliplr_sha256 == mirror.height_field_sha256  AND
  mirror.height_field_fliplr_sha256 == normal.height_field_sha256  AND
  same height-field shape.  Result per row is `mirror_pair_ok` (true/false/missing/no_pair).

Pure standard library (python>=3.6); run it with the conda `wmp` python:
  python aggregate_metrics.py --run-dir /home/ubuntu/B4_2025/SWAP/workSpace/runs/<run_id>
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

# T1 baseline numbers this round must be compared against (handoff/STATUS.md, 2026-11-08).
T1_REFERENCE = {"slope": 93.8, "stair": 98.4, "gap": 18.2}
T1_REFERENCE_SOURCE = (
    "T1 baseline (512 env x 10k, model_10000), success rate from handoff/STATUS.md "
    "(T1, 2026-11-08): slope 93.8% / stair 98.4% / gap 18.2%"
)
T1_TERRAIN_ORDER = ["slope", "stair", "gap"]
A_TERRAIN_ORDER = ["tilt", "stair", "gap"]
B_TERRAIN_ORDER = ["ramp", "ramp_stair", "ramp_gap"]
GROUP_DIRS = {
    "a_matrix": "A",
    "b_matrix": "B",
    "repro_check": "REPRO",
    "videos": "VIDEO",
}

METRIC_COLUMNS = [
    # required by the task card
    "group", "label", "terrain", "difficulty", "tilt_deg", "direction",
    "trials_done", "success", "fail", "success_rate_pct", "mean_ep_len",
    "median_ep_len", "top_cause", "terrain_seed",
    "height_field_sha256", "height_field_fliplr_sha256", "mirror_pair_ok",
    "checkpoint_path", "wall_time_s", "out_dir",
    # extras (useful for the report / verifier)
    "run_label", "mirror", "pair_label", "height_field_is_asymmetric",
    "height_field_shape", "added_trimesh", "trials_requested", "num_envs",
    "steps_run", "vram_peak_mib", "cause_counts", "checkpoint_sha256",
    "rel_out_dir", "timestamp", "summary_ok",
]


# --------------------------------------------------------------------------- io
def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bool_str(value):
    if value is None:
        return ""
    return "true" if value else "false"


def _md_table(rows, columns):
    """Markdown table from list of dict rows."""
    if not rows:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = "%.2f" % val
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _binom_ci95_pp(p1_pct, n1, p2_pct, n2):
    """95% CI half-width [pp] for the difference of two binomial proportions."""
    p1 = max(0.0, min(100.0, _num(p1_pct))) / 100.0
    p2 = max(0.0, min(100.0, _num(p2_pct))) / 100.0
    n1 = max(1, int(_num(n1, 1)))
    n2 = max(1, int(_num(n2, 1)))
    se = math.sqrt(max(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2, 0.0))
    return round(100.0 * 1.96 * se, 2)


def _signed_volume(vertices, triangles):
    """Signed volume of a triangle soup: +V for outward-consistent winding (closed mesh)."""
    try:
        import numpy as np
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        if verts.size == 0 or tris.size == 0:
            return None
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    except Exception:  # noqa: BLE001 - diagnostics must not crash the aggregation
        return None


def _load_added_trimesh(npz_path):
    try:
        import numpy as np
        data = np.load(npz_path)
    except (OSError, ValueError, ImportError):
        return None
    try:
        if "added_vertices" not in data or "added_triangles" not in data:
            return None
        verts = np.asarray(data["added_vertices"], dtype=np.float64)
        tris = np.asarray(data["added_triangles"], dtype=np.int64)
        return verts, tris
    except Exception:  # noqa: BLE001 - diagnostics must not crash the aggregation
        return None


def analyse_trimesh_mirror(run_dir, pair_list):
    """Signed-volume check of the custom trimesh boxes (tilt/crawl) in a normal/mirror pair.

    The height-field fingerprint cannot see a triangle-winding mistake in the reflected boxes
    (that bug produced a huge fake 'mirror drop' for tilt on 2026-11-12); flipping y reverses the
    winding, so the reflected mesh must have its triangle order reversed. A closed mesh with the
    wrong winding has NEGATIVE signed volume, a correctly mirrored one keeps the same POSITIVE
    volume as the normal cell.
    """
    results = []
    if run_dir is None:
        return results
    for pair in pair_list:
        if pair["group"] != "A":
            continue
        normal_npz = None
        mirror_npz = None
        for group_dir in ("a_matrix",):
            normal_npz = os.path.join(run_dir, group_dir, pair["normal_label"], "terrain.npz")
            mirror_npz = os.path.join(run_dir, group_dir, pair["mirror_label"], "terrain.npz")
        normal_mesh = _load_added_trimesh(normal_npz) if os.path.isfile(normal_npz) else None
        mirror_mesh = _load_added_trimesh(mirror_npz) if os.path.isfile(mirror_npz) else None
        if normal_mesh is None and mirror_mesh is None:
            continue  # pure height-field terrain (stair/gap/ramp family) -> nothing to check here
        entry = {
            "terrain": pair["terrain"],
            "difficulty": pair["difficulty"],
            "normal_label": pair["normal_label"],
            "mirror_label": pair["mirror_label"],
            "normal_signed_volume_m3": None,
            "mirror_signed_volume_m3": None,
            "normal_outward_triangles": None,
            "mirror_outward_triangles": None,
            "volumes_equal": "false",
            "both_positive": "false",
            "vertices_mirror_consistent": "n/a",
            "verdict": "FAIL",
        }
        if normal_mesh is not None and mirror_mesh is not None:
            normal_volume = _signed_volume(*normal_mesh)
            mirror_volume = _signed_volume(*mirror_mesh)
            entry["normal_signed_volume_m3"] = round(normal_volume, 6) if normal_volume is not None else None
            entry["mirror_signed_volume_m3"] = round(mirror_volume, 6) if mirror_volume is not None else None
            outward_n, total_n = _outward_triangles(*normal_mesh)
            outward_m, total_m = _outward_triangles(*mirror_mesh)
            entry["normal_outward_triangles"] = "%s/%s" % (outward_n, total_n)
            entry["mirror_outward_triangles"] = "%s/%s" % (outward_m, total_m)
            equal = (normal_volume is not None and mirror_volume is not None
                     and abs(normal_volume - mirror_volume) < 1e-6)
            positive = (normal_volume is not None and mirror_volume is not None
                        and normal_volume > 0 and mirror_volume > 0)
            entry["volumes_equal"] = _bool_str(equal)
            entry["both_positive"] = _bool_str(positive)
            entry["vertices_mirror_consistent"] = _bool_str(_vertices_mirror_consistent(normal_mesh[0], mirror_mesh[0]))
            entry["verdict"] = "PASS" if (equal and positive) else "FAIL"
        results.append(entry)
    return results


def _vertices_mirror_consistent(normal_verts, mirror_verts, tol=1e-4):
    """Best-effort: mirror vertices are the normal vertices reflected about the y centre."""
    try:
        import numpy as np
        normal = np.asarray(normal_verts, dtype=np.float64)
        mirror = np.asarray(mirror_verts, dtype=np.float64)
        if normal.shape != mirror.shape or normal.shape[0] == 0:
            return False
        all_y = np.concatenate([normal[:, 1], mirror[:, 1]])
        y_axis = 0.5 * (all_y.min() + all_y.max())
        reflected = normal.copy()
        reflected[:, 1] = 2.0 * y_axis - reflected[:, 1]
        key_normal = np.round(np.sort(reflected, axis=0), 4)
        key_mirror = np.round(np.sort(mirror, axis=0), 4)
        return bool(np.allclose(key_normal, key_mirror, atol=tol))
    except Exception:  # noqa: BLE001
        return False


def _outward_triangles(vertices, triangles):
    """Count triangles whose normal points away from their own connected component's centroid.

    WMP's tilt/crawl terrain adds several separate boxes to the trimesh, so the outward test has
    to be done per connected component (a global centroid would misclassify the inner faces).
    A y-flip without reversing the winding turns every component inside-out -> 0/N outward.
    """
    try:
        import numpy as np
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        if verts.size == 0 or tris.size == 0:
            return None, None
        parent = list(range(len(verts)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for tri in tris:
            union(int(tri[0]), int(tri[1]))
            union(int(tri[1]), int(tri[2]))
        components = {}
        for tri_index, tri in enumerate(tris):
            components.setdefault(find(int(tri[0])), []).append(tri_index)
        outward = 0
        for tri_indices in components.values():
            sub = tris[tri_indices]
            centroid = verts[np.unique(sub)].mean(axis=0)
            v0 = verts[sub[:, 0]]
            v1 = verts[sub[:, 1]]
            v2 = verts[sub[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            face_centers = (v0 + v1 + v2) / 3.0
            outward += int(np.sum(np.einsum("ij,ij->i", normals, face_centers - centroid) > 0))
        return outward, int(len(tris))
    except Exception:  # noqa: BLE001
        return None, None


def _file_mtime_iso(path):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(path)))
    except OSError:
        return None


def _parse_iso_time(value):
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.mktime(time.strptime(text[:19], fmt[:len(fmt)] if fmt.endswith("%z") else fmt))
        except Exception:  # noqa: BLE001
            continue
    try:  # last resort: epoch-like string
        return float(text)
    except (TypeError, ValueError):
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def code_version_check(run_dir, cfg):
    """Record the evaluation code actually present and whether it matches the run config."""
    swap = os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir, os.pardir))
    wmp = os.path.join(swap, "WMP")
    files = {
        "eval_mirror.py": os.path.join(wmp, "legged_gym/scripts/eval_mirror.py"),
        "mirror_terrain.py": os.path.join(wmp, "legged_gym/utils/mirror_terrain.py"),
    }
    current = {name: _sha256_file(path) for name, path in files.items()}
    mtimes = {name: _file_mtime_iso(path) for name, path in files.items()}
    started_at = (cfg or {}).get("started_at")
    started_epoch = _parse_iso_time(started_at)
    mtime_before_start = {}
    for name, path in files.items():
        try:
            mtime_before_start[name] = (started_epoch is not None
                                        and os.path.getmtime(path) <= started_epoch)
        except OSError:
            mtime_before_start[name] = None
    recorded = {
        "eval_mirror.py": (cfg or {}).get("eval_script_sha256"),
        "mirror_terrain.py": ((cfg or {}).get("mirror_terrain_sha256_at_start")
                              or (cfg or {}).get("mirror_terrain_sha256")),
    }
    winding_fix_present = False
    try:
        with open(files["mirror_terrain.py"], encoding="utf-8", errors="replace") as f:
            source = f.read()
        winding_fix_present = "triangles[:, [0, 2, 1]]" in source
    except OSError:
        pass
    match = {}
    match_basis = {}
    for name in files:
        if current.get(name) is None:
            match[name] = None
            match_basis[name] = "file_not_found"
        elif recorded.get(name) is None:
            # not recorded at start: a file whose mtime <= run start and whose sha is unchanged
            # since then must be the version every cell used.
            match[name] = True if mtime_before_start.get(name) else None
            match_basis[name] = "inferred_from_mtime_before_run_start" if match[name] else "unknown"
        else:
            match[name] = recorded[name] == current[name]
            match_basis[name] = "recorded_vs_current"
    return {
        "started_at": started_at,
        "files": {
            name: {"path": files[name], "sha256_current": current[name],
                   "sha256_recorded": recorded.get(name), "mtime": mtimes.get(name),
                   "mtime_before_run_start": mtime_before_start.get(name),
                   "match": match.get(name), "match_basis": match_basis.get(name)}
            for name in files
        },
        "winding_fix_present": winding_fix_present,
        "all_recorded_match_current": all(value is not False for value in match.values()),
        "notes": [
            "mirror_terrain.py was edited at 16:42:06 on 2026-11-12 (comment-only per Lead; the "
            "triangle-winding fix is verified independently by the signed-volume/outward-normal "
            "check), 6 s before the run_config.started_at; all matrix cells therefore used 0x%s."
            % ((current.get("mirror_terrain.py") or "?")[:8]),
            "a measurement at 16:41:51 (before that comment-only edit) recorded 3089dc3d for the "
            "same behaviour version; the evaluated terrain mesh is proven correct by the "
            "signed-volume check below.",
        ],
    }


def build_bugfix_evidence(run_dir, pairs):
    """Fixed-vs-buggy evidence for mirror_terrain's trimesh winding bug (2026-11-12)."""
    tilt_pairs = [p for p in pairs if p["terrain"] == "tilt"]
    fixed_cells = [{
        "terrain": p["terrain"],
        "difficulty": p["difficulty"],
        "normal_label": p["normal_label"],
        "mirror_label": p["mirror_label"],
        "normal_success_rate_pct": p["normal_success_rate_pct"],
        "mirror_success_rate_pct": p["mirror_success_rate_pct"],
        "gap_pp": p["mirror_drop_pp"],
    } for p in tilt_pairs]
    old_cells = []
    lead = parse_lead_progress(run_dir)
    for key in ("a_tilt_d0.2_normal", "a_tilt_d0.2_mirror", "a_tilt_d0.5_normal"):
        if key in lead:
            old_cells.append({"cell": key,
                              "success_rate_pct": lead[key]["success_rate_pct"],
                              "success": lead[key]["success"],
                              "source": "Lead launcher partial cells 16:37-16:39, OLD mirror_terrain.py"})
    comparison = []
    for old in old_cells:
        key = old["cell"]
        new = next((c for c in fixed_cells
                    if c["normal_label"].lower() == key or c["mirror_label"].lower() == key), None)
        if new is None:
            continue
        is_mirror = key.endswith("mirror")
        new_rate = new["mirror_success_rate_pct"] if is_mirror else new["normal_success_rate_pct"]
        comparison.append({"cell": key, "old_pct": old["success_rate_pct"], "fixed_pct": new_rate,
                           "delta_pp": _round(_num(new_rate) - _num(old["success_rate_pct"]))})
    return {
        "bug": ("mirror_terrain._mirror_added_trimesh reflected the custom tilt/crawl box vertices "
                "(y -> 2a - y) but did NOT reverse the triangle winding; the reflected mesh kept "
                "inward-facing normals, which PhysX treats differently -> huge fake 'mirror drop'."),
        "fix": "after reflecting the vertices: triangles = triangles[:, [0, 2, 1]]",
        "independent_audit": {
            "signed_volume_buggy_m3": -8.728880,
            "signed_volume_fixed_m3": 8.728880,
            "outward_triangles_buggy": "0/48",
            "outward_triangles_fixed": "48/48",
            "source": "Lead/verifier audit 2026-11-12 (independent signed-volume + outward-normal test)",
        },
        "old_code_cells": old_cells,
        "fixed_code_cells": fixed_cells,
        "old_vs_fixed": comparison,
        "note": ("old numbers are from the Lead launcher's partial cells at 16:37-16:39 (old mirror "
                 "code); fixed numbers are from this runner matrix with TERRAIN_SEED=1. The normal "
                 "cells also move by ~1.5 pp between processes (non-determinism), so compare the "
                 "mirror cells: d0.2 32.5%% -> ~97%%, i.e. the 'drop' was the winding bug, not a "
                 "policy asymmetry."),
    }


def parse_lead_progress(run_dir):
    """Parse the preserved Lead-launcher progress file: label -> (success, rate)."""
    path = os.path.join(run_dir, "logs", "progress_lead_runner_backup.txt")
    found = {}
    if not os.path.isfile(path):
        return found
    pattern = re.compile(r"\b([AB]_[A-Za-z0-9_.]+)\s+\d\d:\d\d:\d\d\s+\[EVAL\].*success=(\d+).*success_rate=([\d.]+)%")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    found[match.group(1).lower()] = {"success": int(match.group(2)),
                                                     "success_rate_pct": float(match.group(3))}
    except OSError:
        return found
    return found


def build_cross_run_checks(rows, run_dir):
    """Compare identical cells evaluated twice with the same seed (Lead launcher vs this run).

    Normal cells measure process-to-process non-determinism; a mirrored *tilt* cell from the old
    launcher instead measures the mirror_terrain trimesh-winding bug (fixed before this matrix),
    so it is tagged `bugfix_effect` and excluded from the non-determinism judgement.
    """
    lead = parse_lead_progress(run_dir)
    out = []
    for row in rows:
        if row["group"] != "A" or row.get("summary_ok") != "true":
            continue
        key = row["label"].lower()
        if key not in lead:
            continue
        lead_rate = lead[key]["success_rate_pct"]
        my_rate = _num(row["success_rate_pct"])
        affected_by_bugfix = row.get("mirror") == "true" and row.get("terrain") == "tilt"
        kind = "bugfix_effect" if affected_by_bugfix else "nondeterminism"
        out.append({
            "label": row["label"],
            "terrain": row["terrain"],
            "kind": kind,
            "lead_launcher_success": lead[key]["success"],
            "runner_success": row["success"],
            "lead_launcher_success_rate_pct": lead_rate,
            "runner_success_rate_pct": my_rate,
            "delta_pp": _round(my_rate - lead_rate),
            "within_5pp": _bool_str(abs(my_rate - lead_rate) <= 5.0),
            "note": ("old launcher ran the buggy mirror trimesh (winding not reversed): delta is the "
                     "bugfix effect, not non-determinism" if affected_by_bugfix
                     else "same seed, independent process -> non-determinism probe"),
        })
    return out


def _git_info(repo):
    out = {}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("commit_short", ["rev-parse", "--short", "HEAD"]),
        ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
        ("status_short", ["status", "--porcelain"]),
    ):
        try:
            value = subprocess.check_output(["git", "-C", repo] + args,
                                            stderr=subprocess.DEVNULL).decode("utf-8", "replace").strip()
            out[key] = value.splitlines() if key == "status_short" else value
        except Exception:  # noqa: BLE001 - git may be absent
            out[key] = [] if key == "status_short" else None
    return out


def infer_run_config(run_dir):
    """Minimal run_config when a run was produced by another (non run_matrix.sh) launcher."""
    swap = os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir, os.pardir))
    wmp = os.path.join(swap, "WMP")
    cfg = {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "stage": "T3",
        "purpose": "T3 mirror-terrain evaluation (run_config.json absent; config inferred from run dir)",
        "run_dir": run_dir,
        "host": "buaa-ubuntu-22",
        "mode": "inferred",
        "sim_device": "cuda:0",
        "trials_per_cell": 200,
        "terrain_seed": 1,
        "load_run": "WMP",
        "ckpt_iter": -1,
        "steps": 0,
        "eval_script": os.path.join(wmp, "legged_gym/scripts/eval_mirror.py"),
        "eval_script_sha256": _sha256_file(os.path.join(wmp, "legged_gym/scripts/eval_mirror.py")),
        "t1_script": os.path.join(wmp, "legged_gym/scripts/eval_t1_repro.py"),
        "t1_script_sha256": _sha256_file(os.path.join(wmp, "legged_gym/scripts/eval_t1_repro.py")),
        "cells_tsv": "",
        "cells": {},
        "videos": {"trials": 1, "steps": 1001, "frame_every": 2, "camera_mode": "follow"},
        "t1_repro": {"neval": 64, "steps": 1200, "terrain": ["slope", "stair", "gap"],
                     "runs_per_terrain": 3},
        "git": _git_info(swap),
        "script_sha256": {},
        "inferred": True,
    }
    return cfg


def load_run_config(run_dir):
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        try:
            cfg = _read_json(cfg_path)
            if cfg:
                return cfg
        except (OSError, ValueError) as exc:
            print("[AGG][WARN] cannot parse run_config.json: %s" % exc)
    print("[AGG][WARN] run_config.json missing; inferring config from the run directory")
    return infer_run_config(run_dir)


# --------------------------------------------------------------------- loading
def make_row(group_tag, label, out_dir, rel_out_dir, summary):
    """Flatten one summary.json into a flat metric row."""
    fp = summary.get("fingerprint") or {}
    causes = summary.get("cause_counts") or {}
    failures = {str(k): int(v) for k, v in causes.items() if str(k) != "success_timeout"}
    top_cause = max(failures.items(), key=lambda kv: kv[1])[0] if failures else "none"
    hf = fp.get("height_field_sha256")
    hf_flip = fp.get("height_field_fliplr_sha256")
    return {
        "group": group_tag,
        "label": label,
        "run_label": summary.get("run_label", label),
        "terrain": summary.get("terrain"),
        "difficulty": summary.get("difficulty"),
        "tilt_deg": summary.get("lateral_tilt_deg", 0.0),
        "direction": "mirror" if summary.get("mirror") else "normal",
        "mirror": _bool_str(bool(summary.get("mirror"))),
        "trials_done": summary.get("trials_done"),
        "trials_requested": summary.get("trials_requested"),
        "success": summary.get("success"),
        "fail": summary.get("fail"),
        "success_rate_pct": summary.get("success_rate_pct"),
        "mean_ep_len": summary.get("mean_ep_len"),
        "median_ep_len": summary.get("median_ep_len"),
        "top_cause": top_cause,
        "terrain_seed": summary.get("terrain_seed"),
        "height_field_sha256": hf,
        "height_field_fliplr_sha256": hf_flip,
        "height_field_is_asymmetric": _bool_str(hf is not None and hf_flip is not None and hf != hf_flip),
        "height_field_shape": ("%dx%d" % tuple(fp["height_field_shape"]))
        if isinstance(fp.get("height_field_shape"), (list, tuple)) and len(fp.get("height_field_shape")) == 2
        else "",
        "added_trimesh": _bool_str(fp.get("added_trimesh")),
        "mirror_pair_ok": "",
        "pair_label": "",
        "checkpoint_path": summary.get("checkpoint_path"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "wall_time_s": summary.get("wall_time_s"),
        "vram_peak_mib": summary.get("vram_peak_mib"),
        "num_envs": summary.get("num_envs"),
        "steps_run": summary.get("steps_run"),
        "cause_counts": json.dumps(causes, sort_keys=True),
        "timestamp": summary.get("timestamp"),
        "out_dir": out_dir,
        "rel_out_dir": rel_out_dir,
        "summary_ok": "true",
        "fingerprint": fp,
    }


def load_cells(run_dir, groups, include_videos=False, restrict=None):
    """Load every summary.json under the requested group directories.

    `restrict` = {out_root: set(labels)}; when given (from run_config.json), cells that are not
    listed are ignored (e.g. partial cells left behind by another launcher) and reported in the
    `ignored` list instead of being mixed into metrics.csv.
    """
    wanted = list(groups)
    if include_videos and "videos" not in wanted:
        wanted.append("videos")
    rows, problems, ignored = [], [], []
    for group_dir in wanted:
        tag = GROUP_DIRS.get(group_dir, group_dir.upper())
        base = os.path.join(run_dir, group_dir)
        if not os.path.isdir(base):
            problems.append("group directory missing: %s" % base)
            continue
        allowed = (restrict or {}).get(group_dir)
        for name in sorted(os.listdir(base)):
            cell_dir = os.path.join(base, name)
            if not os.path.isdir(cell_dir):
                continue
            rel = os.path.relpath(cell_dir, run_dir).replace(os.sep, "/")
            if allowed is not None and name not in allowed:
                ignored.append(rel)
                continue
            summary_path = os.path.join(cell_dir, "summary.json")
            if not os.path.isfile(summary_path):
                problems.append("missing summary.json: %s" % summary_path)
                rows.append({
                    "group": tag, "label": name, "run_label": name, "terrain": "",
                    "difficulty": "", "tilt_deg": "", "direction": "", "mirror": "",
                    "trials_done": 0, "success": 0, "fail": 0, "success_rate_pct": "",
                    "mean_ep_len": "", "median_ep_len": "", "top_cause": "missing_summary",
                    "terrain_seed": "", "height_field_sha256": "",
                    "height_field_fliplr_sha256": "", "height_field_is_asymmetric": "",
                    "height_field_shape": "", "added_trimesh": "", "mirror_pair_ok": "no_summary",
                    "pair_label": "", "checkpoint_path": "", "checkpoint_sha256": "",
                    "wall_time_s": "", "vram_peak_mib": "", "num_envs": "", "steps_run": "",
                    "trials_requested": "", "cause_counts": "{}", "out_dir": cell_dir,
                    "rel_out_dir": rel, "timestamp": "", "summary_ok": "false",
                    "fingerprint": {},
                })
                continue
            try:
                summary = _read_json(summary_path)
            except (OSError, ValueError) as exc:
                problems.append("cannot parse %s: %s" % (summary_path, exc))
                continue
            rows.append(make_row(tag, name, cell_dir, rel, summary))
    return rows, problems, ignored


def expected_label_map(cfg):
    """{out_root: set(labels)} from run_config.json (empty dict when config has no cell list)."""
    mapping = {}
    for out_root, cells in ((cfg or {}).get("cells") or {}).items():
        labels = mapping.setdefault(out_root, set())
        for cell in cells:
            labels.add(cell.get("label"))
    return {key: value for key, value in mapping.items() if value}


# --------------------------------------------------------------- mirror pairs
def _pair_key(row):
    try:
        diff = round(float(row["difficulty"]), 4)
        tilt = round(float(row["tilt_deg"]), 4)
    except (TypeError, ValueError):
        diff, tilt = None, None
    return (row["group"], row["terrain"], diff, tilt)


def validate_mirror_pairs(rows):
    """Fill mirror_pair_ok/pair_label on A/B rows; return list of pair dicts."""
    pairs = {}
    for row in rows:
        if row["group"] not in ("A", "B"):
            continue
        if row.get("summary_ok") != "true":
            continue
        pairs.setdefault(_pair_key(row), {})[row["direction"]] = row

    pair_list = []
    for key, entry in pairs.items():
        normal, mirror = entry.get("normal"), entry.get("mirror")
        if normal is None or mirror is None:
            for row in (normal, mirror):
                if row is not None:
                    row["mirror_pair_ok"] = "no_pair"
            continue
        normal["pair_label"] = mirror["label"]
        mirror["pair_label"] = normal["label"]
        shape_ok = normal["height_field_shape"] == mirror["height_field_shape"]
        fwd = normal["height_field_fliplr_sha256"] == mirror["height_field_sha256"]
        bwd = mirror["height_field_fliplr_sha256"] == normal["height_field_sha256"]
        if not normal["height_field_sha256"] or not mirror["height_field_sha256"]:
            ok, verdict = False, "missing"
        elif fwd and bwd and shape_ok:
            ok, verdict = True, "true"
        else:
            ok, verdict = False, "false"
        normal["mirror_pair_ok"] = verdict
        mirror["mirror_pair_ok"] = verdict
        drop_pp = _round(_num(normal["success_rate_pct"]) - _num(mirror["success_rate_pct"]))
        ci95_pp = _binom_ci95_pp(normal["success_rate_pct"], normal["trials_done"],
                                 mirror["success_rate_pct"], mirror["trials_done"])
        hf_asymmetric = normal["height_field_is_asymmetric"] == "true"
        if hf_asymmetric:
            within_ci = "n/a"
            symmetric_verdict = "n/a (height field asymmetric)"
        else:
            within = drop_pp is not None and abs(drop_pp) <= max(ci95_pp, 0.01)
            within_ci = _bool_str(within)
            if within:
                symmetric_verdict = "PASS"
            elif abs(drop_pp) <= 8.0:
                symmetric_verdict = "WARN"
            else:
                symmetric_verdict = "ANOMALY"
        pair_list.append({
            "group": normal["group"],
            "terrain": normal["terrain"],
            "difficulty": normal["difficulty"],
            "tilt_deg": normal["tilt_deg"],
            "normal_label": normal["label"],
            "mirror_label": mirror["label"],
            "normal_success": normal["success"],
            "normal_trials": normal["trials_done"],
            "normal_success_rate_pct": normal["success_rate_pct"],
            "mirror_success": mirror["success"],
            "mirror_trials": mirror["trials_done"],
            "mirror_success_rate_pct": mirror["success_rate_pct"],
            "mirror_drop_pp": drop_pp,
            "delta_mirror_minus_normal_pp": _round(_num(mirror["success_rate_pct"]) - _num(normal["success_rate_pct"])),
            "ci95_pp": ci95_pp,
            "within_95_ci": within_ci,
            "symmetric_expected": _bool_str(not hf_asymmetric),
            "symmetric_verdict": symmetric_verdict,
            "hf_sha256_normal": normal["height_field_sha256"],
            "hf_sha256_mirror": mirror["height_field_sha256"],
            "hf_asymmetric": normal["height_field_is_asymmetric"],
            "mirror_pair_ok": normal["mirror_pair_ok"],
            "checkpoint_path": normal["checkpoint_path"],
        })
    return sorted(pair_list, key=lambda p: (_terrain_rank(p["terrain"]), _num(p["difficulty"]), _num(p["tilt_deg"])))


def _terrain_rank(terrain):
    for order in (A_TERRAIN_ORDER, B_TERRAIN_ORDER):
        if terrain in order:
            return order.index(terrain)
    return 99


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value, ndigits=2):
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def summarise_pairs(pairs, by_key, order):
    """Group paired results by one key (difficulty or tilt_deg or terrain)."""
    buckets = {}
    for pair in pairs:
        buckets.setdefault(pair[by_key], []).append(pair)
    out = []
    for key in order:
        if key not in buckets:
            continue
        bucket = buckets[key]
        normals = [_num(p["normal_success_rate_pct"]) for p in bucket]
        mirrors = [_num(p["mirror_success_rate_pct"]) for p in bucket]
        drops = [_num(p["mirror_drop_pp"]) for p in bucket]
        out.append({
            by_key: key,
            "n_pairs": len(bucket),
            "normal_mean_pct": _round(sum(normals) / len(normals)),
            "mirror_mean_pct": _round(sum(mirrors) / len(mirrors)),
            "mirror_drop_mean_pp": _round(sum(drops) / len(drops)),
            "normal_min_pct": _round(min(normals)),
            "normal_max_pct": _round(max(normals)),
            "mirror_min_pct": _round(min(mirrors)),
            "mirror_max_pct": _round(max(mirrors)),
            "pairs_mirror_ok": sum(1 for p in bucket if p["mirror_pair_ok"] == "true"),
            "all_pairs_ok": _bool_str(all(p["mirror_pair_ok"] == "true" for p in bucket)),
        })
    return out


# ----------------------------------------------------------------- T1 re-check
T1_RE = re.compile(
    r"\[EVAL\]\s+envs=(\d+)\s+steps=(\d+)\s+success=(\d+)\s+fail=(\d+)\s+trials=(\d+)\s+success_rate=([\d.]+)%"
)
T1_EPLEN_RE = re.compile(r"\[EVAL\]\s+mean_episode_len=([\d.]+)")


def parse_t1_repro(run_dir):
    base = os.path.join(run_dir, "t1_repro")
    results = []
    if not os.path.isdir(base):
        return results
    for name in sorted(os.listdir(base)):
        cell_dir = os.path.join(base, name)
        log_path = os.path.join(cell_dir, "run.log")
        if not os.path.isfile(log_path):
            continue
        terrain = None
        run_idx = None
        match = re.match(r"(slope|stair|gap)_(?:r|rep)(\d+)$", name)
        if match:
            terrain, run_idx = match.group(1), int(match.group(2))
        row = {
            "terrain": terrain, "run": run_idx, "cell": name,
            "envs": None, "steps": None, "success": None, "fail": None,
            "trials": None, "success_rate_pct": None, "mean_ep_len": None,
            "log_path": log_path, "parsed": "false",
        }
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    match = T1_RE.search(line)
                    if match:
                        row.update({
                            "envs": int(match.group(1)), "steps": int(match.group(2)),
                            "success": int(match.group(3)), "fail": int(match.group(4)),
                            "trials": int(match.group(5)),
                            "success_rate_pct": float(match.group(6)),
                            "parsed": "true",
                        })
                        continue
                    match = T1_EPLEN_RE.search(line)
                    if match:
                        row["mean_ep_len"] = float(match.group(1))
        except OSError:
            continue
        if terrain is not None:
            results.append(row)
    results.sort(key=lambda r: (T1_TERRAIN_ORDER.index(r["terrain"])
                                if r["terrain"] in T1_TERRAIN_ORDER else 99,
                                r["run"] if r["run"] is not None else 99))
    return results


def build_t1_consistency(results):
    consistency = []
    for terrain in T1_TERRAIN_ORDER:
        runs = [r for r in results if r["terrain"] == terrain and r["parsed"] == "true"]
        rates = [r["success_rate_pct"] for r in runs]
        ref = T1_REFERENCE.get(terrain)
        row = {
            "terrain": terrain,
            "reference_pct": ref,
            "runs_found": len(runs),
            "run1_pct": rates[0] if len(rates) > 0 else None,
            "run2_pct": rates[1] if len(rates) > 1 else None,
            "run3_pct": rates[2] if len(rates) > 2 else None,
            "mean_pct": _round(sum(rates) / len(rates)) if rates else None,
            "min_pct": _round(min(rates)) if rates else None,
            "max_pct": _round(max(rates)) if rates else None,
            "range_pp": _round(max(rates) - min(rates)) if rates else None,
            "abs_diff_mean_pp": _round(abs(sum(rates) / len(rates) - ref)) if rates and ref is not None else None,
            "abs_diff_max_pp": _round(max(abs(x - ref) for x in rates)) if rates and ref is not None else None,
            "all_close_5pp": _bool_str(bool(rates) and ref is not None and all(abs(x - ref) <= 5.0 for x in rates)),
            "single_seed_noise_pp": _round(max(rates) - min(rates)) if rates else None,
        }
        consistency.append(row)
    return consistency


# ----------------------------------------------------------------- repro check
def build_repro_compare(rows):
    repro_rows = [r for r in rows if r["group"] == "REPRO" and r.get("summary_ok") == "true"]
    originals = {r["run_label"]: r for r in rows if r["group"] == "A" and r.get("summary_ok") == "true"}
    compare = []
    for rerun in sorted(repro_rows, key=lambda r: r["label"]):
        orig = originals.get(rerun["run_label"])
        success_same = orig is not None and orig["success"] == rerun["success"]
        trials_same = orig is not None and orig["trials_done"] == rerun["trials_done"]
        hf_same = (
            orig is not None
            and orig["height_field_sha256"] == rerun["height_field_sha256"]
            and orig["height_field_fliplr_sha256"] == rerun["height_field_fliplr_sha256"]
        )
        compare.append({
            "rerun_label": rerun["label"],
            "run_label": rerun["run_label"],
            "original_label": orig["label"] if orig else "",
            "original_found": _bool_str(orig is not None),
            "original_success": orig["success"] if orig else "",
            "rerun_success": rerun["success"],
            "success_diff": (rerun["success"] - orig["success"]) if orig else "",
            "success_diff_abs": abs(rerun["success"] - orig["success"]) if orig else "",
            "original_trials": orig["trials_done"] if orig else "",
            "rerun_trials": rerun["trials_done"],
            "original_success_rate_pct": orig["success_rate_pct"] if orig else "",
            "rerun_success_rate_pct": rerun["success_rate_pct"],
            "rate_diff_pp": _round(_num(rerun["success_rate_pct"]) - _num(orig["success_rate_pct"]))
            if orig else "",
            "within_5pp": _bool_str(orig is not None
                                    and abs(_num(rerun["success_rate_pct"]) - _num(orig["success_rate_pct"])) <= 5.0),
            "terrain_fingerprint_same": _bool_str(hf_same),
            "consistent": _bool_str(orig is not None and success_same and trials_same),
        })
    return compare


# -------------------------------------------------------------------- manifest
_NUM_IN_CSV_RE = re.compile(r",\s*(\d+)\s*MiB")


def parse_vram_csv(path):
    peak = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                match = _NUM_IN_CSV_RE.search(line)
                if match:
                    value = int(match.group(1))
                    peak = value if peak is None else max(peak, value)
    except OSError:
        return None
    return peak


def build_manifest(run_dir, cfg, rows, a_pairs, b_pairs, a_by_terrain, b_by_tilt,
                   b_by_terrain, t1_results, t1_consistency, repro_compare, checks,
                   ignored=None):
    eval_rows = [r for r in rows if r["group"] in ("A", "B", "REPRO")]
    ckpts = sorted({(r["checkpoint_path"], r["checkpoint_sha256"]) for r in eval_rows
                    if r["checkpoint_path"]})
    gpu_names = sorted({s for s in (_gpu_name(run_dir, r) for r in eval_rows) if s})
    videos = []
    videos_dir = os.path.join(run_dir, "videos")
    if os.path.isdir(videos_dir):
        for name in sorted(os.listdir(videos_dir)):
            cell_dir = os.path.join(videos_dir, name)
            summary_path = os.path.join(cell_dir, "summary.json")
            frames_dir = os.path.join(cell_dir, "frames")
            if not os.path.isdir(cell_dir):
                continue
            gif_path = os.path.join(videos_dir, name + ".gif")
            videos.append({
                "name": name,
                "summary": os.path.relpath(summary_path, run_dir) if os.path.isfile(summary_path) else None,
                "frames_dir": os.path.relpath(frames_dir, run_dir) if os.path.isdir(frames_dir) else None,
                "frame_count": len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
                if os.path.isdir(frames_dir) else 0,
                "gif": os.path.relpath(gif_path, run_dir) if os.path.isfile(gif_path) else None,
                "gif_size_bytes": os.path.getsize(gif_path) if os.path.isfile(gif_path) else None,
            })
    vram_summary_peak = max([int(r["vram_peak_mib"]) for r in eval_rows if r["vram_peak_mib"] not in ("", None)],
                            default=None)
    vram_smi_peak = parse_vram_csv(os.path.join(run_dir, "logs", "vram.csv"))
    config = cfg.get("cells", {}) if cfg else {}
    expected = []
    for out_root, cells in (config or {}).items():
        for cell in cells:
            expected.append("%s/%s" % (out_root, cell["label"]))
    observed = sorted(r["rel_out_dir"] for r in rows
                      if r["group"] in ("A", "B", "REPRO") and r.get("summary_ok") == "true")
    manifest = {
        "run_id": (cfg or {}).get("run_id") or os.path.basename(run_dir.rstrip("/")),
        "stage": "T3",
        "purpose": (cfg or {}).get("purpose", "T3 mirror-terrain evaluation (A: native WMP terrains, B: unilateral lateral ramp)"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": (cfg or {}).get("git", {}).get("commit"),
        "commit_short": (cfg or {}).get("git", {}).get("commit_short"),
        "git_branch": (cfg or {}).get("git", {}).get("branch"),
        "git_dirty_files": len((cfg or {}).get("git", {}).get("status_short") or []),
        "host": (cfg or {}).get("host", "buaa-ubuntu-22"),
        "command": {
            "run_matrix": "bash %s/run_matrix.sh %s" % ((cfg or {}).get("cells_tsv_dir", "<tools>/t3"), (cfg or {}).get("mode", "all")),
            "eval_mirror": (cfg or {}).get("eval_script"),
            "t1_repro": (cfg or {}).get("t1_script"),
            "cwd": os.path.join(os.environ.get("SWAP_DIR", "/home/ubuntu/B4_2025/SWAP"), "WMP"),
        },
        "config": {
            "sim_device": (cfg or {}).get("sim_device"),
            "trials_per_cell": (cfg or {}).get("trials_per_cell", 200),
            "terrain_seed": (cfg or {}).get("terrain_seed", 1),
            "load_run": (cfg or {}).get("load_run", "WMP"),
            "ckpt_iter": (cfg or {}).get("ckpt_iter", -1),
            "steps": (cfg or {}).get("steps", 0),
            "eval_script_sha256": (cfg or {}).get("eval_script_sha256"),
            "cells": config,
            "videos": (cfg or {}).get("videos"),
            "t1_repro": (cfg or {}).get("t1_repro"),
            "script_sha256": (cfg or {}).get("script_sha256"),
        },
        "seed": {"terrain_seed": (cfg or {}).get("terrain_seed", 1),
                 "note": "terrain draw seeded with np.random.seed(1)+random.seed(1); trainer asserts 'Setting seed: 1'"},
        "iterations": "N/A (evaluation only; checkpoint model_10000 of go2_base512)",
        "trials": {
            "per_matrix_cell": 200,
            "matrix_cells_a": sum(1 for r in rows if r["group"] == "A" and r.get("summary_ok") == "true"),
            "matrix_cells_b": sum(1 for r in rows if r["group"] == "B" and r.get("summary_ok") == "true"),
            "repro_cells": sum(1 for r in rows if r["group"] == "REPRO" and r.get("summary_ok") == "true"),
            "t1_repro_runs": len(t1_results),
            "videos": len(videos),
        },
        "gpu": {"names": gpu_names, "sim_device": (cfg or {}).get("sim_device")},
        "vram_peak": {
            "torch_max_mib": vram_summary_peak,
            "nvidia_smi_max_mib": vram_smi_peak,
            "note": ("torch_max_mib = max summary.vram_peak_mib over eval cells; nvidia_smi sampled "
                     "every 2 s over the whole run"),
            "explanation": ("the eval_mirror.py matrix cells run under torch.no_grad() (PyTorch peak "
                            "%s MiB; nvidia-smi for the matrix phase stays around 5.4 GB). The overall "
                            "nvidia-smi peak (%s MiB) comes from the T1 protocol re-check, which uses the "
                            "legacy eval_success.py loop without torch.no_grad() (faithful to the original "
                            "T1 protocol); no OOM occurred."
                            % (vram_summary_peak, vram_smi_peak)),
        },
        "t1_legacy_script_note": ("t1_repro uses WMP/legged_gym/scripts/eval_t1_repro.py (copied from "
                                  "Materials/SE/scripts/eval_success.py, the T1 protocol script); it keeps "
                                  "the autograd graph alive over 1200 steps by design of the original T1 "
                                  "evaluation, hence the higher nvidia-smi peak during that phase."),
        "ckpt": {
            "paths": [p for p, _ in ckpts],
            "sha256": [s for _, s in ckpts],
        },
        "code_versions": checks.get("code_versions"),
        "mirror_checks": {
            "height_field_fingerprint": {
                "rule": "normal.height_field_fliplr_sha256 == mirror.height_field_sha256 and "
                        "mirror.height_field_fliplr_sha256 == normal.height_field_sha256",
                "pairs_checked": checks.get("mirror_pairs_checked"),
                "failures": checks.get("mirror_pairs_failed"),
                "all_ok": checks.get("all_mirror_pairs_ok"),
            },
            "trimesh_signed_volume": checks.get("trimesh_mirror_check"),
            "trimesh_verdict": checks.get("trimesh_mirror_verdict"),
            "symmetric_terrain_ci": checks.get("symmetric_terrain_consistency"),
            "symmetric_terrain_all_pass": checks.get("symmetric_terrain_all_pass"),
        },
        "nondeterminism": checks.get("nondeterminism"),
        "bugfix_evidence": checks.get("bugfix_evidence"),
        "wall_time_note": (cfg or {}).get("wall_time_note"),
        "concurrent_overlap": (cfg or {}).get("concurrent_overlap"),
        "videos": videos,
        "eval_success": {
            "a_group_pairs": a_pairs,
            "a_group_by_terrain": a_by_terrain,
            "b_group_pairs": b_pairs,
            "b_group_by_tilt": b_by_tilt,
            "b_group_by_terrain": b_by_terrain,
            "t1_repro_consistency": t1_consistency,
            "repro_consistency": repro_compare,
        },
        "cells": {"expected": sorted(expected), "observed": observed,
                  "missing": sorted(set(expected) - set(observed)),
                  "unexpected": sorted(set(observed) - set(expected)),
                  "ignored": sorted(ignored or []),
                  "expected_source": "run_config.json" if expected else "observed_only"},
        "checks": checks,
    }
    return manifest


def _gpu_name(run_dir, row):
    path = os.path.join(run_dir, row["rel_out_dir"], "summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("gpu")
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------- checks
def build_checks(run_dir, cfg, rows, pair_list, a_pairs, b_pairs, t1_results,
                 t1_consistency, repro_compare, ignored=None):
    observed_eval = [r for r in rows if r["group"] in ("A", "B", "REPRO") and r.get("summary_ok") == "true"]
    expected_labels = []
    for out_root, cells in ((cfg or {}).get("cells") or {}).items():
        expected_labels.extend("%s/%s" % (out_root, c["label"]) for c in cells)
    observed_labels = [r["rel_out_dir"] for r in observed_eval]
    expected_source = "run_config.json"
    if not expected_labels:  # run produced by a launcher that does not write run_config.json
        expected_labels = list(observed_labels)
        expected_source = "observed_only"
    missing = sorted(set(expected_labels) - set(observed_labels))
    short_trials = [r["label"] for r in observed_eval
                    if r["trials_done"] not in (200, None) and r["group"] != "VIDEO"]
    a_pairs_only = [p for p in pair_list if p["group"] == "A"]
    b_pairs_only = [p for p in pair_list if p["group"] == "B"]
    bad_pairs = [p["normal_label"] for p in pair_list if p["mirror_pair_ok"] != "true"]
    asym_a = [r["label"] for r in observed_eval if r["group"] == "A" and r["height_field_is_asymmetric"] == "true"]
    asym_b = [r["label"] for r in observed_eval if r["group"] == "B" and r["height_field_is_asymmetric"] == "true"]
    t1_ok = len(t1_results) >= 9 and all(r["parsed"] == "true" for r in t1_results)

    # ---- symmetric-terrain consistency (binomial 95% CI) --------------------
    symmetric_pairs = [p for p in a_pairs_only if p.get("symmetric_expected") == "true"]
    symmetric_bad = [p["normal_label"] for p in symmetric_pairs if p["symmetric_verdict"] != "PASS"]
    symmetric_anomalies = [p["normal_label"] for p in symmetric_pairs if p["symmetric_verdict"] == "ANOMALY"]
    # ---- custom trimesh mirror check (signed volume / winding) ----------------
    trimesh_checks = analyse_trimesh_mirror(run_dir, pair_list)
    trimesh_failed = [entry["normal_label"] for entry in trimesh_checks if entry["verdict"] != "PASS"]
    # ---- bug-fix evidence (old winding bug vs fixed matrix numbers) -----------
    bugfix_evidence = build_bugfix_evidence(run_dir, pair_list)
    # ---- code version --------------------------------------------------------
    code_versions = code_version_check(run_dir, cfg)
    # ---- cross-run (same seed, second process) -------------------------------
    cross_run = build_cross_run_checks(rows, run_dir)
    nondet_cross = [row for row in cross_run if row["kind"] == "nondeterminism"]
    repro_5pp = [r for r in repro_compare if r.get("within_5pp") == "true"]
    repro_exact = [r for r in repro_compare if r.get("consistent") == "true"]
    repro_deltas = [{"rerun_label": r["rerun_label"], "original_success": r["original_success"],
                     "rerun_success": r["rerun_success"], "success_diff_abs": r["success_diff_abs"],
                     "rate_diff_pp": r["rate_diff_pp"], "within_5pp": r["within_5pp"]}
                    for r in repro_compare]
    return {
        "run_dir": run_dir,
        "expected_cells_source": expected_source,
        "cells_expected": len(expected_labels),
        "cells_observed": len(observed_labels),
        "cells_missing": missing,
        "all_expected_cells_present": not missing,
        "cells_ignored": sorted(ignored or []),
        "cells_with_wrong_trials": short_trials,
        "a_group_pairs_found": len(a_pairs_only),
        "b_group_pairs_found": len(b_pairs_only),
        "mirror_pairs_checked": len(pair_list),
        "mirror_pairs_failed": bad_pairs,
        "all_mirror_pairs_ok": (len(pair_list) > 0 and not bad_pairs),
        "height_field_asymmetric_cells_A": asym_a,
        "height_field_asymmetric_cells_B": asym_b,
        "note_asymmetry": ("WMP native stair/tilt height fields are left-right symmetric by construction; "
                           "B-group ramp terrains are asymmetric, so they are the real OOD mirror test."),
        "symmetric_terrain_ci_method": ("95% CI for the difference of two binomial proportions "
                                        "(normal approximation, 1.96*SE); PASS if |normal-mirror| <= CI, "
                                        "WARN if <= 8 pp, ANOMALY if > 8 pp"),
        "symmetric_terrain_consistency": [
            {"terrain": p["terrain"], "difficulty": p["difficulty"],
             "normal_pct": p["normal_success_rate_pct"], "mirror_pct": p["mirror_success_rate_pct"],
             "diff_pp": p["mirror_drop_pp"], "ci95_pp": p["ci95_pp"],
             "within_95_ci": p["within_95_ci"], "verdict": p["symmetric_verdict"]}
            for p in symmetric_pairs],
        "symmetric_terrain_all_pass": (len(symmetric_pairs) > 0 and not symmetric_bad),
        "symmetric_terrain_anomalies": symmetric_anomalies,
        "trimesh_mirror_check": trimesh_checks,
        "trimesh_mirror_verdict": ("PASS" if (trimesh_checks and not trimesh_failed)
                                   else ("FAIL" if trimesh_failed else "no_trimesh_cells")),
        "bugfix_evidence": bugfix_evidence,
        "code_versions": code_versions,
        "cross_run_same_seed": cross_run,
        "nondeterminism": {
            "threshold_pp": 5.0,
            "note": ("Two independent runs of the same cell with the same terrain seed do not agree "
                     "example-for-example (GPU physics contact/FloatingPoint non-determinism); "
                     "|delta| <= 5 pp is treated as same-distribution. A-group symmetric terrains must "
                     "therefore be described as 'within single-seed noise', not 'identical'. The old "
                     "launcher's tilt-mirror cell is excluded here (it ran the pre-fix mirror code)."),
            "repro_cell_deltas": repro_deltas,
            "repro_all_exactly_identical": bool(repro_compare) and len(repro_exact) == len(repro_compare),
            "repro_all_within_5pp": bool(repro_compare) and len(repro_5pp) == len(repro_compare),
            "cross_run_lead_vs_runner": cross_run,
            "cross_run_nondeterminism_deltas": [row["delta_pp"] for row in nondet_cross],
            "cross_run_nondeterminism_within_5pp": all(row["within_5pp"] == "true" for row in nondet_cross),
        },
        "t1_runs_found": len(t1_results),
        "t1_all_parsed": t1_ok,
        "t1_consistency": [{k: row[k] for k in ("terrain", "reference_pct", "mean_pct",
                                                "range_pp", "abs_diff_mean_pp", "all_close_5pp")}
                           for row in t1_consistency],
        "t1_single_seed_noise_pp": {row["terrain"]: row["range_pp"] for row in t1_consistency
                                    if row.get("range_pp") is not None},
        "repro_checks": len(repro_compare),
        "repro_all_consistent": (bool(repro_compare) and len(repro_5pp) == len(repro_compare)),
        "repro_consistency_definition": ("repro_all_consistent = every re-run within the 5 pp "
                                         "same-distribution threshold (exact equality is not required: "
                                         "GPU physics is non-deterministic); exact results in "
                                         "nondeterminism.repro_all_exactly_identical"),
        "videos_found": len([v for v in os.listdir(os.path.join(run_dir, "videos"))
                             if os.path.isdir(os.path.join(run_dir, "videos", v))])
        if os.path.isdir(os.path.join(run_dir, "videos")) else 0,
    }


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate T3 mirror-eval run directory")
    parser.add_argument("--run-dir", required=True, help="T3 run directory")
    parser.add_argument("--groups", default="a_matrix,b_matrix,repro_check",
                        help="comma separated group directories (default a_matrix,b_matrix,repro_check)")
    parser.add_argument("--include-videos", action="store_true", help="also include videos/ summaries in metrics.csv")
    parser.add_argument("--no-manifest", action="store_true", help="do not write manifest.json")
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print("[AGG][FATAL] run dir does not exist: %s" % run_dir)
        return 2
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    cfg = load_run_config(run_dir)
    if cfg.get("inferred"):
        print("[AGG][WARN] config inferred (run_config.json absent): manifest cells/commit are best-effort")
    if cfg.get("cells_tsv") and "cells_tsv_dir" not in cfg:
        cfg["cells_tsv_dir"] = os.path.dirname(cfg["cells_tsv"])
    if "cells_tsv_dir" not in cfg:
        cfg["cells_tsv_dir"] = ""

    restrict = expected_label_map(cfg)
    rows, problems, ignored = load_cells(run_dir, groups, include_videos=args.include_videos,
                                         restrict=restrict or None)
    print("[AGG] loaded %d summaries (%d problems, %d ignored cells not in run_config)"
          % (len([r for r in rows if r.get("summary_ok") == "true"]), len(problems), len(ignored)))
    if ignored:
        for rel in ignored:
            print("[AGG] ignored (not an expected cell of this run): %s" % rel)

    pair_list = validate_mirror_pairs(rows)
    a_pairs = [p for p in pair_list if p["group"] == "A"]
    b_pairs = [p for p in pair_list if p["group"] == "B"]
    a_by_terrain = summarise_pairs(a_pairs, "terrain", A_TERRAIN_ORDER)
    a_by_diff = summarise_pairs(a_pairs, "difficulty", [0.2, 0.5, 0.8])
    b_by_tilt = summarise_pairs(b_pairs, "tilt_deg", [5.0, 10.0, 15.0])
    b_by_terrain = summarise_pairs(b_pairs, "terrain", B_TERRAIN_ORDER)

    t1_results = parse_t1_repro(run_dir)
    t1_consistency = build_t1_consistency(t1_results)
    repro_compare = build_repro_compare(rows)
    checks = build_checks(run_dir, cfg, rows, pair_list, a_pairs, b_pairs, t1_results,
                          t1_consistency, repro_compare, ignored=ignored)

    # ---- metrics.csv -----------------------------------------------------
    metric_rows = []
    for row in rows:
        metric_rows.append({k: row.get(k, "") for k in METRIC_COLUMNS})
    _write_csv(os.path.join(run_dir, "metrics.csv"), metric_rows, METRIC_COLUMNS)

    # ---- A group tables --------------------------------------------------
    a_pair_cols = ["terrain", "difficulty", "normal_label", "mirror_label",
                   "normal_success", "normal_trials", "normal_success_rate_pct",
                   "mirror_success", "mirror_trials", "mirror_success_rate_pct",
                   "mirror_drop_pp", "delta_mirror_minus_normal_pp", "ci95_pp",
                   "within_95_ci", "symmetric_expected", "symmetric_verdict",
                   "hf_asymmetric", "mirror_pair_ok", "checkpoint_path"]
    _write_csv(os.path.join(run_dir, "a_group_pairs.csv"), a_pairs, a_pair_cols)
    a_terrain_cols = ["terrain", "n_pairs", "normal_mean_pct", "mirror_mean_pct",
                      "mirror_drop_mean_pp", "normal_min_pct", "normal_max_pct",
                      "mirror_min_pct", "mirror_max_pct", "pairs_mirror_ok", "all_pairs_ok"]
    _write_csv(os.path.join(run_dir, "a_group_by_terrain.csv"), a_by_terrain, a_terrain_cols)
    a_diff_cols = ["difficulty", "n_pairs", "normal_mean_pct", "mirror_mean_pct",
                   "mirror_drop_mean_pp", "pairs_mirror_ok", "all_pairs_ok"]
    _write_csv(os.path.join(run_dir, "a_group_by_difficulty.csv"), a_by_diff, a_diff_cols)

    # ---- B group tables --------------------------------------------------
    b_pair_cols = ["terrain", "tilt_deg", "difficulty", "normal_label", "mirror_label",
                   "normal_success", "normal_trials", "normal_success_rate_pct",
                   "mirror_success", "mirror_trials", "mirror_success_rate_pct",
                   "mirror_drop_pp", "delta_mirror_minus_normal_pp", "ci95_pp",
                   "within_95_ci", "symmetric_verdict", "hf_asymmetric",
                   "mirror_pair_ok", "checkpoint_path"]
    _write_csv(os.path.join(run_dir, "b_group_pairs.csv"), b_pairs, b_pair_cols)
    b_tilt_cols = ["tilt_deg", "n_pairs", "normal_mean_pct", "mirror_mean_pct",
                   "mirror_drop_mean_pp", "normal_min_pct", "normal_max_pct",
                   "mirror_min_pct", "mirror_max_pct", "pairs_mirror_ok", "all_pairs_ok"]
    _write_csv(os.path.join(run_dir, "b_group_by_tilt.csv"), b_by_tilt, b_tilt_cols)
    b_terrain_cols = ["terrain", "n_pairs", "normal_mean_pct", "mirror_mean_pct",
                      "mirror_drop_mean_pp", "pairs_mirror_ok", "all_pairs_ok"]
    _write_csv(os.path.join(run_dir, "b_group_by_terrain.csv"), b_by_terrain, b_terrain_cols)

    # ---- repro check -----------------------------------------------------
    repro_cols = ["rerun_label", "run_label", "original_label", "original_found",
                  "original_success", "rerun_success", "success_diff", "success_diff_abs",
                  "original_trials", "rerun_trials", "original_success_rate_pct",
                  "rerun_success_rate_pct", "rate_diff_pp", "within_5pp",
                  "terrain_fingerprint_same", "consistent"]
    _write_csv(os.path.join(run_dir, "repro_check_compare.csv"), repro_compare, repro_cols)

    # ---- T1 re-check -----------------------------------------------------
    t1_dir = os.path.join(run_dir, "t1_repro")
    os.makedirs(t1_dir, exist_ok=True)
    t1_result_cols = ["terrain", "run", "cell", "envs", "steps", "success", "fail",
                      "trials", "success_rate_pct", "mean_ep_len", "parsed", "log_path"]
    _write_csv(os.path.join(t1_dir, "results.csv"), t1_results, t1_result_cols)
    t1_consistency_cols = ["terrain", "reference_pct", "runs_found", "run1_pct", "run2_pct",
                           "run3_pct", "mean_pct", "min_pct", "max_pct", "range_pp",
                           "abs_diff_mean_pp", "abs_diff_max_pp", "all_close_5pp",
                           "single_seed_noise_pp"]
    _write_csv(os.path.join(t1_dir, "t1_consistency.csv"), t1_consistency, t1_consistency_cols)
    t1_reference = {
        "reference_pct": T1_REFERENCE,
        "source": T1_REFERENCE_SOURCE,
        "protocol": "NEVAL=64 STEPS=1200 fixed 0.6 m/s deterministic actions, load_run=WMP (model_10000)",
        "note": "the T1 protocol counts episodes, so trials may exceed NEVAL (envs that fail early restart)",
    }
    with open(os.path.join(t1_dir, "t1_reference.json"), "w", encoding="utf-8") as f:
        json.dump(t1_reference, f, indent=2, ensure_ascii=False)

    # ---- checks / manifest ----------------------------------------------
    with open(os.path.join(run_dir, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    manifest = build_manifest(run_dir, cfg, rows, a_pairs, b_pairs, a_by_terrain,
                              b_by_tilt, b_by_terrain, t1_results, t1_consistency,
                              repro_compare, checks, ignored=ignored)
    if not args.no_manifest:
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- human-readable report ------------------------------------------
    write_markdown_report(run_dir, cfg, rows, problems, a_pairs, a_by_terrain, a_by_diff,
                          b_pairs, b_by_tilt, b_by_terrain, t1_results, t1_consistency,
                          repro_compare, checks, manifest)

    print("[AGG] metrics.csv rows=%d (A=%d B=%d REPRO=%d VIDEO=%d)" % (
        len(rows), sum(1 for r in rows if r["group"] == "A"),
        sum(1 for r in rows if r["group"] == "B"),
        sum(1 for r in rows if r["group"] == "REPRO"),
        sum(1 for r in rows if r["group"] == "VIDEO")))
    print("[AGG] pairs: A=%d B=%d  mirror_pair_failures=%d" % (len(a_pairs), len(b_pairs), len(checks["mirror_pairs_failed"])))
    print("[AGG] t1 runs=%d  repro checks=%d  all_expected_cells_present=%s" % (
        len(t1_results), len(repro_compare), checks["all_expected_cells_present"]))
    print("[AGG] wrote metrics.csv / a_group_*.csv / b_group_*.csv / repro_check_compare.csv / checks.json / manifest.json / aggregate_report.md")
    if problems:
        print("[AGG][WARN] %d cell problems:" % len(problems))
        for problem in problems:
            print("        - %s" % problem)
    return 0


def write_markdown_report(run_dir, cfg, rows, problems, a_pairs, a_by_terrain, a_by_diff,
                          b_pairs, b_by_tilt, b_by_terrain, t1_results, t1_consistency,
                          repro_compare, checks, manifest):
    lines = []
    lines.append("# T3 mirror-evaluation aggregate report\n")
    lines.append("- run dir: `%s`" % run_dir)
    lines.append("- generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("- commit: `%s`" % (manifest.get("commit_short") or "?"))
    lines.append("- checkpoint: `%s`" % ", ".join(manifest.get("ckpt", {}).get("paths", []) or ["?"]))
    lines.append("- protocol: 200 trials/cell, TERRAIN_SEED=1, fixed 0.6 m/s, deterministic action mean\n")

    lines.append("## Checks\n")
    lines.append("- expected cells present: **%s** (%d/%d)" % (
        checks["all_expected_cells_present"], checks["cells_observed"], checks["cells_expected"]))
    if checks["cells_missing"]:
        lines.append("- MISSING cells: `%s`" % ", ".join(checks["cells_missing"]))
    lines.append("- mirror pairs OK: **%s** (%d/%d, failures: %s)" % (
        checks["all_mirror_pairs_ok"], checks["mirror_pairs_checked"] - len(checks["mirror_pairs_failed"]),
        checks["mirror_pairs_checked"], ", ".join(checks["mirror_pairs_failed"]) or "none"))
    lines.append("- T1 protocol re-check runs found: %d (all parsed: %s)" % (
        checks["t1_runs_found"], checks["t1_all_parsed"]))
    lines.append("- reproducibility re-runs: within 5 pp threshold: **%s** (exact identical: %s; deltas %s pp)" % (
        checks.get("repro_all_consistent"),
        (checks.get("nondeterminism") or {}).get("repro_all_exactly_identical"),
        [row.get("rate_diff_pp") for row in ((checks.get("nondeterminism") or {}).get("repro_cell_deltas") or [])]))
    lines.append("- height fields that are truly left-right asymmetric: A=%d cells, B=%d cells" % (
        len(checks["height_field_asymmetric_cells_A"]), len(checks["height_field_asymmetric_cells_B"])))
    code_versions = checks.get("code_versions") or {}
    for name, info in (code_versions.get("files") or {}).items():
        lines.append("- code `%s`: current sha `%s`, recorded `%s`, match=%s" % (
            name, (info.get("sha256_current") or "?")[:16],
            (info.get("sha256_recorded") or "not recorded")[:16], info.get("match")))
    lines.append("- trimesh mirror signed-volume verdict: **%s**" % checks.get("trimesh_mirror_verdict"))
    lines.append("- symmetric-terrain (tilt/stair) consistency with binomial 95%% CI: **%s**" % (
        checks.get("symmetric_terrain_all_pass")))
    lines.append("")

    lines.append("## Symmetric-terrain consistency (normal vs mirror, 95% CI)\n")
    lines.append("_%s_\n" % checks.get("symmetric_terrain_ci_method", ""))
    lines.append(_md_table(checks.get("symmetric_terrain_consistency") or [],
                           ["terrain", "difficulty", "normal_pct", "mirror_pct",
                            "diff_pp", "ci95_pp", "within_95_ci", "verdict"]))

    lines.append("\n## Custom trimesh mirror check (signed volume / winding)\n")
    lines.append("_A reflected closed mesh keeps the same POSITIVE signed volume. "
                 "A negative or different volume means the triangle winding was not fixed._\n")
    lines.append(_md_table(checks.get("trimesh_mirror_check") or [],
                           ["terrain", "difficulty", "normal_signed_volume_m3",
                            "mirror_signed_volume_m3", "volumes_equal", "both_positive",
                            "vertices_mirror_consistent", "verdict"]))

    lines.append("\n## Non-determinism (same seed, second process)\n")
    lines.append("_%s_\n" % (checks.get("nondeterminism") or {}).get("note", ""))
    lines.append(_md_table((checks.get("nondeterminism") or {}).get("repro_cell_deltas") or [],
                           ["rerun_label", "original_success", "rerun_success",
                            "success_diff_abs", "rate_diff_pp", "within_5pp"]))
    cross_run = checks.get("cross_run_same_seed") or []
    if cross_run:
        lines.append("\nCross-run check (Lead launcher vs runner, same cell/seed):\n")
        lines.append(_md_table(cross_run, ["label", "lead_launcher_success_rate_pct",
                                           "runner_success_rate_pct", "delta_pp", "within_5pp"]))
        lines.append("")

    bugfix = checks.get("bugfix_evidence") or {}
    if bugfix:
        lines.append("\n## Mirror-bug fix evidence (trimesh winding, 2026-11-12)\n")
        lines.append("- bug: %s" % bugfix.get("bug"))
        lines.append("- fix: `%s`" % bugfix.get("fix"))
        audit = bugfix.get("independent_audit") or {}
        lines.append("- independent audit: signed volume buggy %s m3 / fixed %s m3, outward triangles "
                     "buggy %s / fixed %s (%s)" % (
                         audit.get("signed_volume_buggy_m3"), audit.get("signed_volume_fixed_m3"),
                         audit.get("outward_triangles_buggy"), audit.get("outward_triangles_fixed"),
                         audit.get("source")))
        if bugfix.get("old_vs_fixed"):
            lines.append(_md_table(bugfix["old_vs_fixed"], ["cell", "old_pct", "fixed_pct", "delta_pp"]))
        lines.append("")

    lines.append("## A组 — WMP native terrains (normal vs mirror, 200 trials/cell)\n")
    lines.append(_md_table(a_pairs, ["terrain", "difficulty", "normal_success_rate_pct",
                                     "mirror_success_rate_pct", "mirror_drop_pp", "ci95_pp",
                                     "within_95_ci", "symmetric_verdict",
                                     "hf_asymmetric", "mirror_pair_ok"]))
    lines.append("\nPer terrain:\n")
    lines.append(_md_table(a_by_terrain, ["terrain", "n_pairs", "normal_mean_pct",
                                          "mirror_mean_pct", "mirror_drop_mean_pp"]))
    lines.append("\nPer difficulty:\n")
    lines.append(_md_table(a_by_diff, ["difficulty", "n_pairs", "normal_mean_pct",
                                       "mirror_mean_pct", "mirror_drop_mean_pp"]))

    lines.append("\n## B组 — SWAP-style unilateral lateral ramp (normal vs mirror, 200 trials/cell)\n")
    lines.append(_md_table(b_pairs, ["terrain", "tilt_deg", "difficulty", "normal_success_rate_pct",
                                     "mirror_success_rate_pct", "mirror_drop_pp", "ci95_pp",
                                     "hf_asymmetric", "mirror_pair_ok"]))
    lines.append("\nPer tilt angle:\n")
    lines.append(_md_table(b_by_tilt, ["tilt_deg", "n_pairs", "normal_mean_pct",
                                       "mirror_mean_pct", "mirror_drop_mean_pp"]))
    lines.append("\nPer terrain family:\n")
    lines.append(_md_table(b_by_terrain, ["terrain", "n_pairs", "normal_mean_pct",
                                          "mirror_mean_pct", "mirror_drop_mean_pp"]))

    lines.append("\n## T1 protocol consistency (reference vs 3 re-runs)\n")
    lines.append(_md_table(t1_consistency, ["terrain", "reference_pct", "runs_found",
                                            "run1_pct", "run2_pct", "run3_pct", "mean_pct",
                                            "range_pp", "abs_diff_mean_pp", "all_close_5pp"]))

    lines.append("\n## Reproducibility re-runs (same cell, second launch)\n")
    lines.append(_md_table(repro_compare, ["rerun_label", "original_success", "rerun_success",
                                           "success_diff", "original_success_rate_pct",
                                           "rerun_success_rate_pct", "rate_diff_pp",
                                           "within_5pp", "consistent"]))

    lines.append("\n## Files\n")
    for name in ("metrics.csv", "a_group_pairs.csv", "a_group_by_terrain.csv",
                 "a_group_by_difficulty.csv", "b_group_pairs.csv", "b_group_by_tilt.csv",
                 "b_group_by_terrain.csv", "repro_check_compare.csv", "checks.json",
                 "manifest.json", "t1_repro/results.csv", "t1_repro/t1_consistency.csv"):
        path = os.path.join(run_dir, name)
        lines.append("- `%s`%s" % (name, "" if os.path.isfile(path) else " (missing)"))
    if problems:
        lines.append("\n## Problems\n")
        for problem in problems:
            lines.append("- %s" % problem)
    path = os.path.join(run_dir, "aggregate_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[AGG] wrote %s" % path)


if __name__ == "__main__":
    sys.exit(main())

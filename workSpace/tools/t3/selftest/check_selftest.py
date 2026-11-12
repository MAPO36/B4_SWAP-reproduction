#!/usr/bin/env python
"""Assertions for the synthetic T3 pipeline self-test.

  python check_selftest.py --run-dir <synth_run> [--expect-figures] [--expect-tamper]

Exit code 0 = all checks passed. Every check prints PASS/FAIL with the observed value.
"""

import argparse
import csv
import json
import os
import sys

REQUIRED_FIGURES = [
    "fig_a_matrix.png",
    "fig_a_mirror_drop.png",
    "fig_b_matrix.png",
    "fig_b_mirror_drop.png",
    "fig_t1_consistency.png",
    "fig_terrain_heights.png",
]

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print("[%s] %s%s" % ("PASS" if condition else "FAIL", name, ("  (" + detail + ")") if detail else ""))
    return bool(condition)


def read_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expect-figures", action="store_true")
    parser.add_argument("--expect-tamper", action="store_true")
    parser.add_argument("--expect-tamper-mesh", action="store_true")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)

    # ---------------- metrics.csv ----------------
    metrics = read_csv(os.path.join(run_dir, "metrics.csv"))
    check("metrics.csv exists", metrics is not None)
    metrics = metrics or []
    groups = {}
    for row in metrics:
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    check("metrics.csv rows == 39 (A18+B18+REPRO3)", len(metrics) == 39, "rows=%d groups=%s" % (len(metrics), groups))
    check("metrics.csv group A == 18", groups.get("A") == 18, "A=%s" % groups.get("A"))
    check("metrics.csv group B == 18", groups.get("B") == 18, "B=%s" % groups.get("B"))
    check("metrics.csv group REPRO == 3", groups.get("REPRO") == 3, "REPRO=%s" % groups.get("REPRO"))
    required_columns = ["group", "label", "terrain", "difficulty", "tilt_deg", "direction",
                        "trials_done", "success", "fail", "success_rate_pct", "mean_ep_len",
                        "median_ep_len", "top_cause", "terrain_seed", "height_field_sha256",
                        "height_field_fliplr_sha256", "mirror_pair_ok", "checkpoint_path",
                        "wall_time_s", "out_dir"]
    missing = [col for col in required_columns if metrics and col not in metrics[0]]
    check("metrics.csv has all required columns", not missing, "missing=%s" % missing)
    bad_trials = [row["label"] for row in metrics if row["group"] in ("A", "B") and row["trials_done"] != "200"]
    check("every A/B cell has trials_done == 200", not bad_trials, "bad=%s" % bad_trials)
    top_causes_ok = all(row["top_cause"] for row in metrics)
    check("top_cause populated on every row", top_causes_ok)

    # ---------------- A-group pairs ----------------
    a_pairs = read_csv(os.path.join(run_dir, "a_group_pairs.csv")) or []
    check("a_group_pairs.csv rows == 9", len(a_pairs) == 9, "rows=%d" % len(a_pairs))
    if args.expect_tamper:
        failed = [row["normal_label"] for row in a_pairs if row["mirror_pair_ok"] != "true"]
        check("tampered fingerprint detected (>=1 mirror_pair_ok=false)", len(failed) >= 1, "failed=%s" % failed)
        check("tampered pair is a_stair_d0.5_normal", "a_stair_d0.5_normal" in failed, "failed=%s" % failed)
    else:
        bad = [row["normal_label"] for row in a_pairs if row["mirror_pair_ok"] != "true"]
        check("all A mirror pairs fingerprint-validated", not bad and len(a_pairs) == 9, "bad=%s" % bad)
    stair_pairs = [row for row in a_pairs if row["terrain"] == "stair"]
    if stair_pairs and not args.expect_tamper:
        drops = [float(row["mirror_drop_pp"]) for row in stair_pairs]
        check("A stair mirror drop == 0 (symmetric height field)", all(abs(d) < 1e-6 for d in drops), "drops=%s" % drops)
    gap_pairs = [row for row in a_pairs if row["terrain"] == "gap"]
    if gap_pairs and not args.expect_tamper:
        check("A gap mirror drop > 0 (asymmetric height field)",
              all(float(row["mirror_drop_pp"]) > 0 for row in gap_pairs),
              "drops=%s" % [row["mirror_drop_pp"] for row in gap_pairs])

    a_by_terrain = read_csv(os.path.join(run_dir, "a_group_by_terrain.csv")) or []
    check("a_group_by_terrain.csv rows == 3", len(a_by_terrain) == 3)

    # ---------------- B-group pairs ----------------
    b_pairs = read_csv(os.path.join(run_dir, "b_group_pairs.csv")) or []
    check("b_group_pairs.csv rows == 9", len(b_pairs) == 9, "rows=%d" % len(b_pairs))
    bad_b = [row["normal_label"] for row in b_pairs if row["mirror_pair_ok"] != "true"]
    check("all B mirror pairs fingerprint-validated", not bad_b, "bad=%s" % bad_b)
    asymmetrics = [row for row in b_pairs if row["hf_asymmetric"] == "true"]
    check("B height fields are asymmetric (real OOD mirror test)", len(asymmetrics) == 9,
          "asymmetric=%d/9" % len(asymmetrics))
    b_by_tilt = read_csv(os.path.join(run_dir, "b_group_by_tilt.csv")) or []
    check("b_group_by_tilt.csv rows == 3", len(b_by_tilt) == 3)
    if len(b_by_tilt) == 3:
        drops = [float(row["mirror_drop_mean_pp"]) for row in b_by_tilt]
        check("B mirror drop increases with tilt (synth data)", drops[0] < drops[1] < drops[2], "drops=%s" % drops)

    # ---------------- repro check ----------------
    repro = read_csv(os.path.join(run_dir, "repro_check_compare.csv")) or []
    check("repro_check_compare.csv rows == 3", len(repro) == 3, "rows=%d" % len(repro))
    bad_repro = [row["rerun_label"] for row in repro if row["consistent"] != "true"]
    check("all repro re-runs identical to first run", not bad_repro, "bad=%s" % bad_repro)

    # ---------------- T1 ----------------
    t1_results = read_csv(os.path.join(run_dir, "t1_repro", "results.csv")) or []
    check("t1_repro/results.csv rows == 9", len(t1_results) == 9, "rows=%d" % len(t1_results))
    check("all T1 [EVAL] lines parsed", all(row["parsed"] == "true" for row in t1_results))
    t1_consistency = read_csv(os.path.join(run_dir, "t1_repro", "t1_consistency.csv")) or []
    check("t1_consistency.csv rows == 3", len(t1_consistency) == 3)
    check("T1 3 runs found per terrain", all(row["runs_found"] == "3" for row in t1_consistency),
          "runs=%s" % [row["runs_found"] for row in t1_consistency])
    check("T1 re-runs within 5pp of reference", all(row["all_close_5pp"] == "true" for row in t1_consistency))

    # ---------------- checks.json / manifest.json ----------------
    checks = read_json(os.path.join(run_dir, "checks.json")) or {}
    check("checks.json: all_mirror_pairs_ok matches expectation",
          checks.get("all_mirror_pairs_ok") is (not args.expect_tamper),
          "all_mirror_pairs_ok=%s" % checks.get("all_mirror_pairs_ok"))
    check("checks.json: all_expected_cells_present", checks.get("all_expected_cells_present") is True)
    check("checks.json: repro_all_consistent", checks.get("repro_all_consistent") is True)

    # ---- new hard checks requested by the Lead ---------------------------------
    code_versions = checks.get("code_versions") or {}
    check("checks.json: code_versions recorded for both files",
          set((code_versions.get("files") or {}).keys()) == {"eval_mirror.py", "mirror_terrain.py"},
          "files=%s" % list((code_versions.get("files") or {}).keys()))
    check("checks.json: code_versions has winding_fix_present flag",
          isinstance(code_versions.get("winding_fix_present"), bool),
          "winding_fix_present=%s" % code_versions.get("winding_fix_present"))

    symmetric = checks.get("symmetric_terrain_consistency") or []
    check("checks.json: symmetric-terrain CI covers 6 A pairs (tilt+stair)", len(symmetric) == 6,
          "pairs=%d" % len(symmetric))
    if not args.expect_tamper and not args.expect_tamper_mesh:
        check("checks.json: symmetric-terrain pairs all PASS",
              checks.get("symmetric_terrain_all_pass") is True,
              "verdicts=%s" % [row.get("verdict") for row in symmetric])
        check("checks.json: no symmetric-terrain anomalies",
              not (checks.get("symmetric_terrain_anomalies") or []),
              "anomalies=%s" % checks.get("symmetric_terrain_anomalies"))
        ci_ok = all(isinstance(row.get("ci95_pp"), (int, float)) and 0 < row["ci95_pp"] < 20
                    for row in symmetric)
        check("checks.json: binomial 95% CI values computed", ci_ok,
              "ci=%s" % [row.get("ci95_pp") for row in symmetric])

    trimesh = checks.get("trimesh_mirror_check") or []
    check("checks.json: trimesh mirror check covers 3 tilt pairs", len(trimesh) == 3,
          "pairs=%d" % len(trimesh))
    if args.expect_tamper_mesh:
        check("checks.json: tampered mesh winding detected (trimesh verdict FAIL)",
              checks.get("trimesh_mirror_verdict") == "FAIL",
              "verdict=%s" % checks.get("trimesh_mirror_verdict"))
        bad = [entry["normal_label"] for entry in trimesh
               if entry.get("verdict") == "FAIL" or entry.get("both_positive") == "false"]
        check("checks.json: failing trimesh pair is a_tilt_d0.5_normal",
              "a_tilt_d0.5_normal" in bad, "failed=%s" % bad)
    else:
        check("checks.json: trimesh signed-volume verdict PASS",
              checks.get("trimesh_mirror_verdict") == "PASS",
              "verdict=%s" % checks.get("trimesh_mirror_verdict"))
        equal_positive = all(entry.get("volumes_equal") == "true"
                             and entry.get("both_positive") == "true" for entry in trimesh)
        check("checks.json: normal/mirror boxes have equal positive signed volume", equal_positive,
              "volumes=%s" % [(entry.get("normal_signed_volume_m3"), entry.get("mirror_signed_volume_m3"))
                              for entry in trimesh])

        def _outward_fully(entry):
            for key in ("normal_outward_triangles", "mirror_outward_triangles"):
                value = str(entry.get(key) or "")
                if "/" not in value:
                    return False
                numerator, denominator = value.split("/")[:2]
                if not (numerator.isdigit() and denominator.isdigit()):
                    return False
                if int(numerator) != int(denominator) or int(denominator) <= 0:
                    return False
            return True

        check("checks.json: trimesh triangle winding fully outward (N/N triangles)",
              all(_outward_fully(entry) for entry in trimesh),
              "outward=%s" % [(entry.get("normal_outward_triangles"),
                               entry.get("mirror_outward_triangles")) for entry in trimesh])

    nondet = checks.get("nondeterminism") or {}
    deltas = nondet.get("repro_cell_deltas") or []
    check("checks.json: nondeterminism repro deltas recorded for 3 cells", len(deltas) == 3,
          "deltas=%d" % len(deltas))
    check("checks.json: repro deltas within 5 pp threshold",
          nondet.get("repro_all_within_5pp") is True,
          "within=%s" % nondet.get("repro_all_within_5pp"))
    check("checks.json: nondeterminism distinguishes exact vs within-5pp",
          isinstance(nondet.get("repro_all_exactly_identical"), bool)
          and isinstance(nondet.get("repro_all_within_5pp"), bool),
          "exact=%s" % nondet.get("repro_all_exactly_identical"))

    bugfix = checks.get("bugfix_evidence") or {}
    check("checks.json: bugfix_evidence present (bug/fix/audit)",
          bool(bugfix.get("bug")) and bool(bugfix.get("fix")) and bool(bugfix.get("independent_audit")))
    audit = bugfix.get("independent_audit") or {}
    check("checks.json: bugfix audit has signed volumes",
          audit.get("signed_volume_fixed_m3") is not None
          and audit.get("signed_volume_buggy_m3") is not None,
          "audit=%s" % {k: audit.get(k) for k in ("signed_volume_buggy_m3", "signed_volume_fixed_m3",
                                                  "outward_triangles_buggy", "outward_triangles_fixed")})

    manifest = read_json(os.path.join(run_dir, "manifest.json"))
    if args.expect_tamper or args.expect_tamper_mesh:
        check("tamper run uses --no-manifest (expected)", True)
    else:
        check("manifest.json exists", manifest is not None)
        manifest = manifest or {}
        for field in ("run_id", "stage", "commit", "command", "config", "seed", "trials",
                      "gpu", "vram_peak", "ckpt", "videos", "eval_success",
                      "code_versions", "mirror_checks", "nondeterminism"):
            check("manifest.json has field '%s'" % field, field in manifest)
        check("manifest videos include a gif", any(video.get("gif") for video in manifest.get("videos", [])))
        check("manifest ckpt path recorded", bool(manifest.get("ckpt", {}).get("paths")))
        check("manifest vram peak parses nvidia-smi log",
              bool((manifest.get("vram_peak") or {}).get("nvidia_smi_max_mib")))
        check("manifest mirror_checks.trimesh_verdict PASS",
              (manifest.get("mirror_checks") or {}).get("trimesh_verdict") == "PASS")

    # ---------------- report ----------------
    report_path = os.path.join(run_dir, "aggregate_report.md")
    check("aggregate_report.md exists", os.path.isfile(report_path))
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8", errors="replace") as f:
            report = f.read()
        for section in ("## A组", "## B组", "## T1", "## Reproducibility"):
            check("report contains section '%s'" % section, section in report)

    # ---------------- figures ----------------
    if args.expect_figures:
        fig_dir = os.path.join(run_dir, "figures")
        for name in REQUIRED_FIGURES:
            path = os.path.join(fig_dir, name)
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            check("figure %s exists (>10 KB)" % name, size > 10 * 1024, "size=%d" % size)
        manifest_figures = (manifest or {}).get("videos")  # noqa: F841  (figures are not in manifest by design)

    # ---------------- GIF (clean run only; the tamper run for the negative test has no frames) ----
    if not (args.expect_tamper or args.expect_tamper_mesh):
        gif_dir = os.path.join(run_dir, "videos")
        gifs = [name for name in sorted(os.listdir(gif_dir)) if name.endswith(".gif")] \
            if os.path.isdir(gif_dir) else []
        check("GIF(s) generated in videos/", len(gifs) >= 2, "gifs=%s" % gifs)
        for name in gifs:
            path = os.path.join(gif_dir, name)
            with open(path, "rb") as f:
                header = f.read(6)
            check("GIF header %s is GIF89a/GIF87a" % name, header in (b"GIF89a", b"GIF87a"), "header=%r" % header)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("")
    if failed:
        print("SELFTEST FAILED: %d/%d checks failed: %s" % (len(failed), len(RESULTS), failed))
        return 1
    print("SELFTEST PASSED: %d/%d checks passed" % (len(RESULTS), len(RESULTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""T3 mirror-evaluation figures (matplotlib, Agg backend, English labels only).

Reads the aggregate tables produced by aggregate_metrics.py and writes:

  <FIG_DIR>/fig_a_matrix.png            A组: 3 terrains x 3 difficulties grouped bars (normal vs mirror),
                                        success rate annotated on every bar
  <FIG_DIR>/fig_a_mirror_drop.png       A组: mirror drop per cell (pp)
  <FIG_DIR>/fig_b_matrix.png            B组: success rate vs lateral tilt [deg], normal vs mirror,
                                        one panel per terrain family
  <FIG_DIR>/fig_b_mirror_drop.png       B组: mirror drop vs tilt [deg]
  <FIG_DIR>/fig_t1_consistency.png      T1 protocol: reference bars + 3 re-run points per terrain
  <FIG_DIR>/fig_terrain_heights.png     terrain.npz height fields: normal | mirror | mirror-fliplr(normal)

Run with the conda `wmp` python (matplotlib 3.5.1 available):
  python make_figures.py --run-dir /home/ubuntu/B4_2025/SWAP/workSpace/runs/<run_id>
"""

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

A_TERRAINS = ["tilt", "stair", "gap"]
A_TERRAIN_TITLES = {"tilt": "tilt (WMP)", "stair": "stair (WMP)", "gap": "gap (WMP)"}
B_TERRAINS = ["ramp", "ramp_stair", "ramp_gap"]
B_TERRAIN_TITLES = {"ramp": "ramp (lateral only)", "ramp_stair": "ramp + stairs", "ramp_gap": "ramp + gap"}
T1_TERRAINS = ["slope", "stair", "gap"]
COLOR_NORMAL = "#4C72B0"
COLOR_MIRROR = "#C44E52"
COLOR_REF = "#8C8C8C"


# ------------------------------------------------------------------ csv utils
def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pairs_by(rows, key):
    out = {}
    for row in rows:
        out.setdefault(row[key], {})[str(row.get("difficulty", ""))] = row
    return out


def annotate_bars(ax, bars, values, fmt="%.1f", fontsize=8, dy=1.5):
    for bar, value in zip(bars, values):
        if value is None or np.isnan(value):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + dy, fmt % value,
                ha="center", va="bottom", fontsize=fontsize)


def annotate_points(ax, xs, ys, fmt="%.1f", fontsize=8, dy=1.5):
    for x, y in zip(xs, ys):
        if y is None or np.isnan(y):
            continue
        offset = dy
        if y + offset < 2.0:  # keep low-value labels inside the axes
            offset = 2.5
        ax.text(x, y + offset, fmt % y, ha="center", va="bottom", fontsize=fontsize)


def binom_ci95_pp(pct, n):
    """Half-width [pp] of the 95% binomial CI of one success rate (normal approximation)."""
    if pct is None or np.isnan(pct):
        return np.nan
    p = min(max(pct, 0.0), 100.0) / 100.0
    n = max(1, int(n))
    return 100.0 * 1.96 * np.sqrt(max(p * (1.0 - p) / n, 0.0))


def errorbars(values, counts):
    return [binom_ci95_pp(value, count) for value, count in zip(values, counts)]


# --------------------------------------------------------------- A matrix fig
def fig_a_matrix(a_pairs, fig_dir):
    if not a_pairs:
        print("[FIG][WARN] no A-group pairs; skip fig_a_matrix")
        return None
    diffs = [0.2, 0.5, 0.8]
    by_terrain = {}
    for row in a_pairs:
        by_terrain.setdefault(row["terrain"], {})[round(fnum(row["difficulty"]), 2)] = row

    fig, axes = plt.subplots(1, len(A_TERRAINS), figsize=(15, 4.8), sharey=True)
    if len(A_TERRAINS) == 1:
        axes = [axes]
    width = 0.36
    x = np.arange(len(diffs))
    for ax, terrain in zip(axes, A_TERRAINS):
        rows = by_terrain.get(terrain, {})
        normal = [fnum(rows[d]["normal_success_rate_pct"]) if d in rows else np.nan for d in diffs]
        mirror = [fnum(rows[d]["mirror_success_rate_pct"]) if d in rows else np.nan for d in diffs]
        normal_n = [fnum(rows[d]["normal_trials"], 200) if d in rows else 200 for d in diffs]
        mirror_n = [fnum(rows[d]["mirror_trials"], 200) if d in rows else 200 for d in diffs]
        bars_n = ax.bar(x - width / 2.0, normal, width, label="normal terrain", color=COLOR_NORMAL)
        bars_m = ax.bar(x + width / 2.0, mirror, width, label="mirrored terrain", color=COLOR_MIRROR)
        ax.errorbar(x - width / 2.0, normal, yerr=errorbars(normal, normal_n), fmt="none",
                    ecolor="#222222", capsize=2.5, linewidth=0.8)
        ax.errorbar(x + width / 2.0, mirror, yerr=errorbars(mirror, mirror_n), fmt="none",
                    ecolor="#222222", capsize=2.5, linewidth=0.8)
        annotate_bars(ax, bars_n, normal)
        annotate_bars(ax, bars_m, mirror)
        ax.set_xticks(x)
        ax.set_xticklabels(["difficulty %.1f" % d for d in diffs])
        ax.set_title(A_TERRAIN_TITLES.get(terrain, terrain))
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("success rate [%]  (200 trials/cell)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("T3 A-group: WMP native terrains, normal vs np.fliplr mirror")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_a_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


def fig_a_mirror_drop(a_pairs, fig_dir):
    if not a_pairs:
        return None
    rows = sorted(a_pairs, key=lambda r: (A_TERRAINS.index(r["terrain"])
                                          if r["terrain"] in A_TERRAINS else 9,
                                          fnum(r["difficulty"])))
    labels = ["%s\nd%.1f" % (r["terrain"], fnum(r["difficulty"])) for r in rows]
    drops = [fnum(r["mirror_drop_pp"]) for r in rows]
    colors = [COLOR_NORMAL if abs(value) < 1e-9 else COLOR_MIRROR for value in drops]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(rows)), 4.2))
    bars = ax.bar(np.arange(len(rows)), drops, color=colors)
    annotate_bars(ax, bars, drops, fmt="%+.1f", fontsize=7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mirror drop = normal - mirror [pp]")
    ax.set_title("T3 A-group mirror drop per cell (positive = mirror terrain worse)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_a_mirror_drop.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


# --------------------------------------------------------------- B matrix fig
def fig_b_matrix(b_pairs, b_by_tilt, fig_dir):
    if not b_pairs:
        print("[FIG][WARN] no B-group pairs; skip fig_b_matrix")
        return None
    by_terrain = {}
    for row in b_pairs:
        by_terrain.setdefault(row["terrain"], {})[round(fnum(row["tilt_deg"]), 1)] = row
    tilts = [5.0, 10.0, 15.0]

    fig, axes = plt.subplots(1, len(B_TERRAINS), figsize=(15, 4.8), sharey=True)
    if len(B_TERRAINS) == 1:
        axes = [axes]
    for ax, terrain in zip(axes, B_TERRAINS):
        rows = by_terrain.get(terrain, {})
        normal = [fnum(rows[t]["normal_success_rate_pct"]) if t in rows else np.nan for t in tilts]
        mirror = [fnum(rows[t]["mirror_success_rate_pct"]) if t in rows else np.nan for t in tilts]
        normal_n = [fnum(rows[t]["normal_trials"], 200) if t in rows else 200 for t in tilts]
        mirror_n = [fnum(rows[t]["mirror_trials"], 200) if t in rows else 200 for t in tilts]
        ax.plot(tilts, normal, marker="o", color=COLOR_NORMAL, label="normal terrain", linewidth=1.8)
        ax.plot(tilts, mirror, marker="s", color=COLOR_MIRROR, label="mirrored terrain",
                linewidth=1.8, linestyle="--")
        ax.errorbar(tilts, normal, yerr=errorbars(normal, normal_n), fmt="none",
                    ecolor=COLOR_NORMAL, capsize=2.5, linewidth=0.8)
        ax.errorbar(tilts, mirror, yerr=errorbars(mirror, mirror_n), fmt="none",
                    ecolor=COLOR_MIRROR, capsize=2.5, linewidth=0.8)
        annotate_points(ax, tilts, normal)
        annotate_points(ax, tilts, mirror, dy=-4.5)
        for tilt, value_n, value_m in zip(tilts, normal, mirror):
            if not (np.isnan(value_n) or np.isnan(value_m)):
                ax.annotate("", xy=(tilt, value_m), xytext=(tilt, value_n),
                            arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6))
        ax.set_xticks(tilts)
        ax.set_xticklabels(["%d deg" % t for t in tilts])
        ax.set_xlabel("unilateral lateral tilt")
        ax.set_title(B_TERRAIN_TITLES.get(terrain, terrain))
        ax.set_ylim(0, 108)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("success rate [%]  (200 trials/cell)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("T3 B-group: SWAP-style unilateral lateral ramp, normal vs mirrored (OOD)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_b_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


def fig_b_mirror_drop(b_pairs, fig_dir):
    if not b_pairs:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.26
    tilts = [5.0, 10.0, 15.0]
    x = np.arange(len(tilts))
    for idx, terrain in enumerate(B_TERRAINS):
        values = []
        for tilt in tilts:
            match = [r for r in b_pairs
                     if r["terrain"] == terrain and abs(fnum(r["tilt_deg"]) - tilt) < 1e-6]
            values.append(fnum(match[0]["mirror_drop_pp"]) if match else np.nan)
        bars = ax.bar(x + (idx - (len(B_TERRAINS) - 1) / 2.0) * width, values, width,
                      label=B_TERRAIN_TITLES.get(terrain, terrain))
        annotate_bars(ax, bars, values, fmt="%+.1f", fontsize=7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["%d deg" % t for t in tilts])
    ax.set_xlabel("unilateral lateral tilt")
    ax.set_ylabel("mirror drop = normal - mirror [pp]")
    ax.set_title("T3 B-group mirror drop vs lateral tilt (positive = mirrored terrain worse)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_b_mirror_drop.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


# ------------------------------------------------------------ T1 consistency fig
def fig_t1_consistency(t1_results, t1_consistency, fig_dir):
    ref = {row["terrain"]: fnum(row["reference_pct"]) for row in t1_consistency}
    if not t1_results and not t1_consistency:
        print("[FIG][WARN] no T1 re-check data; skip fig_t1_consistency")
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(T1_TERRAINS))
    ref_values = [ref.get(t, np.nan) for t in T1_TERRAINS]
    bars = ax.bar(x, ref_values, 0.5, color=COLOR_REF, label="T1 reference (2026-11-08)")
    annotate_bars(ax, bars, ref_values, fmt="%.1f")
    jitter = [-0.18, 0.0, 0.18]
    for idx, terrain in enumerate(T1_TERRAINS):
        runs = [r for r in t1_results if r["terrain"] == terrain]
        points_x = [idx + jitter[k] for k in range(len(runs))]
        points_y = [fnum(r["success_rate_pct"]) for r in runs]
        ax.scatter(points_x, points_y, marker="D", s=46, color=COLOR_MIRROR,
                   zorder=5, label="T1 re-check (3 runs)" if idx == 0 else None)
        for px, py in zip(points_x, points_y):
            ax.text(px, py - 4.6, "%.1f" % py, ha="center", va="top", fontsize=7, color=COLOR_MIRROR,
                    zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(T1_TERRAINS)
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 108)
    ax.set_title("T1 protocol consistency: reference vs 3 re-runs (NEVAL=64, 1200 steps)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_t1_consistency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


# ------------------------------------------------------- terrain height fields
def _pick_height_pairs(a_pairs, b_pairs):
    picked = []
    for terrain in A_TERRAINS:
        rows = [r for r in a_pairs if r["terrain"] == terrain]
        if not rows:
            continue
        preferred = [r for r in rows if abs(fnum(r["difficulty"]) - 0.5) < 1e-6]
        picked.append(("A", preferred[0] if preferred else rows[0]))
    for terrain in B_TERRAINS:
        rows = [r for r in b_pairs if r["terrain"] == terrain]
        if not rows:
            continue
        preferred = [r for r in rows if abs(fnum(r["tilt_deg"]) - 10.0) < 1e-6]
        picked.append(("B", preferred[0] if preferred else rows[0]))
    return picked


def _load_height(run_dir, label):
    candidates = []
    for group in ("a_matrix", "b_matrix", "repro_check"):
        candidates.append(os.path.join(run_dir, group, label, "terrain.npz"))
    for path in candidates:
        if os.path.isfile(path):
            data = np.load(path)
            return np.asarray(data["height_field_raw"])
    return None


def fig_terrain_heights(run_dir, a_pairs, b_pairs, fig_dir):
    picked = _pick_height_pairs(a_pairs, b_pairs)
    panels = []
    for group, pair in picked:
        normal_hf = _load_height(run_dir, pair["normal_label"])
        mirror_hf = _load_height(run_dir, pair["mirror_label"])
        if normal_hf is None or mirror_hf is None:
            print("[FIG][WARN] terrain.npz missing for %s / %s" % (pair["normal_label"], pair["mirror_label"]))
            continue
        expected = np.fliplr(normal_hf)
        diff = mirror_hf.astype(np.int32) - expected.astype(np.int32)
        panels.append({
            "group": group, "pair": pair,
            "normal": normal_hf, "mirror": mirror_hf, "diff": diff,
            "max_abs_diff": int(np.max(np.abs(diff))),
        })
    if not panels:
        print("[FIG][WARN] no height-field pairs; skip fig_terrain_heights")
        return None

    fig, axes = plt.subplots(len(panels), 3, figsize=(12.5, 3.1 * len(panels)))
    if len(panels) == 1:
        axes = np.array([axes])
    for row_idx, panel in enumerate(panels):
        pair = panel["pair"]
        vmin = float(min(panel["normal"].min(), panel["mirror"].min()))
        vmax = float(max(panel["normal"].max(), panel["mirror"].max()))
        images = [
            (panel["normal"], "viridis", None, "normal: %s" % pair["normal_label"]),
            (panel["mirror"], "viridis", None, "mirror: %s" % pair["mirror_label"]),
            (panel["diff"], "RdBu_r", None,
             "mirror - fliplr(normal)  max|d|=%d raw" % panel["max_abs_diff"]),
        ]
        for col_idx, (image, cmap, _, title) in enumerate(images):
            ax = axes[row_idx][col_idx]
            if col_idx == 2:
                limit = max(1.0, float(np.max(np.abs(panel["diff"]))))
                im = ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            else:
                im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(
        "T3 terrain height fields (A rows = WMP native; B rows = unilateral lateral ramp). "
        "Right column ~0 proves the mirror is an exact np.fliplr. Custom trimesh boxes (tilt/crawl) not shown.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(fig_dir, "fig_terrain_heights.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("[FIG] %s" % path)
    return path


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description="T3 mirror-eval figures")
    parser.add_argument("--run-dir", required=True, help="T3 run directory (must contain metrics tables)")
    parser.add_argument("--fig-dir", default=None, help="output dir (default <run-dir>/figures)")
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print("[FIG][FATAL] run dir does not exist: %s" % run_dir)
        return 2
    fig_dir = os.path.abspath(args.fig_dir or os.path.join(run_dir, "figures"))
    os.makedirs(fig_dir, exist_ok=True)

    a_pairs = read_csv(os.path.join(run_dir, "a_group_pairs.csv"))
    b_pairs = read_csv(os.path.join(run_dir, "b_group_pairs.csv"))
    b_by_tilt = read_csv(os.path.join(run_dir, "b_group_by_tilt.csv"))
    t1_results = read_csv(os.path.join(run_dir, "t1_repro", "results.csv"))
    t1_consistency = read_csv(os.path.join(run_dir, "t1_repro", "t1_consistency.csv"))
    if not a_pairs:
        print("[FIG][WARN] a_group_pairs.csv missing/empty; run aggregate_metrics.py first")

    produced = [
        fig_a_matrix(a_pairs, fig_dir),
        fig_a_mirror_drop(a_pairs, fig_dir),
        fig_b_matrix(b_pairs, b_by_tilt, fig_dir),
        fig_b_mirror_drop(b_pairs, fig_dir),
        fig_t1_consistency(t1_results, t1_consistency, fig_dir),
        fig_terrain_heights(run_dir, a_pairs, b_pairs, fig_dir),
    ]
    produced = [path for path in produced if path]
    print("[FIG] done: %d figures in %s" % (len(produced), fig_dir))
    if not produced:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

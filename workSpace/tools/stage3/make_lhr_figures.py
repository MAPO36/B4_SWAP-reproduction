#!/usr/bin/env python
"""Figures for the stage-3 R0-LHR LHR/RHL evaluation (reads aggregate outputs).

Usage:
  python make_lhr_figures.py --run-dir <RUN_DIR>

Produces <RUN_DIR>/figures/:
  fig_lhr_rhl_by_family.png    3 families x 3 levels, LHR vs RHL success (+95% binomial error bars)
  fig_mirror_gap.png           9 pairs mirror gap (LHR - RHL) in pp, sorted by family/level
  fig_lhr_rhl_by_level.png     level means (1/2/3) LHR vs RHL
  fig_training_curve.png       mean reward vs iteration (if train_curve.csv exists)
  fig_lhr_rhl_terrain.png      example LHR vs RHL height fields + difference (from terrain.npz)
"""
import argparse
import csv
import os

import numpy as np


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR")
    ap.add_argument("--eval-subdir", default="eval")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = read_csv(os.path.join(run_dir, "metrics.csv"))
    pairs = read_csv(os.path.join(run_dir, "lhr_rhl_pairs.csv"))
    families = ["box", "stair", "gap"]
    levels = [1, 2, 3]

    def cell(f, lv, d):
        for r in metrics:
            if r["family"] == f and int(r["level"]) == lv and r["direction"] == d:
                return r
        return None

    def ci95(p, n=200):
        p = max(min(p / 100.0, 1.0), 0.0)
        return 1.96 * (p * (1 - p) / n) ** 0.5 * 100.0

    # 1) by family x level
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    x = np.arange(len(levels))
    w = 0.36
    for ax, f in zip(axes, families):
        lhr, rhl, lerr, rerr = [], [], [], []
        for lv in levels:
            l = cell(f, lv, "LHR")
            r = cell(f, lv, "RHL")
            lv_l = float(l["success_rate_pct"]) if l else 0.0
            lv_r = float(r["success_rate_pct"]) if r else 0.0
            lhr.append(lv_l)
            rhl.append(lv_r)
            lerr.append(ci95(lv_l))
            rerr.append(ci95(lv_r))
        ax.bar(x - w / 2, lhr, w, label="LHR (in-distribution)", color="#4C78A8", yerr=lerr, capsize=3)
        ax.bar(x + w / 2, rhl, w, label="RHL (mirror, OOD)", color="#E45756", yerr=rerr, capsize=3)
        for xi, (a, b) in enumerate(zip(lhr, rhl)):
            ax.text(xi - w / 2, a + 2, "%.0f" % a, ha="center", fontsize=8)
            ax.text(xi + w / 2, b + 2, "%.0f" % b, ha="center", fontsize=8)
        ax.set_title("family: %s" % f)
        ax.set_xticks(x)
        ax.set_xticklabels(["level %d" % lv for lv in levels])
        ax.set_ylim(0, 108)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("success rate [%] (200 trials/cell)")
    fig.suptitle("R0-LHR: LHR (training distribution) vs RHL (mirror, OOD) success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_lhr_rhl_by_family.png"), dpi=110)
    plt.close(fig)

    # 2) mirror gap pairs
    if pairs:
        labels = ["%s/L%s" % (r["family"], r["level"]) for r in pairs]
        gaps = [float(r["mirror_gap_pp"]) for r in pairs]
        cis = [float(r["ci95_pp"]) for r in pairs]
        colors = {"box": "#4C78A8", "stair": "#F58518", "gap": "#54A24B"}
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(pairs)), gaps, color=[colors.get(r["family"], "#888") for r in pairs],
                      yerr=cis, capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        for i, (b, g) in enumerate(zip(bars, gaps)):
            ax.text(i, g + (2 if g >= 0 else -4), "%.1f" % g, ha="center", fontsize=8)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("mirror gap = LHR - RHL [pp]")
        ax.set_title("R0-LHR mirror gap per cell (positive = mirrored terrain is worse)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_mirror_gap.png"), dpi=110)
        plt.close(fig)

    # 3) by level
    by_level = read_csv(os.path.join(run_dir, "by_level.csv"))
    if by_level:
        x = np.arange(len(by_level))
        lhr = [float(r["lhr_mean_pct"]) for r in by_level]
        rhl = [float(r["rhl_mean_pct"]) for r in by_level]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - 0.18, lhr, 0.36, label="LHR", color="#4C78A8")
        ax.bar(x + 0.18, rhl, 0.36, label="RHL", color="#E45756")
        for xi, (a, b) in enumerate(zip(lhr, rhl)):
            ax.text(xi - 0.18, a + 1.5, "%.1f" % a, ha="center", fontsize=9)
            ax.text(xi + 0.18, b + 1.5, "%.1f" % b, ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["level %s" % r["level"] for r in by_level])
        ax.set_ylim(0, 108)
        ax.set_ylabel("mean success rate [%]")
        ax.set_title("R0-LHR: LHR vs RHL by difficulty level")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_lhr_rhl_by_level.png"), dpi=110)
        plt.close(fig)

    # 4) training curve
    curve = read_csv(os.path.join(run_dir, "train_curve.csv"))
    if curve:
        its = np.array([float(r["iteration"]) for r in curve])
        rw = np.array([float(r["mean_reward"]) for r in curve])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(its, rw, alpha=0.35, label="mean reward")
        k = max(1, len(rw) // 100)
        if len(rw) > k:
            ma = np.convolve(rw, np.ones(k) / k, mode="valid")
            ax.plot(its[k - 1:], ma, color="tab:red", label="moving average (%d)" % k)
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean reward")
        ax.set_title("R0-LHR training curve (task go2_amp_lhr, fixed 10 deg LHR)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig_training_curve.png"), dpi=110)
        plt.close(fig)

    # 5) example terrain LHR vs RHL
    lhr_dir = None
    rhl_dir = None
    for r in pairs:
        if r.get("lhr_out_dir"):
            lhr_dir = r["lhr_out_dir"]
            rhl_dir = r.get("rhl_out_dir")
            break
    if lhr_dir and rhl_dir:
        try:
            a = np.load(os.path.join(lhr_dir, "terrain.npz"))["height_field_raw"].astype(float) * 0.005
            b = np.load(os.path.join(rhl_dir, "terrain.npz"))["height_field_raw"].astype(float) * 0.005
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for ax, img, title in ((axes[0], a, "LHR terrain (training distribution)"),
                                   (axes[1], b, "RHL terrain (np.fliplr mirror, OOD)"),
                                   (axes[2], np.fliplr(b) - a, "fliplr(RHL) - LHR (should be ~0)")):
                im = ax.imshow(img, origin="upper", cmap="viridis", aspect="auto")
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.suptitle("stage-3 terrain pair example (fixed 10 deg LHR / its RHL mirror)")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "fig_lhr_rhl_terrain.png"), dpi=100)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print("[figures] terrain example skipped: %r" % (exc,))

    print("[figures] saved to %s" % fig_dir)
    print(sorted(os.listdir(fig_dir)))


if __name__ == "__main__":
    main()

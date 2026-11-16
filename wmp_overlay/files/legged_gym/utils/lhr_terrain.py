"""LHR (left-high-right-low) unilateral tilted terrain for SWAP-lite stage 3.

Used only through monkeypatching `legged_robot.Terrain`, so upstream WMP stays untouched:
  * training:  `train_lhr.py`  (task `go2_amp_lhr`, stock 10-row curriculum)
  * evaluation: `eval_lhr.py`  (single family, fixed difficulty, LHR vs RHL mirror)

Conventions (WMP / Isaac Gym): +x = forward, +y = robot's LEFT, +z = up.
LHR means the ground is higher on the robot's LEFT: z increases with +y.
RHL (the mirror distribution) is exactly `np.fliplr(height_field_raw)`.

Terrain families (height-field only, chosen so that a lateral ramp composes exactly):
  box   -> WMP `climb` generator (raised rectangular platforms, slot 6)
  stair -> WMP stairs-UP generator (slot 3, step_height positive by stock mapping)
  gap   -> WMP `gap` generator (slot 5)

Every sub-terrain gets the same fixed LHR lateral ramp (never randomized left/right),
applied on the global height-field map (per patch columns) before mesh conversion.
`env_origins[:, :, 2]` is recomputed from the ramped field so robots spawn correctly.
"""
import json
import os
import random

import numpy as np
from scipy.ndimage import binary_dilation

from legged_gym.utils.terrain import Terrain, convert_heightfield_to_trimesh
from legged_gym.utils import trimesh


class LHRTerrain(Terrain):
    """Fixed single-sided (LHR) tilted terrain, optional RHL mirror for evaluation.

    Class attributes (set by train_lhr.py / eval_lhr.py before env creation):
        LHR_TILT_DEG      : float  fixed lateral roll angle [deg]; +y (left) is higher
        MIRROR            : bool   if True, apply np.fliplr to the final height field (RHL)
        FIXED_DIFFICULTY  : float|None  pin every sub-terrain to this difficulty (eval)
        TERRAIN_SEED      : int|None    seed np.random + random for a reproducible draw
        FAMILY            : str|None    tag saved into the fingerprint ('box'/'stair'/'gap')
    """

    LHR_TILT_DEG = 10.0
    MIRROR = False
    FIXED_DIFFICULTY = None
    TERRAIN_SEED = None
    FAMILY = None

    # ------------------------------------------------------------------ construction
    def __init__(self, cfg, num_robots):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        if cfg.selected:
            raise RuntimeError("LHRTerrain does not support cfg.terrain.selected == True")
        if self.MIRROR and cfg.num_cols != 1:
            raise RuntimeError(
                "RHL mirror evaluation requires cfg.terrain.num_cols == 1 "
                "(the mirror is a global np.fliplr of a single-patch map)"
            )

        if self.TERRAIN_SEED is not None:
            np.random.seed(int(self.TERRAIN_SEED))
            random.seed(int(self.TERRAIN_SEED))

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1]) for i in range(len(cfg.terrain_proportions))
        ]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.added_trimesh = None

        # --- generation: equivalent to Terrain.curriculum()/evaluated_terrain() ----------
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                if self.FIXED_DIFFICULTY is not None:
                    difficulty = float(self.FIXED_DIFFICULTY)
                else:
                    difficulty = i / cfg.num_rows
                choice = j / cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

        # --- stage-3 modification: fixed LHR lateral ramp --------------------------------
        if self.added_trimesh is not None:
            raise RuntimeError(
                "LHRTerrain expects height-field-only families (box=climb, stair up, gap); "
                "custom trimesh terrains (tilt/crawl) are not supported with a lateral ramp"
            )
        if float(self.LHR_TILT_DEG) != 0.0:
            self._apply_lhr_ramp()
        if self.MIRROR:
            self.height_field_raw = np.ascontiguousarray(np.fliplr(self.height_field_raw))
        # spawn heights must follow the (ramped, possibly mirrored) height field
        self._recompute_env_origins_z()

        # --- mesh conversion (same tail as Terrain.__init__) -----------------------------
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
            if self.added_trimesh is not None:
                self.vertices, self.triangles = trimesh.combine_trimeshes(
                    (self.vertices, self.triangles), self.added_trimesh
                )
            half_edge_width = int(1)
            structure = np.ones((half_edge_width * 2 + 1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

        self.lhr_stats = self.lhr_check()

    # ------------------------------------------------------------------ terrain ramp
    def _apply_lhr_ramp(self):
        """Add the fixed LHR roll ramp to every patch (higher on the robot's left, +y).

        Applied on the global map per patch-column range, so it is independent of the
        stock sub-terrain array orientation. The outer border columns stay flat (the robot
        operates inside the 8 x 8 m patches).
        """
        hs = self.cfg.horizontal_scale
        vs = self.cfg.vertical_scale
        slope = np.tan(np.deg2rad(float(self.LHR_TILT_DEG)))
        w = self.width_per_env_pixels
        y_m = (np.arange(w) - (w - 1) / 2.0) * hs  # +y to the right in the array == left of robot
        ramp_raw = np.round(slope * y_m / vs).astype(np.int32)
        hf = self.height_field_raw.astype(np.int32)
        for j in range(self.cfg.num_cols):
            start_y = self.border + j * w
            hf[:, start_y:start_y + w] += ramp_raw.reshape(1, -1)
        self.height_field_raw = hf.astype(np.int16)

    def _recompute_env_origins_z(self):
        """Spawn heights = max height of the central 2 m x 2 m window of each patch."""
        hs = self.cfg.horizontal_scale
        vs = self.cfg.vertical_scale
        x1 = int((self.env_length / 2.0 - 1) / hs)
        x2 = int((self.env_length / 2.0 + 1) / hs)
        y1 = int((self.env_width / 2.0 - 1) / hs)
        y2 = int((self.env_width / 2.0 + 1) / hs)
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                sx = self.border + i * self.length_per_env_pixels
                sy = self.border + j * self.width_per_env_pixels
                patch = self.height_field_raw[sx + x1:sx + x2, sy + y1:sy + y2]
                self.env_origins[i, j, 2] = np.max(patch) * vs

    # ------------------------------------------------------------------ validation
    def lhr_check(self):
        """Measure the actual roll direction/angle of every patch (fail-fast evidence).

        For each patch: least-squares slope of column-mean height vs y (the ramp is the
        only left/right-antisymmetric component; the terrain features are left/right
        symmetric, so the fit recovers the ramp even on top of them). LHR requires a
        positive slope for ALL patches (higher on +y = robot's left).
        """
        hs = self.cfg.horizontal_scale
        vs = self.cfg.vertical_scale
        w = self.width_per_env_pixels
        y_m = (np.arange(w) - (w - 1) / 2.0) * hs
        left_mask = y_m > 0.5
        right_mask = y_m < -0.5
        angles = []
        left_higher = 0
        per_patch = []
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                sx = self.border + i * self.length_per_env_pixels
                sy = self.border + j * w
                patch_m = self.height_field_raw[sx:sx + self.length_per_env_pixels,
                                                sy:sy + w].astype(np.float64) * vs
                col_mean = patch_m.mean(axis=0)
                slope = np.polyfit(y_m, col_mean, 1)[0]
                angle = float(np.degrees(np.arctan(slope)))
                angles.append(angle)
                is_left_higher = bool(
                    patch_m[:, left_mask].mean() > patch_m[:, right_mask].mean()
                )
                left_higher += int(is_left_higher)
                per_patch.append(
                    {"row": i, "col": j, "roll_deg": round(angle, 3), "left_higher": is_left_higher}
                )
        n = len(angles)
        return {
            "n_patches": n,
            "left_higher_count": int(left_higher),
            "direction_consistent": bool(left_higher == n and min(angles) > 0.0),
            "roll_deg_min": round(float(min(angles)), 3),
            "roll_deg_median": round(float(np.median(angles)), 3),
            "roll_deg_max": round(float(max(angles)), 3),
            "requested_deg": float(self.LHR_TILT_DEG),
            "mirror": bool(self.MIRROR),
            "per_patch": per_patch[:6],  # keep the JSON small; full data in the npz
        }

    def terrain_fingerprint(self):
        import hashlib

        hf = np.ascontiguousarray(self.height_field_raw)
        return {
            "height_field_shape": list(hf.shape),
            "height_field_sha256": hashlib.sha256(hf.tobytes()).hexdigest(),
            "height_field_fliplr_sha256": hashlib.sha256(
                np.ascontiguousarray(np.fliplr(hf)).tobytes()
            ).hexdigest(),
            "lhr_tilt_deg": float(self.LHR_TILT_DEG),
            "mirror_rhl": bool(self.MIRROR),
            "family": self.FAMILY,
            "fixed_difficulty": self.FIXED_DIFFICULTY,
            "terrain_seed": self.TERRAIN_SEED,
            "num_rows": int(self.cfg.num_rows),
            "num_cols": int(self.cfg.num_cols),
            "terrain_proportions": [float(x) for x in self.cfg.terrain_proportions],
            "lhr_check": self.lhr_stats,
        }

    def save_lhr_visuals(self, out_dir, prefix="train_terrain"):
        """Save the height field (global + patch samples) and the direction report as PNG/JSON."""
        os.makedirs(out_dir, exist_ok=True)
        report = {
            "fingerprint": self.terrain_fingerprint(),
            "env_origins": np.asarray(self.env_origins).round(4).tolist(),
        }
        with open(os.path.join(out_dir, "%s_report.json" % prefix), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        np.savez_compressed(
            os.path.join(out_dir, "%s_heightfield.npz" % prefix),
            height_field_raw=self.height_field_raw,
            env_origins=np.asarray(self.env_origins),
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001 - visualization must never break training
            print("[LHR] matplotlib unavailable, skip PNG previews: %r" % (exc,), flush=True)
            return report
        hf = self.height_field_raw.astype(np.float64) * self.cfg.vertical_scale
        stride = max(1, hf.shape[1] // 420)
        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(hf[::stride, ::stride], origin="upper", cmap="viridis", aspect="auto")
        ax.set_title(
            "LHR training terrain (fixed %.1f deg roll, left=+y is higher); shape=%s"
            % (self.LHR_TILT_DEG, tuple(self.height_field_raw.shape))
        )
        ax.set_xlabel("y (left of robot is larger y, toward the right of the image)")
        ax.set_ylabel("x (forward)")
        fig.colorbar(im, ax=ax, label="height [m]")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "%s_heightfield.png" % prefix), dpi=90)
        plt.close(fig)

        # patch samples from 3 different columns/rows (three terrain families)
        n_show = min(4, self.cfg.num_rows)
        n_cols_show = min(4, self.cfg.num_cols)
        fig, axes = plt.subplots(n_show, n_cols_show, figsize=(4.2 * n_cols_show, 3.6 * n_show))
        for a in range(n_show):
            for b in range(n_cols_show):
                i = int(round(a * (self.cfg.num_rows - 1) / max(1, n_show - 1)))
                j = int(round(b * (self.cfg.num_cols - 1) / max(1, n_cols_show - 1)))
                sx = self.border + i * self.length_per_env_pixels
                sy = self.border + j * self.width_per_env_pixels
                patch = self.height_field_raw[sx:sx + self.length_per_env_pixels,
                                              sy:sy + self.width_per_env_pixels].astype(np.float64)
                patch *= self.cfg.vertical_scale
                ax = axes[a][b] if n_show > 1 and n_cols_show > 1 else axes
                im = ax.imshow(patch, origin="upper", cmap="viridis", aspect="auto")
                ax.set_title("row %d col %d" % (i, j), fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle("LHR patch samples (all left-high right-low)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "%s_patches.png" % prefix), dpi=90)
        plt.close(fig)
        return report

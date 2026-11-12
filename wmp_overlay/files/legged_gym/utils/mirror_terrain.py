"""Evaluation-only terrain generator for T3 mirror evaluation.

This module adds three things on top of WMP's stock `legged_gym.utils.terrain.Terrain`,
used ONLY by `legged_gym/scripts/eval_mirror.py` (which monkeypatches
`legged_robot.Terrain` before env creation). Training code paths are untouched, so the
T1 baseline recipe stays reproducible.

Features
--------
1. Pinned difficulty: every sub-terrain uses one fixed `difficulty` value instead of
   `i / num_rows`, so a cell (terrain type x difficulty) can be evaluated in isolation.
2. True left-right mirror: after generation, the whole height field is flipped with
   `np.fliplr` (the exact operation requested by the T3 task card / plan stage 3), and
   the custom trimesh obstacles (WMP `tilt` / `crawl` boxes) are reflected about the
   robot's sagittal plane (y = env_origin_y).
3. Optional unilateral lateral ramp (SWAP Fig.3 style "左高右低" terrain): a roll slope
   is added to the height field BEFORE mirroring, so `MIRROR=1` gives the exact
   "右高左低" counterpart.

Constraints (evaluation only):
* `num_cols` must be 1 (single terrain row), `curriculum` must be False, `selected` False.
* lateral ramp + custom-trimesh terrains (tilt/crawl) is rejected (boxes would need to be
  tilted as well; out of scope for this round).
* Some WMP terrains are left-right symmetric by construction (stair / tilt / crawl);
  for those, normal and mirrored terrain are identical up to float noise. This is a
  feature: it is used as a self-check of the mirror implementation, and it is the central
  finding of the T3 report.

All coordinates follow WMP's convention: +x is forward, +y is the robot's left,
+z is up. `height_field_raw` is `int16` with `vertical_scale` metres per unit.
"""
import random

import numpy as np
from scipy.ndimage import binary_dilation

from legged_gym.utils.terrain import Terrain, convert_heightfield_to_trimesh
from legged_gym.utils import trimesh


class EvalTerrain(Terrain):
    """Terrain used only for evaluation (see module docstring).

    Class attributes are set by the caller before env creation:
        FIXED_DIFFICULTY : float | None   pinned `difficulty` (None = stock i/num_rows)
        MIRROR           : bool           left-right flip of the final height field
        LATERAL_TILT_DEG : float          unilateral lateral ramp [deg]; 0 = off
        TERRAIN_SEED     : int | None     seeds both `np.random` and `random` for the draw
    """

    FIXED_DIFFICULTY = None
    MIRROR = False
    LATERAL_TILT_DEG = 0.0
    TERRAIN_SEED = None

    def __init__(self, cfg, num_robots):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        if cfg.curriculum:
            raise RuntimeError("EvalTerrain requires cfg.terrain.curriculum == False")
        if cfg.selected:
            raise RuntimeError("EvalTerrain does not support cfg.terrain.selected == True")
        if cfg.num_cols != 1:
            raise RuntimeError("EvalTerrain mirror evaluation requires num_cols == 1")

        if self.TERRAIN_SEED is not None:
            # The stock generator mixes np.random (terrain_utils) and Python's `random`
            # (stair step width); seed both for a reproducible terrain draw.
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

        # --- stock generation, but with a pinned difficulty -------------------------
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = (
                    float(self.FIXED_DIFFICULTY)
                    if self.FIXED_DIFFICULTY is not None
                    else i / cfg.num_rows
                )
                choice = j / cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)

        # --- evaluation-only modifications (post-generation, pre-mesh) --------------
        if self.LATERAL_TILT_DEG:
            if self.added_trimesh is not None:
                raise RuntimeError(
                    "lateral tilt is only supported for pure height-field terrains "
                    "(not tilt/crawl, whose boxes would need to be tilted too)"
                )
            self._add_lateral_ramp()
            # spawn heights must include the ramp in BOTH directions so that the
            # normal/mirror pair starts at the same height (symmetry of the protocol)
            self._recompute_env_origins_z()
        if self.MIRROR:
            self._mirror_height_field()
            self._mirror_added_trimesh()
            self._recompute_env_origins_z()

        # --- mesh conversion (identical to Terrain.__init__ tail) -------------------
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
                    (self.vertices, self.triangles),
                    self.added_trimesh,
                )
            half_edge_width = int(1)
            structure = np.ones((half_edge_width * 2 + 1, 1))
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

    # ------------------------------------------------------------------ helpers
    def _add_lateral_ramp(self):
        """Add a unilateral roll ramp inside the terrain patches: +y (robot's left) is higher.

        The ramp is written into the sub-terrain columns only: the y-direction side strips
        (outside the patch columns) stay at z=0, while the 25 m x-direction border rows do
        carry the ramp at the patch columns. Robots only ever touch the patch area itself.
        """
        ramp_raw = self._lateral_ramp_raw()
        hf = self.height_field_raw.astype(np.int32)
        for j in range(self.cfg.num_cols):
            start_y = self.border + j * self.width_per_env_pixels
            end_y = start_y + self.width_per_env_pixels
            hf[:, start_y:end_y] += ramp_raw[start_y:end_y].reshape(1, -1)
        self.height_field_raw = hf.astype(np.int16)

    def _lateral_ramp_raw(self):
        """Raw (vertical_scale) units of the lateral ramp for every map column."""
        n_cols = self.tot_cols
        y_m = (np.arange(n_cols) - (n_cols - 1) / 2.0) * self.cfg.horizontal_scale
        slope = np.tan(np.deg2rad(float(self.LATERAL_TILT_DEG)))
        return np.round(slope * y_m / self.cfg.vertical_scale).astype(np.int32)

    def _mirror_height_field(self):
        """Exact left-right flip of the height field (this is `np.fliplr`, as specified)."""
        self.height_field_raw = np.ascontiguousarray(np.fliplr(self.height_field_raw))

    def _mirror_added_trimesh(self):
        """Reflect WMP's custom boxes about the robot's sagittal plane (y = env origin).

        The stock tilt/crawl boxes are laid out symmetrically about
        `y = env_origin_y = (j + 0.5) * terrain_width`; the mirror plane of the robot is
        exactly that plane, so reflecting about it is the physically correct mirror
        (reflecting about the map centre instead would introduce a half-pixel shift
        because WMP's env origin sits half a pixel off the patch centre).
        Only valid for num_cols == 1, which is enforced in __init__.

        A reflection flips the triangle winding, which flips every face normal of the
        box mesh. Physics *does* notice (measured: tilt normal 96.5% vs mirror 32.5%
        success before this fix), so the triangle winding is reversed as well to keep
        the face normals pointing outwards exactly like in the unmirrored mesh.
        """
        if self.added_trimesh is None:
            return
        vertices, triangles = self.added_trimesh
        y_axis = self.cfg.border_size + 0.5 * self.env_width
        vertices = np.array(vertices, dtype=np.float32, copy=True)
        vertices[:, 1] = 2.0 * y_axis - vertices[:, 1]
        triangles = np.array(triangles, copy=True)
        triangles = triangles[:, [0, 2, 1]]  # undo the winding flip caused by y -> 2a - y
        self.added_trimesh = (vertices, triangles)

    def _recompute_env_origins_z(self):
        """Spawn heights depend on the (possibly ramped / flipped) height field."""
        x1 = int((self.env_length / 2.0 - 1) / self.cfg.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / self.cfg.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / self.cfg.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / self.cfg.horizontal_scale)
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                start_x = self.border + i * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                patch = self.height_field_raw[
                    start_x + x1: start_x + x2, start_y + y1: start_y + y2
                ]
                self.env_origins[i, j, 2] = np.max(patch) * self.cfg.vertical_scale

    # ------------------------------------------------------------------ reporting
    def terrain_fingerprint(self):
        """Small dict used to prove that a normal/mirror pair is a true mirror pair."""
        import hashlib
        hf = np.ascontiguousarray(self.height_field_raw)
        flipped = np.ascontiguousarray(np.fliplr(hf))
        fp = {
            "height_field_shape": list(hf.shape),
            "height_field_sha256": hashlib.sha256(hf.tobytes()).hexdigest(),
            "height_field_fliplr_sha256": hashlib.sha256(flipped.tobytes()).hexdigest(),
            "added_trimesh": self.added_trimesh is not None,
        }
        if self.added_trimesh is not None:
            verts = np.ascontiguousarray(np.asarray(self.added_trimesh[0], dtype=np.float32))
            # rotation/order independent fingerprint of the box vertex cloud
            rounded = np.round(verts, 3)
            order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
            fp["added_vertices_sha256"] = hashlib.sha256(
                np.ascontiguousarray(rounded[order]).tobytes()
            ).hexdigest()
            fp["added_num_vertices"] = int(verts.shape[0])
            fp["added_vertices_min"] = rounded.min(axis=0).tolist()
            fp["added_vertices_max"] = rounded.max(axis=0).tolist()
        return fp

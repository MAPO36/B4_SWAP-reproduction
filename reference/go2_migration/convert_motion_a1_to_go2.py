#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert WMP's A1 AMP motion clips to Go2 joint order.

Why this exists
---------------
WMP's AMP discriminator does NOT see foot positions or root height. It sees
exactly 30 numbers per frame:

    [joint_pos(12), base_lin_vel(3), base_ang_vel(3), joint_vel(12)]

so the *only* thing that must match Go2 is the 12-joint ordering and the
joint-angle values. A1 and Go2 happen to share the same joint names, the same
12-DoF topology and (almost) the same joint limits, so no retargeting of the
angles is needed.

The one real difference is ordering:

    A1 URDF  : FR, FL, RR, RL   (this is the order stored in the .txt files)
    Go2 URDF : FL, FR, RL, RR   (this is the order Isaac Gym gives Go2)

Applying that permutation to every 12-wide leg block fixes it. Base velocities
are order-independent and are left alone (optionally scaled, see below).

File format expected by WMP (datasets/mocap_motions/*.txt)
----------------------------------------------------------
    {
      "LoopMode": "Wrap",
      "FrameDuration": 0.02,
      "MotionWeight": 1.0,
      "Frames": [[61 floats], ...]
    }

Frame layout (61):
    [0:3]   root position
    [3:7]   root quaternion (xyzw)
    [7:19]  joint position   (12)  <- reorder
    [19:31] toe position     (12)  <- reorder (unused by the discriminator)
    [31:34] base linear velocity (3)
    [34:37] base angular velocity (3)
    [37:49] joint velocity   (12)  <- reorder
    [49:61] toe velocity     (12)  <- reorder (unused by the discriminator)

Usage
-----
Run from the WMP repository root:

    python /path/to/convert_motion_a1_to_go2.py

or explicitly:

    python convert_motion_a1_to_go2.py \
        --src datasets/mocap_motions \
        --dst datasets/mocap_motions_go2 \
        --lin-vel-scale 1.0

Notes
-----
* `--lin-vel-scale 1.065` makes the recorded base speed match what Go2 actually
  achieves at the same joint angles (Go2 legs are 0.213 m vs A1 0.200 m, so
  +6.5%). 1.0 is the safest first try; only change it if the AMP reward/gait
  looks too slow or too fast.
* Go2 joint limits comfortably contain every A1 clip (checked on all four WMP
  files): hip [-0.41, 0.26] vs [-1.047, 1.047]; front thigh [0.69, 1.28] vs
  [-1.571, 3.491]; rear thigh [0.43, 1.18] vs [-0.524, 4.538]; calf
  [-2.33, -0.94] vs [-2.723, -0.838].
"""

import argparse
import json
import os
import shutil

# A1 index -> Go2 index for each leg block of 12 joints.
# A1 : FR(0,1,2)  FL(3,4,5)  RR(6,7,8)  RL(9,10,11)
# Go2: FL(0,1,2)  FR(3,4,5)  RL(6,7,8)  RR(9,10,11)
GO2_FROM_A1 = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

FRAME_LEN = 61
JOINT_POS = slice(7, 19)
TOE_POS = slice(19, 31)
LIN_VEL = slice(31, 34)
JOINT_VEL = slice(37, 49)
TOE_VEL = slice(49, 61)


def reorder_block(values, idx):
    return [values[i] for i in idx]


def convert_frame(frame, lin_vel_scale):
    if len(frame) != FRAME_LEN:
        raise ValueError(
            "expected %d floats per frame, got %d" % (FRAME_LEN, len(frame))
        )

    out = list(frame)  # copy everything, then overwrite the 12-wide blocks
    out[JOINT_POS] = reorder_block(frame[JOINT_POS], GO2_FROM_A1)
    out[TOE_POS] = reorder_block(frame[TOE_POS], GO2_FROM_A1)
    out[JOINT_VEL] = reorder_block(frame[JOINT_VEL], GO2_FROM_A1)
    out[TOE_VEL] = reorder_block(frame[TOE_VEL], GO2_FROM_A1)

    if lin_vel_scale != 1.0:
        out[LIN_VEL] = [v * lin_vel_scale for v in frame[LIN_VEL]]
    return out


def convert_file(src_path, dst_path, lin_vel_scale):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["Frames"]
    data["Frames"] = [convert_frame(fr, lin_vel_scale) for fr in frames]

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return len(frames)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="datasets/mocap_motions",
        help="source folder with A1-order .txt clips (default: datasets/mocap_motions)",
    )
    parser.add_argument(
        "--dst",
        default="datasets/mocap_motions_go2",
        help="output folder (default: datasets/mocap_motions_go2)",
    )
    parser.add_argument(
        "--lin-vel-scale",
        type=float,
        default=1.0,
        help="multiply base linear velocity; 1.065 compensates Go2's longer legs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the destination folder if it already exists",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(
            "source folder not found: %s\n"
            "Run this from the WMP repository root." % os.path.abspath(args.src)
        )

    if os.path.exists(args.dst):
        if not args.overwrite:
            raise SystemExit(
                "destination already exists: %s\n"
                "Pass --overwrite to replace it." % os.path.abspath(args.dst)
            )
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)

    files = sorted(f for f in os.listdir(args.src) if f.endswith(".txt"))
    if not files:
        raise SystemExit("no .txt motion files found in %s" % os.path.abspath(args.src))

    print("A1 -> Go2 permutation:", GO2_FROM_A1)
    print("linear-velocity scale:", args.lin_vel_scale)
    for name in files:
        src = os.path.join(args.src, name)
        dst = os.path.join(args.dst, name)
        n = convert_file(src, dst, args.lin_vel_scale)
        print("  %-12s %4d frames -> %s" % (name, n, dst))

    print("\nDone. Check with:")
    print("  python %s --help" % os.path.join(here, "check_motion_order.py"))


if __name__ == "__main__":
    main()

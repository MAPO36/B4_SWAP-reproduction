#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail loudly if the Go2 motion files are not a correct permutation of A1's.

Run from the WMP repository root after convert_motion_a1_to_go2.py:

    python check_motion_order.py

It checks:
  1. every frame still has 61 columns;
  2. the 12 joint positions / velocities are exactly the A1 values permuted by
     [3,4,5, 0,1,2, 9,10,11, 6,7,8];
  3. base linear/angular velocity is unchanged (unless --lin-vel-scale was
     used, in which case lin vel is checked with the same scale);
  4. every joint angle lies inside the Go2 URDF limits.

Exit code 0 = all good, 1 = a check failed.
"""

import argparse
import json
import os
import sys

GO2_FROM_A1 = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

JOINT_POS = slice(7, 19)
TOE_POS = slice(19, 31)
LIN_VEL = slice(31, 34)
ANG_VEL = slice(34, 37)
JOINT_VEL = slice(37, 49)
TOE_VEL = slice(49, 61)

# Go2 URDF joint limits, in Go2 order: FL, FR, RL, RR (hip, thigh, calf each).
# source: resources/robots/go2/urdf/go2.urdf
GO2_LOWER = [
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
]
GO2_UPPER = [
    1.0472, 3.4907, -0.83776,
    1.0472, 3.4907, -0.83776,
    1.0472, 4.5379, -0.83776,
    1.0472, 4.5379, -0.83776,
]
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_pair(src_path, dst_path, lin_vel_scale):
    with open(src_path, "r", encoding="utf-8") as f:
        src = json.load(f)
    with open(dst_path, "r", encoding="utf-8") as f:
        dst = json.load(f)

    errs = []
    if len(src["Frames"]) != len(dst["Frames"]):
        errs.append("frame count differs: %d vs %d" % (len(src["Frames"]), len(dst["Frames"])))

    for fi, (a, b) in enumerate(zip(src["Frames"], dst["Frames"])):
        if len(b) != 61:
            errs.append("frame %d has %d columns, expected 61" % (fi, len(b)))
            break
        expected_jp = [a[JOINT_POS][i] for i in GO2_FROM_A1]
        expected_jv = [a[JOINT_VEL][i] for i in GO2_FROM_A1]
        if not all(close(x, y) for x, y in zip(b[JOINT_POS], expected_jp)):
            errs.append("frame %d joint positions are not the expected permutation" % fi)
            break
        if not all(close(x, y) for x, y in zip(b[JOINT_VEL], expected_jv)):
            errs.append("frame %d joint velocities are not the expected permutation" % fi)
            break
        if not all(close(b[LIN_VEL][i], a[LIN_VEL][i] * lin_vel_scale) for i in range(3)):
            errs.append("frame %d base linear velocity mismatch" % fi)
            break
        if not all(close(x, y) for x, y in zip(b[ANG_VEL], a[ANG_VEL])):
            errs.append("frame %d base angular velocity changed" % fi)
            break

    # Joint limit check over the whole clip.
    lo = [float("inf")] * 12
    hi = [float("-inf")] * 12
    for fr in dst["Frames"]:
        for j, v in enumerate(fr[JOINT_POS]):
            lo[j] = min(lo[j], v)
            hi[j] = max(hi[j], v)
    for j in range(12):
        if lo[j] < GO2_LOWER[j] - 1e-6 or hi[j] > GO2_UPPER[j] + 1e-6:
            errs.append(
                "%s out of Go2 limits: [%.3f, %.3f] not inside [%.3f, %.3f]"
                % (JOINT_NAMES[j], lo[j], hi[j], GO2_LOWER[j], GO2_UPPER[j])
            )

    name = os.path.basename(src_path)
    if errs:
        print("FAIL %s" % name)
        for e in errs:
            print("     - %s" % e)
        return False, lo, hi

    print("OK   %-12s %4d frames, joints inside Go2 limits" % (name, len(dst["Frames"])))
    print("     hip   range %+.2f .. %+.2f" % (min(lo[0], lo[3], lo[6], lo[9]), max(hi[0], hi[3], hi[6], hi[9])))
    print("     thigh range %+.2f .. %+.2f" % (min(lo[1], lo[4], lo[7], lo[10]), max(hi[1], hi[4], hi[7], hi[10])))
    print("     calf  range %+.2f .. %+.2f" % (min(lo[2], lo[5], lo[8], lo[11]), max(hi[2], hi[5], hi[8], hi[11])))
    return True, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="datasets/mocap_motions")
    parser.add_argument("--dst", default="datasets/mocap_motions_go2")
    parser.add_argument("--lin-vel-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.isdir(args.src) or not os.path.isdir(args.dst):
        raise SystemExit("run this from the WMP root, or pass --src/--dst")

    names = sorted(f for f in os.listdir(args.src) if f.endswith(".txt"))
    if not names:
        raise SystemExit("no .txt files in %s" % os.path.abspath(args.src))

    ok = True
    for name in names:
        src = os.path.join(args.src, name)
        dst = os.path.join(args.dst, name)
        if not os.path.exists(dst):
            print("FAIL %s: missing in destination" % name)
            ok = False
            continue
        good, _, _ = check_pair(src, dst, args.lin_vel_scale)
        ok = ok and good

    print()
    if ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())

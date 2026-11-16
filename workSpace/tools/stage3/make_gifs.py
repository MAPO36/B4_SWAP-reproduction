#!/usr/bin/env python
"""Collect viewer PNG frames from a --videos-dir tree and write one GIF per subdir.

Used after the stage-3 eval/video pass. Pure PIL (system python3 works):
  python /tmp/make_gifs.py --videos-dir <RUN>/videos --num-frames 60 --fps 10
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", default="640x360")
    args = ap.parse_args()
    from PIL import Image

    w, h = [int(x) for x in args.size.lower().split("x")]
    made = []
    for sub in sorted(glob.glob(os.path.join(args.videos_dir, "*"))):
        frames = sorted(glob.glob(os.path.join(sub, "frames", "*.png")))
        if not frames:
            continue
        idx = [int(round(i * (len(frames) - 1) / (args.num_frames - 1)))
               for i in range(args.num_frames)] if len(frames) > 1 else [0] * args.num_frames
        imgs = []
        for i in idx:
            im = Image.open(frames[i]).convert("RGB").resize((w, h))
            imgs.append(im)
        out = os.path.join(args.videos_dir, os.path.basename(sub) + ".gif")
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
        made.append((out, os.path.getsize(out), len(frames)))
    for m in made:
        print("[gif] %s  %d bytes  (from %d frames)" % m)
    print("[gif] total %d gifs" % len(made))


if __name__ == "__main__":
    main()

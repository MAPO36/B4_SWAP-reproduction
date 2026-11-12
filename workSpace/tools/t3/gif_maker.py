#!/usr/bin/env python
"""Build a GIF from a directory of viewer PNG frames (Pillow only, no other deps).

Compatible with the system Pillow 9.0.1 on the 4090 (/usr/bin/python3) and newer Pillow.

  python gif_maker.py FRAMES_DIR OUT.gif [--fps 10] [--width 640] [--sample 60] [--pattern "*.png"]

--sample N  uniformly subsample N frames out of all frames (0/absent = keep all).
--width W   resize frames to width W (aspect kept); 0/absent = keep original size.
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported to the caller
    print("[GIF][FATAL] Pillow (PIL) is not importable by %s" % sys.executable)
    sys.exit(2)

RESAMPLING = getattr(Image, "Resampling", Image)
LANCZOS = RESAMPLING.LANCZOS if hasattr(RESAMPLING, "LANCZOS") else Image.LANCZOS


def uniform_sample(items, count):
    """Pick `count` items uniformly spaced over the list (always keeps first and last)."""
    total = len(items)
    if count <= 0 or count >= total:
        return list(items)
    if count == 1:
        return [items[0]]
    indices = []
    for i in range(count):
        index = int(round(i * (total - 1) / float(count - 1)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return [items[i] for i in indices]


def main(argv=None):
    parser = argparse.ArgumentParser(description="frames/*.png -> animated GIF")
    parser.add_argument("frames_dir")
    parser.add_argument("out_gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--pattern", default="*.png")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.frames_dir):
        print("[GIF][FATAL] frames dir does not exist: %s" % args.frames_dir)
        return 2
    paths = sorted(glob.glob(os.path.join(args.frames_dir, args.pattern)))
    if len(paths) < 2:
        print("[GIF][FATAL] need >= 2 frames in %s, found %d" % (args.frames_dir, len(paths)))
        return 2
    selected = uniform_sample(paths, args.sample)
    duration = max(20, int(round(1000.0 / max(1, args.fps))))

    images = []
    size = None
    for path in selected:
        image = Image.open(path).convert("RGB")
        if args.width and image.width != args.width:
            height = max(2, int(round(image.height * args.width / float(image.width))))
            image = image.resize((args.width, height), LANCZOS)
        if size is None:
            size = image.size
        elif image.size != size:  # keep every frame the same size for a stable GIF
            image = image.resize(size, LANCZOS)
        images.append(image)
    if not images:
        print("[GIF][FATAL] no readable frames in %s" % args.frames_dir)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.out_gif))
    os.makedirs(out_dir, exist_ok=True)
    images[0].save(
        args.out_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
        format="GIF",
    )
    size_bytes = os.path.getsize(args.out_gif)
    print("[GIF] wrote %s  size=%.1f KB  frames=%d (sampled %d/%d)  %dx%d  %d fps"
          % (args.out_gif, size_bytes / 1024.0, len(images), len(selected), len(paths),
             size[0], size[1], args.fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())

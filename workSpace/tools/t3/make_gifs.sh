#!/bin/bash
# =============================================================================
# T3 video-frame -> GIF helper (Pillow only; no new dependencies are installed).
#
# Two modes:
#   bash make_gifs.sh --all <VIDEOS_ROOT> [--sample 60] [--fps 10] [--width 640]
#       batch mode: for every <VIDEOS_ROOT>/<name>/frames/*.png writes <VIDEOS_ROOT>/<name>.gif
#   bash make_gifs.sh <FRAMES_DIR> <OUT_GIF> [--sample 60] [--fps 10] [--width 640]
#
# Interpreter priority: /usr/bin/python3 (system Pillow 9.0.1 on the 4090) -> conda `wmp`
# python -> python3/python on PATH.  Override with PYTHON=/path/to/python.
# =============================================================================
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIF_MAKER="$SCRIPT_DIR/gif_maker.py"

pick_python() {
  local candidate
  if [ -n "${PYTHON:-}" ]; then
    candidate="$PYTHON"
    if "$candidate" -c "import PIL" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
    echo "[GIF][WARN] PYTHON=$candidate cannot import PIL; trying fallbacks" >&2
  fi
  for candidate in /usr/bin/python3 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c "import PIL" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

PY=$(pick_python) || {
  echo "[GIF][FATAL] no python with Pillow found (tried \$PYTHON, /usr/bin/python3, python3, python)" >&2
  exit 2
}
PY_VERSION=$("$PY" -c "import sys, PIL; print('python %s / Pillow %s' % (sys.version.split()[0], PIL.__version__))" 2>/dev/null)
echo "[GIF] interpreter: $PY ($PY_VERSION)"

if [ ! -f "$GIF_MAKER" ]; then
  echo "[GIF][FATAL] gif_maker.py not found next to make_gifs.sh ($GIF_MAKER)" >&2
  exit 2
fi

if [ "${1:-}" = "--all" ]; then
  ROOT="${2:-}"
  shift 2 || true
  if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
    echo "[GIF][FATAL] usage: make_gifs.sh --all <VIDEOS_ROOT> [--sample N] [--fps N] [--width N]" >&2
    exit 2
  fi
  rc_all=0
  found=0
  for cell_dir in "$ROOT"/*/; do
    [ -d "$cell_dir" ] || continue
    name=$(basename "$cell_dir")
    frames="$cell_dir/frames"
    [ -d "$frames" ] || continue
    count=$(ls -1 "$frames"/*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "${count:-0}" -lt 2 ]; then
      echo "[GIF][WARN] $name: only ${count} frame(s) in $frames; skipped"
      continue
    fi
    found=$((found + 1))
    if "$PY" "$GIF_MAKER" "$frames" "$ROOT/$name.gif" "$@"; then
      echo "[GIF] OK   $name ($count frames)"
    else
      echo "[GIF] FAIL $name ($count frames)"
      rc_all=1
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "[GIF][WARN] no <name>/frames/*.png found under $ROOT"
    exit 1
  fi
  echo "[GIF] batch done: $found video cell(s) under $ROOT"
  exit "$rc_all"
fi

if [ "${1:-}" = "" ] || [ "${2:-}" = "" ]; then
  echo "usage: make_gifs.sh --all <VIDEOS_ROOT> [--sample N] [--fps N] [--width N]" >&2
  echo "       make_gifs.sh <FRAMES_DIR> <OUT_GIF> [--sample N] [--fps N] [--width N]" >&2
  exit 2
fi
FRAMES_DIR="$1"
OUT_GIF="$2"
shift 2
"$PY" "$GIF_MAKER" "$FRAMES_DIR" "$OUT_GIF" "$@"

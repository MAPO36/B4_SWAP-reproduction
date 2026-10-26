#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWAP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WMP_DIR="$SWAP_ROOT/WMP"
WMP_COMMIT="$(tr -d '[:space:]' < "$SCRIPT_DIR/WMP_COMMIT")"

if [[ "${1:-}" == "--ref" ]]; then
  REF="${2:?usage: apply.sh [--ref <tag>]}"
  echo "[apply] restoring wmp_overlay/ from $REF"
  git -C "$SWAP_ROOT" checkout "$REF" -- wmp_overlay/
fi

if [[ ! -d "$WMP_DIR" ]]; then
  echo "[apply] cloning upstream WMP"
  if ! git clone https://github.com/bytedance/WMP "$WMP_DIR"; then
    echo "[apply] ERROR: cannot clone. Copy a WMP snapshot to $WMP_DIR and rerun." >&2
    exit 1
  fi
fi

if [[ -d "$WMP_DIR/.git" ]]; then
  echo "[apply] fetching tags (offline is okay if commit exists)"
  git -C "$WMP_DIR" fetch --tags origin || true

  if [[ -n "$(git -C "$WMP_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "[apply] ERROR: WMP has tracked local changes." >&2
    echo "[apply] Move them to wmp_overlay/files/ before applying." >&2
    git -C "$WMP_DIR" status --short >&2
    exit 1
  fi

  echo "[apply] checking out $WMP_COMMIT"
  git -C "$WMP_DIR" checkout --detach "$WMP_COMMIT"
else
  echo "[apply] WARN: $WMP_DIR has no .git; using it as a vendored copy (skip checkout)."
fi

if [[ -d "$SCRIPT_DIR/files" ]]; then
  echo "[apply] copying overlay files"
  cp -a "$SCRIPT_DIR/files/." "$WMP_DIR/"
fi

GO2_ASSETS="$SWAP_ROOT/unitree_rl_gym/resources/robots/go2"
if [[ -d "$GO2_ASSETS" ]]; then
  echo "[apply] copying Go2 assets from unitree_rl_gym"
  mkdir -p "$WMP_DIR/resources/robots/go2"
  cp -a "$GO2_ASSETS/." "$WMP_DIR/resources/robots/go2/"
fi

if [[ -d "$WMP_DIR/.git" ]]; then
  echo "[apply] WMP HEAD: $(git -C "$WMP_DIR" rev-parse HEAD)"
  echo "[apply] WMP status:"
  git -C "$WMP_DIR" status --short
fi

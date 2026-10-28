#!/usr/bin/env bash
# 按 AGENTS.md §6 打包一次交接备份到指定目录（通常是 U 盘挂载点）
# 用法：bash workSpace/tools/backup_usb.sh /media/ubuntu/USBNAME
set -euo pipefail

DEST_ARG="${1:-}"
if [ -z "$DEST_ARG" ]; then
  echo "用法: bash workSpace/tools/backup_usb.sh <目标目录>" >&2
  echo "示例: bash workSpace/tools/backup_usb.sh /media/ubuntu/KINGSTON" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M)"
OUT="$DEST_ARG/SWAP-backup-$STAMP"

mkdir -p "$OUT"
cd "$ROOT"

HEAD_SHA="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STAGE="$(sed -n 's/^当前阶段：//p' handoff/STATUS.md | head -1)"

cat > "$OUT/README.txt" <<TXT
SWAP 交接备份
日期     : $(date '+%F %T %Z')
HEAD     : $HEAD_SHA ($BRANCH)
当前阶段 : $STAGE
工作人员 : 陈旺（2026-10-28）
主机     : $(hostname)

内容
  repo.bundle   : git bundle --all（完整历史，可 git clone 还原）
  snapshot/     : 项目快照（排除 WMP/.git、runs 里的 ckpt、__pycache__、unitree_rl_gym/logs）
  wmp_overlay/  : 对 WMP 的全部改动（含 go2 配置、train_capped、check_go2_depth）
  runs/         : 当前 run 的 manifest / 日志 / 显存采样 / 深度图
  SHA256SUMS    : 校验清单

说明
  训练 ckpt（113 MB/个）默认不备份，路径见 runs/manifest.json。
  还原代码：git clone repo.bundle SWAP && git -C SWAP checkout $HEAD_SHA
TXT

echo "[1/5] git bundle"
git bundle create "$OUT/repo.bundle" --all 2>&1 | tail -2

echo "[2/5] snapshot"
mkdir -p "$OUT/snapshot"
tar czf "$OUT/snapshot/repo-snapshot.tar.gz" \
  --exclude='./WMP/.git' \
  --exclude='./WMP/logs' \
  --exclude='./workSpace/runs/*/ckpt' \
  --exclude='./unitree_rl_gym/logs' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='./.git' \
  --exclude='./reference/*.pdf' \
  . 2>/dev/null

echo "[3/5] wmp_overlay"
mkdir -p "$OUT/wmp_overlay"
cp -a wmp_overlay/. "$OUT/wmp_overlay/"

echo "[4/5] runs"
mkdir -p "$OUT/runs"
if [ -d workSpace/runs/stage-0 ]; then
  cp -a workSpace/runs/stage-0/. "$OUT/runs/" 2>/dev/null || true
fi
rm -f "$OUT"/runs/*.pt 2>/dev/null || true

echo "[5/5] SHA256SUMS"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS )

echo ""
echo "完成: $OUT"
du -sh "$OUT"
ls -1 "$OUT"

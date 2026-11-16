#!/bin/bash
# Commit #2: record the watcher PID and backfill the stage-3 commit hash into logs.md.
set -uo pipefail
SWAP=/home/ubuntu/B4_2025/SWAP
HASH=8cab403
WATCHER_PID=$(cat "$SWAP/workSpace/runs/R0_LHR/logs/watcher.pid" 2>/dev/null || echo "")
cd "$SWAP" || exit 1

python3 - "$HASH" "$WATCHER_PID" <<'PY'
import sys
h, pid = sys.argv[1], sys.argv[2]
swap = "/home/ubuntu/B4_2025/SWAP"
# 1) logs.md: fill the stage-3 WIP line commit placeholder
p = swap + "/handoff/logs.md"
lines = open(p, encoding="utf-8").read().splitlines()
hits = 0
for i in range(len(lines) - 1, -1, -1):
    if "| T1b | 王 |" in lines[i] and "【commit 待回填】" in lines[i]:
        lines[i] = lines[i].replace("【commit 待回填】", h)
        hits += 1
        break
if hits != 1:
    sys.exit("ERROR: T1b log line placeholder not found")
open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("logs.md commit backfilled")
# 2) STATUS.md: record the watcher PID
p = swap + "/handoff/STATUS.md"
text = open(p, encoding="utf-8").read()
old = "- watcher：`/tmp/wait_gpu_then_train.sh`，日志 `workSpace/runs/R0_LHR/logs/watcher.log`，PID 见 `logs/watcher.pid`"
new = ("- watcher：`/tmp/wait_gpu_then_train.sh`，**PID %s**（2026-11-16 11:06 启动，见 `logs/watcher.pid`）；"
       "日志 `workSpace/runs/R0_LHR/logs/watcher.log`" % (pid or "unknown"))
if old not in text:
    sys.exit("ERROR: STATUS watcher line not found")
text = text.replace(old, new)
# 3) record stage-3 commit in STATUS §六 style note (append to §三 code line)
old2 = "- 本轮实现（代码，overlay 为准）："
new2 = ("- 本轮提交：`%s`（代码 + 11-16 方案重命名 + STATUS/logs）；工作区干净，未 push。\n"
        "- 本轮实现（代码，overlay 为准）：" % h)
if old2 in text:
    text = text.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(text)
print("STATUS.md updated")
PY

chmod 755 handoff/logs.md
chmod 644 handoff/STATUS.md
git add handoff/logs.md handoff/STATUS.md
git -c user.name='王' -c user.email='wang@local' commit -m "handoff(T1b): 记录显存守卫 watcher PID ${WATCHER_PID} 与 commit ${HASH}（王，2026-11-16）" >/dev/null || echo "nothing to commit"
echo "=== log ==="
git log --oneline -3
echo "=== status ==="
git status --short
echo "=== HEAD ==="
git rev-parse --short HEAD
echo "=== logs tail ==="
tail -1 handoff/logs.md | cut -c1-260
echo "=== STATUS watcher line ==="
grep -n 'watcher：' handoff/STATUS.md | cut -c1-220
echo "=== watcher alive? ==="
W=$(cat "$SWAP/workSpace/runs/R0_LHR/logs/watcher.pid")
kill -0 "$W" 2>/dev/null && echo "watcher RUNNING pid=$W" || echo "watcher NOT running"
echo "=== watcher log tail ==="
tail -3 "$SWAP/workSpace/runs/R0_LHR/logs/watcher.log"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

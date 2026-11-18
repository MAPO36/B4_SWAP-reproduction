#!/bin/bash
# Upload STATUS + append the stage-3 WIP line to handoff/logs.md, then commit #1.
set -uo pipefail
SWAP=/home/ubuntu/B4_2025/SWAP
cd "$SWAP" || exit 1

python3 - <<'PY'
line = open('/tmp/logs_new_line.txt', encoding='utf-8').read().strip()
p = '/home/ubuntu/B4_2025/SWAP/handoff/logs.md'
text = open(p, encoding='utf-8').read()
if not text.endswith('\n'):
    text += '\n'
if line in text:
    print('logs line already present, skip')
else:
    text += line + '\n'
    open(p, 'w', encoding='utf-8').write(text)
    print('logs.md appended')
PY

chmod 755 handoff/logs.md
chmod 644 handoff/STATUS.md
echo "=== git status before commit ==="
git status --short
git add -A
echo "=== staged stat ==="
git diff --cached --stat | tail -15
git -c user.name='王' -c user.email='wang@local' commit -m 'T1b/stage3: LHR 训练地形（10°左高右低+平台箱/上行楼梯/沟壑）+ go2_amp_lhr + train_lhr + 训练链；方案切换 11-16 版；GPU 被他人占用，显存守卫 watcher 待启动（王，2026-11-16）'
echo "=== committed ==="
git log -1 --format='%h %ci %an %s'
git rev-parse --short HEAD
echo "=== status after ==="
git status --short

# AGENTS.md — SWAP 工作流唯一规则

> 任何 agent（Claude Code / Codex / 其他）进入本仓库，先完整读本文件，再读 `handoff/STATUS.md`。
> 本文件只规定“怎么干活”；技术方案在 `reference/复现方案_SWAP轻量版.md`，按需 grep 小节，不要整篇复述。

## 0. 环境与事实

- 目标：复现 SWAP 轻量版（WMP 底座 + 对称等变，Unitree Go2）。
- 工作机：实验室 4090（Ubuntu 22.04，4 人共用一个 Linux 账号）。
- 同一时间只有 1 人在 4090 上操作；线性接力，不开分支、不 push、不用 GitHub。
- 本机 5060 笔记本只用于读写文档、静态检查；实质训练一律在 4090。
- WMP 在 `WMP/`（gitignore）；所有 WMP 改动以 `wmp_overlay/` 为准。
- 路径约定：先 `export SWAP=/home/l/workSpace/SWAP`；WMP 在 `$SWAP/WMP`。
- `unitree_rl_gym` 没有 `.git`，父仓库只有 gitlink；不要删除它，备份必须整目录快照。

## 1. 每次会话开始（按顺序，不许跳）

1. 读 `handoff/STATUS.md`。
2. 读 STATUS 指向的当前任务卡（`handoff/tasks/*.md`）。
3. 读 `handoff/logs.md` 最后 5 行。
4. 检查 `git status`、`nvidia-smi`、STATUS 里有没有 running 的训练。
5. 向使用者复述，然后问“是否按任务卡继续？”：

```
【当前】阶段 X，工作人员 Y
【下一步】任务卡 ...，第一个动作 ...
【缺失证据】...
【运行中训练】无 / run_id ...，预计 ... 完成
【需要你确认】是否继续？
```

## 2. 能不能接手

- STATUS 的“当前工作人员”为空：你可以接手；先把名字和开始时间写进 STATUS。
- 是你：继续。
- 是别人：停止操作，只提醒使用者“当前由 X 负责，请先交接”。
- 不允许同时开第二个 GPU 训练。夜间训练可以跨人交接，但接手人只监控，不启动新任务。

## 3. 硬规则（违反即停）

1. 不跳过任务卡的退出条件；5 iter 冒烟不算完成；核心阶段必须有真实训练 + 评估 + 视频 + manifest。
2. 训练超过 10 分钟、关闭阶段、打 tag、回滚、删除产物、使用别人的 ckpt → 先问使用者。
3. WMP 的改动必须以 `wmp_overlay/files/` 为准；在 WMP 里改完后必须同步回 overlay，再用 `apply.sh` 验证。
4. 所有训练/评估必须登记到 STATUS + logs；夜间跑写 run_id、路径、预计完成、监控命令。
5. 不伪造结果；没跑就写“未验证”；不改测试/验收脚本让它通过。
6. 不改 `reference/`；不读 reference 里的 go2walker 工作日志；要整篇读复现方案先问使用者。
7. 不 push；不做 `git reset --hard` 改写已交接的历史。
8. 不把 ckpt、大视频、`reference/*.pdf` 提交进 git；训练产物放 `workSpace/runs/`。
9. 交接时必须 commit（WIP 也可以）；关闭阶段/打 tag 必须先问使用者。
10. 每次结束提醒使用者：做 U 盘备份、确认下一位工作人员。
11. 使用者要求跳过流程时，先指出违反哪条并请其确认；不要默默跳过。

## 4. 标准动作

### 4.1 训练/评估登记

- run_id 格式：`s<stage>-<YYYYMMDD-HHMM>-<purpose>`，例如 `s1-20261027-1430-baseline`。
- 目录：`workSpace/runs/<run_id>/`，至少放 `manifest.json`、日志、`metrics.csv`、视频。
- `manifest.json` 至少包含：run_id/stage/commit/command/config/seed/iterations/gpu/vram_peak/ckpt/videos/eval_success。
- 夜间跑在 STATUS 写：run_id、命令、预计完成时间、监控命令（如 `tail -f ...`）。

### 4.2 交接

1. `git add -A && git commit -m "handoff(stage-X): ..."`。
2. 更新 `handoff/STATUS.md`：清空“当前工作人员”，写“上一步”、证据路径、下一步、风险。
3. `handoff/logs.md` 追加一行：

   `YYYY-MM-DD HH:MM | stage-X | 姓名 | 一句话结果 | next: 任务卡 | commit 短哈希 | backup 目录名`

4. 提醒使用者做 U 盘备份（见 §6）。
5. 阶段门禁全过才打 tag：`git tag stage-X-done`（打 tag 前先问使用者）。

### 4.3 回滚

- 代码：`git revert <bad-commit>`；不要 reset 已交接历史。
- WMP：`bash wmp_overlay/apply.sh --ref <tag>` 还原 overlay，再重跑任务卡验收。
- 回滚后更新 STATUS/logs。

## 5. 任务卡

- `00-env.md`：4090 环境 + WMP 冒烟
- `05-go2.md`：Go2 迁移 + 200~500 iter 短训练
- `10-baseline.md`：完整基线 + 训练地形评估
- `30-mirror-eval.md`：镜像地形评估
- `40-symmetric.md`：对称 actor/critic
- `60-report.md`：评估、消融、报告
- 可选阶段 2/5/8/9 不建卡，需要时再加；不要偷偷做。

## 6. 备份（U 盘）

每次交接至少备份到新的日期目录，旧目录不覆盖：

```
SWAP-backup-YYYYMMDD-HHMM/
  README.txt        # 日期 / HEAD / 阶段 / owner / 内容
  repo.bundle       # git bundle --all
  snapshot/         # 整项目快照（排除 WMP/.git、unitree_rl_gym/logs、runs/*/ckpt、__pycache__）
  wmp_overlay/      # WMP overlay + patches
  runs/             # 当前 run 的 manifest、metrics.csv、视频
  SHA256SUMS
```

默认不备份 ckpt，只记路径 + sha256；U 盘空间够时每个阶段额外放 best ckpt。

## 7. 禁止

- 在本机 5060 跑实质训练。
- 用冒烟/短训练代替阶段成果。
- 改 WMP 上游后不写回 overlay。
- 跳过 STATUS/logs 直接开始干活。
- 把“未验证”说成“已完成”。

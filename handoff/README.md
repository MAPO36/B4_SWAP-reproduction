# 给人看的简要指南

## 一、每次开始

1. 打开 `handoff/STATUS.md`，看“当前工作人员”。
2. 不是你就别动；是空的就写你的名字和开始时间。
3. 看“下一步”指向的任务卡，照着做。
4. 让 agent 先读 `AGENTS.md`，它会复述当前阶段、下一步、缺什么证据。

## 二、每次结束 / 换人

1. 提交代码：`git add -A && git commit -m "handoff(stage-X): ..."`。
2. 更新 `STATUS.md`：上一步结果、证据路径、下一步、风险；清空“当前工作人员”。
3. `logs.md` 追加一行。
4. 做 U 盘备份：新建日期文件夹，旧版本不覆盖。
5. 把工作交给下一个人，并口头提醒他先看 STATUS。

## 三、四个人怎么轮

- 同一时间只有 1 人操作 4090。
- 夜间训练可以挂着，下一个人只监控，不启动第二个训练。
- 交接以 `STATUS.md` 为准，不靠口头。
- 阶段没做完也可以交接：提交 WIP，写清“从哪里继续”。

## 四、三条红线

- 不跳过任务卡的退出条件；冒烟不算完成。
- 不伪造结果；没跑就写“未验证”。
- WMP 只通过 `wmp_overlay/` 改，不直接改上游后不管。

## 五、常用命令

```bash
export SWAP=/home/l/workSpace/SWAP

# 看状态
git status
nvidia-smi
tail -f workSpace/runs/<run_id>/train.log

# 看历史
git log --oneline
git tag

# 提交
git add -A && git commit -m "stage-X: ..."

# 回滚
git revert <commit>
bash wmp_overlay/apply.sh --ref <tag>
```

## 六、备份什么

- `git bundle --all`
- 整个 `SWAP/` 快照（排除 `unitree_rl_gym/logs`、`workSpace/runs/*/ckpt`、`__pycache__`）
- 当前 run 的 `manifest.json`、`metrics.csv`、视频
- `SHA256SUMS`
- 每次一个日期文件夹，保留多版本。

## 七、卡住怎么办

- 先看 `STATUS.md` 的“风险/阻塞”和 `logs.md` 最后几行。
- 再问当前负责人；不要直接改别人的任务。
- 要跳流程时，先在群里说清违反哪条、为什么，得到确认再动。

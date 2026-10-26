# 交接状态（唯一状态源）

更新时间：2026-10-26 22:00
当前工作人员：空
当前阶段：0

## 上一步

- 阶段/门禁：T0（旧编号）/ 阶段 0 前置
- 结果：本机部分完成；实验室未验证
- 工作人员：MAPO
- 日期：2026-10-26
- 说明：WMP 已克隆到 `c232c115`；代码地图已写；本机 5060 + Isaac Gym Preview 4 无法实质训练，已停止死磕。

## commit / tag

- 基线：`pre-handoff-v2`
- 工作流重构：本次提交（`workflow-v2`）

## 当前 run

- 无

## 证据

- 代码地图：`wmp_overlay/WMP_code_map.md`
- T0 本机日志：`workSpace/runs/stage-0/t0_*.log`
- 冒烟/受控训练脚本：`wmp_overlay/files/train_capped.py`（部署后为 `WMP/train_capped.py`）
- 旧交接单：git 历史 `pre-handoff-v2:workSpace/agentWorkSpace/testBrochure.md`

## 下一步

- 任务卡：`handoff/tasks/00-env.md`
- 第一个动作：在实验室 4090 上运行 `bash wmp_overlay/apply.sh`，然后按 `wmp_overlay/SETUP.md` 建 `wmp` 环境。
- 完成条件：00-env 三项验收全过，显存峰值记录，WMP 无越界改动。

## 风险 / 阻塞

- 4090 环境尚未验证；Isaac Gym Preview 3 / cu117 可能与驱动冲突。
- 上游 `train.py` 硬编码 `max_iterations=100000`；必须用 `train_capped.py` 控制。
- `unitree_rl_gym` 没有 `.git`，父仓库只有 gitlink；不要删除该目录，备份必须整目录快照。

## 备份

- 尚未做；首次交接前按 `AGENTS.md` §6 备份到 U 盘。

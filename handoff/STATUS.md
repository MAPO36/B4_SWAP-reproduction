# 交接状态（唯一状态源）

更新时间：2026-10-28 16:31
当前工作人员：陈旺
当前阶段：0

## 上一步

- 阶段/门禁：T0（环境）
- 结果：**通过**（4090，Preview 4 路线）
- 工作人员：陈旺
- 日期：2026-10-28
- 说明：`wmp` 环境（克隆自 `unitree-rl`）可用；import、环境冒烟、5 iter 受控训练三条全部通过；显存峰值 8946 MiB。

## commit / tag

- 基线：`pre-handoff-v2`
- 本次：见 `git log -1 --oneline`

## 当前 run

- `s0-20261028-1610-t0-env`（已结束，见 `workSpace/runs/stage-0/`）

## 证据

- manifest：`workSpace/runs/stage-0/manifest.json`
- 日志：`t0_env_smoke.log`、`t0_train_5iter.log`、`t0_vram.csv`
- 三条验收的命令、退出码、关键输出：见 manifest 的 `checks` 字段

## 与方案（00-env/SETUP.md）的偏差

1. **Isaac Gym Preview 4（1.0rc4）代替 Preview 3**：机器上不存在 Preview 3，API 面已逐项核对（gymapi/gymtorch/gymutil/terrain_utils），三条验收均通过。
2. **torch 2.3.1+cu121 代替 2.0.1+cu117**：环境克隆自 `unitree-rl`，避免 2GB 下载（官方源实测 88 KB/s）。
3. 需要 conda 激活钩子把 `$CONDA_PREFIX/lib` 加入 `LD_LIBRARY_PATH`，否则 `gym_38.so` 找不到 `libpython3.8.so.1.0`。
4. `ruamel.yaml` 必须 0.17.4；0.18.x 的 `safe_load` 直接抛弃用错误。
5. 运行脚本需 `PYTHONPATH=WMP根目录`，否则 `ModuleNotFoundError: legged_gym`。
6. 上游 `test_env.py` 解包 5 个返回值、`legged_robot.step` 返回 7 个（上游自身不一致）。**未改该测试**，改用等价独立脚本验证环境本身。

## 下一步

- 由使用者决定：继续 Go2 迁移，或先在 4090 补一个 Preview 3 环境做对照。

## 风险 / 阻塞

- Preview 4 与方案要求的 Preview 3 存在版本差异；后续若出现 Isaac Gym 相关异常，优先怀疑此项。
- `handoff/tasks/*.md` 已于 16:04 被移入回收站；git 中完好，可 `git checkout -- handoff/tasks/` 恢复。
- WMP 无越界改动：仅 overlay 文件、`__pycache__/`、`logs/` 为未跟踪。

## 备份

- 尚未做；首次交接前按 `AGENTS.md` §6 备份到 U 盘。

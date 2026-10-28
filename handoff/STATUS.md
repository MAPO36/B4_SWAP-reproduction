# 交接状态（唯一状态源）

更新时间：2026-10-28 16:31
当前工作人员：陈旺
当前阶段：0.5（Go2 迁移）

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

## 阶段 0.5（Go2 迁移）进度

- **G0 通过**：Go2 资产已由 `apply.sh` 复制到 `WMP/resources/robots/go2/`；`go2_amp` 已在 overlay 中注册。
- **G1 通过**：4 个动作文件转换到 `mocap_motions_go2/`，`check_motion_order.py` 输出 `ALL CHECKS PASSED`。
- **G2 通过**：`go2_amp` 冒烟 5 iter，`num_obs=285`、`num_actions=12`，exit 0。
- **G3 通过**：near fraction 0.000（要求 <0.10），`go2_depth.png` 地面从近到远清晰可见。
- **G3 关键修正**：迁移文档把 `y_angle=[-20,-10]` 当作“低头 10~20°”，但代码约定 **正值才是低头**，原值让相机抬头，实测 mean depth 1.98 m、地面只剩左下角一条。已改为 `[12, 22]`，mean 降到 1.41 m、bottom-row 0.56 m。
- **未完成**：300 iter 短训练（需使用者批准，预计超过 10 分钟）与 rollout 视频。

## 下一步

- 等使用者批准后跑 `MAX_ITERS=300 MAX_ENVS=64` 短训练并录 rollout 视频。

## 风险 / 阻塞

- Preview 4 与方案要求的 Preview 3 存在版本差异；后续若出现 Isaac Gym 相关异常，优先怀疑此项。
- `handoff/tasks/*.md` 已于 16:04 被移入回收站；git 中完好，可 `git checkout -- handoff/tasks/` 恢复。
- WMP 无越界改动：仅 overlay 文件、`__pycache__/`、`logs/` 为未跟踪。

## 备份

- 尚未做；首次交接前按 `AGENTS.md` §6 备份到 U 盘。

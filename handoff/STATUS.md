# 交接状态（唯一状态源）

更新时间：2026-11-08 13:09
当前工作人员：空
当前阶段：1（T1 已完成）

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

## 阶段 0.5 进展（2026-11-05 20:51 收尾）

- **G0–G3 全过**（提交 `eae96f8`）。
- **长训练跑了一轮**：`go2_long2000`（256 env / task=go2_amp），跑到 **iter 899/2000** 手动停止。
  训练 reward 0.02 → 11.16，全程 **0 行 nan**，显存峰值约 13.1 GB / 24 GB。
- **最佳 ckpt 已固化**：`workSpace/runs/stage-0/best_ckpt/go2_s05_model800.pt`（306 MB，含 sha256 与 META.txt）。
- **play 评估（climb 地形, 0.6 m/s, seed=1）**：
  | ckpt | total_reward | rew_tracking_lin_vel |
  |---|---|---|
  | iter 300（旧 run） | 0.6403 | 0.154 |
  | iter 400 | 2.9063 | 0.218 |
  | iter 600 | 1.9817 | 0.163 |
  | **iter 800** | **4.0674** | **0.283** |
- **结论**：策略明确在学，但**单次 play 噪声大**（400 高于 600），最终判断必须用多 seed 平均。
- **三地形评估脚本已备**：`workSpace/tools/eval_terrains.sh`（slope/stair/gap × 3 seed），训练结束后跑。

## T1 完成情况（2026-11-08 13:09）

- **训练**：`go2_base512`，512 env × 10000 iter，22h18m（11-07 11:23 → 11-08 09:41），
  最终 reward 15.93，**0 行 nan**，显存峰值 14.8 GB。
- **三地形评估**（64 env × 1200 步，固定 0.6 m/s，确定性动作）：
  | 地形 | 成功率 | 试验数 |
  |---|---|---|
  | slope | **93.8%** | 65 |
  | stair | **98.4%** | 64 |
  | gap | **18.2%** | 148 |
- **结论**：保底档「三地形 > 50%」在 slope/stair 达成，**gap 未达成**（跳跃类技能，需更长训练或专门课程）。
- **材料包**：`/home/ubuntu/B4_2025/Materials/SE/`（327 MB）
  README + T1_report.md + manifest.json + figures/ + logs/ + eval/ + ckpt/ + videos/ + scripts/
- **最终 ckpt**：`ckpt/go2_base512_model9500.pt`（sha256 `e77b12fd…`）

## 下一步

- T2 降显存（当前 14.8 GB，若要 12 GB 卡需降 camera_num_envs）
- T3 镜像地形评估（`np.fliplr` 翻转 height_field_raw，对比训练/镜像成功率）
- gap 若要达标：加跳跃课程或延长训练，而非简单加 iteration

## 下一步（下次开工）

1. 续训：从零重跑更长的基线（建议直接进 T1：10k iter，env 数按显存选 1024–4096），或先补跑到 2000 iter 验证趋势。
2. 跑 `bash workSpace/tools/eval_terrains.sh go2_long2000` 得到三地形对比表（T1 验收的一半）。
3. 录 rollout 视频（T1/0.5 都要求）。
4. 恢复训练前先确认 GPU 空闲、STATUS 无其他工作人员。

## 风险

- 2000 iter 未跑完；当前最优只是 iter 800，后续可能仍有提升空间。
- play 单次结果噪声大，勿用单次数字下结论。
- 环境为 Preview 4 而非方案要求的 Preview 3（详见阶段 0 记录）。

## 阶段 0.5 收尾状态（2026-10-28 17:41 交接）

- **G0–G3 全部通过**（提交 `eae96f8`）。
- **短训练完成**：`go2_capped300`（300 iter / 64 env），reward 0.72→1.99，无 nan，显存峰值约 9.0 GB。
  ckpt：`WMP/logs/go2_amp_example/go2_capped300/model_300.pt`（118 MB）。
- **play 验证已跑通**：窗口能在 4090 显示器显示（需 `DISPLAY=:1` +
  `XAUTHORITY=/run/user/1000/gdm/Xauthority`）；headless 下也能加载策略并打印 total reward。
  日志：`workSpace/runs/stage-0/go2-300iter/train.log`。

### 已知问题（下次先解决）

1. **机器人站不直**：零动作平地测试稳定在 0.274 m、前倾约 9°。默认姿态本身不水平，
   300 iter 的策略不足以补偿 → 加载策略后前栽。属训练量不足，非配置 bug。
2. **无 WASD 遥控**：WMP 只绑定了 ESC（退出）和 V（切同步）；`play.py` 把
   `lin_vel_x` 写死为 0.6。要实现键盘遥控需新增 overlay 脚本。
3. `play.py` 硬编码 `load_run='WMP'`，靠软链接
   `logs/go2_amp_example/WMP -> go2_capped300` 解决（已建，勿删）。
4. `camera_num_envs` 必须 ≤ `crawl_end_idx`；已在 `train_capped.py` 中自动收敛。

## 下一步

- 写 `go2_teleop.py`（overlay）实现 WASD 速度控制，用于人工验证仿真/资产。
- 长训练（2000~5000 iter）让策略真正学会站立与行走，再录 rollout 视频。

## 风险 / 阻塞

- Preview 4 与方案要求的 Preview 3 存在版本差异；后续若出现 Isaac Gym 相关异常，优先怀疑此项。
- `handoff/tasks/*.md` 已于 16:04 被移入回收站；git 中完好，可 `git checkout -- handoff/tasks/` 恢复。
- WMP 无越界改动：仅 overlay 文件、`__pycache__/`、`logs/` 为未跟踪。

## 备份

- 尚未做；首次交接前按 `AGENTS.md` §6 备份到 U 盘。

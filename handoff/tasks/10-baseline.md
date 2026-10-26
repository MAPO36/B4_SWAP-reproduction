# 阶段 1 — WMP/Go2 基线

- 状态：以 `handoff/STATUS.md` 为准
- 入口：`05-go2.md` 退出条件全过
- 出口：完整基线训练 + 训练地形评估 + 视频 + manifest

## 目标
跑出 WMP/Go2 基线，作为对称改动的对照。

## 允许改
- 只通过 overlay 改 WMP；允许调 `num_envs`、`camera_num_envs`、`depth.update_interval`。
- 不改奖励、网络、RSSM、AMP、地形生成。

## 训练
```bash
cd WMP && MAX_ITERS=10000 MAX_ENVS=4096 python train_capped.py \
  --task=go2_amp --headless --sim_device=cuda:0 --num_envs=4096
```
- OOM：overlay 改 `depth.camera_num_envs=256~512`、`num_envs=2048`，再 apply。
- 10k iter 或到收敛；记录 iter time/显存/reward；seed 固定；夜间跑登记 STATUS（run_id/路径/预计完成/监控命令）。

## 评估
- 训练地形（slope/stairs/gap）100~200 episodes；固定命令、确定性动作均值。
- 写 `eval_success.py` 放 overlay，用 `reset_buf` / `time_out_buf` 统计成功率。
- 产出 `metrics.csv`、训练曲线、2 段视频；从 ckpt 重跑一次确认可复现。

## 必须产物
- ckpt、log、`metrics.csv`、视频、`manifest.json`（字段见 `AGENTS.md` §4.1）。

## 退出条件
- 无 nan、显存不爆、成功率表存在、视频可看、重跑评估一致。

## 禁止
- 不收敛就改奖励/网络；用冒烟或短训练代替 10k 基线。

## 下一步
- `handoff/tasks/30-mirror-eval.md`

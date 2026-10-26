# 阶段 3 — 镜像地形评估

- 状态：以 `handoff/STATUS.md` 为准
- 入口：`10-baseline.md` 退出条件全过
- 出口：训练地形 vs 镜像地形成功率对比表 + 视频

## 目标
测出 WMP 基线在镜像地形上的成功率下降，作为 SWAP 要解决的问题。

## 允许改
- 新增评估脚本放 overlay；不改训练代码/奖励/地形生成逻辑。
- 地形翻转只在评估脚本里做（`np.fliplr(height_field_raw)`）。

## 步骤
1. 在 `terrain.py` 找到 `tilt` / `gap` 生成函数，确认评估入口。
2. 写 `eval_mirror.py` 放 `wmp_overlay/files/`：加载阶段 1 ckpt；生成“左高右低/左低右高”两组地形；每组 100~200 episodes。
3. 固定命令、确定性动作均值；记录成功率、失败原因、episode 长度。
4. 产出对比 CSV、柱状图/折线图、2 段视频（训练地形/镜像地形）。
5. 重跑一次确认结果一致。

## 必须产物
- `metrics.csv`（训练/镜像两列）、对比图、2 段视频、`manifest.json`。
- 脚本在 `wmp_overlay/files/`，可被 `apply.sh` 还原。

## 退出条件
- 训练地形成功率与阶段 1 一致；镜像地形成功率明显下降；脚本可复现。

## 禁止
- 改地形生成器让镜像地形变容易；改验收脚本让它通过。
- 只跑 10 次就下结论。

## 下一步
- `handoff/tasks/40-symmetric.md`

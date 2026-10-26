# 阶段 4 — 对称 actor/critic

- 状态：以 `handoff/STATUS.md` 为准
- 入口：`30-mirror-eval.md` 退出条件全过
- 出口：单测通过 + 2k~5k iter 训练 + 镜像地形评估不低于基线

## 目标
在 actor/critic 上加对称等变（SE-MLP），让策略在镜像地形上不退化。

## 允许改
- 新增 `symmetry.py`；通过 overlay 改 `actor_critic_wmp.py`。
- 只动 actor/critic；暂不动世界模型；不改奖励/地形/RSSM。

## 步骤
1. 按论文 Table I 列镜像：线速度 `(vx,-vy,vz)`；角速度 `(-wx,wy,-wz)`；gravity `(gx,-gy,gz)`；命令 `(vx,-vy,-wz)`；关节位置/速度/动作左右交换；Go2 顺序 `[3,4,5,0,1,2,9,10,11,6,7,8]`，hip `[0,3,6,9]` 取负；接触标志左右交换。
2. 写 `F_obs` / `F_act`；actor/critic MLP 换 SE-MLP；critic 输出保持镜像不变。
3. 单测 100 次：`actor(F_obs(o)) == F_act(actor(o))`，`critic(F_obs(o)) == critic(o)`，atol=1e-5。
4. 训练 2k~5k iter（夜间可跑），用阶段 3 脚本评估训练/镜像地形。
5. 与基线对比，产出成功率表、视频、曲线。

## 必须产物
- 单测 log、ckpt、`metrics.csv`、对比图、视频、`manifest.json`。

## 退出条件
- 单测通过；策略能走；镜像地形成功率不低于基线（趋势即可）。
- 训练崩：先只加 actor 对称、critic 保持普通，并在 STATUS 记录。

## 禁止
- 改镜像地形/验收脚本让它通过；跳过单测直接训练。
- 同时改世界模型和策略（归因不清）。

## 下一步
- `handoff/tasks/60-report.md`

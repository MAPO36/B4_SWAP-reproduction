# Go2 最小改动迁移清单（SWAP / WMP）

> 一句话：**WMP 眼里的 Go2 和 A1 是同一个接口**——12 个关节、4 只脚、8 个 thigh/calf 刚体。
> 所以网络、观测、动作、世界模型的张量形状**一个都不用改**。
> 真正要动的只有 6 个地方：URDF 路径、初始高度、站立高度、相机位置、动作顺序、任务名。

这份清单配套 `SWAP/go2_migration/` 里的 4 个文件使用。
先按 `README.md` 一步步做，做到哪一步不确定就回来查这里。

---

## 一、为什么说“改动很小”

WMP 的输入输出是**按张量形状写死**的，不是按机器人型号写死的：

| 东西 | A1 | Go2 | 要改吗 |
|---|---|---|---|
| 关节数 | 12 | 12 | 否 |
| 脚的个数 | 4 | 4 | 否 |
| 腿的刚体数（thigh+calf） | 8 | 8 | 否 |
| 本体观测 `prop_dim` | 33 | 33 | 否 |
| 特权观测 `privileged_dim` | 53 | 53 | 否 |
| 高度图 `height_dim` | 187 | 187 | 否 |
| 动作维度 | 12 | 12 | 否 |
| 总观测 `num_observations` | 285 | 285 | 否 |
| AMP 判别器输入 | 30 | 30 | 否 |
| RSSM 世界模型维度 | 固定 | 固定 | 否 |
| 地形 / 课程 / 奖励函数 | 通用 | 通用 | 否 |
| 对称变换（左右腿交换） | 4 条腿 | 4 条腿 | 否 |

唯一变的是“同一个 12 维向量里，哪个数对应哪条腿”，以及几个和尺寸有关的常数。
这就是为什么我们敢说“最小改动”。

---

## 二、本地能直接复用的东西（重点）

### 2.1 直接复制就能用

| 本地位置 | 复制到 WMP 的位置 | 用途 | 备注 |
|---|---|---|---|
| `unitree_rl_gym/resources/robots/go2/` | `WMP/resources/robots/go2/` | Isaac Gym 的 URDF 和视觉 mesh | **整个文件夹复制**，URDF 用相对路径找 `../dae/` |
| `unitree_rl_gym/resources/robots/go2/urdf/go2.urdf` | 同上 | 12 个转动关节 + 4 个脚；关节限位、effort、惯量都在里面 | 关节顺序 FL, FR, RL, RR |
| `unitree_rl_gym/resources/robots/go2/dae/` | 同上 | 视觉模型 | Isaac Gym 加载的是这个 |
| `unitree_rl_gym/resources/robots/go2/go2.xml`、`scene.xml` | 先不用 | 阶段 8 的 MuJoCo sim2sim | 到那时再复制 |

> Go2 的碰撞体不是 mesh，而是 box/cylinder/sphere（URDF 里写好的），
> 这对 Isaac Gym 是好事：比凸分解快，也不会因为 mesh 太薄而穿模。

### 2.2 参数可以抄，但配置不能直接用

| 本地文件 | 能抄什么 | 为什么不能直接注册成 WMP 任务 |
|---|---|---|
| `unitree_rl_gym/legged_gym/envs/go2/go2_config.py` | 初始高度 `pos.z=0.42`；默认关节角；`action_scale=0.25`；`decimation=4`；`stiffness=20`、`damping=0.5`（备用） | 它是 47 维**盲走**策略，观测布局、奖励、网络都和 WMP 不一样 |
| `unitree_rl_gym/deploy/deploy_mujoco/configs/go2.yaml` | PD 增益、关节顺序、观测拼接顺序、键盘控制 | 这是 MuJoCo 部署配置，不是训练配置 |
| `go2_1` ~ `go2_8`、`rough_go2` 的 checkpoint | 只能当“Go2 在 Isaac Gym 里能跑”的证据 | 47 维输入，和 WMP 的 285 维完全不兼容 |

**结论**：本地 Go2 的价值是“模型 + 尺寸 + 默认姿态 + 部署参数”，
不是“训练代码”。训练代码用 WMP 的，一行算法都不用改。

### 2.3 本次新增的文件

| 文件 | 作用 |
|---|---|
| `SWAP/go2_migration/go2_amp_config.py` | 从 `a1_amp_config.py` 改出的 Go2 配置 |
| `SWAP/go2_migration/convert_motion_a1_to_go2.py` | A1 关节顺序 → Go2 关节顺序 |
| `SWAP/go2_migration/check_motion_order.py` | 验证转换正确 + 关节不超限 |
| `SWAP/go2_migration/check_go2_depth.py` | 存一张深度图，验证相机没被头挡住 |
| `SWAP/go2_migration/README.md` | 按顺序执行的操作手册 |

---

## 三、逐文件改动（就这 5 个文件）

### 改动 1：新增 `legged_gym/envs/go2/go2_amp_config.py`

从 WMP 的 `legged_gym/envs/a1/a1_amp_config.py` 复制，只改下面这些地方：

| 字段 | A1 原值 | Go2 新值 | 原因 |
|---|---|---|---|
| `MOTION_FILES` | `datasets/mocap_motions/*` | `datasets/mocap_motions_go2/*` | 关节顺序不同 |
| 类名 | `A1AMPCfg` / `A1AMPCfgPPO` | `GO2AMPCfg` / `GO2AMPCfgPPO` | 避免和 A1 冲突 |
| `init_state.pos` | `[0, 0, 0.35]` | `[0, 0, 0.42]` | 本地 `go2_config.py` 的出生高度 |
| `depth.position` | `[0.27, 0, 0.03]` | `[0.28, 0, 0.11]` | 原位置在 Go2 头壳里面 |
| `depth.y_angle` | `[-5, 5]` | `[-20, -10]` | 低头看前方地面 |
| `asset.file` | `.../a1/urdf/a1.urdf` | `.../go2/urdf/go2.urdf` | 换模型 |
| `asset.name` | `"a1"` | `"go2"` | 换模型 |
| `normalization.base_height` | `0.3` | `0.33` | Go2 站立髋高 |
| `rewards.base_height_target` | `0.3` | `0.33` | 该项奖励权重是 0，保持数值一致 |
| `runner.experiment_name` | `a1_amp_example` | `go2_amp_example` | 日志分开放 |

**没有改**：`prop_dim`、`privileged_dim`、`height_dim`、`num_observations`、
RSSM 配置、地形比例、奖励系数、AMP 系数、PPO 超参数。

### 改动 2：注册任务 `legged_gym/envs/__init__.py`

加 import：

```python
from .go2.go2_amp_config import GO2AMPCfg, GO2AMPCfgPPO
```

加注册：

```python
task_registry.register( "go2_amp", LeggedRobot, GO2AMPCfg(), GO2AMPCfgPPO() )
```

### 改动 3：复制资产 `resources/robots/go2/`

只复制文件夹，不改 URDF。URDF 里的 `dont_collapse="true"` 是 Isaac Gym 的扩展，
用来保留 4 个脚和 2 个头壳刚体；Go2 的 URDF 已经写好了。

### 改动 4：生成 `datasets/mocap_motions_go2/`

用 `convert_motion_a1_to_go2.py` 生成，**不要手改**。
生成后必须用 `check_motion_order.py` 验证。

### 改动 5：`go2_amp_config.py` 里的 PD 增益（先不动）

保留 WMP 的 `40 / 1.0`。本地 Go2 盲走策略用 `20 / 0.5`，但那是另一套奖励和动作缩放调出来的。
只有在训练抖动/扭矩饱和时才整体换，并且 MuJoCo 部署必须跟着换。

---

## 四、AMP 动作为什么只要“换个顺序”

WMP 的 AMP 判别器每帧只吃 30 个数：

```
[关节角 12] + [机身线速度 3] + [机身角速度 3] + [关节角速度 12]
```

**脚的位置、机身高度、机身朝向都不进判别器**（代码里被注释掉了，
`get_full_amp_observations` 写了但训练没用）。
所以不需要 IK 重定向、不需要重新录动作，只要保证“第 i 个关节角”对应 Go2 的第 i 个关节。

### 4.1 两边顺序

| 腿 | A1 在 12 维里的位置 | Go2 在 12 维里的位置 |
|---|---|---|
| 左前 FL | 3, 4, 5 | 0, 1, 2 |
| 右前 FR | 0, 1, 2 | 3, 4, 5 |
| 左后 RL | 9, 10, 11 | 6, 7, 8 |
| 右后 RR | 6, 7, 8 | 9, 10, 11 |

所以 Go2 的第 0 个数 = A1 的第 3 个数，Go2 的第 1 个 = A1 的第 4 个……：

```python
GO2_FROM_A1 = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
```

每条腿内部（hip, thigh, calf）顺序不变。

### 4.2 为什么不用改符号

我们对比过两边的 URDF：

- hip 关节轴都是 `1 0 0`；
- thigh / calf 关节轴都是 `0 1 0`；
- 默认角完全一样：hip ±0.1，前 thigh 0.8、后 thigh 1.0，calf -1.5。

轴一样、默认角一样，就说明正方向一样，**不需要给 hip 外摆取负**。
`check_motion_order.py` 还会检查所有角度是否落在 Go2 限位内，我们已经跑过 4 个动作，
全部通过：

```
OK   hop1.txt      501 frames, joints inside Go2 limits
OK   hop2.txt      501 frames, joints inside Go2 limits
OK   trot1.txt     501 frames, joints inside Go2 limits
OK   trot2.txt     501 frames, joints inside Go2 limits

ALL CHECKS PASSED
```

### 4.3 转换脚本到底改了什么

每帧 61 个数，只重排 4 个 12 维块，其他原样复制：

| 下标 | 内容 | 处理 |
|---|---|---|
| 0:3 | 机身位置 | 不动 |
| 3:7 | 机身四元数 | 不动 |
| 7:19 | 12 个关节角 | **重排** |
| 19:31 | 12 个脚的位置 | 重排（判别器不用，但保持一致） |
| 31:34 | 机身线速度 | 默认不动；可选 ×1.065 |
| 34:37 | 机身角速度 | 不动 |
| 37:49 | 12 个关节角速度 | **重排** |
| 49:61 | 12 个脚的速度 | 重排（判别器不用） |

线速度 ×1.065 是可选优化：Go2 小腿 0.213 m，A1 是 0.200 m，同样的关节角 Go2 实际更快 6.5%。
第一版**不要开**，等基线步态明显偏慢再试。

### 4.4 SWAP 对称变换在 Go2 上的索引（写 `symmetry.py` 时用）

Go2 的 12 维关节顺序是：

```
FL: 0,1,2    FR: 3,4,5    RL: 6,7,8    RR: 9,10,11
```

左右镜像就是“交换 FL↔FR、RL↔RR”，索引排列和转换脚本完全一样：

```python
LEFT_RIGHT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
HIP_INDICES     = [0, 3, 6, 9]      # 这 4 个 hip 外摆角要取负
```

所以写 `F_act(action)` 时：

```python
def mirror_action(a):
    out = a[..., LEFT_RIGHT_PERM].clone()
    out[..., HIP_INDICES] *= -1
    return out
```

`F_obs` 里的关节块、关节速度块、上一帧动作块都用同一个 `LEFT_RIGHT_PERM`；
机身线速度 `(vx,-vy,vz)`、角速度 `(-wx,wy,-wz)`、gravity `(gx,-gy,gz)` 的符号规则不变。
**不要**把 A1 的 `FR,FL,RR,RL` 顺序抄进对称代码，否则左右会反两次、等于没反。

---

## 五、相机：唯一需要“眼见为实”的地方

A1 的相机 `[0.27, 0, 0.03]` 放到 Go2 上会出问题：

- Go2 身体前缘到 `x=0.332`；
- 头顶到 `z≈0.089`；
- 相机在 `[0.27, 0, 0.03]` 时，正好在身体/头壳内部。

配置里改成 `[0.28, 0, 0.11]`、俯仰 `[-20, -10]` 度。
但这是按 mesh 尺寸估的，**必须跑一次 `check_go2_depth.py` 看 PNG**。

判断标准：

| 指标 | 合格 | 不合格的含义 |
|---|---|---|
| `near fraction (<0.10 m)` | < 0.10 | > 0.30 说明画面一大块是机器人自己 |
| `go2_depth.png` | 能看到地面从近到远 | 全黑/全白 = 没有深度返回；下半屏巨大机身 = 相机太低 |
| `bottom-row mean` | 比 top 小，但不小于 0.1 m | 接近 0.05 m 说明正对着自己的头 |

调参只改 `class depth` 的 `position` 和 `y_angle`，每次改完重跑脚本。
这一步最多花半天，但能避免“训练一周，策略在看后脑勺”的惨剧。

---

## 六、Go2 的尺寸和指标换算

论文的难度是按机器人尺寸归一化的，不是绝对米数：

- **L = 前后髋关节中心的距离**；
- **H = 站立时的髋关节高度**。

### 6.1 Go2 的 L 和 H

从本地 URDF 量出来：

```
前髋 x = +0.1934 m
后髋 x = -0.1934 m
L = 0.3868 m
```

站立高度（thigh 0.213 m、calf 0.213 m，默认角 0.8 / -1.5，脚球半径 0.022 m）：

```
H = 0.213·cos(0.8) + 0.213·cos(0.7) + 0.022
  = 0.1484 + 0.1629 + 0.022
  ≈ 0.333 m
```

### 6.2 难度阶梯（Go2 版）

| 归一化倍数 | 沟宽（3.0 那一列用 L） | 台高（用 H） |
|---|---|---|
| 1.0× | 0.39 m | 0.33 m |
| 1.5× | 0.58 m | 0.50 m |
| 2.0× | 0.77 m | 0.66 m |
| 2.5× | 0.97 m | 0.83 m |
| 3.0× | **1.16 m** | **1.00 m** |

和论文 Apollo 的 2.13 m / 1.63 m 是同一把尺子，只是 Go2 小一号。

### 6.3 轻量复现的目标怎么定

| 档位 | Go2 目标 | 说明 |
|---|---|---|
| 保底 | 跑通 WMP 基线，slope/stairs/gap 能走 | gap 用默认难度，不追米数 |
| 标准 | 对称 actor/critic，镜像地形提升 | 不涉及极限米数 |
| 冲刺 | 完整 SWAP-lite | 同上 |
| 游刃有余（可选） | 至少一个技能 ≥ **2.0×**（0.77 m 沟 / 0.66 m 台），最好 3.0×（1.16 / 1.00） | 达不到就按 1.5× → 1.0× 降 |

**不要拿 A1 的 1.10 / 0.90 直接套 Go2**，两者差 5~10%，评估时会误判。

---

## 七、Go2 比 A1 “力气小”这件事

| | A1 | Go2 |
|---|---|---|
| 质量 | 12.45 kg | 15.02 kg |
| 髋/大腿扭矩 | 20 N·m | 23.7 N·m |
| 小腿扭矩 | 30 N·m | 35.55 N·m |
| 大腿 扭矩/体重 | 2.41 N·m/kg | 1.58 N·m/kg |
| 小腿 扭矩/体重 | 2.41 N·m/kg | 2.37 N·m/kg |

大腿的“单位体重扭矩”明显更低，所以：

1. 论文里 Apollo 的 3.0× 跳跃对 Go2 更难，**2.0× 是更现实的目标**；
2. 如果训练中出现“蹲不下去、起跳没力”，先降难度，不要先怀疑世界模型；
3. 不要为了追米数把 `action_scale` 调大，那会让 AMP 动作完全失真。

---

## 八、验收闸门（每一步不过，不准进下一步）

| 闸门 | 做什么 | 通过标准 | 不过怎么办 |
|---|---|---|---|
| G0 | 复制资产、配置、注册任务 | `ls resources/robots/go2/urdf/go2.urdf` 存在；`--task=go2_amp` 能被识别 | 检查 `__init__.py` 拼写 |
| G1 | 转动作 + 检查 | `ALL CHECKS PASSED` | 看 `FAIL` 的关节名；不要手改 |
| G2 | 16 环境跑 5 iter | 不报错；`num_obs=285`、`num_actions=12` | 按 README 的报错表排查 |
| G3 | 深度图 | `near fraction < 0.10`；PNG 能看到地面 | 调 `position.z` / `y_angle` |
| G4 | 200 iter 短训练 | reward 在涨，无 nan，显存稳定 | 降 `camera_num_envs` |
| G5 | 2k iter 基线 | 能站、能走，不原地抖 | 检查 PD、动作顺序、AMP reward |
| G6 | 完整基线 | slope/stairs/gap 能走，有视频 | 回到 G4，不要改算法 |

**G1 和 G3 是最容易被跳过、也最容易埋雷的两步。**
跳过 G1 → 腿的顺序错，策略学出“顺拐”；跳过 G3 → 世界模型学后脑勺。
这两个错误都不会让程序报错，只会让最后结果莫名其妙地差。

---

## 九、Agent 任务板（Go2 版）

在原来的 T0~T13 前面插入 4 个迁移任务，原来的 T1 顺延为“Go2 基线”。

| 编号 | 任务 | 依赖 | 验收 |
|---|---|---|---|
| **G0** | 复制 Go2 资产 + 配置 + 注册 `go2_amp` | T0 | 能 `--task=go2_amp` 建环境 |
| **G1** | 转 4 个动作文件 + 跑检查脚本 | G0 | `ALL CHECKS PASSED` |
| **G2** | 16 环境冒烟 5 iter | G1 | 不报错，维度 285/12 |
| **G3** | 深度图检查 + 调相机 | G2 | `near fraction < 0.10`，PNG 有地面 |
| T1' | Go2 基线 10k iter | G3 | slope/stairs/gap 能走，视频 |
| T2' | 降显存 | T1' | 12 GB 内能跑，趋势一致 |
| T3' | 镜像地形评估 | T1' | 基线成功率表 |
| T4~T11 | 同原计划（对称代码与机器人无关） | - | 同原计划 |
| T12' | 爬高/跳远地形 + `go2_parkour` 配置 | T1' | 地形可视化 + 短训练不崩 |
| T13' | 两技能对比训练与评估 | T9,T12' | 成功率/极限能力表 + 对比视频 |

### 可以直接复制的提示词

**G0**
```
在 WMP 仓库里新建 legged_gym/envs/go2/，放入 go2_amp_config.py，
在 legged_gym/envs/__init__.py 注册 go2_amp。
把 unitree_rl_gym/resources/robots/go2/ 整个复制到 WMP/resources/robots/go2/。
不要改 WMP 的算法代码。
验收：python legged_gym/scripts/train.py --task=go2_amp --headless --num_envs=16 能跑 5 iter，
日志里 num_obs=285, num_actions=12。
```

**G1**
```
运行 convert_motion_a1_to_go2.py，把 datasets/mocap_motions 转到 mocap_motions_go2，
再运行 check_motion_order.py。
要求：输出 ALL CHECKS PASSED；把 4 个文件的关节范围打印给我。
如果 FAIL，不要修改脚本去“让它通过”，先把失败的那一帧原始数据打印出来。
```

**G3**
```
运行 check_go2_depth.py，把 go2_depth.png 和终端统计发给我。
如果 near fraction > 0.10，只允许改 go2_amp_config.py 的 depth.position 和 depth.y_angle，
每次改完重跑，并记录改了什么、near fraction 变成多少。
不要改相机 FOV、far_clip、update_interval。
```

---

## 十、时间估计

| 步骤 | 时间 |
|---|---|
| G0 资产 + 配置 + 注册 | 0.5 天 |
| G1 转动作 + 检查 | 0.5 天（脚本已写好，主要是跑） |
| G2 冒烟 + 修报错 | 0.5~1 天 |
| G3 相机检查 + 调参 | 0.5 天 |
| 后面接原计划阶段 1 | 不变 |

总共为 Go2 迁移多花 **2 天左右**，比“重新做一套 Go2 训练代码”省 2~3 周。

---

## 十一、已知风险（以及为什么可以接受）

| 风险 | 说明 | 处理 |
|---|---|---|
| URDF 视觉用 `.dae` | Isaac Gym 对 dae 的支持取决于版本 | 你本地 `unitree_rl_gym` 用同一个 URDF 训练成功过（go2_1~go2_8），说明这个环境能加载；如果新服务器报错，把 visual 的 dae 换成 `meshes/*.obj` 或直接删掉 visual（碰撞不受影响） |
| Go2 刚体数 19，A1 是 17 | 多了 2 个头壳刚体 | WMP 全部用 `num_bodies` 动态取，AMP 也只按名字找脚，已验证没有写死 17 |
| `dont_collapse` 固定关节 | 脚和头壳被保留 | `self_collisions=1` 关闭自碰撞，不影响训练 |
| 动作文件机身高度是 A1 的 | A1 约 0.30 m，Go2 约 0.33 m | 训练里 `reference_state_initialization=False`，动作只用于 AMP 判别器，不用于重置；差 3 cm 不影响“像不像” |
| Go2 大腿单位体重扭矩低 | 极限跳跃更难 | 指标按 2.0× 定，不硬追 3.0× |
| 相机位置是估算的 | 可能被头挡 | G3 必须看图，不许跳过 |

---

## 十二、一句话总结

**Go2 迁移 = 换一个 URDF + 换一个动作顺序 + 挪一下相机 + 改 4 个常数。**
WMP 的算法、SWAP 的对称代码、世界模型、奖励、地形，全部原样复用。
先把 G0~G3 四个闸门走完，再开长训练；这四步总共 2 天，能省掉后面两周的“玄学调参”。

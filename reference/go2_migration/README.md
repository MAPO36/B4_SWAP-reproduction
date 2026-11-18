# Go2 迁移包（WMP / SWAP-lite 用）

这个文件夹只放“从 A1 换到 Go2 需要新增或修改的东西”，**不改 WMP 的算法代码**。
一共 4 个文件：

| 文件 | 作用 |
|---|---|
| `go2_amp_config.py` | Go2 的训练配置，从 WMP 的 `a1_amp_config.py` 改出来 |
| `convert_motion_a1_to_go2.py` | 把 A1 关节顺序的 AMP 动作文件转成 Go2 顺序 |
| `check_motion_order.py` | 检查转换有没有转错（强烈建议每次转完都跑） |
| `check_go2_depth.py` | 单独跑 1 个环境，把深度图存成 PNG，确认相机没被 Go2 的头挡住 |

下面按顺序做，每一步都有“看到什么算成功”。

---

## 第 0 步：前提

WMP 已经按原方案装好，`--task=a1_amp` 能跑起来（哪怕只跑 5 个 iteration）。
**先不要动 WMP 的算法代码。** Go2 迁移只碰配置、资产、动作文件。

下面的命令都假设你在 WMP 仓库根目录：

```bash
cd ~/WMP          # 换成你自己的路径
conda activate wmp
```

---

## 第 1 步：把 Go2 的模型文件放进 WMP

WMP 自带的是 A1，路径是 `resources/robots/a1/`。
Go2 的 URDF 和 mesh 用你本地 `unitree_rl_gym` 里现成的，**整个文件夹原样复制**：

```bash
# 在 WMP 根目录执行
mkdir -p resources/robots
cp -r /path/to/B4_sim2sim/unitree_rl_gym/resources/robots/go2 resources/robots/go2
```

复制完检查一下：

```bash
ls resources/robots/go2/urdf/go2.urdf
ls resources/robots/go2/dae/ | head
```

**为什么必须整个文件夹复制**：URDF 里写的是相对路径 `../dae/xxx.dae`，
只复制 `go2.urdf` 会找不到模型。`meshes/`、`go2.xml`、`scene.xml` 是给 MuJoCo 用的，
Isaac Gym 暂时用不到，但一起复制不碍事，后面 sim2sim 还会用。

> 注意：Go2 的 URDF 关节顺序是 **FL, FR, RL, RR**，A1 是 **FR, FL, RR, RL**。
> 这是整个迁移里唯一会“静默出错”的地方，第 2 步会处理。

---

## 第 2 步：转换 AMP 动作文件

WMP 的 AMP 判别器只看 30 个数：

```
[12 个关节角, 3 个机身线速度, 3 个机身角速度, 12 个关节角速度]
```

它**不看脚的位置、不看机身高度**，所以不需要做复杂的动作重定向（retargeting），
只要把 12 个关节的左右前后顺序换一下。

在 WMP 根目录执行：

```bash
python /path/to/SWAP/go2_migration/convert_motion_a1_to_go2.py \
    --src datasets/mocap_motions \
    --dst datasets/mocap_motions_go2
```

会看到：

```
A1 -> Go2 permutation: [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
linear-velocity scale: 1.0
  hop1.txt      501 frames -> datasets/mocap_motions_go2/hop1.txt
  hop2.txt      501 frames -> datasets/mocap_motions_go2/hop2.txt
  trot1.txt     501 frames -> datasets/mocap_motions_go2/trot1.txt
  trot2.txt     501 frames -> datasets/mocap_motions_go2/trot2.txt
```

然后**一定要检查**：

```bash
python /path/to/SWAP/go2_migration/check_motion_order.py \
    --src datasets/mocap_motions \
    --dst datasets/mocap_motions_go2
```

正确输出是四行 `OK` 加最后一行：

```
ALL CHECKS PASSED
```

它会做三件事：
1. 确认每一帧还是 61 列；
2. 确认关节顺序确实按 `[3,4,5,0,1,2,9,10,11,6,7,8]` 换了；
3. 确认所有关节角都在 Go2 的 URDF 限位内。

我们已经在 WMP 自带的 4 个动作上跑过，四个都是 `ALL CHECKS PASSED`。
如果这里报 `FAIL`，**不要继续训练**，先看报错里的关节名和角度范围。

### 线速度要不要乘 1.065？

先不要。默认 `--lin-vel-scale 1.0`。
Go2 的小腿比 A1 长 6.5%，同样的关节角理论上走得更快，但 AMP 只是“像不像”，
差 6.5% 通常看不出来。等基线跑起来、步态明显偏慢/偏快时再试：

```bash
python convert_motion_a1_to_go2.py --lin-vel-scale 1.065 --overwrite
python check_motion_order.py --lin-vel-scale 1.065
```

---

## 第 3 步：放入配置并注册任务

```bash
# 在 WMP 根目录执行
mkdir -p legged_gym/envs/go2
cp /path/to/SWAP/go2_migration/go2_amp_config.py legged_gym/envs/go2/
touch legged_gym/envs/go2/__init__.py
```

然后编辑 `legged_gym/envs/__init__.py`，在 A1 的 import 下面加一行：

```python
from .go2.go2_amp_config import GO2AMPCfg, GO2AMPCfgPPO
```

在文件最后的 `task_registry.register(...)` 下面加一行：

```python
task_registry.register( "go2_amp", LeggedRobot, GO2AMPCfg(), GO2AMPCfgPPO() )
```

改完应该长这样（只列出相关部分）：

```python
from .a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO
from .go2.go2_amp_config import GO2AMPCfg, GO2AMPCfgPPO
...
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_amp", LeggedRobot, A1AMPCfg(), A1AMPCfgPPO() )
task_registry.register( "go2_amp", LeggedRobot, GO2AMPCfg(), GO2AMPCfgPPO() )
```

> 想偷懒也可以不建 `go2/` 文件夹，直接把配置改名放进 `a1/`，
> 但建议保持 `legged_gym/envs/go2/`，以后加 `go2_parkour` 配置时不会乱。

---

## 第 4 步：冒烟测试（先别开大训练）

```bash
python legged_gym/scripts/train.py \
    --task=go2_amp --headless --sim_device=cuda:0 --num_envs=16
```

**成功的样子**：
- 不报 `KeyError: 'FL_hip_joint'`；
- 不报 `FileNotFoundError: .../go2.urdf`；
- 日志里 `num_obs` 是 **285**，`num_actions` 是 **12**（和 A1 完全一样）；
- 能打印 iteration 0、1、2……的 loss；
- 显存没有爆。

**如果报这些，基本就是对应原因**：

| 报错 | 原因 | 怎么办 |
|---|---|---|
| `KeyError: 'FL_hip_joint'` | 注册的配置还是 A1 | 检查 `--task=go2_amp` 和 `__init__.py` |
| `FileNotFoundError ... go2.urdf` | 资产没复制对 | 回到第 1 步，确认 `urdf/` 和 `dae/` 都在 |
| `KeyError: 'FL_hip'` 或 body 名对不上 | 用了 MuJoCo 的 `go2.xml` | Isaac Gym 必须用 `urdf/go2.urdf` |
| 显存爆 | `camera_num_envs=1024` 太大 | 先改成 64 或 128 |
| `IndexError` / 全 `-inf` 深度图 | 相机位置在 Go2 头里面 | 做第 5 步 |

---

## 第 5 步：确认深度相机没被 Go2 的头挡住（重要）

A1 的相机位置是 `[0.27, 0, 0.03]`，直接抄给 Go2 会**卡在头壳里面**：
Go2 的身体前缘到 x=0.332，头顶到 z≈0.089。
所以配置里已经改成：

```python
position = [0.28, 0, 0.11]
y_angle = [-20, -10]      # 低头 10~20 度
```

但这个值只是按 mesh 尺寸估算的，**必须亲眼看一张图**。
把 `check_go2_depth.py` 复制到 WMP 根目录，执行：

```bash
cp /path/to/SWAP/go2_migration/check_go2_depth.py .
python check_go2_depth.py --task=go2_amp --headless
```

它会：
1. 只建 1 个环境；
2. 空动作走 10 个物理步（机器人会自然站住）；
3. 把深度图存成 `go2_depth.png`；
4. 在终端打印：

```
depth range (m): 0.12 .. 2.00
mean depth (m): 0.85
near fraction (<0.10 m): 0.03
bottom-row mean (m): 0.90
top-row mean (m): 1.40
saved: go2_depth.png
```

**怎么判断**：
- `near fraction` 应该 **小于 0.10**。如果大于 0.30，说明画面里一大块是机器人自己（头/身体），
  相机还在模型里面或者太低；
- `go2_depth.png` 应该能看出**地面从近到远**，不是纯黑、纯白、或者下半屏一个巨大的机身；
- `bottom-row mean` 比 `top-row mean` 小是正常的（下边是近处地面），但不应小到 0.1 m。

**不合格怎么调**（只改 `go2_amp_config.py` 的 `class depth`）：

| 现象 | 改法 |
|---|---|
| 画面下半部分全是机身 | `position` 的 z 再加 0.02~0.04 |
| 完全看不到地面，只看到远处 | `y_angle` 改成 `[-30, -20]`（低头更多） |
| 相机在头里面，深度全是 0.05 左右 | `position` 的 x 减到 0.20，或 z 加到 0.13 |
| 画面左右不对称 | 确认 `position` 的 y 是 0 |

调完重新跑一次 `check_go2_depth.py`，直到 `near fraction < 0.10` 且 PNG 里能看到地面。

> 为什么这么较真：深度图是世界模型的输入。相机位置错了，训练照样能跑、loss 照样下降，
> 但策略学到的是“机器人的后脑勺”，最后就是不走。这是最典型的静默失败。

---

## 第 6 步：正式训练

深度图确认没问题后：

```bash
# 先按轻量配置跑，确认曲线正常
python legged_gym/scripts/train.py \
    --task=go2_amp --headless --sim_device=cuda:0 \
    --num_envs=1024 --max_iterations=200
```

看 200 个 iteration：
- episode reward 在涨；
- `amp_reward` 不是一直 0；
- 没有大量 `nan`。

然后再恢复长时间训练（按你的显存选）：

```bash
python legged_gym/scripts/train.py \
    --task=go2_amp --headless --sim_device=cuda:0 --num_envs=2048
```

可视化：

```bash
python legged_gym/scripts/play.py \
    --task=go2_amp --sim_device=cuda:0 --terrain=climb
```

> `play.py` 默认 `load_run='WMP'`。如果训练时没有改 `run_name`，
> checkpoint 在 `logs/go2_amp_example/WMP/` 下，能直接加载。
> 如果改过 `run_name`，要么改回 `WMP`，要么给 `play.py` 加 `--load_run=你的名字`。

---

## 第 7 步：只改这些，别改别的

Go2 和 A1 都是 12 自由度、4 只脚、8 个 thigh/calf 刚体，所以下面这些**一个都不用改**：

- 观测维度（`prop_dim=33`、`privileged_dim=53`、`height_dim=187`、总 285）；
- 动作维度（12）；
- AMP 判别器输入（30）；
- RSSM 世界模型的所有维度；
- 地形生成、课程、奖励函数；
- SWAP 的对称变换（左右腿交换 + hip 外摆取负），它只依赖“4 条腿”这个结构。

真正改过的只有：

| 项目 | A1 | Go2 | 原因 |
|---|---|---|---|
| URDF | `resources/robots/a1/urdf/a1.urdf` | `resources/robots/go2/urdf/go2.urdf` | 换机器人 |
| 初始高度 | 0.35 | **0.42** | Go2 站立更高，用本地 `go2_config.py` 的值 |
| `base_height` | 0.30 | **0.33** | Go2 站立髋高 |
| 相机 | `[0.27,0,0.03]`，俯仰 ±5° | `[0.28,0,0.11]`，俯仰 -20~-10° | A1 的位置在 Go2 头里 |
| 动作文件 | `mocap_motions/` | `mocap_motions_go2/` | 关节顺序不同 |
| 实验名 | `a1_amp_example` | `go2_amp_example` | 方便区分 |
| 任务名 | `a1_amp` | `go2_amp` | 方便区分 |
| PD 增益 | 40 / 1.0 | **暂用 40 / 1.0** | 见下 |

### PD 增益 40 / 1.0 还是 20 / 0.5？

- WMP 的 A1 AMP 配方用 **40 / 1.0**，Go2 的强化学习社区常用 **20 / 0.5**；
- 配置里**先保留 40 / 1.0**，因为这是 WMP 调过的、和奖励/动作缩放配套的值；
- 如果训练出现“抖、弹、扭矩饱和”，再整体换成 20 / 0.5（改 `control.stiffness/damping`）；
- **无论用哪组，后面 MuJoCo sim2sim 必须用同一组**，否则策略会完全不同。

### Go2 比 A1 重，动作会更“吃力”

- A1 约 12.5 kg，Go2 约 15.0 kg；
- Go2 髋/大腿扭矩 23.7 N·m、小腿 35.55 N·m（A1 是 20 / 30）；
- 单位体重扭矩反而更低，所以论文里 Apollo 那种“3 倍身长跳远”在 Go2 上更难。
- 轻量复现的指标已经按 Go2 尺寸重算（见 `Go2迁移改动清单.md`），不要拿 A1 的米数硬套。

---

## 一页纸检查清单

- [ ] `resources/robots/go2/urdf/go2.urdf` 和 `dae/` 都在
- [ ] `legged_gym/envs/go2/go2_amp_config.py` 在，且 `__init__.py` 注册了 `go2_amp`
- [ ] `datasets/mocap_motions_go2/` 有 4 个文件
- [ ] `check_motion_order.py` 输出 `ALL CHECKS PASSED`
- [ ] `train.py --task=go2_amp --num_envs=16` 能跑 5 个 iteration
- [ ] `check_go2_depth.py` 的 `near fraction < 0.10`，PNG 能看到地面
- [ ] 200 iteration 曲线正常，再开长训练

全打勾之后，Go2 迁移就完成了。后面写 SWAP 的对称代码时，**不用再碰任何机器人相关的配置**。

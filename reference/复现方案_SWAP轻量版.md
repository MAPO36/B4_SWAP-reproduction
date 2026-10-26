# 复现选型与实施方案：SWAP 轻量版（基于 WMP）

> **结论：选 SWAP，做“轻量复现”。**
> 关键原因：SWAP 的前作 **WMP 已经开源了完整底座**（Isaac Gym + 深度相机 + RSSM 世界模型 + 地形课程 + AMP），
> Apache-2.0，410 stars，2026 年 10 月还在更新。
> SWAP 只在这个底座上加“对称等变”，而论文里的 `SWAP (w/o Eq)` 基线**就是 WMP**。
> 复现可以沿着论文的消融逐级做，每一步都有现成参考。
>
> **2026-10 更新**：机器人已由 A1 改为 **Unitree Go2**（导师要求；Go2 数据集更多，实验室也有真机）。
> 两者同为 12 自由度，迁移约 2 天，详见 `Go2迁移改动清单.md`；下文命令和指标已同步为 Go2。
>
> **前提**：服务器显存最好 ≥24 GB（WMP 默认 4096 环境约 23 GB）。
> 如果只有 8~12 GB，可以把环境降到 1024、相机环境降到 256，仍然能跑；
> 如果只有 8 GB 笔记本，看文末的 PointWalker-lite 备选方案。

---

## 一、人话版：两个工作到底在干什么

### SWAP

先让网络在“潜空间”里学一个世界模型（RSSM）：看当前观测，预测下一步会看到什么；
再要求这个潜空间、演员网络、评论家网络都满足左右镜像对称（等变网络）；
还要用 AMP（对抗运动先验）模仿动物步态；最后把世界模型和 PPO 策略一起训练。

**比喻**：闭着眼睛在脑子里放一部“小电影”，预测下一秒的画面；
还要求这部电影左右镜像之后也成立。电影放得越准，走路越稳。

### WMP（SWAP 的底座）

WMP 是 2024 年的论文 *World Model-based Perception for Visual Legged Locomotion*。
它已经实现了：
- 深度相机（Isaac Gym camera）；
- RSSM 世界模型（DreamerV3 风格，潜变量 32×32）；
- 地形课程（wave/slope/stairs/discrete/gap/pit/tilt/crawl）；
- AMP 对抗运动先验（自带 trot/hop 动作数据）；
- PPO + 世界模型特征的训练循环；
- 机器狗的 URDF 和配置（自带 A1；本项目已迁移到 **Go2**，见 `Go2迁移改动清单.md`）。

SWAP 论文里的 `SWAP (w/o Eq)` 基线，本质上就是 WMP。
所以复现 SWAP 不是“从零造世界模型”，而是“**把 WMP 改造成左右对称版**”。


## 二、关键发现：

### 发现 1：WMP 是完整可用的开源底座

- 仓库：https://github.com/bytedance/WMP
- License：Apache-2.0
- 规模：410 stars，36 forks，2026-10-20 还在更新
- 内容：
  - `dreamer/models.py`、`dreamer/networks.py`：RSSM 世界模型（约 1100 行）；
  - `rsl_rl/modules/actor_critic_wmp.py`：世界模型 + 策略的 actor-critic；
  - `rsl_rl/runners/wmp_runner.py`：训练循环；
  - `rsl_rl/algorithms/amp_ppo.py`、`amp_discriminator.py`：AMP + PPO；
  - `legged_gym/envs/base/legged_robot.py`：深度相机、地形、课程；
  - `datasets/mocap_motions/`：trot/hop 参考动作；
  - `resources/robots/a1/`：A1 的 URDF 和 mesh（保留作对照）；
  - `resources/robots/go2/`：本项目新增的 Go2 URDF 和 mesh。
- 训练命令（Go2）：
  ```bash
  python legged_gym/scripts/train.py --task=go2_amp --headless --sim_device=cuda:0
  ```
- 可视化命令（Go2）：
  ```bash
  python legged_gym/scripts/play.py --task=go2_amp --sim_device=cuda:0 --terrain=climb
  ```

### 发现 2：论文的消融就是你的复现路线图

SWAP 论文里的四组配置：

| 论文里的名字 | 实际是什么 | 你的复现阶段 |
|---|---|---|
| `SWAP (w/o Eq)` | 标准 WMP，没有任何对称约束 | **阶段 A：跑通 WMP** |
| `SWAP (w/o Eq-Policy)` | 世界模型等变，actor-critic 普通 | 阶段 B 的进阶 |
| `SymLoss` | 普通网络 + 对称性惩罚项 | 可选的对比基线 |
| `SWAP` | 世界模型 + actor-critic 都等变 | **阶段 C：完整轻量版** |

这意味着你不必一上来就啃最难的“等变 RSSM”。
先跑 WMP，再加“对称 actor/critic”，最后再加“对称世界模型”，每一步都是一个正式消融，
每一步都能写进报告。

### 发现 3：对称策略有现成参考代码

- **SymmLoco**（IROS 2024）：https://github.com/HybridRobotics/SymmLoco
  - Isaac Gym + rsl_rl + MorphoSymm；
  - 提供 `emlp`（等变 MLP）的 actor-critic；
  - 训练命令 `--task=cyber2_push_door_emlp`；
  - 64 stars，2026-10-18 还在更新。
- **morphosymm-rsl-rl**：https://github.com/iit-DLSLab/morphosymm-rsl-rl
  - RSL-RL 的形态对称扩展（IsaacLab 版）。
- **escnn**：https://github.com/QUVA-Lab/escnn
  - 标准的 Z2 等变 CNN 库，用来做深度图编码器/解码器。
- **mdp-homomorphic-networks**：https://github.com/ElisevanderPol/mdp-homomorphic-networks
  - MDP 同态网络参考实现。

你不需要从数学公式开始推导，可以先“抄结构 + 写单元测试”。

### 发现 4：论文缺的超参数，WMP 代码里有

SWAP 论文没给潜变量维度、学习率、batch、序列长度、β、更新间隔 k。
WMP 的 `dreamer/configs.yaml` 给了：
- `dyn_hidden: 512`、`dyn_deter: 512`、`dyn_stoch: 32`、`dyn_discrete: 32`；
- `batch_size: 16`、`batch_length: 64`、`train_ratio: 512`；
- `model_lr: 1e-4`、`kl_free: 1.0`、`dyn_scale: 0.5`、`rep_scale: 0.1`；
- `pretrain: 100`。

这些数值不一定和 SWAP 完全一致，但足够让你跑起来，不用瞎猜。

### 发现 5：评价实验在仿真里就能做

SWAP 的核心实验是“**只在单侧倾斜地形上训练，直接测试镜像地形**”（论文 Fig. 3）。
WMP 的地形生成器已经有 `tilt` 和 `gap`，你只需要在评估时把高度图左右翻转，
就能复现这个实验。不需要真机、不需要 Apollo。

---

## 三、为什么选 SWAP（对着你的两个标准）

### 标准 1：已有框架成熟，少写额外代码

| 模块 | SWAP（基于 WMP） | PointWalker |
|---|---|---|
| 训练算法 | WMP 已实现 PPO + AMP + 世界模型训练循环 | 标准 PPO 要自己接注意力 |
| 世界模型 | WMP 已实现 RSSM（约 1100 行） | 不需要 |
| 深度相机 | WMP 已实现 Isaac Gym camera | 需要自己写点云射线 |
| 地形 | WMP 已实现 9 类地形 + 课程 | 你的仓库只有平面，需移植 |
| 步态先验 | WMP 自带 AMP + trot/hop 数据 | 不需要 |
| 对称网络 | SymmLoco/MorphoSymm 可参考 | 不需要 |
| 复用现有代码 | **WMP 直接就是底座** | 只能复用 PPO 和 MuJoCo |
| 需要从零写的部分 | 等变层 + 集成（约 500~1000 行） | 点云 + 注意力 + PPO 集成（约 1000 行） |
| 有没有“跑通”的参考 | **有：先跑 WMP 基线** | 没有，第一步就是新代码 |

### 标准 2：任务能拆给 agent 执行

SWAP 的复现可以拆成一条很清晰的流水线：

1. 装 WMP，跑 5 个 iteration；
2. 跑完整 WMP 基线（10k iterations）；
3. 降显存（num_envs、camera_num_envs）；
4. 做镜像地形评估，拿到基线成功率；
5. 把 actor/critic 换成 SE-MLP，写等变性单元测试；
6. 训练，和基线对比；
7. 把 encoder/decoder 换成 SE-CNN；
8. 把 GRU 换成 SE-GRU；
9. 完整 SWAP-lite 训练；
10. 评估、画图、写报告。

每一步都有明确的输入输出和验收标准，而且**前一步永远是一个能跑的版本**。
就算第 7~8 步做不出来，你也有第 1~6 步的完整结果（WMP + 对称策略），
这就是论文里的 `SWAP (w/o Eq-Policy)`，是一个合理的轻量复现交付。

PointWalker 没有这种“保底版本”：点云写错、注意力不收敛、PPO 接错，
你手里可能一个能走的策略都没有。

### 标准 3：算力和时间

| | SWAP / WMP | PointWalker |
|---|---|---|
| 并行环境 | 4096（WMP 默认） | 8192（论文） |
| 显存 | 约 23 GB（默认） | 未公开；轻量版 6~10 GB |
| 单次训练 | 4090 约 14 s/iter；10k iter ≈ 39 h | 4×5090 约 50 h（论文） |
| 降配后 | 1024 envs + 256 相机 envs，约 10 h/次 | 512~1024 envs，约 2~6 h/次 |
| 主要耗时 | 深度图渲染 | 点云射线 + 注意力 |

**诚实说**：WMP 默认训练一次要 30~40 小时，比 PointWalker-lite 的单次训练久。
但它的“第一次能跑”来得早：装好就能看 WMP 走路。
PointWalker 要自己写 2~3 周才第一次看到策略。

另外，可以把 `depth.camera_num_envs` 从 4096 降到 256：
WMP 作者在 issue #18 里确认，主要耗时是深度图渲染；
相机环境数降 16 倍，渲染时间也会大幅下降，单次训练可以压到 10 小时左右。

### 标准 4：失败时的退路

SWAP 的退路是**层层降级**：
- 完整 SWAP 做不出来 → 做 `SWAP (w/o Eq-Policy)`（只对称 actor/critic）；
- 对称 actor/critic 也做不出来 → 保留 WMP 基线，加一个“镜像地形泛化”评估；
- WMP 跑不起来 → 再考虑 PointWalker-lite。

PointWalker 的退路是“换输入表示”：
- 注意力不行 → mean pooling；
- 点云不行 → 高度图；
- 但换完以后，离论文的核心越来越远。

---

## 四、轻量化复现指标（不上真机）

### 机器人

直接用 **Unitree Go2** 跑通。WMP 自带的是 A1，但导师要求换 Go2；
两者都是 12 自由度、4 只脚、8 个 thigh/calf 刚体，网络输入输出维度完全一样，
所以迁移只改 URDF 路径、初始高度、站立高度、相机位置、动作顺序、任务名 6 处，
约 2 天完成，详见 `Go2迁移改动清单.md` 和 `go2_migration/README.md`。
A1 的配置和资产保留在仓库里，作为“迁移失败时的对照组”，不要删除。

### 四档指标

| 档位 | 做什么 | 验收指标 | 时间 |
|---|---|---|---|
| **保底** | 跑通 WMP 基线（Go2） | 3 种地形（slope/stairs/gap）能走；成功率 > 50%；有视频 | 1~2 周 |
| **标准** | WMP + 对称 actor/critic | 镜像地形成功率比基线高（例如 +10 个百分点）；复现论文 Fig. 3 的趋势 | 3~4 周 |
| **冲刺** | 完整 SWAP-lite（对称世界模型） | 镜像地形上深度图重构误差不明显上升（论文 Fig. 3c）；gap/box 成功率接近基线 | 5~6.5 周 |
| **游刃有余**（可选） | 爬高 / 跳远**全量冲刺**：Go2 先追归一化 3.0×L / 3.0×H（约 1.16 m / 1.00 m），达不到再降到 2.0× / 1.5× | 至少一个技能达到 ≥2.0×；同难度下 SWAP-lite ≥ WMP 基线；有对比视频 | 6.5~8 周（仅缓冲期） |

### 明确不做的事

- 不做 Apollo 真机；
- 核心阶段不追 2.13 m gap、1.63 m platform；“游刃有余”阶段先追 Go2 归一化全量 3.0×L / 3.0×H（约 1.16 m / 1.00 m），绝对 2.13 / 1.63 需更大机器人（见阶段 9）；
- 不做 6000 环境、4×5090、50 小时；
- 不做室外 2.5 km；
- 不做 50 kg 负载；
- 不做 RK3588 部署。

这些都不影响你复现“世界模型 + 对称等变”的核心链路。
答辩时主动说明“这是轻量复现，砍掉了哪些工程细节”即可。

---

## 五、全部流程（照着做）

> 时间按“课余时间、有 agent 帮忙”估算；全职做可以压缩一半。

### 阶段 0：环境体检（1~2 天）

**做什么**
1. 确认实验室服务器（你之前的日志在一台 `buaa-ubuntu-22` 上）能跑 Isaac Gym；
2. 确认当时训练 Go2 的 conda 环境还能正常用（只作为参考，不要在里面装新东西）；
3. WMP 要求 Python 3.6~3.8、Isaac Gym Preview 3、PyTorch cu117：
   ```bash
   conda create -n wmp python=3.8
   conda activate wmp
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   # 安装 Isaac Gym Preview 3
   cd isaacgym/python && pip install -e .
   cd ../..
   pip install setuptools==59.5.0 ruamel_yaml==0.17.4
   pip install -r requirements.txt
   ```
4. 如果服务器已经能跑你的 Go2 训练，先不要动那个环境；
   WMP 单独建一个环境，避免把现有项目搞坏。
5. 服务器是 4090（你说 32 GB；标准 4090 是 24 GB，先跑 `nvidia-smi` 确认）。
   24 GB 跑 WMP 默认 4096 环境约 23 GB，比较紧；32 GB 会宽裕很多。
   不管哪种，先跑默认配置，爆了再按阶段 2 降。

**验收**
- `python -c "import isaacgym; import torch; print(torch.__version__)"` 不报错；
- `python legged_gym/tests/test_env.py` 能跑；
- 跑 5 个 iteration 不报错。

**卡住怎么办**
- Isaac Gym Preview 3 在 RTX 50 系显卡上可能不兼容（需要 CUDA 11.x 的 PyTorch），
  优先用实验室的 3090/4090 服务器；笔记本 8 GB 显存不够。
- 如果服务器只有 50 系显卡，考虑 Isaac Lab + Isaac Sim，但 WMP 是 Isaac Gym 代码，
  移植成本高，不建议在时间紧的时候换。

---

### 阶段 0.5：把 A1 换成 Go2（约 2 天）

**做什么**

按 `go2_migration/README.md` 执行，只有 4 个闸门：

1. **G0 资产与配置**：把本地 `unitree_rl_gym/resources/robots/go2/` 整个复制到
   `WMP/resources/robots/go2/`；放入 `go2_amp_config.py`；在 `legged_gym/envs/__init__.py`
   注册 `go2_amp`。
2. **G1 动作转换**：运行 `convert_motion_a1_to_go2.py`，再运行 `check_motion_order.py`，
   必须看到 `ALL CHECKS PASSED`。这一步只换 12 个关节的左右前后顺序，不做动作重定向。
3. **G2 冒烟**：`--task=go2_amp --num_envs=16` 跑 5 个 iteration，
   确认 `num_obs=285`、`num_actions=12`。
4. **G3 深度图**：运行 `check_go2_depth.py`，看 `go2_depth.png`，
   确认 `near fraction < 0.10`（相机没被 Go2 的头挡住）。

**验收**
- 四个闸门全过；
- 有一条“Go2 站在地面上”的深度图/截图；
- WMP 的算法代码 `git diff` 为空（只允许改配置、注册表、资产、动作文件）。

**卡住怎么办**
- 动作检查 FAIL：不要改检查脚本让它通过，先打印失败帧的原始 61 个数；
- 深度图全黑/全是机身：只调 `depth.position` 和 `depth.y_angle`，不要动 FOV 和 `far_clip`；
- URDF 加载失败：先确认 `dae/` 目录在，再考虑把 visual 换成 `meshes/*.obj`；
- 任何一步超过 1 天：回到 `Go2迁移改动清单.md` 的风险表逐条排查，不要开始改算法。

---

### 阶段 1：跑通 WMP / Go2 基线（2~4 天，含训练）

**做什么**
1. 先用小环境数跑通：
   ```bash
   python legged_gym/scripts/train.py --task=go2_amp --headless --sim_device=cuda:0 --num_envs=64
   ```
   注意：`--num_envs` 会覆盖配置，但深度相机的 `camera_num_envs` 可能仍按配置来，
   先看显存；如果爆，改 `go2_amp_config.py` 里的 `depth.camera_num_envs`。
2. 确认能跑后，恢复 4096 环境（或 2048），跑完整基线：
   ```bash
   python legged_gym/scripts/train.py --task=go2_amp --headless --sim_device=cuda:0
   ```
3. 训练 10k iterations（约 1~2 天，取决于相机环境数）；
4. 可视化：
   ```bash
   python legged_gym/scripts/play.py --task=go2_amp --sim_device=cuda:0 --terrain=climb
   ```
5. 记录：成功率、训练曲线、显存峰值、单 iteration 时间。

**验收**
- Go2 能在 slope/stairs/gap 上走；
- 有视频；
- 有基线成功率表。

**卡住怎么办**
- 显存爆：`num_envs` 4096→1024，`camera_num_envs` 4096→256，
  `depth.update_interval` 调大（比如 2→4）；
- 训练不收敛：先确认 WMP 原配置能跑，不要一上来改奖励/网络；
- 深度图全黑/全 -inf：检查 camera 的 near/far clip 和 `attach_camera`。

---

### 阶段 2：降显存、加速（1~2 天）

**做什么**
1. 4090 24/32 GB 可以先保留 `num_envs=4096`，只把 `camera_num_envs` 降到 256~512——
   深度图渲染是主要耗时，降相机环境通常比降总环境更划算；
2. 如果显存还是紧，再把 `num_envs` 降到 1024~2048；
3. 确认 `depth.update_interval`（相机更新间隔）和策略频率匹配；
4. 重新跑 500~1000 iterations，确认成功率曲线和 4096 环境时趋势一致。

**验收**
- 显存留出至少 2 GB 余量（12 GB 卡则降到 12 GB 以内）；
- 单 iteration 时间明显下降；
- 策略仍能走。

**卡住怎么办**
- 环境数太少导致课程学不动：保持 2048，优先降相机环境；
- 相机环境太少导致世界模型学不好：先保证 512，再降其他。

---

### 阶段 3：镜像地形评估（2~3 天）

**做什么**
1. 在 `terrain.py` 里找到 `tilt` / `gap` 地形的生成函数；
2. 写一个评估脚本：
   - 加载训练好的 WMP 策略；
   - 生成“左高右低”和“左低右高”两组地形（后者是前者的左右翻转）；
   - 每组跑 100~200 次，固定命令，记录成功率；
3. 记录基线在“训练地形”和“镜像地形”上的成功率差。

**验收**
- 一张对比表：WMP 在训练地形 vs 镜像地形的成功率；
- 镜像地形成功率明显下降（这就是 SWAP 要解决的问题）。

**卡住怎么办**
- 地形翻转不方便：直接在 `height_field_raw` 上做 `np.fliplr`，再生成 mesh；
- 评估脚本不会写：先手动改地形配置，用 `play.py` 跑几次，确认能复现；
- 成功率统计：用 `reset_buf` / `time_out_buf`，参考 `legged_robot.py` 的 episode 逻辑。

---

### 阶段 4：对称 actor/critic（1~2 周，核心）

**做什么**
1. 先读论文 Table I，列出每个观测/动作分量的镜像变换：
   - base 线速度：`(vx, -vy, vz)`；
   - base 角速度：`(-wx, wy, -wz)`；
   - projected gravity：`(gx, -gy, gz)`；
   - 命令：`(vx, -vy, -wz)`；
   - 关节位置/速度/动作：左右腿交换，hip abduction 取负；
     Go2 顺序是 FL,FR,RL,RR，左右交换排列为 `[3,4,5,0,1,2,9,10,11,6,7,8]`，
     hip 下标 `[0,3,6,9]` 取负（详见 `Go2迁移改动清单.md` 4.4 节）；
   - 接触标志：左右脚交换；
   - 随机化参数：左右交换/取负。
2. 写一个 `symmetry.py`，实现：
   - `F_obs(obs)`：观测镜像；
   - `F_act(action)`：动作镜像；
   - `F_latent(z)`：潜变量成对交换；
3. 把 `actor_critic_wmp.py` 里的 actor/critic MLP 换成 SE-MLP：
   - actor 输出满足 `mu(F_obs(o)) = F_act(mu(o))`；
   - critic 输出满足 `V(F_obs(o)) = V(o)`（不变）；
4. 写单元测试：
   ```python
   for _ in range(100):
       o = torch.randn(batch, obs_dim)
       assert torch.allclose(actor(F_obs(o)), F_act(actor(o)), atol=1e-5)
       assert torch.allclose(critic(F_obs(o)), critic(o), atol=1e-5)
   ```
5. 训练 2k~5k iterations，和基线对比。

**验收**
- 单元测试通过；
- 策略能走；
- 镜像地形成功率比基线高（趋势即可，不要求论文的绝对值）。

**卡住怎么办**
- 不会写 SE-MLP：先抄 SymmLoco 的 `emlp` 实现，改成你的观测/动作维度；
- 等变性测试不过：先检查镜像变换的符号（hip abduction、yaw）；
- 训练崩：先把对称约束只加在 actor 上，critic 保持普通；再逐步加。

---

### 阶段 5：对称世界模型（2~4 周，可选/冲刺）

**做什么**
1. **SE-CNN 编码器/解码器**：
   - 用 `escnn` 的 Z2 等变卷积替换 WMP 的 depth encoder/decoder；
   - 第一层把 trivial field 提升到 regular representation（成对镜像通道）；
   - 验证：`decoder(encoder(F_img(x))) = F_img(decoder(encoder(x)))`。
2. **SE-GRU**：
   - 把 RSSM 的 GRU 输入/输出按左右配对排列；
   - 权重约束成块对角/反对角，保证 `f(F_l(h), F_l(z), F_a(a)) = F_l(f(h,z,a))`；
3. **配对潜变量**：
   - 把 `dyn_stoch` / `dyn_deter` 设为偶数；
   - 相邻两维是一对左右特征；
   - prior/posterior 的均值和方差也按对约束。
4. 先只改 encoder/decoder，再改 GRU，最后改 prior/posterior；
5. 每改一块，跑 1k iterations，确认重构误差和策略都没崩。

**验收**
- 深度图重构误差在训练地形上不显著变差；
- 镜像地形上重构误差比 WMP 低（论文 Fig. 3c 的趋势）；
- 策略能走。

**卡住怎么办**
- escnn 和 Isaac Gym 的 PyTorch 版本冲突：先用纯 PyTorch 手写 Z2 等变卷积（只有 2 个群元素，不难）；
- SE-GRU 太难：先不做，保留普通 GRU；报告里说明“只做了 encoder/decoder 等变”；
- 训练不稳定：冻结世界模型，只训策略；或先训世界模型，再训策略。

---

### 阶段 6：评估与消融（3~5 天）

**做什么**
1. 三组对比：
   - **WMP 基线**（无对称）；
   - **+ SE actor/critic**（策略对称）；
   - **+ SE world model**（完整轻量版，如果做了）。
2. 指标：
   - 训练地形成功率；
   - 镜像地形成功率；
   - 镜像地形深度图重构误差；
   - gap/box 难度曲线（可选）。
3. 每组 100~200 次试验，固定命令，确定性动作均值；
4. 画柱状图/折线图。

**验收**
- 一张成功率表；
- 一张镜像地形对比图；
- 至少一个“对称版比基线好”的结论。

**卡住怎么办**
- 时间不够就只做 WMP vs +SE actor/critic；
- 成功率不高也能写：“趋势与论文一致，绝对数值受轻量化影响”。

---

### 阶段 7：整理报告（2~3 天）

**交付物**
- 代码仓库（新分支或新文件夹）；
- README（怎么安装、训练、评估）；
- 训练曲线；
- 成功率表 + 消融图；
- 2~3 段视频/GIF；
- 一页“与论文的差异”说明（砍了什么、为什么砍）。

---

### 阶段 8（可选）：MuJoCo sim2sim

WMP 没有现成的 MuJoCo 部署。如果导师要求 sim2sim：
1. Go2 的 MuJoCo XML 本地已有（`unitree_rl_gym/resources/robots/go2/go2.xml` 和 `scene.xml`），
   可以直接复用，不用重新转 URDF；
2. 复用你现有的 `deploy_mujoco.py` 结构；
3. 观测：WMP 的策略输入包含深度图/RSSM 潜变量，MuJoCo 里需要渲染深度图并跑世界模型；
4. 这一步工作量不小，建议作为“加分项”，不要作为必须交付。

---

### 阶段 9（可选，最后的“游刃有余”）：爬高 / 跳远全量冲刺

**什么时候才能启动**

只有同时满足下面三条才启动，否则直接跳过：

1. 阶段 0~7 的核心交付全部完成（基线、镜像评测、对称策略、报告初稿）；
2. 报告和视频已经能交差，就算这个阶段完全失败也不影响成绩；
3. 服务器还有 ≥5 天连续可用时间。

**目标：先追全量，达不到再降级**

SWAP 论文的“全量”不是随便一个绝对米数，而是按机器人尺寸归一化的：

- 跨沟距离 = **3.0 × L**（L = 前后髋距）；
- 爬台高度 = **3.0 × H**（H = 站立髋高）。

对论文的 Apollo（70×55 cm，72 kg）：3.0×L = **2.13 m**，3.0×H = **1.63 m**。
对本项目的 Go2（URDF 里 L = 0.3868 m，站立 H ≈ 0.33 m）：

- 3.0×L ≈ **1.16 m**；
- 3.0×H ≈ **1.00 m**。

所以“追全量”：

   用 Go2 追“归一化全量” **3.0×L / 3.0×H**（约 1.16 m / 1.00 m）。
   这和论文的 2.13 m / 1.63 m 是同一把尺子，只是机器人小一号。

**止损线**

- 缓冲期内至少做到 **2.0×L / 2.0×H**（Go2 约 0.77 m / 0.66 m）；
- 做不到就按 2.0× → 1.5× → 1.0× 逐级降，最后退到小尺寸定性演示；
- 不要为了 3.0× 无限调参，核心交付永远优先。

**为什么放最后**

这两个任务奖励难调、训练容易崩、单次训练时间长，是“锦上添花”，不是“雪中送炭”。
核心交付没冻结之前，绝对不要碰。

**怎么做（按顺序）**

1. **地形直接用米数，不要只调 difficulty**
   - WMP 已有 `gap_terrain(gap_size)` 和 `climb_terrain(depth)`；
   - 全量目标：`gap_size = 1.16`（3.0×L）、`depth = 1.00`（3.0×H）；
   - 注意：配置注释里第 7 类可能写作 `pit`，但 `make_terrain` 实际调用的是
     `climb_terrain`，别被名字骗了；
   - 先跑 `play.py --terrain=climb` / `--terrain=gap` 看默认难度下的表现。

2. **做一个 `go2_parkour` 配置**
   - 复制 `go2_amp_config.py`；
   - 地形比例：gap 0.4、climb 0.4、stairs up/down 各 0.1；
   - 环境数先 512~1024，相机环境 128~256；
   - 奖励先完全复用 WMP，不要一上来加新项。

3. **只加最小的奖励/终止条件**
   - 跳远：前进速度 + 起跳向上的冲量 + 落地稳定；掉进沟里立即终止；
   - 爬高：base 高度上升 + 前进到平台；从平台掉下去立即终止；
   - 其他奖励（姿态、能耗、AMP）保持 WMP 原样。

4. **用课程 + 微调，不要从零训**
   - 从阶段 1 的 WMP checkpoint 初始化，微调 2000~5000 iterations；
   - 课程从 1.5× 开始，逐步加到 3.0×；
   - SWAP-lite 从阶段 4/5 的 checkpoint 初始化，同样微调；
   - 两组用完全相同的 iteration 数、地形种子、命令采样。

5. **难度阶梯评估**
   - 档位按归一化倍数：1.0× / 1.5× / 2.0× / 2.5× / 3.0×；
   - Go2 对应：沟宽 0.39 / 0.58 / 0.77 / 0.97 / 1.16 m；
     台高 0.33 / 0.50 / 0.66 / 0.83 / 1.00 m；
   - 每档 100 次，固定命令，确定性动作；
   - 同时记录绝对米数和归一化倍数，方便和论文 Table IV 对比；
   - 再做一次左右镜像版本，看对称版是否更稳。

**验收**

- 至少一个技能达到 **≥ 2.0×**（Go2 约 0.77 m 沟 / 0.66 m 台），最好到 3.0×；
- 同一难度下，SWAP-lite 的成功率或极限能力 ≥ WMP 基线；
- 有 2 段侧视对比视频，标注起跳/落地；
- 有一张“归一化难度-成功率”曲线；
- 如果只做到 1.5× 甚至更低，也算完成：交付地形、评估脚本、负结果分析，
  报告里写清楚“卡在哪、为什么”。

**卡住怎么办**

- Go2 力量不够、打滑、翻车：降到 2.0× 或 1.5×，不要中途换机器人；
- 训练不收敛：退回单一技能，固定地形，降低难度；
- 奖励难调：完全用 WMP 原奖励，只加“掉下去终止”；
- 时间不够：只做 3 个难度点 + 定性视频，不做完整曲线；
- 导师坚持绝对 2.13 / 1.63：停下来讨论换机器人，不要伪造数字；
- 任何情况下，不要为了这个阶段改核心代码、不要重跑核心实验。

**预计时间**

- 微调 + 评估：5~7 天；
- 从零训练：2~3 周（不建议）；
- 换更大机器人：2~4 周（超出 8 周范围，需单独立项）。

---

## 六、Agent 任务板（可直接复制）

> 原则：**一次只让 agent 改一个模块**；每个任务都要有“单元测试 + 短训练 + 可视化”三件套验收。
> 不要让 agent 直接跑 10k iteration 的长训练。

| 编号 | 任务 | 依赖 | 验收 |
|---|---|---|---|
| T0 | 装 WMP，跑 5 iter | - | 不报错，显存记录 |
| G0 | 复制 Go2 资产 + 配置 + 注册 `go2_amp` | T0 | `--task=go2_amp` 能建环境 |
| G1 | 转动作 + `check_motion_order.py` | G0 | `ALL CHECKS PASSED` |
| G2 | 16 环境冒烟 5 iter | G1 | 不报错，`num_obs=285`、`num_actions=12` |
| G3 | 深度图检查 + 调相机 | G2 | `near fraction < 0.10`，PNG 有地面 |
| T1 | 跑 Go2 基线 10k iter | G3 | slope/stairs/gap 能走，视频 |
| T2 | 降显存（num_envs/camera_num_envs） | T1 | 12 GB 内能跑，趋势一致 |
| T3 | 镜像地形评估脚本 | T1 | 基线成功率表 |
| T4 | 写 `symmetry.py`（F_obs/F_act/F_latent） | - | 单元测试通过 |
| T5 | 把 actor/critic 换成 SE-MLP | T3,T4 | 等变性测试 + 短训练能走 |
| T6 | 训练对称策略，和基线对比 | T5 | 镜像成功率提升 |
| T7 | SE-CNN encoder/decoder（escnn 或手写） | T1 | 重构误差不崩 |
| T8 | SE-GRU / 配对潜变量 | T7 | 等变性测试 + 短训练 |
| T9 | 完整 SWAP-lite 训练 | T6,T8 | 成功率 + 重构误差 |
| T10 | 评估/画图/消融 | T9 | 表 + 图 |
| T11 | README + 报告 + 视频 | 全部 | 可交付 |
| T12 | 爬高/跳远地形 + `go2_parkour` 配置 | T1 | 地形可视化 + 短训练不崩 |
| T13 | 两技能对比训练与评估 | T9,T12 | 成功率/极限能力表 + 对比视频 |

### 几条通用提示词模板

**T0 环境**
```
在服务器上按 WMP 的 README 安装 Python 3.8 + Isaac Gym Preview 3 + torch cu117。
不要动现有的 Go2 环境，新建 conda env。
验收：import isaacgym/torch 成功；跑 5 个 iteration 不报错；记录显存峰值。
```

**T4 对称变换**
```
新建 symmetry.py，实现观测/动作/潜变量的左右镜像变换。
参考 SWAP 论文 Table I：base 线速度 (vx,-vy,vz)，角速度 (-wx,wy,-wz)，
gravity (gx,-gy,gz)，命令 (vx,-vy,-wz)，关节按左右腿交换、hip abduction 取负。
Go2 关节顺序是 FL,FR,RL,RR，左右交换排列为 [3,4,5,0,1,2,9,10,11,6,7,8]，
hip 下标是 [0,3,6,9]。
要求：写单元测试，验证 F(F(x))=x；把每个分量的变换打印出来给我核对。
```

**T5 SE-MLP**
```
把 actor_critic_wmp.py 里的 actor/critic MLP 换成 SE-MLP。
要求：
- actor 满足 mu(F_obs(o)) = F_act(mu(o))；
- critic 满足 V(F_obs(o)) = V(o)；
- 写 100 次随机输入的等变性测试；
- 先只改 actor，critic 保持普通，跑 100 iter 确认能走。
验收：测试通过 + 100 iter 日志 + 视频。
```

**T7 SE-CNN**
```
用 escnn（或手写 Z2 等变卷积）替换 WMP 的 depth encoder/decoder。
要求：
- 第一层把 trivial field 提升到 regular representation；
- 保持输入/输出通道数和 WMP 一致；
- 写重构等变性测试；
- 跑 100 iter，确认重构误差不崩。
验收：测试 + loss 曲线。
```

---

## 七、最容易翻车的 8 个地方

1. **显存爆**
   WMP 默认 4096 环境 + 深度相机约 23 GB。先把 `camera_num_envs` 降到 256，
   再把 `num_envs` 降到 1024~2048，不要一上来改网络。
2. **Isaac Gym 版本**
   WMP 要 Preview 3，你的仓库可能用 Preview 4。两个环境分开建，别混。
3. **Go2 关节顺序转错（静默失败）**
   A1 是 FR/FL/RR/RL，Go2 是 FL/FR/RL/RR。转错程序不报错，
   但 AMP 会学出“顺拐”。必须跑 `check_motion_order.py`，看到 `ALL CHECKS PASSED` 才继续。
4. **Go2 深度相机被头挡住（静默失败）**
   A1 的相机位置 `[0.27,0,0.03]` 在 Go2 头壳里面。必须跑 `check_go2_depth.py`
   看 `go2_depth.png`，`near fraction < 0.10` 才继续。否则世界模型学的是后脑勺。
5. **对称性写错但程序不报错**
   等变性错误是“静默失败”：训练照样跑，效果就是不好。
   一定要写 `f(F(x)) == F(f(x))` 的单元测试，每个模块都测。
6. **一上来就改世界模型**
   先跑 WMP 基线，再加 actor/critic 对称，最后再改 RSSM。
   顺序反了，你很难判断问题出在哪。
7. **agent 说“完成了”但没验证**
   永远要求它给：单元测试、loss 曲线、短视频。没有这三样，不算完成。
8. **为了“游刃有余”牺牲核心交付**
   阶段 9 只能在核心报告冻结后启动。爬高/跳远失败不丢人，核心实验没做完才丢人。

---



## 九、参考链接

- SWAP 论文：https://arxiv.org/abs/2606.19928
- SWAP 项目页：https://swap-parkour.github.io （Code: coming soon）
- WMP 论文：https://arxiv.org/abs/2409.16784
- WMP 代码：https://github.com/bytedance/WMP （Apache-2.0）
- WMP 项目页：https://wmp-loco.github.io/
- WMP issue #18（训练时间/显存）：https://github.com/bytedance/WMP/issues/18
- SymmLoco（等变 PPO）：https://github.com/HybridRobotics/SymmLoco
- morphosymm-rsl-rl：https://github.com/iit-DLSLab/morphosymm-rsl-rl
- escnn：https://github.com/QUVA-Lab/escnn
- mdp-homomorphic-networks：https://github.com/ElisevanderPol/mdp-homomorphic-networks
- PointWalker 论文：https://arxiv.org/abs/2610.15548
- PointWalker 视频：https://youtu.be/QnCFGh7QXIw
- PointWalker B 站讲解：https://www.bilibili.com/video/BV1GQJo6YEW7/
- legged_gym：https://github.com/leggedrobotics/legged_gym
- rsl_rl：https://github.com/leggedrobotics/rsl_rl

---

## 十、一句话总结

**SWAP 有现成的 WMP 底座，PointWalker 没有代码。**
你现在的目标不是“从零证明自己会写世界模型”，
而是“站在 WMP 上，把对称等变这一块补出来，做出论文里的对比曲线”。
先跑 WMP，再加对称策略，最后碰世界模型——这就是最稳的轻量复现路线。

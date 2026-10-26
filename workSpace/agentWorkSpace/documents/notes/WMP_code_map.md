# WMP 代码地图（T0）

- 上游：`https://github.com/bytedance/WMP`
- 本机 clone：`/home/l/workSpace/SWAP/WMP`（gitignored）
- commit：`c232c115ada4517453ebded5019078ba055456de`（2024-12-18，仓库仅 1 个 commit）
- 规模：68 个文件；核心 Python 约 4.6k 行（不含 assets/datasets）
- 论文：World Model-based Perception for Visual Legged Locomotion（arXiv 2409.16784）
- WMP 本身**不含对称/等变机制**（全仓库无 symmetry/mirror/equivariance 实现），SWAP 需另加。

## 1. 目录与职责

| 路径 | 职责 |
|---|---|
| `legged_gym/` | Isaac Gym 环境；A1 配置、地形、相机、AMP 观测 |
| `rsl_rl/` | PPO/AMP runner、actor-critic、motion 数据集、存储 |
| `dreamer/` | RSSM 世界模型（从 dreamerv3-torch 改） |
| `datasets/mocap_motions/` | AMP 专家动作：hop1/hop2/trot1/trot2（JSON） |
| `resources/robots/a1/` | A1 URDF + 5 个 .dae mesh |
| `licenses/` | 第三方 license |

## 2. 关键文件

### 环境层

| 文件 | 作用 / 关键符号 |
|---|---|
| `legged_gym/envs/base/legged_robot.py:67` | `LeggedRobot`：仿真、地形、相机、观测、奖励、AMP 观测 |
| `legged_robot.py:428-479` | `compute_observations()`：拼 285 维特权 obs；`obs_buf` 直接克隆 privileged |
| `legged_robot.py:481-491` | `get_amp_observations()` = joint_pos12+lin_vel3+ang_vel3+joint_vel12 = 30 |
| `legged_robot.py:493-504` | `get_full_amp_observations()`（训练未调用） |
| `legged_robot.py:59-64` | A1 的 `COM_OFFSET/HIP_OFFSETS`（仅 full AMP obs 用，训练未调用） |
| `legged_robot.py:900-924` | `_get_noise_scale_vec()`：12+12+12 的 dof/action 噪声布局 |
| `legged_robot.py:1029-1053` | A1 腿长/腿序正运动学（训练未调用） |
| `legged_robot.py:1127-1155` | 相机挂载 `create_camera_sensor` |
| `legged_robot.py:1166-1185` | 从 `cfg.asset.file` 解析 URDF 并 `load_asset` |
| `legged_robot.py:1236` | `camera_num_envs = min(camera_num_envs, num_envs)` |
| `legged_robot.py:1339/1355/1371/1409/1447` | 高度图/前向高度图采样 |
| `legged_gym/envs/a1/a1_amp_config.py:41` | `A1AMPCfg`：obs/reward/terrain/depth/PPO 全部配置 |
| `legged_gym/envs/a1/a1_amp_config.py:292` | `A1AMPCfgPPO` |
| `legged_gym/envs/__init__.py:42-43` | 注册 `a1`、`a1_amp` |
| `legged_gym/utils/task_registry.py:162` | `make_wmp_runner()`；log 路径 `<WMP>/legged_gym/logs/<experiment>/<run>` |
| `legged_gym/scripts/train.py:49-58` | 硬编码 `run_name='WMP'`、`max_iterations=100000`（**忽略 `--max_iterations`**） |

### RL / WMP 层

| 文件 | 作用 / 关键符号 |
|---|---|
| `rsl_rl/runners/wmp_runner.py:61` | `WMPRunner`：WMP 训练主循环 |
| `wmp_runner.py:63-140` | 构造：world model → depth predictor → `ActorCriticWMP` → AMPLoader/discriminator → `AMPPPO` |
| `wmp_runner.py:142-181` | `_build_world_model()`；**用 `sys.argv[0]` 推 `dreamer/configs.yaml`（:145-147）**，并二次 `parse_args()`（:168） |
| `wmp_runner.py:183-382` | `learn()`：rollout → PPO → 周期性 WM/depth predictor 训练 → 保存 |
| `wmp_runner.py:240-253` | 每 `depth.update_interval=5` 步做 RSSM `obs_step`，取 `deter(512)` 作 `wm_feature` |
| `wmp_runner.py:364-368` | `sum_wm_dataset_size > train_start_steps(10000)` 才开始训 WM/深度预测器 |
| `wmp_runner.py:382` | `learn()` 结束**总会**存一次 ckpt（与 `save_interval` 无关） |
| `wmp_runner.py:384-418` | `init_wm_dataset()`：每 env 的 prop/action/reward/(image/forward_height_map) 序列 |
| `wmp_runner.py:419-438` | `train_depth_predictor()`：525 维前向高度图 + 33 维 prop → 64×64 深度图 |
| `wmp_runner.py:440-477` | `train_world_model()`：无相机 env 用深度预测器补图 |
| `wmp_runner.py:558` | `save()`：actor_critic / alg / world_model / depth_predictor |
| `rsl_rl/modules/actor_critic_wmp.py:42` | `ActorCriticWMP` |
| `actor_critic_wmp.py:45-153` | history encoder（210→256→128→latent 35）、wm encoder（512→64→64→32）、actor/critic |
| `actor_critic_wmp.py:184-191` | `act()`：`[history_latent(35), command(obs[59:62]), wm_latent(32)]` → actor |
| `actor_critic_wmp.py:197-200` | `get_linear_vel()`：history latent 末 3 维（被监督预测 base lin vel） |
| `actor_critic_wmp.py:214-221` | `evaluate()`：`[critic_obs(285), wm_latent(32)]` |
| `rsl_rl/algorithms/amp_ppo.py:43` | `AMPPPO`：PPO + AMP 判别器 + 线速度预测损失 |
| `amp_ppo.py:124-142` | `act()`：记录 history/wm_feature/amp_obs |
| `amp_ppo.py:169-304` | `update()`：surrogate/value/vel/AMP/grad-penalty 合计一次反传 |
| `rsl_rl/algorithms/amp_discriminator.py:40` | `AMPDiscriminator`；reward 映射 `:94-99`；**`_lerp_reward` 实际忽略 task lerp，直接相加** `:101-104` |
| `rsl_rl/modules/depth_predictor.py:24` | `DepthPredictor`：558→[256,128]→转置卷积→64×64×1（forward `:123`） |
| `rsl_rl/datasets/motion_loader.py:35` | `AMPLoader`：专家数据加载、插值采样、滑窗 |
| `motion_loader.py:28` | `from pybullet_utils import transformations`（quaternion 工具，本机缺包） |
| `motion_loader.py:37-68` | 单帧 61 维：pos3 + quat4 + joint_pos12 + toe_pos12 + linvel3 + angvel3 + jointvel12 + toe_vel12 |
| `motion_loader.py:154-184` | `reorder_from_pybullet_to_isaac()`（FR/FL/RR/RL → FL/FR/RL/RR，**当前调用被注释** `:101-102`） |
| `motion_loader.py:364-367` | `observation_dim = trajectory_dim - 12` → A1 为 30 |
| `rsl_rl/datasets/motion_util.py:96` | `quaternion_about_axis`；四元数约定 **[x,y,z,w]**（`pose3d.py:96,219`） |

### 世界模型层（dreamer，来自 dreamerv3-torch）

| 文件 | 作用 / 关键符号 |
|---|---|
| `dreamer/models.py:34` | `WorldModel`：encoder/decoder/dynamics + reward/cont heads |
| `dreamer/models.py:117-180` | `_train()`：RSSM observe → KL + 重建 + reward/cont loss |
| `dreamer/networks.py:36` | `RSSM`：`observe/obs_step/img_step/kl_loss` |
| `dreamer/networks.py:178` | `get_feat()` = stoch32×32 + deter512 = 1536 |
| `dreamer/networks.py:185` | `get_deter_feat()` = 512（策略侧 `wm_feature` 用这个） |
| `dreamer/networks.py:320/396` | `MultiEncoder/MultiDecoder`（prop MLP + image CNN） |
| `dreamer/networks.py:486/541` | `ConvEncoder/ConvDecoder`（64×64×1 深度图） |
| `dreamer/networks.py:815` | `GRUCell`（deter 状态） |
| `dreamer/configs.yaml` | 世界模型超参（见下） |

## 3. 训练数据流（一次 iteration）

1. `WMPRunner.learn()` 初始化 5 帧 history 与 `wm_dataset`（`wmp_runner.py:183-235`）。
2. 每 `num_steps_per_env=24` 步：
   - 每 5 步：深度图/prop → `WorldModel.encoder` → RSSM `obs_step` → `wm_feature = deter(512)`（:240-250）。
   - `history(210)` + `obs(285)` + `amp_obs(30)` → `AMPPPO.act()` → `ActorCriticWMP.act/evaluate` → 环境 step。
   - 存 PPO 转移与 AMP 转移；env reset 时把该段序列归档进 `wm_dataset`。
3. PPO/AMP 更新（`amp_ppo.update`），一次反传含 surrogate/value/vel/AMP/grad-pen。
4. 当累计帧数 > 10000：每 10 iter 训深度预测器（1000 inner steps），每 iter 训世界模型（10 inner steps，batch 16×64）。
5. 每 `save_interval` 保存；`learn()` 结束再保存一次。

## 4. 观测/动作布局（A1 / a1_amp，285 维）

由 `legged_robot.py:431-466` 拼接顺序推出（全部随机化项开启时）：

| 索引 | 内容 | 维度 |
|---|---|---|
| `[0:8]` | 被惩罚刚体接触标志（thigh+calf，8 个） | 8 |
| `[8:20]` | 四足 force sensor 三轴力 | 12 |
| `[20:32]` | d_gain 归一化 | 12 |
| `[32:44]` | p_gain 归一化 | 12 |
| `[44:47]` | com | 3 |
| `[47]` / `[48]` / `[49]` | added mass / restitution / friction | 1+1+1 |
| `[50:53]` | **base_lin_vel（特权，最后 3 维）** | 3 |
| `[53:56]` | base_ang_vel | 3 |
| `[56:59]` | projected gravity | 3 |
| `[59:62]` | commands | 3 |
| `[62:74]` | dof_pos | 12 |
| `[74:86]` | dof_vel | 12 |
| `[86:98]` | actions | 12 |
| `[98:285]` | heightmap | 187 |

- `privileged_dim=53` = `[0:53]`；`prop_dim=33` = `obs[53:86]`（ang_vel+gravity+cmd+dof_pos+dof_vel）。
- history 每帧 42 维 = `[53:59]` + `[62:98]`；5 帧 = 210（`wmp_runner.py:97,207-213`）。

### 其他关键维度

| 项 | 值 | 出处 |
|---|---|---|
| 关节数 / action | 12 | `a1_amp_config.py:46` |
| height_dim | 187（17×11） | `a1_amp_config.py:48,65-67` |
| forward_height_dim | 525（21×25） | `a1_amp_config.py:49,72-75` |
| num_observations | 285 | `a1_amp_config.py:50` |
| actor 输入 | latent35 + cmd3 + wm_latent32 = 70 | `actor_critic_wmp.py:73,184-191` |
| critic 输入 | 285 + 32 = 317 | `actor_critic_wmp.py:74,214-221` |
| wm_feature | `dyn_deter` = 512 | `wmp_runner.py:180` |
| RSSM feat | stoch32×32 + deter512 = 1536 | `networks.py:178` |
| depth image | 64×64×1，FOV 58°，near 0 / far 2 | `a1_amp_config.py:153-159` |
| 相机位置 | `[0.27, 0, 0.03]`（Go2 会撞头） | `a1_amp_config.py:146` |
| camera_num_envs | 1024（env 内再 cap 到 num_envs） | `a1_amp_config.py:142`、`legged_robot.py:1236` |
| update_interval | 5 | `a1_amp_config.py:151` |
| episode | 20 s / dt 0.005 / decimation 4 → 1000 控制步 | `legged_robot_config.py:47` |
| PPO | 24 steps/env，5 epochs，4 mini-batches | `legged_robot_config.py:281,292` |
| AMP | 判别器输入 60，hidden [1024,512]，reward coef 0.01，预载 2M | `a1_amp_config.py:324-328` |
| 世界模型 | hidden/deter 512，stoch 32，discrete 32；batch 16×64；lr 1e-4；train_start 10000 | `dreamer/configs.yaml` |
| 深度预测器 | lr 3e-4，interval 10，iters 1000，batch 1024，loss_scale 100 | `a1_amp_config.py:332-339` |

## 5. 换 Go2 时优先改的点（T0 预览，G0 再细化）

### 机器人配置
- `a1_amp_config.py:166`：`asset.file` → Go2 URDF；新增 `resources/robots/go2/`。
- `a1_amp_config.py:92`：初始高度 `[0,0,0.35]`（Go2 通常 0.42）。
- `a1_amp_config.py:94-107`：`default_joint_angles` 的 A1 关节名（Go2 名称相同，URDF 顺序不同）。
- `a1_amp_config.py:132-137`：PD `40/1.0`、`action_scale=0.25`、`decimation=4`（Go2 常用 `20/0.5`，需决策）。
- `a1_amp_config.py:146-154`：相机 `position`、pitch、`update_interval`、64×64（Go2 头壳遮挡）。
- `a1_amp_config.py:167-174`：`foot_name`、`penalize_contacts_on`、`terminate_after_contacts_on`、`self_collisions`。
- `a1_amp_config.py:182-202`：A1 质量/质心/推力随机化范围。
- `a1_amp_config.py:220,241-242`：`base_height=0.3`、`base_height_target=0.25`、`foot_height_target=0.15`。
- `a1_amp_config.py:38`：`MOTION_FILES = glob('datasets/mocap_motions/*')`。

### 资产与关节顺序
- 只有 `resources/robots/a1/`，需新增 Go2 URDF/mesh。
- A1 URDF 关节顺序 FR→FL→RR→RL（`resources/robots/a1/urdf/a1.urdf:139,261,383,505`）。
- Go2 URDF 顺序 FL→FR→RL→RR（`unitree_rl_gym/resources/robots/go2/urdf/go2.urdf:159,418,677,936`）。
- `motion_loader.py:101-102` 的 PyBullet→Isaac 重排被注释；`datasets/mocap_motions/*` 关节块必须按 Go2 顺序重排（G1 任务）。
- 本仓库已有参考（只读）：`reference/go2_migration/go2_amp_config.py`、`reference/go2_migration/convert_motion_a1_to_go2.py`、`reference/Go2迁移改动清单.md`、`unitree_rl_gym/resources/robots/go2/`。注意 WMP 没有 `asset.name` 字段，参考里的该字段对 WMP 无效。

### 基类/env 隐含假设
- `legged_robot.py:59-64`：A1 的 COM/HIP offsets（仅 full AMP obs 用，训练未调用）。
- `legged_robot.py:1029-1053`：A1 腿长/腿序（训练未调用）。
- `legged_robot.py:900-924`：12+12+12 噪声布局；改关节数会错位。
- `legged_robot.py:953`：`view(num_envs, 4, 6)` 硬编码 4 足（Go2 仍 4）。
- `legged_robot.py:1234-1243`：camera env 索引与 tilt/crawl terrain 区间绑定。

### runner / dreamer / 注册
- `wmp_runner.py:379`：保存日志时硬编码复制 `a1_amp_config.py`。
- `legged_gym/envs/__init__.py:32-35`：新增 Go2 config import / `task_registry.register("go2_amp", ...)`。
- `dreamer/configs.yaml:10-11`：`task: 'a1'`（标签）、`num_actions: 12`（Go2 仍 12）。
- `ActorCriticWMP` 的 `latent_dim/wm_latent_dim/height_dim/privileged_dim` 由 runner 注入，换 obs 维度会自动跟随；但 `policy` 配置 `latent_dim=32+3` 要核对。

## 6. 本机 T0 冒烟结论

- 本机（RTX 5060 8G / Preview 4 / torch 2.4.1+cu121）**不能跑 WMP 训练**：torch CUDA 报 `no kernel image`（sm_120 不在 cu121 wheel）。
- Isaac Gym Preview 4 的基础 PhysX 场景（16 env）可跑；问题在 torch 与依赖，不是 PhysX。
- WMP import 还缺 `pybullet_utils/ruamel_yaml/sklearn/cv2/trimesh/matplotlib`，且 numpy 1.20 与 Pillow 不兼容。
- 上游 `train.py` 硬编码 `max_iterations=100000`，忽略 `--max_iterations`；冒烟用 `artifacts/t0_smoke.py`（`T0_MAX_ITERS` 控制，env 数 cap 16）。
- 实验室安装/验收见 `testBrochure.md` 的 T0 交接单。

## 7. 死代码 / 不确定点

1. `reference_state_initialization=False`（`a1_amp_config.py:52`），env 内 `amp_loader`、`_reset_dofs_amp`、`_reset_root_states_amp` 训练时不执行；启用时需处理 Go2 关节顺序/四元数。
2. `configs.yaml` 的 `pretrain:100`、`train_ratio:512`、`compile`、`video_pred_log`、`cont_head`、`discount` 在训练主路径未检索到使用点（疑似遗留）。
3. `policy_class_name='ActorCritic'`（`a1_amp_config.py:320`）被 `WMPRunner` 忽略，runner 直接构造 `ActorCriticWMP`。
4. `AMPDiscriminator._lerp_reward` 忽略 `amp_task_reward_lerp=0.3`，直接 `disc_r + task_r`（`amp_discriminator.py:101-104`）；是否有意不确定。
5. `extras["depth"] = depth_buffer[:, -2]`（`legged_robot.py:195`）取上一帧深度，时序意图不确定。
6. `get_full_amp_observations()` / `HIP_OFFSETS` / 腿部 FK 训练未调用，A1 腿长当前不影响训练；启用时需按 Go2 几何改。
7. `use_camera=False` 时 CNN 输入零填充、decoder 不输出 image（`networks.py:384-388,461`）；SWAP 若想去相机需确认 loss 分支语义。
8. Go2 相机安装参数（参考配置 `[0.28,0,0.11]`、pitch `[-20,-10]`）未在 WMP 内验证，待 G0/实验室实测。
9. `WorldModel.preprocess` 用旧式 `torch.Tensor(v)`（`models.py:197`），`ConvEncoder.forward` 原地改输入（`networks.py:525`）；共享张量副作用未验证。

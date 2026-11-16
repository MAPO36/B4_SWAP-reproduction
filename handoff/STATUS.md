# 交接状态（唯一状态源）

更新时间：2026-11-16 11:15
当前工作人员：**空**（王 2026-11-16 11:15 会话暂停；训练由后台 watcher 自动等待启动）
当前阶段：**3**（T1b：R0-LHR 重训；T3：LHR/RHL 评估——训练**待 GPU 空闲后由 watcher 自动启动**，评估待训练完成）

---

## 一、方案版本（重要）

**以 `reference/复现方案_SWAP轻量版(11月16日优化版)(以此版爲準）.md` 为准（11-16 优化版）**。
11-07 版已改名为 `（旧版-非必要勿读）`，不再作准。

11-16 版相对 11-07 版的关键变化：

1. 改为**论文同款四组消融**：R0 基线、R1 SymLoss 软约束、R2 等变世界模型 + 普通 AC、R3 等变世界模型 + 等变 AC。
2. 四组**统一在固定的“左高右低（LHR）”单侧倾斜地形上训练**（倾斜箱/台阶/沟壑）。
3. 评估分两套：**同向 LHR（训练分布内）** 与 **镜像 RHL（右高左低，分布外）**，必须分列；镜像 gap = LHR − RHL。
4. **旧 10k 基线（T1）不是固定 LHR，降级为 R0-pilot**，不进入四组任何表/图/视频。
5. 正式 **R0 = R0-LHR**：阶段 3 用固定 LHR 从零重训 512 env × 10k ≈ 24 h。
6. 评估强制 6 项标注：条件 / 训练地形 / 模型 / 评估地形 / 地形与难度 / 试验数（200 次/格）。
7. 任务板：T1（R0-pilot，已完成）→ **T1b（R0-LHR 重训）→ T3（LHR/RHL 评估脚本）** → T4/T5（symmetry.py + SymLoss）→ …

## 二、历史摘要（已完成）

| 阶段 | 结果 | 证据 |
|---|---|---|
| T0 环境 | ✅ 通过（Isaac Gym Preview 4 路线） | `workSpace/runs/stage-0/manifest.json` |
| G0–G3 Go2 迁移 | ✅ 通过 | `go2_amp` 注册；`check_motion_order.py` ALL CHECKS PASSED；`go2_depth.png` |
| T1 基线 10k（**现记 R0-pilot**） | ✅ 完成（陈旺 2026-11-08）：512 env × 10k，22h18m，reward 15.93，0 nan；slope 93.8% / stair 98.4% / gap 18.2% | `workSpace/runs/stage-1/`、`Materials/SE/`、ckpt `WMP/logs/go2_amp_example/go2_base512/model_10000.pt`（sha256 `46cbef12…`） |
| T3 镜像评估（旧口径 A+B） | ✅ 完成（王 2026-11-12）：A 组 WMP 原地形镜像掉点≈0（地形对称）；B 组单侧横坡 OOD 无方向性掉点；修复 trimesh 绕序 bug | `workSpace/runs/s3-20261112-1620-mirror-eval/`、`Materials/SE/T3/`、commit `c833743` |
| 备份 | ✅ U 盘 `SWAP-backup-20261112-1809`（582 文件，SHA 全过） | U 盘 `/media/ubuntu/__/SWAP-backup-20261112-1809/` |

> 注意：T1 的旧 10k 与旧 T3 评估都**不是固定 LHR 口径**，从本轮起只作 R0-pilot/方法学参考，**不得进入四组消融表**。

## 三、本轮：阶段 3（王，2026-11-16，进行中）

- 方案口径（用户 2026-11-16 确认）：
  - LHR = **固定 10° 左高右低横向坡度**叠加三类 WMP heightfield 地形：**平台箱（climb 升高平台）/ 上行楼梯（stairs up）/ 沟壑（gap）**；训练三类各 1/3、课程 10 档；方向固定、不随机左右翻转。
  - 评估：3 地形 × 3 难度（level 1/2/3 = 特征难度 0.2/0.5/0.8）× 2 方向（LHR / RHL=np.fliplr）× **200 次/格 = 18 格**。
  - 训练：512 env × 10k 从零（**不加载 R0-pilot checkpoint**），约 22~24 h；另做 1k 冒烟。
- 本轮提交：`8cab403`（代码 + 11-16 方案重命名 + STATUS/logs）；工作区干净，未 push。
- 本轮实现（代码，overlay 为准）：
  - `legged_gym/utils/lhr_terrain.py`（LHRTerrain：训练/评估两用；固定 LHR 横坡；RHL 镜像；方向与指纹校验）
  - `legged_gym/envs/go2/go2_amp_lhr_config.py` + `legged_gym/envs/__init__.py` 注册 `go2_amp_lhr`
  - `train_lhr.py`（训练入口；训练前打印/保存高度场可视化，确认 LHR 方向固定）
  - 脚本归档：`workSpace/tools/stage3/`（训练链、显存守卫 watcher、CPU 不变量测试、上传检查；运行中副本在 `/tmp/`）
- **T3 评估工具链（已就绪，CPU 合成数据自测 PASS）**：`eval_lhr.py`（3 地形 × 3 难度 × LHR/RHL × 200 次，输出带 6 项标注）+ `aggregate_lhr.py`（18 格完整性 / 镜像对 / 训练曲线 / manifest）+ `make_lhr_figures.py`（5 张图）；工具归档在 `workSpace/tools/stage3/`；训练完成后运行 `eval_lhr.py` 写 `workSpace/runs/R0_LHR/eval/`，再 `aggregate_lhr.py --run-dir workSpace/runs/R0_LHR`。
- 训练产物：`workSpace/runs/R0_LHR/`（含 smoke + 正式 10k；checkpoint 依方案命名为 `R0_LHR/seed0/model_10000.pt`）
- 运行中训练：**训练已武装、尚未启动**——共享 4090 当前被**另一位同学的项目**占用（`/home/ubuntu/桌面/WMP-master` 的 `go2_amp --run_name=go2_stair`，约 19.9 GB / 24 GB）。已部署显存守卫 watcher，GPU 连续 3 次空闲 ≥18 GB 时**自动启动**训练链（见 §四）。
- 本轮剩余（训练完成后由下一位/下一会话执行）：LHR/RHL 18 格评估（`eval_lhr.py`）、图表/视频/manifest、T3 报告与 Materials、四组表基线列。

## 四、运行中训练登记（夜间/长训练）

- run_id：`s4-20261116-r0lhr`（目录 `workSpace/runs/R0_LHR/`）
- 状态：**已武装待启动（armed）**。共享 4090 被另一位同学的项目占用（`/home/ubuntu/桌面/WMP-master`，
  其 `train.py --task go2_amp --run_name=go2_stair` 进程，GPU 约 19.9 GB），因此训练不能立即开始。
  **显存守卫 watcher 已在后台运行**：每 60 s 检查 1 次，GPU **连续 3 次**空闲 ≥ 18000 MiB 时自动启动训练链；
  它不会杀/不碰别人的进程，最多等 ~72 h。
- watcher：`/tmp/wait_gpu_then_train.sh`，**PID 1018480**（2026-11-16 11:06 启动，见 `logs/watcher.pid`）；日志 `workSpace/runs/R0_LHR/logs/watcher.log`
- 链式命令：`/tmp/run_r0_lhr_chain.sh`（自动执行以下 3 步，任一步失败即中止并写日志）
  1. 集成冒烟：32 env × 20 iter（验证 `go2_amp_lhr` + LHR 地形 + `train_lhr.py` 全链路）
  2. 1k 冒烟：512 env × 1000 iter（稳定性门槛：退出码 0、无 NaN、ckpt 存在）
  3. **R0-LHR 正式 10k**：512 env × 10000 iter，约 22~24 h，从零训练（不加载 R0-pilot）
- 预计完成：**GPU 空闲后** 约 5 min + 2.2 h + 22~24 h（若对方长跑，则整体顺延）
- 监控命令：
  - `tail -f /home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR/logs/watcher.log`
  - `tail -f /home/ubuntu/B4_2025/SWAP/workSpace/runs/R0_LHR/logs/chain.log`
  - `nvidia-smi`；显存采样 `workSpace/runs/R0_LHR/logs/vram.csv`
- 日志/ckpt 路径：`R0_LHR/logs/{integration,smoke,full}.log`；
  ckpt 在 `WMP/logs/go2_lhr_example/{R0_LHR_integration,R0_LHR_smoke,R0_LHR}/`，
  训练完成后自动复制最终权重到 `workSpace/runs/R0_LHR/seed0/model_10000.pt`（附 sha256）
- 训练地形证据（训练启动时自动生成）：`workSpace/runs/R0_LHR/terrain_check/`（高度场 PNG/npz + LHR 方向 JSON）
- 风险：与另一位同学的 GPU 任务并发导致 OOM（训练内已设显存守卫：每阶段开跑前要求空闲 ≥17 GB，
  最多等 30 min，不够则中止）；若中止，见 `logs/*.log` 与最近 ckpt 再决定续训/重跑。

## 四之二、会话暂停记录（2026-11-16 11:15）

- 用户要求**暂停自动续跑**（本会话结束）。已完成：阶段3 代码/LHR 地形/训练链/评估工具链，全部 CPU 验证并提交（HEAD `eb09b4b`）。
- **后台 watcher 仍在运行**（PID 见 `workSpace/runs/R0_LHR/logs/watcher.pid`）：GPU 连续 3 次空闲 ≥18 GB 时会自动启动链式训练（集成→1k→10k R0-LHR）。训练由它自动开始，不需要人工守着。
- 下次开工顺序：①看 `workSpace/runs/R0_LHR/logs/watcher.log` 与 `chain.log`、`full.log`；②训练完成后跑 `bash workSpace/tools/stage3/run_lhr_eval_matrix.sh`（18 格 LHR/RHL + 6 视频 + 聚合/图）；③填阶段3 报告与 `Materials/SE/`，更新 STATUS/logs，git 提交（署名 王，2026-11-16）。
- 未做：训练监控、LHR/RHL 评估、报告/Materials、U 盘备份（本轮未做）。

## 五、下一步（训练完成后）

1. 若训练链仍在跑：用 §四 的监控命令查看；**不要同时启动第二个 GPU 任务**。
2. 训练完成后先确认：10k 完成、无 NaN、`model_10000.pt` 存在且 sha256 记录；`play`/评估能走三种 LHR 地形。
3. 跑 LHR/RHL 18 格 × 200 次评估（T3）；输出带 6 项标注；R0-pilot 不得混入。
4. 写 T3 报告 + Materials/SE/R0_LHR 材料包；更新 STATUS/logs；git 提交（署名 王，2026-11-16）。
5. 之后进入 R1（SymLoss）等后续组（见 11-16 方案任务板）。

## 六、风险 / 约束

- 单 GPU：训练期间禁止任何其他 GPU 任务。
- LHR 方向必须训练前可视化确认（左高右低、不随机翻转）；否则 10k 白跑。
- 10k ≈ 22~24 h，若中途崩溃：保留日志与最近 ckpt，记入 logs 后再决定续训/重跑。
- 本轮**未做**：10k 之后的评估/报告/材料（按用户指示“进入训练阶段后即可退出”）。
- U 盘备份：本轮结束需补做（参照 `AGENTS.md` §6，新日期目录，不覆盖 11-12 备份）。

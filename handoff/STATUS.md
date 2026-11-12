# 交接状态（唯一状态源）

更新时间：2026-11-12 17:55
当前工作人员：空
当前阶段：2（T1 已完成；T3 镜像地形评估已完成）

---

## 一、T2 核查结论（2026-11-12，王）

**问题**：旧 STATUS/PROJECT_STATE 把 T2 记为“未开始”，需确认 T2 是否真的完成。

**结论：T2 在 11-07 优化版方案里已取消，无待办工作，视为完成。**

依据：

1. 新版方案 `reference/复现方案_SWAP轻量版(11月7日优化版)(以此版为准).md`：
   - 阶段 2 标题即“降显存、加速（已合并到阶段 1，不再单独做）”；
   - 任务板 T2：“**已合并到阶段 1，不再单独做**；后续组复用 512 env”。
2. 新版 T2 对应的约束已在 T1 中满足，证据：
   - `num_envs=512` 已冻结（训练日志 `[train_capped] task=go2_amp num_envs=512`）；
   - 显存峰值 **14774 MiB / 24564 MiB**（`workSpace/runs/stage-1/go2-base512/vram.csv` 末行）；
   - `camera_num_envs` 由 overlay `train_capped.py` 自动收敛 1024 → 487（`train.log:12`），无 OOM。
3. 旧版方案“12 GB 内能跑”指标不适用于本机（24 GB 4090）；旧版已标注“非必要勿读”，不作准。

**遗留**：`handoff/PROJECT_STATE.md` 曾把 T2 记为“未开始”，**已在本轮一并向修正（2026-11-12）**；以本文件为准。

---

## 二、已完成阶段（历史摘要）

| 阶段 | 结果 | 证据 |
|---|---|---|
| T0 环境 | ✅ 通过（Isaac Gym Preview 4 + torch cu121 路线） | `workSpace/runs/stage-0/manifest.json` |
| G0–G3 Go2 迁移 | ✅ 通过 | `go2_amp` 注册；`check_motion_order.py` ALL CHECKS PASSED；`go2_depth.png` |
| T1 基线 10k | ✅ 完成（陈旺 2026-11-08） | 见下 |
| **T3 镜像地形评估** | ✅ 完成（王 2026-11-12） | 见下 + `Materials/SE/T3/` |

### T1 关键结果（陈旺，2026-11-08）

- 训练：`go2_base512`，512 env × 10000 iter，22h18m，reward 15.93，0 nan，显存峰值 14.8 GB
- 评估（play.py 派生脚本，64 env × 1200 步，ckpt `model_10000.pt`）：

| 地形 | 成功率 |
|---|---|
| slope | 93.8%（61/65） |
| stair | 98.4%（63/64） |
| gap | 18.2%（27/148） |

- ckpt：`WMP/logs/go2_amp_example/go2_base512/model_10000.pt`（sha256 `46cbef12…`，T3 轮补录核对）
- 材料包：`/home/ubuntu/B4_2025/Materials/SE/`（T1_report.md / manifest.json / figures / eval / videos）

---

## 三、本轮：T3 镜像地形评估（王，2026-11-12，已完成）

- run_id：`s3-20261112-1620-mirror-eval`
- 目录：`workSpace/runs/s3-20261112-1620-mirror-eval/`（474 MB，含视频原始帧）
- 材料包：`/home/ubuntu/B4_2025/Materials/SE/T3/`
- 状态：**完成**（2026-11-12 17:22 矩阵结束；52/52 GPU 任务成功；GPU 已释放）
- 代码：评测代码 `eval_mirror.py`（sha256 `82d960e5…`）+ `mirror_terrain.py`（sha256 `e7de4724…`，修复版）；评测时仓库 HEAD `cabcf50`（T3 代码当时为未提交工作区，本轮结束由 Lead 统一提交，commit 见下）
- ckpt：`model_10000.pt`（sha256 `46cbef12…`）

### 本轮口径（用户 2026-11-12 确认 A+B）

- **A 组**（WMP 原地形）：`tilt` / `stair` / `gap` × DIFF 0.2/0.5/0.8 × normal/mirror × 200 次/格 = 18 格。
- **B 组**（SWAP 式单侧横向倾斜，OOD 对照）：`ramp` / `ramp_stair` / `ramp_gap` × 5°/10°/15° × normal/mirror × 200 次/格 = 18 格。
- 真镜像：同一 terrain seed 后 `np.fliplr(height_field_raw)`，自定义 trimesh 绕机器人矢状面反射**并翻转三角形绕序**；每格记录 `height_field_sha256` / `height_field_fliplr_sha256` 供配对校验。
- 另：T1 协议复现 3 次/地形、A 组抽 3 格重跑、4 段视频（A/B × normal/mirror）。

### 关键结果

**A 组（WMP 原地形）：镜像掉点 ≈ 0（在单种子噪声内）**

- `tilt`：d0.2 97.5/97.5（0.0 pp）、d0.5 98.5/96.5（+2.0）、d0.8 98.5/99.0（−0.5）
- `stair`：d0.2 99.5/98.5（+1.0）、d0.5 99.5/99.0（+0.5）、d0.8 97.5/98.0（−0.5）
- `gap`（高度场本身非对称）：d0.2 84.5/88.0（−3.5）、d0.5 75.0/79.5（−4.5）、d0.8 0.0/0.0（该难度过难）
- 对称地形 6/6 对差异均在二项 95% CI 内（≤2 pp 量级）；**不能写“完全一致”，只能写“在单种子噪声内”**。
- **任务卡 `30-mirror-eval` 的退出门槛「镜像成功率明显下降」在 WMP 原地形上不成立**（任务卡假设 WMP 地形有左右难度属性，实测没有；`stair`/`tilt` 高度场翻转逐像素一致）。这是本轮关键结论，已写入 T3 报告 §5/§7。

**B 组（自建单侧横坡，OOD）**：

- `ramp`：5° 98.0/97.0（+1.0）、10° 90.5/96.5（−6.0）、15° 58.0/60.5（−2.5）
- `ramp_stair`：5° 97.0/100.0（−3.0）、10° 91.5/99.0（−7.5）、15° 81.5/82.5（−1.0）
- `ramp_gap`：5° 75.5/61.5（**+14.0**）、10° 36.5/37.5（−1.0）、15° 8.5/14.0（−5.5）
- **无一致方向性掉点**：9 对里 7 对 mirror 更好，平均 drop −1.28 pp、平均 |drop| 4.61 pp；唯一明显掉点是 ramp_gap 5° 的 +14.0 pp。横坡增大主要让两方向一起变难（normal 均值 90.17→49.33%）。
- 口径：B 组是 **OOD 镜像敏感性对照，不能直接等同论文 Fig.3**（论文是同分布单侧倾斜训练后的镜像测试）。

**T1 一致性 / 可复现性**：

- T1 协议复现 3 次：slope 96.9/92.4/92.3（均值 93.87 vs 93.8）、stair 100.0/98.5/98.4（98.97 vs 98.4）、gap 22.2/18.4/19.9（20.17 vs 18.2）→ 全部 ±5 pp 内，T1 可复现；极差 4.6/1.6/3.8 pp 作为单种子噪声。
- A 组抽 3 格重跑：+1.5/−1.0/−0.5 pp（同分布，非逐例一致——GPU 物理非确定性）。
- `checks.json`：39/39 cells 齐全（1 ignored=首轮遗留格）、镜像配对 18/18 PASS、对称地形 6/6 PASS、trimesh 有符号体积 3/3 PASS、代码版本链 match、videos=4、无失败格。

### 本轮实现修复（必须写进交接）

- **trimesh 绕序 bug（静默失败，已修复）**：镜像自定义 trimesh 时只反射顶点、未翻转三角形绕序 → 面法向反转；修复前 A 组 tilt d0.2 normal 97.0% / mirror 32.5%（假掉点 64.5 pp）。修复：`triangles[:, [0, 2, 1]]`；用**有符号体积**验证（修复前 mirror 为负 −8.72888 m³，修复后 tilt/crawl normal/mirror 均正：tilt +8.728880 m³、crawl +4.534485 m³，外向三角形 0/48 → 48/48）。
- 修复版 `mirror_terrain.py` sha256 `e7de4724…`；修复前 16:37 首轮尝试的数据**全部作废**，正式矩阵 16:42 重启（`run_config.json.started_at=2026-11-12T16:42:12+0800`）。
- 不变量测试（含面法向/有符号体积）：168/168 PASS（定稿版；修复前 160/160 未覆盖绕序）。
- 教训：镜像正确性不能只看高度场/顶点集合，必须校验有方向的量（面法向/有符号体积）。

### 证据

- 汇总表：`workSpace/runs/s3-20261112-1620-mirror-eval/metrics.csv`（39 行）
- 自校验：`checks.json`；人读版：`aggregate_report.md`；元数据：`manifest.json`（含 `code_versions`/`mirror_checks`/`bugfix_evidence`/`nondeterminism`）
- 图：`figures/fig_a_matrix.png`、`fig_a_mirror_drop.png`、`fig_b_matrix.png`、`fig_b_mirror_drop.png`、`fig_t1_consistency.png`、`fig_terrain_heights.png`
- 视频：`videos/*.gif`（4 段）
- 逐格原始数据：`a_matrix/*/summary.json`、`b_matrix/*/summary.json`、`trials.jsonl`、`terrain.npz`、`run.log`
- 材料包：`/home/ubuntu/B4_2025/Materials/SE/T3/`（含 `SHA256SUMS`）
- 报告：`Materials/SE/T3/T3_report.md`（本地起草版 `lab/work/docs/T3_report.md`）
- 修复证据：`lab/work/docs/evidence/prefix_aborted_run_20261112/`（修复前 tilt 数据，作废）、`evidence/mirror_terrain_remote_fixed_20261112.py`
- 数据有效性：仅使用 16:42 重启后的运行；首轮遗留 `a_matrix/A_tilt_d0.2_normal` 已被聚合忽略；修复前数据只作 bug 证据、未进入结果表

### 评测代码（评测专用，不改训练/地形生成）

- `WMP/legged_gym/utils/mirror_terrain.py`（`EvalTerrain`：固定难度 / 真镜像 / 单侧横坡 / 指纹；修复版）
- `WMP/legged_gym/scripts/eval_mirror.py`（200 试次/格、首回合判定、失败原因、地形 npz）
- overlay 副本：`wmp_overlay/files/legged_gym/{utils,scripts}/`；`apply.sh` 可还原
- 管线：`workSpace/tools/t3/{run_matrix.sh, aggregate_metrics.py, make_figures.py, make_gifs.sh, cells.tsv}`

---

## 四、下一步（本轮完成后）

1. **若要复现论文 Fig.3（同分布镜像）**：在**单侧横向倾斜地形**上重训 R0（建议 2k 冒烟确认能走 → 10k 全量；建议用固定、干净的斜坡基座，避免 B 组 ramp 的“波浪+横坡”混淆），然后：
   - 用同一评估协议测 R0 在同分布 normal/mirror 上的 drop；
   - 再按方案阶段 4~6 做 R1/R2/R3，与 R0 同配置对比（镜像 gap 指标）。
   - 本轮 B 组只是 OOD 敏感性预演，不能替代这一步。
2. **T4/T5（对称变换 + SE-MLP）**：可先做 CPU 单元测试（`symmetry.py` + 等变测试），不占 GPU；T3 的镜像评测脚本可直接复用（把 `LOAD_RUN`/`CKPT_ITER` 换成新组 ckpt）。
3. **gap 低成功率（T1 遗留）**：如需达标，加跳跃课程/延长训练，不要靠堆 iteration。
4. **文档同步**（本轮已完成，供核对）：
   - `handoff/STATUS.md`：本文件（T2 结论 + T3 结果）
   - `handoff/logs.md`：已追加本轮日志行
   - `handoff/PROJECT_STATE.md`：已修正“T2 未开始”过时表述、更新 T3 状态与关键数字
5. **U 盘备份**：按 `AGENTS.md` §6 新建 `SWAP-backup-20261112-XXXX/`（含 `repo.bundle`、快照、`wmp_overlay/`、本轮 `runs/` 的 manifest/metrics/videos、`SHA256SUMS`），旧目录不覆盖。
6. **下一位工作人员**：由 Lead 指定；交接前确认 GPU 空闲、无后台评测进程（本轮结束时 784 MiB / 0%）。

## 五、风险 / 运行中训练

- 运行中训练：无（GPU 空闲）
- 阻塞：无
- 风险：
  1. A 组“镜像掉点≈0”可能被误读为“策略稳健”；实为地形对称（报告第 2/5/7 章已写明）。
  2. B 组为 OOD，不能等同论文 Fig.3；后续引用时注意口径。
  3. 单地形种子 + 单 ckpt + 单次评估：200 试次的二项误差约 ±4 个百分点（p≈0.9），掉点小的差异不显著；同种子重跑不逐例一致（|Δ|≤1.5 pp）。
  4. `ramp` 基座地形实际是 wave+横坡（不是纯平地+横坡），已按实际地形表述（报告 7.8）。
  5. 评测地形布局与训练课程不同（单地形块 vs 10×20 课程；报告 7.5）。

---

## 六、commit / 备份

- 本次 T3 提交：`c833743`（Lead 统一提交后回填；评测时仓库 HEAD 为 `cabcf50`）
- 备份：✅ 已完成 → U 盘 `/media/ubuntu/__/SWAP-backup-20261112-1809/`（`repo.bundle` + 快照 + `wmp_overlay/` + 本轮 runs 产物 + Materials + best ckpt，`SHA256SUMS` 已校验）
- 运行中训练/评测：无（矩阵 17:22 结束，GPU 已释放）

# 项目状态快照（压缩版）

> 这份文件取代"翻聊天记录"。任何 agent 接手先读它，再读 `handoff/STATUS.md`。
> 最后更新：2026-11-12，代码版本 `cabcf50`（T3 提交后 `c833743`）。

## 一、这台机器上有什么

| 项 | 位置 |
|---|---|
| 项目仓库 | `/home/ubuntu/B4_2025/SWAP` |
| WMP 上游 | `SWAP/WMP`（gitignored，commit `c232c115`，改动只走 `wmp_overlay/`） |
| 交付材料 | `/home/ubuntu/B4_2025/Materials/SE/`（T1）+ `/home/ubuntu/B4_2025/Materials/SE/T3/`（T3） |
| conda 环境 | `wmp` = py3.8.20 + **torch 2.4.1+cu121（实测 2026-11-12；旧文档写 2.3.1）** + Isaac Gym **Preview 4**（1.0rc4） |
| 训练产物 | `SWAP/workSpace/runs/`（gitignored） |

**环境激活必须先 source**：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate wmp`
（`wmp` 环境带激活钩子自动设置 `LD_LIBRARY_PATH`，否则 `gym_38.so` 找不到 `libpython3.8`）

## 二、进度

| 阶段 | 状态 |
|---|---|
| T0 环境 | ✅ 通过（Preview 4 路线，非方案要求的 Preview 3） |
| G0–G3 Go2 迁移 | ✅ 通过（资产/注册/动作转换/冒烟/深度相机） |
| **T1 基线 10k iter** | ✅ 完成（512 env，22h18m，reward 15.93，0 nan） |
| T2 降显存 | ✅ 已合并进阶段 1、不再单独做（512 env 冻结；显存峰值 14774 MiB / 24564 MiB；camera_num_envs 由 train_capped.py 自动 1024→487，无 OOM） |
| T3 镜像地形评估 | ✅ 完成（王 2026-11-12）：A 组 WMP 原地形掉点≈0（地形左右对称，任务卡退出门槛不成立）+ B 组自建单侧横坡 OOD 对照；材料见 `Materials/SE/T3/` |

### T1 关键数字

| 地形 | 成功率 | play 总 reward |
|---|---|---|
| slope | 93.8% | 32.12 |
| stair | 98.4% | 24.87 |
| **gap** | **18.2%** | 16.31 |

保底档要求「三地形 > 50%」：slope/stair 达成，**gap 未达成**（跳跃类技能，需专门课程）。

### T3 关键数字（王，2026-11-12）

| 组 | 设置 | 结果 |
|---|---|---|
| A（WMP 原地形） | tilt/stair/gap × DIFF 0.2/0.5/0.8 × normal/mirror，200 次/格 | 对称地形 6/6 对 drop 全部在二项 95% CI 内（tilt 0.0/+2.0/−0.5 pp，stair +1.0/+0.5/−0.5 pp）；gap −3.5/−4.5/0.0 pp；**任务卡“镜像成功率明显下降”退出门槛不成立**（WMP 原地形左右对称，非策略稳健） |
| B（自建单侧横坡，OOD） | ramp/ramp_stair/ramp_gap × 5/10/15° × normal/mirror，200 次/格 | 无一致方向性掉点：9 对里 7 对 mirror 更好、平均 drop −1.28 pp、平均 |drop| 4.61 pp；唯一明显掉点 ramp_gap 5° +14.0 pp。**OOD 对照，不能直接等同论文 Fig.3** |
| T1 一致性 | 同协议复跑 3 次/地形 | slope 93.87 / stair 98.97 / gap 20.17%（vs 93.8/98.4/18.2，全部 ±5 pp 内）→ T1 可复现 |
| 可复现性 | A 组抽 3 格重跑 | +1.5/−1.0/−0.5 pp（同分布；同种子不逐例一致 = GPU 物理非确定性） |

- 评测代码：`WMP/legged_gym/utils/mirror_terrain.py`（sha256 `e7de4724…`）+ `WMP/legged_gym/scripts/eval_mirror.py`（`82d960e5…`）；评测专用，不改训练。
- 运行目录：`workSpace/runs/s3-20261112-1620-mirror-eval/`；`checks.json`: 39/39 cells、配对 18/18、对称地形 6/6、trimesh 3/3 PASS。
- 本轮 ckpt：`go2_base512/model_10000.pt`（sha256 `46cbef12…`；T1 评估实际加载的也是 10000，T1 材料包放的 9500 是历史遗留）。
- c833743：评测时仓库 HEAD `cabcf50`，T3 提交由 Lead 完成后回填。

## 三、必须知道的坑（都踩过）

1. **`camera_num_envs` 必须 ≤ `crawl_end`**：已在 `wmp_overlay/files/train_capped.py` 里自动收敛，别再手改配置。
2. **`play.py` 硬编码 `load_run='WMP'`**：靠软链接 `logs/go2_amp_example/WMP -> <run>` 切换。
3. **`play.py` 的 `--seed` 不被 CLI 接受**：只能单次多环境评估，不能多 seed 平均。
4. **`max_episode_length = 1001`**：评估至少跑 1200 步才等得到超时判定。
5. **自定义脚本放 `/tmp` 会崩**：`WMPRunner` 用 `sys.argv[0]` 向上三级找 `dreamer/configs.yaml`，必须放进 `WMP/legged_gym/scripts/`。
6. **`mesh_type='plane'` 会崩**：`check_termination` 无条件读 `terrain_levels`，直线地形才创建。
7. **`ruamel.yaml` 必须 0.17.4**：0.18.x 的 `safe_load` 直接抛错。
8. **`scp` 必须加 `-O`**：SFTP 模式在高延迟下慢 10 倍。
9. **SSH**：`ssh ubuntu-ip`（固定 IP 最稳），mDNS 名字时好时坏。
10. **WMP 原地形左右近似对称**：stair/tilt 高度场翻转逐像素一致，slope 只差 ±5 cm 噪声，gap 只差坑沿 1 像素。不要期待“镜像地形”在 WMP 原地形上掉点——这是地形性质，不是策略稳健。要测真镜像敏感性，用单侧横坡（T3 的 B 组）并做同分布训练（重训 R0）。
11. **镜像评测必须单地形行**：`EvalTerrain` 要求 `num_rows/num_cols=1`、`curriculum=False`，并 monkeypatch `legged_robot.Terrain`；不要拿去改训练路径。
12. **评测 step 上限**：`max_episode_length=1001`，`STEPS=0` 时脚本自动跑 1004 步；成功判定是 `ep_len >= max_episode_length-1`（首回合 timeout），每 env 只计首回合。
13. **B 组 `ramp` 口径**：当前 `eval_mirror.TERRAIN_PROPS["ramp"]=[1.0,0,...]` 命中 WMP `make_terrain` 的 `wave_terrain` 分支（波浪地形+横坡），并非“纯平地+横坡”。若后续改成纯横坡需 props 全 0（走 `else` 分支），并重跑 B 组才可比。
14. **镜像 trimesh 必须翻转三角形绕序**：只反射顶点会让面法向反转，PhysX 碰撞行为改变（曾导致 tilt normal 97% vs mirror 32.5% 的假掉点）；反射后要 `triangles[:, [0,2,1]]`，并用**有符号体积/面法向**验证，不能只比高度场或顶点集合。
15. **镜像反射轴是整幅地图几何中心**（世界 y≈3.95 m），与机器人矢状面 y=4.0 m 差 0.05 m；stair/tilt 无影响，gap 走廊 1 像素、ramp 零点 0.05 m 偏移；配对指纹仍严格成立。

## 四、常用命令

```bash
# 看训练进度（面板）
cd ~/B4_2025/SWAP && bash workSpace/tools/watch_train.sh

# 开训练（示例：512 env × 10k）
cd ~/B4_2025/SWAP/WMP
PYTHONPATH=$PWD MAX_ITERS=10000 MAX_ENVS=512 SAVE_INTERVAL=500 RUN_NAME=<name> \
  python train_capped.py --task=go2_amp --headless --num_envs=512 --sim_device=cuda:0

# 三地形评估（脚本在 Materials/SE/scripts/）
cp /home/ubuntu/B4_2025/Materials/SE/scripts/eval_success.py ~/B4_2025/SWAP/WMP/legged_gym/scripts/
ln -sfn <run> ~/B4_2025/SWAP/WMP/logs/go2_amp_example/WMP
NEVAL=64 STEPS=1200 PYTHONPATH=$PWD python legged_gym/scripts/eval_success.py \
  --task=go2_amp --headless --sim_device=cuda:0 --terrain=slope

# T3 镜像评测（36 格 + T1 复现 + 重跑 + 视频，约 45-60 min）
bash /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/run_matrix.sh all
python /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/aggregate_metrics.py --run-dir <RUN_DIR>
python /home/ubuntu/B4_2025/SWAP/workSpace/tools/t3/make_figures.py --run-dir <RUN_DIR>

# U 盘交接备份
bash ~/B4_2025/SWAP/workSpace/tools/backup_usb.sh /media/ubuntu/<U盘名>
```

## 五、下一步建议

1. **复现论文 Fig.3（同分布镜像）**：在单侧横向倾斜地形上重训 R0（先 2k 冒烟，再 10k），用 T3 的评测脚本测同分布 normal/mirror drop；这是把 T3 的“镜像评测能力”升级为“论文核心证据”的必要一步。T3 的 B 组只是 OOD 预演，不能替代。
2. **T4/T5（对称变换 + SE-MLP）**：先做 CPU 单元测试（symmetry.py + 等变测试），不占 GPU；评测复用 T3 脚本（换 LOAD_RUN / CKPT_ITER）。
3. **gap 低成功率（T1 遗留 18.2%；T3 的 gap d0.8 双向 0%）**：若要达标，加跳跃课程或延长训练，不要靠堆 iteration。
4. **文档对齐**：STATUS / logs / PROJECT_STATE 三处对 T2、T3 的口径已同步（2026-11-12）。

## 六、材料包

- （T1）`/home/ubuntu/B4_2025/Materials/SE/`（331 MB）：README、T1_report.md、manifest.json、figures/、logs/、eval/、ckpt/、videos/、scripts/
- （T3）`/home/ubuntu/B4_2025/Materials/SE/T3/`：README.md、T3_report.md、manifest.json、figures/、eval/、videos/、scripts/、SHA256SUMS

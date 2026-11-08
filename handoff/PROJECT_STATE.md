# 项目状态快照（压缩版）

> 这份文件取代"翻聊天记录"。任何 agent 接手先读它，再读 `handoff/STATUS.md`。
> 最后更新：2026-11-08，代码版本 `c6fe3bb`。

## 一、这台机器上有什么

| 项 | 位置 |
|---|---|
| 项目仓库 | `/home/ubuntu/B4_2025/SWAP` |
| WMP 上游 | `SWAP/WMP`（gitignored，commit `c232c115`，改动只走 `wmp_overlay/`） |
| 交付材料 | `/home/ubuntu/B4_2025/Materials/SE/` |
| conda 环境 | `wmp` = py3.8 + torch 2.3.1+cu121 + Isaac Gym **Preview 4** |
| 训练产物 | `SWAP/workSpace/runs/`（gitignored） |

**环境激活必须先 source**：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate wmp`
（`wmp` 环境带激活钩子自动设置 `LD_LIBRARY_PATH`，否则 `gym_38.so` 找不到 `libpython3.8`）

## 二、进度

| 阶段 | 状态 |
|---|---|
| T0 环境 | ✅ 通过（Preview 4 路线，非方案要求的 Preview 3） |
| G0–G3 Go2 迁移 | ✅ 通过（资产/注册/动作转换/冒烟/深度相机） |
| **T1 基线 10k iter** | ✅ 完成（512 env，22h18m，reward 15.93，0 nan） |
| T2 降显存 | ⬜ 未开始（当前 14.8 GB） |
| T3 镜像地形评估 | ⬜ 未开始（这是 SWAP 的核心证据） |

### T1 关键数字

| 地形 | 成功率 | play 总 reward |
|---|---|---|
| slope | 93.8% | 32.12 |
| stair | 98.4% | 24.87 |
| **gap** | **18.2%** | 16.31 |

保底档要求「三地形 > 50%」：slope/stair 达成，**gap 未达成**（跳跃类技能，需专门课程）。

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

# U 盘交接备份
bash ~/B4_2025/SWAP/workSpace/tools/backup_usb.sh /media/ubuntu/<U盘名>
```

## 五、下一步建议

1. **T3 镜像地形评估**（最高优先级）：在 `height_field_raw` 上 `np.fliplr` 翻转地形，
   对比训练地形 vs 镜像地形成功率。这是 SWAP 要解决的问题、也是论文 Fig.3 的核心。
2. **T2 降显存**：若要用 12 GB 卡，降 `camera_num_envs` 到 256~512 再验证趋势一致。
3. gap 若要达标：加跳跃课程或延长训练，不要指望单纯加 iteration。

## 六、材料包

`/home/ubuntu/B4_2025/Materials/SE/`（331 MB）：
README、T1_report.md、manifest.json、figures/、logs/、eval/、ckpt/、videos/、scripts/

# 阶段 0 — 4090 环境与 WMP 冒烟
- 状态：以 `handoff/STATUS.md` 为准
- 入口：无
- 出口：本卡退出条件全过
- 说明：唯一豁免“真实训练”的入场阶段；通过后才能开始 0.5。
## 目标
在实验室 4090 装好 WMP 环境（Py3.8 + Isaac Gym Preview 3 + torch cu117），跑通 `test_env` 和 5 iter 冒烟。
## 允许改
- 环境可以装；不改仓库代码。
- WMP 改动只能通过 `wmp_overlay/`。
## 步骤
1. `bash wmp_overlay/apply.sh`
2. 按 `wmp_overlay/SETUP.md` 建 `wmp` 环境。
3. 验收 1：`python -c "import isaacgym, torch, numpy; print(torch.__version__, torch.cuda.is_available())"` → CUDA True。
4. 验收 2：
```bash
cd WMP && python legged_gym/tests/test_env.py --task=a1_amp --headless --num_envs=16
echo EXIT:$?   # 预期 Done, EXIT:0
```
5. 验收 3：
```bash
cd WMP && MAX_ITERS=5 MAX_ENVS=16 python train_capped.py \
  --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0
# 预期 [train_capped] PASS: 5 iterations completed
```
6. 采显存峰值；日志复制到 `workSpace/runs/stage-0/`。
## 必须产物
- 三条验收命令、退出码、关键输出。
- `nvidia-smi` 型号/显存、显存峰值、`git -C WMP status --porcelain`。
## 退出条件
- 三条验收全过；无 nan；WMP 无越界改动；STATUS 已更新。
## 禁止
- 不装 Preview 4；不用 50 系本机做实质训练；不改 WMP 算法。
- 不把 5 iter 冒烟当阶段成果。
## 下一步
- `handoff/tasks/05-go2.md`

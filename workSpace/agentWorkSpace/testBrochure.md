# 实验室交接单（testBrochure）

> 声明：实验室电脑使用不便，本机主要用于快速推进工作；验证项将集中在实验室电脑上进行。
> 因此，既往待办默认标记为“不清楚，但默认正确”，不在本机反复追问；
> 所有未验证事项汇总在下方“验证欠账”，到实验室逐项验证、逐项修改。

## 当前待办

- [ ] T0：在实验室机器新建 `wmp` 环境（Python 3.8 + Isaac Gym Preview 3 + torch cu117），按下方 T0 交接单跑通 `test_env.py` 与 5 iter 冒烟，回传显存峰值与日志。
- [ ] T0 之后：在实验室单独 clone WMP 并 checkout 到 commit `c232c115`（WMP 被 gitignore，不会随主仓库同步）。

## 验证欠账

| 任务 | 事项 | 本机状态 | 实验室验证方法 | 预期 | 结果 |
|---|---|---|---|---|---|
| T0 | `import isaacgym, torch` | 未验证（本机预览 4，且 torch 无 sm_120 内核） | T0 交接单“验收 1” | 无报错，CUDA 可用 | 待做 |
| T0 | `legged_gym/tests/test_env.py` | 未验证（本机缺依赖，import 失败） | T0 交接单“验收 2” | 打印 `Done`，退出码 0 | 待做 |
| T0 | WMP 5 iter 冒烟 + 显存峰值 | 未验证（本机版本冲突，见交接单“本机结论”） | T0 交接单“验收 3” | 5 iter 无报错，记录峰值显存 | 待做 |
| T0 | `train.py --max_iterations` 是否生效 | 本机静态确认**无效**：`train.py` 硬编码 100000 | 用交接单 wrapper 跑，确认实际只跑 5 iter | 不被硬编码覆盖 | 已知问题 |
| T0 | 世界模型/深度预测器训练是否触发 | 未验证（5 iter×16 env 仅约 1920 帧 < 10000 门限） | 后续用更大 env/iter 跑，看日志 `World_model/*`、`DepthPredictor/loss` | 门限后正常训练 | 待做 |
| T0 | `t0_smoke.py` 的 argv/config 路径修正 | 静态模拟通过；本机缺依赖未端到端跑 | 实验室按交接单跑 5 iter | 正常构造 WMPRunner | 待做 |

---

## T0 — 环境安装与 WMP 冒烟（交接单）

**状态**：本机部分完成，实验室部分待验证
**上游**：`https://github.com/bytedance/WMP`，commit `c232c115ada4517453ebded5019078ba055456de`（2024-12-18，仓库仅 1 个 commit）
**任务卡**：`documents/tasks/T0.md`

### 本机结论（2026-10-26）

本机：WSL Ubuntu 22.04 / RTX 5060 Laptop 8 GB / conda env `unitree-rl`
（Python 3.8.20，torch 2.4.1+cu121，Isaac Gym 1.0rc4 = Preview 4，numpy 1.20.0）。

1. **torch CUDA 不可用（决定性阻塞）**：`torch.cuda.is_available()` 为 True，但任何 CUDA 算子报
   `RuntimeError: CUDA error: no kernel image is available for execution on the device`
   （cu121 wheel 不含 sm_120 内核）。→ 本机无法做 WMP 实质训练。
2. **Isaac Gym 本体可用（隔离测试）**：Preview 4 在 5060 上创建 16 env 的 PhysX 场景并 step 10 次成功
   （`artifacts/t0_isaacgym_probe.py`，EXIT 0）。
3. **WMP import 失败**：任务卡原命令在 `import torchvision → PIL` 处因 numpy 1.20 无
   `numpy.typing.NDArray` 失败（EXIT 1）；加运行时 shim 后继续到
   `ModuleNotFoundError: pybullet_utils`。本机 conda 还缺 `ruamel_yaml/sklearn/cv2/trimesh/matplotlib`。
   按红线不装包，不继续死磕。
4. **上游 `train.py` 忽略 `--max_iterations`**：`train.py` 写死
   `train_cfg.runner.max_iterations = 100000`，直接按任务卡命令跑会无界训练。本机改用
   `artifacts/t0_smoke.py`（默认 2 iter，可用 `T0_MAX_ITERS` 覆盖；`WMPRunner.learn`
   结束时总会存一次 ckpt，2~5 iter 的 ckpt 很小，不影响冒烟）。wrapper 已修正
   `WMPRunner` 对 `sys.argv[0]`/二次 argparse 的依赖，本机用同一套 config 默认值模拟验证通过；
   端到端仍需实验室验证。

证据：`artifacts/t0_train_card.log`、`t0_smoke.log`、`t0_torch_cuda.log`、`t0_isaacgym_probe.log`。
代码地图（285 维观测布局、RSSM/AMP/深度预测器、Go2 改动点）：`documents/notes/WMP_code_map.md`。

### 实验室前置

- GPU：**3090 / 4090（Ampere/Ada）**。若只有 50 系，Preview 3/cu117 路线大概率跑不起来，
  先停下回报，不要自行换 Isaac Lab（移植成本高）。
- 不动现有 Go2 环境；WMP 单独建 `wmp` 环境。
- 磁盘预留 ≥10 GB（Isaac Gym + torch + conda 包）。
- 本仓库同步到实验室；若路径不同，替换下方 `SWAP=...`。
- **`WMP/` 在 `.gitignore` 中，不会随主仓库同步**：实验室需单独 clone 并 checkout 到
  `c232c115ada4517453ebded5019078ba055456de`（见安装命令第 1 步）。

### 安装命令（照抄；在实验室 Ubuntu 22.04 执行）

```bash
export SWAP=/home/l/workSpace/SWAP

# 1) 单独 clone WMP 并对齐 commit（gitignored，不随主仓库同步）
git clone https://github.com/bytedance/WMP "$SWAP/WMP"
git -C "$SWAP/WMP" checkout c232c115ada4517453ebded5019078ba055456de
git -C "$SWAP/WMP" rev-parse HEAD   # 应输出 c232c115...

sudo apt-get install -y build-essential ninja-build libgl1-mesa-glx

conda create -n wmp python=3.8 -y
conda activate wmp

# torch（cu117；Python 3.8 下通常解析到 2.0.1+cu117）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Isaac Gym Preview 3（需从 NVIDIA 开发者页面下载后解压；Preview 2/4 都不行）
cd ~/isaacgym/python && pip install -e .
cd "$SWAP/WMP"

pip install setuptools==59.5.0 ruamel_yaml==0.17.4
pip install opencv-contrib-python

# requirements.txt 里有 sklearn==0.0（废弃占位包，部分源会报错），先剔除再装
grep -v '^sklearn==' requirements.txt > /tmp/wmp_requirements.txt
pip install -r /tmp/wmp_requirements.txt
```

预期版本：Python 3.8.x、torch 2.0.1+cu117、isaacgym 1.0rc3（Preview 3）、numpy 1.22.3、
pybullet 3.2.5、trimesh 3.20.1。以 `pip show` 实际输出为准并回传。

### 验收 1：import 检查

```bash
python -c "import isaacgym, torch, numpy; \
print('isaacgym', isaacgym.__file__); \
print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print('numpy', numpy.__version__)"
```

预期：无报错，`torch.cuda.is_available()` 为 `True`，numpy 为 1.22.3。

### 验收 2：test_env

```bash
cd "$SWAP/WMP"
python legged_gym/tests/test_env.py --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0
echo EXIT:$?
```

说明：`test_env.py` 先把 env 数压到 ≤10，但随后 `make_env` 又会用 `--num_envs` 覆盖，
所以实际是 16 env；跑 10×`max_episode_length`（=10000）步，不训练，可能需要几分钟。
预期：打印 `Done`，`EXIT:0`。

### 验收 3：5 iter 冒烟 + 显存峰值（使用受控 wrapper）

**不要**直接用上游 `train.py --max_iterations=5`：该参数会被硬编码的 100000 覆盖。
使用本仓库的 `artifacts/t0_smoke.py`（镜像 `train.py`，用 `T0_MAX_ITERS=5` 控制迭代数，env 数 cap 16）：

```bash
cd "$SWAP/WMP"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > /tmp/wmp_vram.csv &
VRAM_PID=$!

T0_MAX_ITERS=5 python "$SWAP/workSpace/agentWorkSpace/artifacts/t0_smoke.py" \
  --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0 2>&1 | tee /tmp/wmp_5iter.log
EC=${PIPESTATUS[0]}

kill $VRAM_PID
echo EXIT:$EC
awk -F',' 'NR>1 {gsub(/ MiB/,"",$2); print $2}' /tmp/wmp_vram.csv | sort -n | tail -1
```

预期：日志出现 `[t0_smoke] PASS: 5 iterations completed`，`EXIT:0`；
最后一行是采样到的显存峰值（MiB）。把 `/tmp/wmp_5iter.log`、峰值、`nvidia-smi` GPU 型号回传。

参考：WMP README 称默认 4096 env 约 23 GB；本次 16 env 冒烟峰值应远低于此。

注意：16 env × 5 iter × 24 steps ≈ 1920 帧，低于 `train_start_steps=10000`，因此这次冒烟
**不会触发**世界模型/深度预测器训练，只验证 env、rollout、PPO、AMP 与 RSSM 前向。WM 训练留到后续阶段。

### 失败排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `numpy.typing` has no attribute `NDArray` | numpy < 1.21 | `pip install numpy==1.22.3` |
| `No module named pybullet_utils` | pybullet 未装 | `pip install pybullet==3.2.5` |
| `ImportError gym_38.so` / 导入到 Preview 4 路径 | 环境里混入了别的 isaacgym | 在 `wmp` 环境重装 Preview 3 的 `isaacgym/python`；`python -c "import isaacgym; print(isaacgym.__file__)"` 确认路径 |
| `CUDA error: no kernel image ...` | GPU 架构与 cu117/torch 不匹配（50 系必现；4090 偶见 sm_89 不在 wheel 内） | 50 系换 3090/4090；4090 若报 sm_89，改装 cu118 的 torch 2.0.1 并记录版本差异 |
| CUDA OOM | env 太多/相机太多 | 冒烟用 `--num_envs=16`；正式训练按方案先降 `camera_num_envs=256`、`num_envs=1024~2048` |
| `KeyError: a1_amp` / 找不到 motion | 不在 WMP 根目录运行（motion 路径是相对路径） | `cd "$SWAP/WMP"` 再跑 |
| `sklearn==0.0` 安装报错 | requirements 里的废弃占位包 | 按上面命令剔除该行 |
| `cv2` 导入报错/版本冲突 | `opencv-contrib-python` 与 requirements 的 `opencv-python` 同时装 | 只保留 `opencv-contrib-python`，或统一为一个 |
| 训练停不下来 | `train.py` 硬编码 100000 iter | 用 `t0_smoke.py`；正式训练再讨论改法 |

### 回传清单

1. `nvidia-smi` 型号与显存；
2. 验收 1/2/3 的命令、退出码、关键输出；
3. `/tmp/wmp_5iter.log` 全文；
4. 显存峰值（MiB）与 GPU 型号；
5. 安装过程与预期版本不一致的地方。

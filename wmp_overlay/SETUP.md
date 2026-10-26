# 4090 环境安装与验收

> 目标：Python 3.8 + Isaac Gym Preview 3 + torch cu117；WMP 单独建 `wmp` 环境，不动现有 Go2 环境。

## 前置

- GPU：3090 / 4090（Ampere/Ada）。50 系不要在这条线上死磕。
- 磁盘：≥10 GB。
- 已安装 conda；Ubuntu 22.04。
- 先还原 WMP：`bash wmp_overlay/apply.sh`。

## 安装

```bash
export SWAP=/home/l/workSpace/SWAP
cd "$SWAP"

sudo apt-get install -y build-essential ninja-build libgl1-mesa-glx

conda create -n wmp python=3.8 -y
conda activate wmp

# cu117；Python 3.8 下通常解析到 2.0.1+cu117
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Isaac Gym Preview 3（需从 NVIDIA 开发者页面下载后解压）
cd ~/isaacgym/python && pip install -e .
cd "$SWAP/WMP"

pip install setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python

# requirements 里的 sklearn==0.0 是废弃占位包，先剔除
grep -v '^sklearn==' requirements.txt > /tmp/wmp_requirements.txt
pip install -r /tmp/wmp_requirements.txt
```

预期版本：Python 3.8.x、torch 2.0.1+cu117、isaacgym 1.0rc3、numpy 1.22.3、pybullet 3.2.5、trimesh 3.20.1。以 `pip show` 实际输出为准。

## 验收

### 1. import

```bash
python -c "import isaacgym, torch, numpy; \
print('isaacgym', isaacgym.__file__); \
print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print('numpy', numpy.__version__)"
```

预期：无报错，CUDA 为 True，numpy 1.22.3。

### 2. test_env

```bash
cd "$SWAP/WMP"
python legged_gym/tests/test_env.py --task=a1_amp --headless --num_envs=16
echo EXIT:$?
```

预期：打印 `Done`，`EXIT:0`。可能需要几分钟。

### 3. 受控 5 iter 冒烟 + 显存峰值

上游 `train.py` 硬编码 `max_iterations=100000`，**不要**直接用 `--max_iterations`。使用 overlay 里的 `train_capped.py`：

```bash
cd "$SWAP/WMP"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > /tmp/wmp_vram.csv &
VRAM_PID=$!

MAX_ITERS=5 MAX_ENVS=16 python train_capped.py \
  --task=a1_amp --headless --num_envs=16 --sim_device=cuda:0 \
  2>&1 | tee /tmp/wmp_5iter.log
EC=${PIPESTATUS[0]}

kill $VRAM_PID
echo EXIT:$EC
awk -F',' 'NR>1 {gsub(/ MiB/,"",$2); print $2}' /tmp/wmp_vram.csv | sort -n | tail -1
```

预期：`[train_capped] PASS: 5 iterations completed`，`EXIT:0`；最后一行是显存峰值（MiB）。

## 显存提示

- 16 env 冒烟峰值应远低于 23 GB。
- 正式基线 4096 env 约 23 GB；24 GB 卡较紧。OOM 时先把 `depth.camera_num_envs` 降到 256~512，再把 `num_envs` 降到 2048。
- 16 env × 5 iter 不会触发世界模型/深度预测器训练（低于 `train_start_steps=10000`）。

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `numpy.typing` has no `NDArray` | numpy < 1.21 | `pip install numpy==1.22.3` |
| `No module named pybullet_utils` | pybullet 未装 | `pip install pybullet==3.2.5` |
| `ImportError gym_38.so` / 导入到 Preview 4 | 环境混入别的 isaacgym | 在 `wmp` 环境重装 Preview 3；确认 `isaacgym.__file__` |
| `CUDA error: no kernel image` | GPU 架构与 cu117 wheel 不匹配 | 50 系换 3090/4090；4090 报 sm_89 可试 cu118 torch 2.0.1 并记录 |
| CUDA OOM | env/相机太多 | 先降 `camera_num_envs`，再降 `num_envs` |
| `KeyError: a1_amp` / 找不到 motion | 不在 WMP 根目录运行 | `cd "$SWAP/WMP"` |
| `sklearn==0.0` 报错 | requirements 废弃占位包 | 按上面命令剔除 |
| `cv2` 冲突 | opencv 两套包 | 只保留 `opencv-contrib-python` |
| 训练停不下来 | `train.py` 硬编码 100000 | 用 `train_capped.py`，`MAX_ITERS=...` |

## 回传

- `nvidia-smi` 型号与显存。
- 1/2/3 三条验收的命令、退出码、关键输出。
- `/tmp/wmp_5iter.log` 与显存峰值。
- 安装过程与预期版本不一致的地方。

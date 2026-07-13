# Play 数据采集与监督训练说明

这两个脚本用于把已经训练好的 RSL-RL policy 当作 teacher，在 `play` 过程中采集监督学习数据，然后训练一个直接的状态到动作映射网络。

## 1. 采集数据：`collect_play_dataset.py`

脚本位置：

```bash
scripts/collect_play_dataset.py
```

它会像 `scripts/rsl_rl/play.py` 一样加载 checkpoint，运行 policy，并在每一步记录：

- 当前状态
- policy 输出动作
- 关节角度
- YOLO 观测信息
- 目标位姿
- 物体当前位姿
- 每条轨迹的初始位姿
- episode / trajectory 标识

默认推荐的监督学习 pair 是：

```text
obs_policy -> action
```

其中 `obs_policy` 是采集时 policy 看到的输入，`action` 是经过 action clip 后真正送进环境的动作。

### 基本采集命令

```bash
source /root/gpufree-data/isaac_ws/IsaacLab/env_isaaclab/bin/activate

python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Distill-Direct-v0 \
  --num_envs 1 \
  --headless \
  --load_run 2026-06-25_15-36-45 \
  --checkpoint model_9999.pt \
  --num_steps 1000 \
  --output_dir datasets/play_supervised
```

输出文件：

```text
datasets/play_supervised/<task>_<time>.pt
datasets/play_supervised/<task>_<time>.json
```

`.pt` 文件结构：

```python
data = torch.load("xxx.pt", map_location="cpu")
metadata = data["metadata"]
tensors = data["tensors"]
```

大部分张量形状都是：

```text
[T, N, ...]
```

其中：

- `T` 是采集步数 `num_steps`
- `N` 是并行环境数 `num_envs`

### 轨迹区分

数据里有：

```python
tensors["env_id"]
tensors["episode_id"]
tensors["episode_step"]
tensors["done"]
```

一条轨迹可以用下面这个 pair 唯一标识：

```text
(env_id, episode_id)
```

例如取第 0 个环境的第 2 条轨迹：

```python
data = torch.load("xxx.pt", map_location="cpu")
t = data["tensors"]

mask = (t["env_id"] == 0) & (t["episode_id"] == 2)

obs_traj = t["obs_policy"][mask]
act_traj = t["action"][mask]
```

### 固定目标角度采集

如果想固定目标角度，比如绕相机视角反方向 / 当前等价于手部局部 Y 轴旋转 45°：

```bash
python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Distill-Direct-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --goal_y_angle_deg 45 \
  --num_steps 1000 \
  --output_dir datasets/play_supervised
```

也可以直接指定 `goal_rot`，四元数顺序是 `w x y z`：

```bash
python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Distill-Direct-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --goal_rot 0.9238795 0.0 0.3826834 0.0 \
  --num_steps 1000
```

### YOLO 字段说明

常用 YOLO 字段：

```text
yolo_position_image          # mask 中心，归一化二维坐标，形状 [T, N, 2]
yolo_angle_image_rad         # mask PCA 主轴角度，形状 [T, N]
yolo_target_angle_features   # [sin(2Δθ), cos(2Δθ)]，形状 [T, N, 2]
yolo_confidence              # YOLO 检测置信度
yolo_mask_area               # mask 前景像素数量，是标量，不是像素图
yolo_visible_ratio           # 当前 mask area / 历史最大 mask area
yolo_position_valid          # 位置观测是否有效
yolo_measurement_valid       # 角度观测是否有效
```

注意：

```text
yolo_mask_area 只是一个标量，表示 mask 里有多少个前景像素。
```

如果你想保存真正的 mask 像素图，需要加：

```bash
--save_yolo_mask_pixels
```

默认保存下采样后的二值 mask：

```text
yolo_mask_pixels: [T, N, 48, 64]
```

值为 `uint8`，每个像素是 `0/1`。

示例：

```bash
python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Distill-Direct-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --num_steps 1000 \
  --save_yolo_mask_pixels \
  --yolo_mask_size 64 48
```

如果要保存原始分辨率 mask：

```bash
--yolo_mask_size 640 480
```

但这会显著增大数据集体积。建议先用 `64×48` 或 `128×96`。

### 只保留 YOLO 有效样本

采集时可以生成 `valid_sample_mask`：

```bash
python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Distill-Direct-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --num_steps 1000 \
  --only_valid_yolo
```

也可以训练时再使用 `--filter_valid_yolo` 过滤。

## 2. 监督训练：`train_supervised_policy.py`

脚本位置：

```bash
scripts/train_supervised_policy.py
```

它是纯 PyTorch 脚本，不会启动 Isaac Sim。

默认训练：

```text
obs_policy -> action
```

### 基本训练命令

```bash
python scripts/train_supervised_policy.py \
  datasets/play_supervised/xxx.pt \
  --output_dir logs/supervised_policy \
  --run_name bc_obs_policy \
  --epochs 100 \
  --batch_size 1024 \
  --lr 3e-4 \
  --device cuda
```

输出目录：

```text
logs/supervised_policy/bc_obs_policy/
```

主要输出：

```text
best.pt          # 验证集最优 checkpoint
last.pt          # 最后一轮 checkpoint
policy_jit.pt    # TorchScript 导出模型
config.json      # 训练配置
summary.json     # 训练结果摘要
```

### 使用指定状态训练

如果要用关节角、YOLO mask 像素、目标位置、目标姿态训练到动作：

```bash
python scripts/train_supervised_policy.py \
  datasets/play_supervised/xxx.pt \
  --input_keys hand_dof_pos,yolo_mask_pixels,goal_pos,goal_rot \
  --target_key action \
  --output_dir logs/supervised_policy \
  --run_name bc_hand_mask_goal \
  --epochs 100 \
  --batch_size 1024 \
  --lr 3e-4 \
  --device cuda
```

这里 `yolo_mask_pixels` 会自动展平成一维向量。例如采集时使用 `64×48`：

```text
hand_dof_pos:      20
yolo_mask_pixels:  64 * 48 = 3072
goal_pos:          3
goal_rot:          4
总输入维度:        3099
```

如果你只是使用 `yolo_mask_area`：

```bash
python scripts/train_supervised_policy.py \
  datasets/play_supervised/xxx.pt \
  --input_keys hand_dof_pos,yolo_mask_area,goal_pos,goal_rot \
  --target_key action \
  --output_dir logs/supervised_policy \
  --run_name bc_hand_maskarea_goal \
  --epochs 100 \
  --device cuda
```

但这只用了 mask 面积这个标量，不包含 mask 的空间形状信息。

### 只训练 YOLO 有效样本

```bash
python scripts/train_supervised_policy.py \
  datasets/play_supervised/xxx.pt \
  --input_keys hand_dof_pos,yolo_mask_pixels,goal_pos,goal_rot \
  --target_key action \
  --filter_valid_yolo \
  --exclude_done \
  --output_dir logs/supervised_policy \
  --run_name bc_hand_mask_goal_valid \
  --epochs 100 \
  --device cuda
```

含义：

- `--filter_valid_yolo`：只保留 YOLO 位置和角度观测有效的样本
- `--exclude_done`：去掉 episode 结束那一步的样本

### 多个数据集一起训练

```bash
python scripts/train_supervised_policy.py \
  datasets/play_supervised/run1.pt \
  datasets/play_supervised/run2.pt \
  datasets/play_supervised/run3.pt \
  --input_keys hand_dof_pos,yolo_mask_pixels,goal_pos,goal_rot \
  --target_key action \
  --run_name bc_multi_runs \
  --epochs 100 \
  --device cuda
```

### 常用训练参数

```text
--input_keys          输入字段，逗号分隔
--target_key          目标字段，默认 action
--epochs              训练轮数
--batch_size          batch size
--lr                  学习率
--hidden_dims         MLP 隐藏层，例如 256,256,128
--activation          elu/relu/silu/tanh/gelu
--loss                mse/smooth_l1/l1
--filter_valid_yolo   只用 YOLO 有效样本
--exclude_done        去掉 done 样本
--max_samples         最多使用多少样本
--device              cuda 或 cpu
```

## 3. 读取训练后的模型

加载普通 checkpoint：

```python
import torch

ckpt = torch.load("logs/supervised_policy/bc_hand_mask_goal/best.pt", map_location="cpu")
print(ckpt["input_keys"])
print(ckpt["input_dims"])
```

加载 TorchScript：

```python
import torch

policy = torch.jit.load("logs/supervised_policy/bc_hand_mask_goal/policy_jit.pt")
policy.eval()

action = policy(obs)
```

其中 `obs` 需要和训练时 `--input_keys` 的拼接顺序完全一致。

## 4. 建议

如果你使用 `yolo_mask_pixels`，当前监督脚本会把 mask 直接展平后送入 MLP。这能跑通，但不是最优结构。

更推荐的下一步是：

```text
mask image -> CNN encoder -> mask feature
mask feature + hand_dof_pos + goal_pos + goal_rot -> MLP -> action
```

如果后续需要，可以再新增一个 CNN 版本的监督训练脚本。

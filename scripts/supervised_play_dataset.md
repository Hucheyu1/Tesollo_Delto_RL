# Play 数据采集与监督训练说明

这两个脚本用于把已经训练好的 RSL-RL policy 当作 teacher，在 `play` 过程中采集监督学习数据，然后训练一个直接的状态到动作映射网络。

## 1. 采集数据：`collect_play_dataset.py`
### 基本采集命令

```bash
source /root/gpufree-data/isaac_ws/IsaacLab/env_isaaclab/bin/activate

python scripts/collect_play_dataset.py \
  --task Tesollo-Delto-DG5F-Direct-v0 \
  --num_envs 16 \
  --headless \
  --load_run 2026-06-30_11-55-12 \
  --checkpoint /home/amlrobotics/hcy_ws/Tesollo_Delto_RL_main/logs/rsl_rl/TesolloDelto/2026-06-30_11-55-12/model_8000.pt \
  --yolo_model_path /home/amlrobotics/hcy_ws/Tesollo_Delto_RL_main/scripts/yolo_seg_sim/best.pt \
  --num_steps 5000 \
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
  datasets/play_supervised/Tesollo-Delto-DG5F-Direct-v0_20260715_143300.pt \
  --output_dir logs/supervised_policy \
  --run_name history_5_new \
  --history_len 5 \
  --history_stride 1 \
  --epochs 2000 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --mask_embedding_dim 64 \
  --frame_embedding_dim 128 \
  --gru_hidden_dim 256 \
  --gru_layers 1 \
  --head_hidden_dims 256,128 \
  --dropout 0.1 \
  --loss smooth_l1 \
  --split_mode episode \
  --device cuda:0
```

```bash
python scripts/play_supervised_policy.py \
    --task Tesollo-Delto-DG5F-Direct-v0 \
    --num_envs 1 \
    --policy_jit ./logs/supervised_policy/history_10/policy_jit.pt \
    --history_len 10 \
    --history_stride 1 \
    --yolo_model_path /home/amlrobotics/hcy_ws/Tesollo_Delto_RL_main/scripts/yolo_seg_sim/best.pt \
    --supervised_action_mode delta \
    --headless \
    --video \
    --video_length 500
```

delta / absolute

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
  datasets/play_supervised/Tesollo-Delto-DG5F-Direct-v0_20260714_105857.pt \
  --vector_input_keys xxx \
  --target_key target_pos \
  --mask_key yolo_mask_pixels \
  --output_dir logs/supervised_policy \
  --run_name xxx \
  --history_len 10 \
  --history_stride 1 \
  --epochs 500 \
  --batch_size 256 \
  --lr 3e-4 \
  --loss smooth_l1 \
  --device cuda
```


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

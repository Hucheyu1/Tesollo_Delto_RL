# Tesollo Delto RL

这是一个基于 Isaac Lab 的强化学习扩展项目，用于在仿真中训练 Tesollo/Delto DG5F 右手机器人完成灵巧手操作任务。项目当前采用 Direct RL 环境，主要面向手内物体操作，并提供 RSL-RL、RL-Games 和视觉观测相关的脚本与配置。

## 当前状态

- 机器人资产已切换为 `source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/dg5f_right.usd`。
- DG5F 右手配置位于 `tasks/direct/tesollo_delto_rl/delto_cfg.py`，包含 USD 加载、初始姿态、关节初始值和 actuator 设置。
- 主环境配置位于 `tasks/direct/tesollo_delto_rl/tesollo_delto_rl_env_cfg.py`。
- 普通策略观测维度为 `84`，动作维度为 `20`。
- OpenAI 风格观测维度为 `47`，critic state 维度为 `84`。
- 蒸馏任务基于普通环境动力学，student observation 为 `47`，teacher observation 为 `84`。
- 视觉环境配置位于 `tasks/direct/tesollo_delto_rl/tesollo_delto_rl_vision_env.py`，policy observation 为 `118`，critic state 为 `111`。
- Gym 任务注册入口已整理到 `tasks/direct/tesollo_delto_rl/__init__.py`，任务名前缀为 `Tesollo-Delto-DG5F`。

## 已注册任务

| 任务名 | 环境 | 配置 |
| --- | --- | --- |
| `Tesollo-Delto-DG5F-Direct-v0` | `TesolloDeltoRlEnv` | `TesolloDeltoRlEnvCfg` |
| `Tesollo-Delto-DG5F-Distill-Direct-v0` | `TesolloDeltoRlEnv` | `TesolloDeltoRlDistillEnvCfg` |
| `Tesollo-Delto-DG5F-OpenAI-FF-Direct-v0` | `TesolloDeltoRlEnv` | `TesolloDeltoRlOpenAIEnvCfg` |
| `Tesollo-Delto-DG5F-OpenAI-LSTM-Direct-v0` | `TesolloDeltoRlEnv` | `TesolloDeltoRlOpenAIEnvCfg` |
| `Tesollo-Delto-DG5F-Vision-Direct-v0` | `TesolloDeltoRlVisionEnv` | `TesolloDeltoRlVisionEnvCfg` |
| `Tesollo-Delto-DG5F-Vision-Direct-Play-v0` | `TesolloDeltoRlVisionEnv` | `TesolloDeltoRlVisionEnvPlayCfg` |

## 环境要求

推荐使用已有 Isaac Lab 环境：

```bash
source /root/isaac_ws/IsaacLab/env_isaaclab/bin/activate
```

项目依赖 Isaac Lab、Isaac Sim、PyTorch、RSL-RL、RL-Games 等组件。若没有现成环境，请先按照 Isaac Lab 官方安装流程安装 Isaac Lab，并确认 `isaaclab.sh -p` 或当前 Python 解释器能够 import Isaac Lab。

## 安装

在仓库根目录执行：

```bash
python -m pip install -e source/Tesollo_Delto_RL
```

如果当前 shell 没有激活 Isaac Lab Python，可以使用 Isaac Lab 的启动脚本：

```bash
/root/IsaacLab/isaaclab.sh -p -m pip install -e source/Tesollo_Delto_RL
```

## 目录结构

```text
Tesollo_Delto_RL/
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── rsl_rl/
│   │   ├── train.py
│   │   └── play.py
│   ├── yolo/
│   │   ├── export_yolo_dataset.py
│   │   └── train_yolo.py
│   └── rl_games/
│       ├── train.py
│       └── play.py
└── source/Tesollo_Delto_RL/
    ├── config/extension.toml
    └── Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/
        ├── delto_cfg.py
        ├── tesollo_delto_rl_env.py
        ├── tesollo_delto_rl_env_cfg.py
        ├── tesollo_delto_rl_vision_env.py
        ├── feature_extractor.py
        ├── agents/
        └── robots/
```

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `delto_cfg.py` | DG5F 右手 `ArticulationCfg`，加载 `dg5f_right.usd`。 |
| `tesollo_delto_rl_env.py` | Direct RL 环境逻辑，包括 reset、action、reward、observation。 |
| `tesollo_delto_rl_env_cfg.py` | 主环境配置、物体配置、随机化事件、奖励参数、观测维度。 |
| `tesollo_delto_rl_vision_env.py` | 带相机和 CNN feature extractor 的视觉版本环境。 |
| `yolo_pose_estimator.py` | YOLO + RGB-D 的物体位置估计工具，用于仿真到真机迁移实验。 |
| `agents/rsl_rl_ppo_cfg.py` | RSL-RL PPO、distillation runner、policy 和算法参数。 |
| `agents/rl_games_ppo_cfg.yaml` | RL-Games PPO 配置。 |
| `robots/dg5f_right.usd` | DG5F 右手机器人主 USD。 |

## DG5F 关节

当前环境控制 20 个 DG5F revolute joints：

```text
rj_dg_1_1, rj_dg_1_2, rj_dg_1_3, rj_dg_1_4
rj_dg_2_1, rj_dg_2_2, rj_dg_2_3, rj_dg_2_4
rj_dg_3_1, rj_dg_3_2, rj_dg_3_3, rj_dg_3_4
rj_dg_4_1, rj_dg_4_2, rj_dg_4_3, rj_dg_4_4
rj_dg_5_1, rj_dg_5_2, rj_dg_5_3, rj_dg_5_4
```

当前观测不再读取指尖刚体状态或指尖接触力。OpenAI 风格 reduced observation 使用 `hand_dof_pos` 替代原来的指尖位置，full observation/state 只保留关节、物体、目标和动作相关量。

## 常用命令

列出当前注册的任务：

```bash
python scripts/list_envs.py
```

零动作 smoke test：

```bash
python scripts/zero_agent.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1 --headless
```

随机动作 smoke test：

```bash
python scripts/random_agent.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1 --headless
```

使用 RSL-RL 训练：

```bash
python scripts/rsl_rl/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 2048 --headless
```

使用 RSL-RL 蒸馏训练：

```bash
python scripts/rsl_rl/train.py --task Tesollo-Delto-DG5F-Distill-Direct-v0 --num_envs 2048 --headless --load_run 2026-06-25_15-36-45 --checkpoint model_9999.pt
```

蒸馏任务使用 `TesolloDeltoRlEnvCfg` 的动力学、reset 和奖励参数；环境返回 `policy` 作为 student reduced observation，返回 `critic` 作为 teacher full observation。teacher checkpoint 应来自 `Tesollo-Delto-DG5F-Direct-v0` 的普通 RSL-RL 训练，例如 `logs/rsl_rl/TesolloDelto/<TEACHER_RUN>/model_*.pt`。如果不传 `--load_run` 和 `--checkpoint`，脚本会从 `logs/rsl_rl/TesolloDelto/` 下按名字选择最新匹配的 checkpoint。

使用 RSL-RL 播放 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 16 --checkpoint <PATH_TO_CHECKPOINT>
```

使用 RL-Games 训练：

```bash
python scripts/rl_games/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1024 --headless
```

蒸馏、OpenAI 风格观测或视觉任务可将 `--task` 替换为上表中的对应任务名。

## YOLO 视觉流程

YOLO 流程用于把仿真中的视觉感知方式迁移到真机。本项目现在支持两种路径：

- 快速位置估计：YOLO detection/segmentation + RGB-D depth，输出物体 3D 中心位置。
- 完整姿态估计：YOLO pose 关键点 + 3D 关键点模板 + OpenCV PnP，输出物体位置和四元数姿态。

安装可选依赖：

```bash
source /root/isaac_ws/IsaacLab/env_isaaclab/bin/activate
uv pip install ultralytics opencv-python
```

### 位置估计

从仿真导出 detection 数据集：

```bash
python scripts/yolo/export_yolo_dataset.py \
  --task Tesollo-Delto-DG5F-Vision-Direct-v0 \
  --num_envs 32 \
  --num_images 5000 \
  --label_type detect \
  --output_dir datasets/tesollo_tomato_yolo \
  --headless
```

导出目录结构：

```text
datasets/tesollo_tomato_yolo/
├── dataset.yaml
├── images/train
├── images/val
├── labels/train
└── labels/val
```

训练 YOLO：

```bash
python scripts/yolo/train_yolo.py \
  --data datasets/tesollo_tomato_yolo/dataset.yaml \
  --task_type detect \
  --epochs 100 \
  --imgsz 320 \
  --batch 32 \
  --device 0
```

训练完成后，默认权重路径为：

```text
runs/tesollo_yolo/tomato_detect/weights/best.pt
```

在仿真中使用 YOLO + depth 估计物体位置：

```python
from .yolo_pose_estimator import YoloPoseEstimator, YoloPoseEstimatorCfg

self.yolo_pose = YoloPoseEstimator(
    YoloPoseEstimatorCfg(
        model_path="runs/tesollo_yolo/tomato_detect/weights/best.pt",
        class_id=0,
        confidence_threshold=0.35,
        device=self.device,
    )
)

estimate = self.yolo_pose.estimate_from_tiled_camera(
    self._tiled_camera.data,
    env_origins=self.scene.env_origins,
)

object_pos_from_yolo = estimate.position_env
valid = estimate.valid
```

### 完整姿态估计

完整姿态估计需要 YOLO-pose 模型。导出脚本会把仿真物体的 3D 包围盒 8 个角点投影成关键点标签，角点顺序和 `YoloPoseEstimatorCfg(object_size=...)` 内部模板一致。

导出 pose 数据集：

```bash
python scripts/yolo/export_yolo_dataset.py \
  --task Tesollo-Delto-DG5F-Vision-Direct-v0 \
  --num_envs 32 \
  --num_images 8000 \
  --label_type pose \
  --object_size 0.06 0.06 0.06 \
  --output_dir datasets/tesollo_tomato_yolo_pose \
  --headless
```

训练 YOLO-pose：

```bash
python scripts/yolo/train_yolo.py \
  --data datasets/tesollo_tomato_yolo_pose/dataset.yaml \
  --task_type pose \
  --epochs 150 \
  --imgsz 320 \
  --batch 32 \
  --device 0
```

默认权重路径：

```text
runs/tesollo_yolo/tomato_pose/weights/best.pt
```

使用完整姿态估计：

```python
from .yolo_pose_estimator import YoloPoseEstimator, YoloPoseEstimatorCfg

self.yolo_pose = YoloPoseEstimator(
    YoloPoseEstimatorCfg(
        model_path="runs/tesollo_yolo/tomato_pose/weights/best.pt",
        class_id=0,
        confidence_threshold=0.35,
        object_size=(0.06, 0.06, 0.06),
        min_keypoints_for_pnp=6,
        device=self.device,
    )
)

estimate = self.yolo_pose.estimate_from_tiled_camera(
    self._tiled_camera.data,
    env_origins=self.scene.env_origins,
)

object_pos_from_yolo = estimate.position_env
object_quat_from_yolo = estimate.quat_w
valid = estimate.valid
```

`quat_w` 使用 Isaac Lab 的 `(w, x, y, z)` 四元数顺序。对于西红柿这种接近球形的物体，外观本身可能没有唯一可观测朝向；代码可以输出完整姿态，但训练效果取决于模型是否能稳定识别出这些仿真关键点。如果实际物体没有明显纹理或几何特征，位置估计通常比姿态更可靠。

真机使用同一个估计器，但需要传入真实 RGB-D 图像、相机内参和手眼标定得到的相机外参：

```python
estimate = yolo_pose.estimate(
    rgb=rgb_tensor,
    depth=depth_tensor,
    intrinsics=K_tensor,
    camera_pos_w=camera_pos_w,
    camera_quat_w=camera_quat_w,
)
```

## 日志与输出

RSL-RL 默认日志目录：

```text
logs/rsl_rl/TesolloDelto/
```

蒸馏任务默认也使用 `TesolloDelto` 作为实验目录，并把 run name 后缀设为 `distill`，方便从同一个目录加载普通 full-observation teacher。OpenAI-FF 和视觉任务分别使用 `TesolloDelto_openai_ff`、`TesolloDelto_vision` 作为实验目录。训练过程中会保存 runner 配置、环境配置和 checkpoint。`play.py` 会从对应目录查找最新 checkpoint，或使用 `--checkpoint` 显式指定路径。

## 开发与格式检查

项目使用 `ruff`、`pyright` 和 pre-commit 模板。安装 pre-commit：

```bash
pip install pre-commit
```

运行全部检查：

```bash
pre-commit run --all-files
```

只做 Python 语法检查时，可以执行：

```bash
python -m py_compile \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/__init__.py \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/delto_cfg.py \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/agents/rsl_rl_ppo_cfg.py \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/tesollo_delto_rl_env_cfg.py \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/tesollo_delto_rl_vision_env.py \
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/yolo_pose_estimator.py \
  scripts/yolo/export_yolo_dataset.py \
  scripts/yolo/train_yolo.py \
  scripts/list_envs.py
```

## VS Code

如果 Pylance 无法找到 Isaac Lab 或 Isaac Sim 模块，请检查 `.vscode/settings.json` 的 `python.analysis.extraPaths`，确保包含：

```json
{
  "python.analysis.extraPaths": [
    "<repo>/source/Tesollo_Delto_RL",
    "<isaaclab>/source/isaaclab",
    "<isaaclab>/source/isaaclab_tasks",
    "<isaaclab>/source/isaaclab_assets",
    "<isaaclab>/source/isaaclab_rl"
  ]
}
```

也可以运行 VS Code task `setup_python_env` 重新生成 `.vscode/.python.env`。

## 待办

- 根据 DG5F 手型和目标物体重新调试 `object_cfg.init_state`、`goal_pos`、奖励尺度和 reset 噪声。
- 在 Isaac Sim 中运行 `zero_agent.py` 和 `random_agent.py` 做环境实例化 smoke test。

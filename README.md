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
source /root/gpufree-data/isaac_ws/IsaacLab/env_isaaclab/bin/activate
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
        ├── yolo_seg_image_estimator.py
        ├── foundationpose_estimator.py
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
| `yolo_seg_image_estimator.py` | 批量 YOLO-seg 二维中心、主轴角度与遮挡时序跟踪，用于蒸馏 student observation。 |
| `foundationpose_estimator.py` | FoundationPose + RGB-D + mask 的物体 6D 位姿估计封装，用于仿真到真机迁移实验。 |
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
python scripts/rsl_rl/train.py --task Tesollo-Delto-DG5F-Distill-Direct-v0 --num_envs 16 --headless --load_run 2026-06-25_15-36-45 --checkpoint model_9999.pt
```

蒸馏任务使用 `TesolloDeltoRlEnvCfg` 的动力学、reset 和奖励参数。student 的 54 维 `policy` observation 为：关节位置 20 维、YOLO mask 归一化图像中心 2 维、绕手部局部 Y 轴的目标角误差 `[sin(2Δθ), cos(2Δθ)]` 2 维、二值触觉 10 维、上一时刻动作 20 维。二维位置以图像中心为 `(0, 0)`，范围约为 `[-1, 1]`，正 X 向右、正 Y 向下。

环境返回与现有 teacher checkpoint 匹配的 84 维 `critic` 真值状态（不含新增的 10 维触觉），奖励和终止条件也仍使用仿真真值。双角形式是因为分割 mask 的 PCA 主轴具有 180° 等价性；`Δθ` 与 `Δθ+π` 会映射到相同特征。Distill 的旋转奖励、成功判定和 teacher 目标也统一为绕 Y 轴、模 180° 的最近等价姿态，避免同一 student observation 对应互相冲突的 teacher 动作。仿真仅在首次有效测量时估计一次共享的相机角度偏置，后续 reset 不再使用姿态真值；真机使用时应将标定结果填入 `yolo_angle_offset_rad`。teacher checkpoint 应来自 `Tesollo-Delto-DG5F-Direct-v0` 的 84 维普通 RSL-RL 训练，例如 `logs/rsl_rl/TesolloDelto/<TEACHER_RUN>/model_*.pt`。如果不传 `--load_run` 和 `--checkpoint`，脚本会从 `logs/rsl_rl/TesolloDelto/` 下按名字选择最新匹配的 checkpoint。视觉蒸馏包含相机渲染和 YOLO 推理，建议先从 16 个环境开始，再按显存和吞吐量调整 `--num_envs`；训练和播放脚本会为 Distill 任务自动启用相机。

使用 RSL-RL 播放 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 16 --checkpoint <PATH_TO_CHECKPOINT>
```

使用 RL-Games 训练：

```bash
python scripts/rl_games/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1024 --headless
```

蒸馏、OpenAI 风格观测或视觉任务可将 `--task` 替换为上表中的对应任务名。

## FoundationPose 视觉流程

当前视觉迁移路线改为 FoundationPose：不再导出检测数据集，也不训练检测网络。FoundationPose 直接使用 `RGB + depth + mask + 相机内参 + 物体 mesh` 输出物体 6D 位姿。

需要准备的输入：

- RGB-D 图像：仿真中由 `Tesollo-Delto-DG5F-Vision-Direct-v0` 的 `TiledCamera` 输出，真机上来自 RGB-D 相机。
- 目标 mask：仿真中由 semantic segmentation 生成，真机上可以来自任意分割模块或人工/规则分割。
- 相机内参和外参：仿真中从 `camera.data.intrinsic_matrices`、`camera.data.pos_w`、`camera.data.quat_w_ros` 读取；真机上来自相机标定和手眼标定。
- 目标 mesh：FoundationPose 通常需要 `.obj`、`.ply` 或 `.stl`。当前仿真物体是 `robots/tomato.usd`，实际使用前建议导出一个同尺度的 tomato mesh，例如 `assets/tomato.obj`。

安装 FoundationPose 后设置路径：

```bash
source /root/isaac_ws/IsaacLab/env_isaaclab/bin/activate
export FOUNDATIONPOSE_ROOT=/root/gpufree-data/FoundationPose
export PYTHONPATH=$FOUNDATIONPOSE_ROOT:$PYTHONPATH
```

FoundationPose 依赖和编译方式以 NVLabs FoundationPose 官方仓库为准。项目内的 `foundationpose_estimator.py` 只做接口封装，不把 FoundationPose 源码复制进来。

### 仿真中使用

视觉环境的相机已经配置为输出 `rgb`、`depth` 和非彩色 `semantic_segmentation`，并用 `semantic_filter="class:tomato"` 尽量只保留目标物体 mask。

```python
from .foundationpose_estimator import (
    FoundationPoseEstimator,
    FoundationPoseEstimatorCfg,
    fixed_axis_twist_angle,
    mask_from_semantic,
)

self.pose_estimator = FoundationPoseEstimator(
    FoundationPoseEstimatorCfg(
        foundationpose_root="/root/gpufree-data/FoundationPose",
        mesh_path="<PATH_TO_TOMATO_MESH_OBJ_OR_PLY>",
        device="cuda:0",
    )
)

mask = mask_from_semantic(self._tiled_camera.data, semantic_label="tomato", foreground_fallback=True)
estimate = self.pose_estimator.estimate_from_tiled_camera(
    self._tiled_camera.data,
    mask=mask,
    env_origins=self.scene.env_origins,
)

object_pos = estimate.position_env
object_quat = estimate.quat_w
# 以手/初始姿态为参考，提取手局部 Y 轴的有符号 twist angle。
angle_y_deg = fixed_axis_twist_angle(
    estimate.quat_w,
    axis=(0.0, 1.0, 0.0),
    reference_quat_w=hand_or_initial_quat_w,
)
valid = estimate.valid
```

`quat_w` 使用 Isaac Lab 的 `(w, x, y, z)` 顺序。`estimate.pose_w` 是世界坐标系下的 4x4 齐次变换矩阵，`estimate.pose_camera` 是相机坐标系下的 4x4 变换矩阵。

### 真机迁移

真机侧也调用同一个估计器，只需要把仿真相机数据替换成真实 RGB-D、真实 mask、相机内参 `K` 和手眼标定后的相机位姿：

```python
estimate = pose_estimator.estimate(
    rgb=rgb_tensor,
    depth=depth_tensor,
    mask=mask_tensor,
    intrinsics=K_tensor,
    camera_pos_w=camera_pos_w,
    camera_quat_w=camera_quat_w,
)
```

如果相机固定在手外，`camera_pos_w/camera_quat_w` 来自外参标定；如果相机装在手上，则每步需要用机器人当前末端位姿更新相机世界位姿。FoundationPose 对 mask、深度尺度和 mesh 尺寸非常敏感，仿真和真机应尽量保持同一物体尺度、同一相机内参约定和同一坐标系约定。

### YOLO mask 角度的遮挡鲁棒基线

如果暂时没有可用的 FoundationPose mesh/RGB-D 标定，可以继续使用已有 YOLO segmentation，但不再直接把单帧 PCA 当成最终角度。`scripts/yolo_seg_sim/angle_estimator.py` 会根据 mask 面积、PCA 各向异性、轮廓 solidity 和检测置信度拒绝低质量测量，并在遮挡期间用带速度衰减的模 180°时序预测。

交互查看：

```bash
python scripts/yolo_seg_sim/detect_seg_cal_angel_from_jpg.py \
  --images datasets/test_picture \
  --model scripts/yolo_seg_sim/best.pt
```

无界面批处理并输出质量报告：

```bash
python scripts/yolo_seg_sim/detect_seg_cal_angel_from_jpg.py \
  --images datasets/test_picture \
  --model scripts/yolo_seg_sim/best.pt \
  --no-show \
  --csv outputs/tomato_angles.csv \
  --output-dir outputs/tomato_angle_frames
```

当前画面中未遮挡 mask 面积约为 `18000` 像素时，可以显式传入 `--reference-area 18000`，使序列第一帧恰好被遮挡时仍能判断可见比例。黄色轴是原始 PCA 测量，绿色轴是接受测量后的滤波结果，红色轴表示当前测量被拒绝、正在使用时序预测。该基线只能减少中短时遮挡误差；近圆、无纹理物体在完全遮挡下仍需要 RGB-D 位姿跟踪、额外标记或第二视角。

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
  source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/foundationpose_estimator.py \
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

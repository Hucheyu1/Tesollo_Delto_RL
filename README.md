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
- VTDexManip 桌面与手内重定向环境使用项目内置的
  `vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210` 权重，policy observation 为 `424`，critic state 为
  `104`；瓶盖任务使用相同的 `424` 维 actor 输入和包含瓶盖关节真值的 `82` 维 critic state。
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
| `Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0` | `TesolloDeltoVTDexTomatoEnv` | `TesolloDeltoVTDexTomatoEnvCfg` |
| `Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0` | `TesolloDeltoVTDexEnv` | `TesolloDeltoVTDexEnvCfg` |
| `Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0` | `TesolloDeltoVTDexReorientUpEnv` | `TesolloDeltoVTDexReorientUpEnvCfg` |
| `Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0` | `TesolloDeltoVTDexBottleCapEnv` | `TesolloDeltoVTDexBottleCapEnvCfg` |

`Tesollo-Delto-DG5F-VTDex-Direct-v0` 保留为原番茄任务的兼容别名。

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
        ├── tesollo_delto_vtdex_env.py
        ├── tesollo_delto_vtdex_reorient_up_env.py
        ├── tesollo_delto_vtdex_bottle_cap_env.py
        ├── vtdex_encoder.py
        ├── vtdex_policy.py
        ├── vtdex_pretrained/
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
| `tesollo_delto_vtdex_env.py` | VTDexManip `reorient_down-vt_all_cls` 到 Isaac Lab + DG5F 的桌面重定向适配。 |
| `tesollo_delto_vtdex_reorient_up_env.py` | VTDexManip `reorient_up` 到 Isaac Lab + DG5F 的无桌面手内重定向适配。 |
| `tesollo_delto_vtdex_bottle_cap_env.py` | VTDexManip `bottle_cap` 到 Isaac Lab + DG5F 的固定瓶身、单关节瓶盖旋转适配。 |
| `tesollo_delto_vtdex_tomato_env.py` | 独立保留的 DG5F 番茄位姿调整任务，使用 471 维 VTDex policy observation。 |
| `vtdex_encoder.py` | 冻结的 VTDex VT-JointPretrain 与 V-CLIP 纯视觉编码器统一封装。 |
| `vtdex_policy.py` | 对齐上游 `ActorCriticVTEncoder` 的 40→128 本体状态分支和 384→128 VTDex CLS 分支。 |
| `vtdex_pretrained/` | 已复制到本项目的 VTDex 模型、checkpoint、`reorient_down/reorient_up/bottle_cap` 参考配置及三项任务涉及的物体资产。 |
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

蒸馏任务使用 `TesolloDeltoRlEnvCfg` 的动力学、reset 和奖励参数。student 的 54 维 `policy` observation 为：关节位置 20 维、YOLO mask 归一化图像中心 2 维、绕相机视角反方向（当前配置下等价于手部局部 Y 轴）的目标角误差 `[sin(2Δθ), cos(2Δθ)]` 2 维、二值触觉 10 维、上一时刻动作 20 维。二维位置以图像中心为 `(0, 0)`，范围约为 `[-1, 1]`，正 X 向右、正 Y 向下。

环境返回与现有 teacher checkpoint 匹配的 84 维 `critic` 真值状态（不含新增的 10 维触觉），奖励和终止条件也仍使用仿真真值。双角形式是因为分割 mask 的 PCA 主轴具有 180° 等价性；`Δθ` 与 `Δθ+π` 会映射到相同特征。Distill 的旋转奖励、成功判定和 teacher 目标也统一为绕 Y 轴、模 180° 的最近等价姿态，避免同一 student observation 对应互相冲突的 teacher 动作。仿真仅在首次有效测量时估计一次共享的相机角度偏置，后续 reset 不再使用姿态真值；真机使用时应将标定结果填入 `yolo_angle_offset_rad`。teacher checkpoint 应来自 `Tesollo-Delto-DG5F-Direct-v0` 的 84 维普通 RSL-RL 训练，例如 `logs/rsl_rl/TesolloDelto/<TEACHER_RUN>/model_*.pt`。如果不传 `--load_run` 和 `--checkpoint`，脚本会从 `logs/rsl_rl/TesolloDelto/` 下按名字选择最新匹配的 checkpoint。视觉蒸馏包含相机渲染和 YOLO 推理，建议先从 16 个环境开始，再按显存和吞吐量调整 `--num_envs`；训练和播放脚本会为 Distill 任务自动启用相机。

使用 RSL-RL 播放 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 16 --checkpoint <PATH_TO_CHECKPOINT>
```

播放时录制视频会默认使用配置中的 `student_camera`/`tiled_camera` 视角，并在画面中显示番茄物体坐标系和目标坐标系：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Distill-Direct-v0 --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT> --video --video_length 300
```

如果希望保留原来的 viewer 视角，可加 `--video_view viewer`。默认只显示坐标系，避免目标番茄 mesh 被 YOLO 当成第二个番茄；如果只是做可视化展示、希望视频里也显示目标番茄 mesh，可额外加 `--show_video_goal_mesh`。

蒸馏任务播放时可以固定测试目标角度，例如让目标姿态为绕相机视角反方向（当前等价于手部局部 Y 轴）旋转 45°：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Distill-Direct-v0 --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT> --goal_y_angle_deg 45
```

如果需要直接指定 `self.goal_rot`，可以传 `w x y z` 顺序的四元数：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Distill-Direct-v0 --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT> --goal_rot 0.9238795 0.0 0.3826834 0.0
```

使用 RL-Games 训练：

```bash
python scripts/rl_games/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1024 --headless
```

蒸馏、OpenAI 风格观测或视觉任务可将 `--task` 替换为上表中的对应任务名。

## VTDexManip 桌面物体位姿调整

该环境复现 VTDexManip 的 `reorient_down-vt_all_cls` 任务语义，同时将 Shadow Hand 替换为 DG5F 右手：

- 环境按上游顺序循环使用 apple、Rubik's cube、colored wood blocks、doorknob、potted meat can、cups、
  toy airplane、rubber duck、plum 和 master chef can；主物体/目标物体尺度分别为 `0.05`/`0.005`，颜色和
  `0.8` 物体摩擦系数也与上游一致。
- 桌面尺寸保持上游的 `1×1×0.6 m`。为使用户指定的侧视机位不落入原坐标系的桌体，整套场景统一下移
  `0.29 m`：桌面顶面、物体和目标 z 分别为 `0.31/0.38/0.35 m`，三者的相对几何与上游
  `0.60/0.67/0.64 m` 完全一致。
- VTDex policy camera 使用 `224×224`、约 45° HFOV，环境局部 eye 为 `(0.11, 0.36, 0.36)`，
  look-at 为 `(0.0, 0.02, 0.38)`，从侧面正对物体和 DG5F 指尖。
- 物体在桌面上以随机 yaw 重置，目标是相对初始姿态绕桌面法向旋转 180°。
- 策略以 60 Hz 输出 20 维绝对关节位置动作。
- actor 输入为 DG5F 关节位置 20 维、关节速度 20 维和冻结的 VTDex CLS 特征 384 维；前两者与 CLS
  分别投影为 128 维后再进入控制 MLP，与上游 `ActorCriticVTEncoder` 的融合拓扑一致。仿真物体位姿仅供
  critic、奖励与终止条件使用。
- VTDex 的 20 路二值触觉按“小指、无名指、中指、食指、拇指”的顺序，从末端、中节、近节到根部映射到 DG5F 20 个 link。
- 奖励保留上游任务的平面位置、姿态、物体 Z 角速度、指尖高度、动作惩罚和成功奖励；成功、平面漂移过大、倾倒、离开桌面或 600 步超时会结束 episode。

指定 checkpoint 作为冻结的视觉—触觉表征初始化使用。由于 Shadow Hand 与 DG5F 的运动学、关节限制和动作含义不同，
下游 PPO actor/critic 必须在 DG5F 环境中重新训练，不能直接复用 Shadow Hand 的控制策略权重。

### 联合模型与纯视觉模型对照

所有四个 VTDex 任务都支持统一的 `--model` 参数：

- `--model joint`（默认）：使用原项目 VT-JointPretrain
  `vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210`，输入 RGB 与 20 路二值触觉。
- `--model vision`：使用论文六任务平均结果最好的预训练纯视觉基线 V-CLIP（CLIP ViT-B/16），只输入 RGB，
  不调用触觉编码分支。论文报告其 Seen/Unseen 平均成功率为 `61.3%/49.4%`，高于原项目自训
  V-Pretrain 的 `54.0%/46.1%`。

VT-JointPretrain 输出 384 维 CLS，V-CLIP 输出 512 维图像特征；命令行会同步调整环境 observation 和
Down/Up/Bottle Cap 的可训练视觉投影层，因此两者都先投影到 128 维再与本体特征融合，和原项目各自的 policy
拓扑一致。Tomato 的 20 维触觉占位在纯视觉模式中始终填零。critic 的仿真特权状态、物理接触、奖励和终止条件
不变，从而只比较 actor 的预训练感知模态。纯视觉训练自动写入原实验名加 `_vision` 的目录，防止与 joint
checkpoint 混用。

例如训练纯视觉 Reorient Down：

```bash
python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --model vision --num_envs 10 --headless
```

其余任务只需替换 `--task`：

```text
Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0
Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0
Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0
```

播放纯视觉策略时也必须使用相同模式：

```bash
python scripts/rsl_rl/play.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --model vision --num_envs 1 --checkpoint <VISION_CHECKPOINT>
```

项目内已复制原项目的 CLIP ViT-B/16 模型代码和 checkpoint，默认不依赖外部 VTDexManip 路径。若要把同布局的
VTDexManip 模型目录放到别处，可设置 `TESOLLO_VTDEX_VISION_REPO_ROOT`；普通联合模型仍使用原来的
`TESOLLO_VTDEX_REPO_ROOT` 与 `TESOLLO_VTDEX_MODEL_ID`。这些覆盖对四个 VTDex 任务都有效。

先做单环境相机与接触 smoke test：

```bash
python scripts/zero_agent.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --num_envs 1 --headless --max_steps 10 \
  --save_camera_frame outputs/vtdex_reorient_down_camera.png
```

训练：

```bash
python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --num_envs 10 --headless
```

播放 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT>
```

屏蔽策略触觉输入做消融测试（物理接触和原始触觉力日志仍保留）：

```bash
python scripts/rsl_rl/play.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Down-Direct-v0 \
  --num_envs 10 \
  --checkpoint logs/rsl_rl/TesolloDelto_vtdex/2026-07-27_21-52-00/model_5099.pt \
  --mask_vtdex_tactile --max_steps 600
```

此开关会将送入 VTDex encoder 的 20 路触觉全部置零，但不会关闭 ContactSensor，也不会改变接触动力学、
奖励或终止条件。因而 `vtdex_tactile_active_ratio` 仍表示物理上测得的真实触觉，
`vtdex_policy_tactile_active_ratio` 则应始终为 `0`。

预训练 checkpoint 较大且由 Git LFS 管理。克隆项目后若该文件仍是 LFS pointer，需要执行 `git lfs pull`。首次训练建议从
10 个环境开始，以覆盖一轮完整物体集合；扩大规模时建议使用 10 的倍数。

### VTDexManip 瓶盖旋转

`Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0` 保留上游 `bottle_cap-vt_all_cls` 的任务结构：

- 按原顺序循环使用 10 个瓶子；相关 URDF/OBJ、原环境代码和 task/PPO YAML 均已复制到
  `vtdex_pretrained/`，运行时不依赖外部 VTDexManip 仓库。
- 瓶身固定且关闭重力，瓶盖是绕世界 Z 轴、范围 `0～6.28 rad` 的被动 revolute joint；策略需要通过
  DG5F 接触推动瓶盖正向旋转。
- DG5F 的掌心朝下姿态、X/Y 对齐、关节限位和相机 eye 沿用 Reorient Down；瓶盖任务单独使用环绕瓶盖的
  轻预抓取手型，并将手根基准高度设为 `0.55 m`。其余 9 个瓶子的手高只保留上游数据中的相对差值，
  相机 look-at 抬高到瓶盖区域。
- Bottle Cap 单独把 DG5F 仿真力矩上限从共享配置的 `0.1 N·m` 提高到 `0.8 N·m`，接近上游
  Shadow Hand 执行器的最低量级；动作移动平均系数由 `0.2` 提高为 `0.5`，瓶盖摩擦系数显式设为
  `1.0`。这些设置只对 Bottle Cap 生效，不会改变 Down、Up 或 Tomato。
- actor 输入为 `40` 维 DG5F 本体状态 + 冻结预训练特征（`joint` 为 `384` 维，`vision` 为
  `512` 维）；critic 额外使用瓶盖角度/速度、原始 20 路触觉和动作。
- 正向瓶盖速度只有在至少一个 DG5F 触觉 link 确实接触瓶盖时才产生奖励；碰到瓶身或桌面不会通过该
  门控。奖励速度由相邻 60 Hz 控制步的实际瓶盖角度差计算，并忽略小于 `1e-4 rad` 的求解器抖动；
  PhysX 原始关节速度仅用于诊断，避免策略通过关节限位附近的速度尖峰刷奖励。奖励权重继续采用上游
  数值：累计角度 `0.5`、角速度 `1.0`、指尖高度 shaping `0.5`、成功奖励 `5.0`。瓶盖角度超过
  `6 rad` 记为成功，超过 `6.15 rad` 结束 episode，最多 500 个 60 Hz 控制步。
- `debug_visualization=False`，策略 RGB 中没有手部、瓶子或目标坐标轴；DG5F 独立白色硅胶 tip visual
  仍按 Reorient Down 的方式隐藏，碰撞 link 与触觉不受影响。

上述动力学和初始手型已经改变了 PPO 所面对的 MDP，因此不要续训修改前的 Bottle Cap checkpoint；
请新建一次训练。TensorBoard 中重点观察 `bottle_cap_angle_rad`、`bottle_cap_angle_max_rad`、
`bottle_cap_velocity_rad_s`、`bottle_cap_raw_joint_velocity_rad_s`、
`bottle_cap_positive_rotation_ratio`、`bottle_cap_contact_count` 和 `bottle_cap_success_rate`：
前 3 项能够区分“偶尔碰一下”与“持续正向拧动”。

先保存一帧策略相机图像并做零动作检查：

```bash
python scripts/zero_agent.py \
  --task Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0 \
  --num_envs 1 --headless --max_steps 10 \
  --save_camera_frame outputs/vtdex_bottle_cap_camera.png
```

训练：

```bash
python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0 \
  --num_envs 10 --headless --model joint
```

纯视觉对照训练时只需将最后一项改成 `--model vision`。

播放：

```bash
python scripts/rsl_rl/play.py \
  --task Tesollo-Delto-DG5F-VTDex-Bottle-Cap-Direct-v0 \
  --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT> --model joint
```

播放纯视觉 checkpoint 时也必须对应使用 `--model vision`。

### VTDexManip 手内 Reorient Up

`Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0` 独立复现上游
`In-hand Reorientation / reorient_up`，不会覆盖 Tomato 或 Reorient Down：

- 场景中没有桌面或地面，DG5F 手根固定，物体由手指在重力下保持；
- 按上游顺序循环 cups-c、tennis ball、strawberry、cracker box、apple、Rubik's cube、
  colored wood blocks、potted meat can、lemon 和 rubber duck，尺度均为 `0.04`；
- 使用 DG5F 已验证的手内物体相对位姿与抓持初始关节值，同时保留完整物理关节范围供策略重定向；
- 策略仍以 60 Hz 输出 20 维绝对关节位置，actor 输入仍为 40 维本体状态和 384 维冻结 VTDex CLS；
- 原代码显示的目标姿态为相对初始 yaw `+90°`，但实际奖励以相对初始 yaw 的绝对值大于
  `3 rad`（约 `172°`）判成功；本实现按实际奖励语义保留这一行为；
- 原奖励中的水平漂移、竖直掉落、物体倾斜、Z 角速度、五指高度和动作惩罚均保留；
- policy camera 为 `224×224` 侧视相机，debug 坐标轴、相机 debug visual 和脱离的白色硅胶指尖 visual
  均关闭；环境间距为 `2.0 m`，避免并行环境串景。

先检查单环境初始抓持和相机：

```bash
python scripts/zero_agent.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0 \
  --num_envs 1 --headless --max_steps 10 \
  --save_camera_frame outputs/vtdex_reorient_up_camera.png
```

覆盖一轮十类物体训练：

```bash
python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0 \
  --num_envs 10 --headless
```

播放 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task Tesollo-Delto-DG5F-VTDex-Reorient-Up-Direct-v0 \
  --num_envs 1 --checkpoint <PATH_TO_CHECKPOINT>
```

若要使用自行预训练的 DG5F VTDex encoder，可在启动前设置
`TESOLLO_VTDEX_REPO_ROOT` 和 `TESOLLO_VTDEX_MODEL_ID`；未设置时使用项目内置官方模型。

### 保留的番茄 VTDex 任务

之前的番茄位姿调整代码已独立保存在 `tesollo_delto_vtdex_tomato_env.py`，没有被十物体场景覆盖。该任务继续使用
`robots/tomato.usd`、两个表面方向点、随机目标位置/旋转和 471 维 actor observation。旧任务名和显式任务名均可使用：

```bash
python scripts/zero_agent.py \
  --task Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0 \
  --num_envs 1 --headless --max_steps 10 \
  --save_camera_frame outputs/vtdex_tomato_camera.png

python scripts/rsl_rl/train.py \
  --task Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0 \
  --num_envs 16 --headless
```

### 用 DG5F 策略数据重新做 VT-JointPretrain

项目现已包含一条不覆盖原任务和官方权重的完整流程：用
`Tesollo-Delto-DG5F-Direct-v0` 的已训练策略采集 224×224 RGB 与 20 路
DG5F link 接触，训练 VT-JointPretrain，再通过环境变量把新 encoder 接入
`Tesollo-Delto-DG5F-VTDex-Tomato-Direct-v0`。

采集时会强制关闭手、物体和目标的调试坐标轴，并默认隐藏目标番茄 mesh。
相机、触觉顺序和番茄表面方向点与 Tomato 下游任务共用同一份常量，避免
训练/使用时的输入协议漂移。详细命令、数据检查、训练、模型校验和切换方式见
[DG5F VTDex 预训练说明](scripts/vtdex_pretraining/README.md)。

Isaac Sim 5.1 的 `isaacsim-kernel` 要求 `numpy==1.26.0`；本项目的安装依赖已固定该版本。如果相机初始化出现
`Unable to write from unknown dtype`，请确认当前激活环境不是 NumPy 2.x。

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

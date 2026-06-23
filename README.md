# Tesollo Delto RL

这是一个基于 Isaac Lab 的强化学习扩展项目，用于在仿真中训练 Tesollo/Delto DG5F 右手机器人完成灵巧手操作任务。项目当前采用 Direct RL 环境，主要面向手内物体操作，并提供 RSL-RL、RL-Games 和视觉观测相关的脚本与配置。

## 当前状态

- 机器人资产已切换为 `source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/dg5f_right.usd`。
- DG5F 右手配置位于 `tasks/direct/tesollo_delto_rl/delto_cfg.py`，包含 USD 加载、初始姿态、关节初始值和 actuator 设置。
- 主环境配置位于 `tasks/direct/tesollo_delto_rl/tesollo_delto_rl_env_cfg.py`。
- 普通策略观测维度为 `149`，动作维度为 `20`。
- OpenAI 风格观测维度为 `42`，critic state 维度为 `179`。
- 视觉环境配置位于 `tasks/direct/tesollo_delto_rl/tesollo_delto_rl_vision_env.py`，policy observation 为 `156 + 27`，critic state 为 `179 + 27`。
- Gym 任务注册入口已整理到 `tasks/direct/tesollo_delto_rl/__init__.py`，任务名前缀为 `Tesollo-Delto-DG5F`。

## 已注册任务

| 任务名 | 环境 | 配置 |
| --- | --- | --- |
| `Tesollo-Delto-DG5F-Direct-v0` | `TesolloDeltoRlEnv` | `TesolloDeltoRlEnvCfg` |
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
| `agents/rsl_rl_ppo_cfg.py` | RSL-RL PPO runner、policy 和算法参数。 |
| `agents/rl_games_ppo_cfg.yaml` | RL-Games PPO 配置。 |
| `robots/dg5f_right.usd` | DG5F 右手机器人主 USD。 |

## DG5F 关节与指尖

当前环境控制 20 个 DG5F revolute joints：

```text
rj_dg_1_1, rj_dg_1_2, rj_dg_1_3, rj_dg_1_4
rj_dg_2_1, rj_dg_2_2, rj_dg_2_3, rj_dg_2_4
rj_dg_3_1, rj_dg_3_2, rj_dg_3_3, rj_dg_3_4
rj_dg_4_1, rj_dg_4_2, rj_dg_4_3, rj_dg_4_4
rj_dg_5_1, rj_dg_5_2, rj_dg_5_3, rj_dg_5_4
```

用于 fingertip observation/contact force 的 5 个刚体为：

```text
rl_dg_1_tip
rl_dg_2_tip
rl_dg_3_tip
rl_dg_4_tip
rl_dg_5_tip
```

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
python scripts/rsl_rl/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1024 --headless
```

使用 RSL-RL 播放 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 16 --checkpoint <PATH_TO_CHECKPOINT>
```

使用 RL-Games 训练：

```bash
python scripts/rl_games/train.py --task Tesollo-Delto-DG5F-Direct-v0 --num_envs 1024 --headless
```

OpenAI 风格观测或视觉任务可将 `--task` 替换为上表中的对应任务名。

## 日志与输出

RSL-RL 默认日志目录：

```text
logs/rsl_rl/TesolloDelto/
```

OpenAI-FF 和视觉任务分别使用 `TesolloDelto_openai_ff`、`TesolloDelto_vision` 作为实验目录。训练过程中会保存 runner 配置、环境配置和 checkpoint。`play.py` 会从对应目录查找最新 checkpoint，或使用 `--checkpoint` 显式指定路径。

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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 200
    experiment_name = "delto_walnut"
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependent_std=False,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 价值函数损失系数
        use_clipped_value_loss=True,  # 启用价值函数裁剪
        clip_param=0.2,               # PPO裁剪参数
        entropy_coef=0,               # 熵正则化系数
        num_learning_epochs=5,        # 每次更新的训练轮数
        num_mini_batches=4,           # 小批量数量
        learning_rate=5.0e-4,         # 初始学习率
        schedule="adaptive",          # 自适应学习率调度
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE参数
        desired_kl=0.016,             # 目标KL散度
        max_grad_norm=1.0,            # 梯度裁剪阈值
    )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the Tesollo Delto DG5F direct RL tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="Tesollo-Delto-DG5F-Direct-v0",
    entry_point=f"{__name__}.tesollo_delto_rl_env:TesolloDeltoRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tesollo_delto_rl_env_cfg:TesolloDeltoRlEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TesolloDeltoPPORunnerCfg",
    },
)

gym.register(
    id="Tesollo-Delto-DG5F-OpenAI-FF-Direct-v0",
    entry_point=f"{__name__}.tesollo_delto_rl_env:TesolloDeltoRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tesollo_delto_rl_env_cfg:TesolloDeltoRlOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TesolloDeltoAsymFFPPORunnerCfg",
    },
)

gym.register(
    id="Tesollo-Delto-DG5F-OpenAI-LSTM-Direct-v0",
    entry_point=f"{__name__}.tesollo_delto_rl_env:TesolloDeltoRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tesollo_delto_rl_env_cfg:TesolloDeltoRlOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

gym.register(
    id="Tesollo-Delto-DG5F-Vision-Direct-v0",
    entry_point=f"{__name__}.tesollo_delto_rl_vision_env:TesolloDeltoRlVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tesollo_delto_rl_vision_env:TesolloDeltoRlVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TesolloDeltoVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

gym.register(
    id="Tesollo-Delto-DG5F-Vision-Direct-Play-v0",
    entry_point=f"{__name__}.tesollo_delto_rl_vision_env:TesolloDeltoRlVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tesollo_delto_rl_vision_env:TesolloDeltoRlVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TesolloDeltoVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

import gymnasium as gym

from . import agents

gym.register(
    id="Template-DeltoWalnut-curriculum-Direct-v0",
    entry_point=f"{__name__}.delto_walnut_curriculum_env:DeltoWalnutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.delto_walnut_curriculum_env_cfg:DeltoWalnutEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

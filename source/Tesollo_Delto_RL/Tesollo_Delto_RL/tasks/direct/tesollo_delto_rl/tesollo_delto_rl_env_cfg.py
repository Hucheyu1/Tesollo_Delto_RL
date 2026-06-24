# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .delto_cfg import TESOLLO_CFG


@configclass
class EventCfg:
    """Domain randomization config."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # DG5F / Tesollo 没有 ShadowHand 的 fixed tendon，不能保留 tendon randomization
    robot_tendon_properties = None

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class TesolloDeltoRlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 5.0

    # action / observation
    action_space = 20
    observation_space = 149
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    action_scale = 0.5

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.3,
            dynamic_friction=0.3,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = TESOLLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 20 个可控关节，顺序建议和实机动作顺序保持一致
    hand_joint_names = [
        "rj_dg_1_1",
        "rj_dg_2_1",
        "rj_dg_3_1",
        "rj_dg_4_1",
        "rj_dg_5_1",
        "rj_dg_1_2",
        "rj_dg_2_2",
        "rj_dg_3_2",
        "rj_dg_4_2",
        "rj_dg_5_2",
        "rj_dg_1_3",
        "rj_dg_2_3",
        "rj_dg_3_3",
        "rj_dg_4_3",
        "rj_dg_5_3",
        "rj_dg_1_4",
        "rj_dg_2_4",
        "rj_dg_3_4",
        "rj_dg_4_4",
        "rj_dg_5_4",
    ]
    # action 顺序就使用 hand_joint_names 顺序
    actuated_joint_names = hand_joint_names

    # 初始关节位置，单位：rad
    hand_position = [
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.7,
        0.5,
        0.5,
        0.5,
        0.0,
        0.7,
        0.7,
        0.7,
        0.7,
        1.57,
        0.3,
        1.0,
        1.0,
        1.0,
        1.57,
    ]
    # 关节上限，单位：deg
    hand_upper_limits = [50, 35, 30, 24, 60, 0, 115, 112, 109, 35, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
    # 关节下限，单位：deg
    hand_lower_limits = [-22, -24, -30, -35, 0, -150, 0, 0, 0, -24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 是否使用上面手动配置的关节初始位置和上下限
    use_manual_joint_cfg = True
    # 这个名字必须和 USD 里的 body name 一致
    fingertip_body_names = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]

    # object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/root/gpufree-data/Tesollo_Delto_RL/source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/tomato.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.107, 0.0, 0.375),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.75,
        replicate_physics=True,
    )

    # 重置配置
    # reset_position_noise = 0.01  # 位置重置噪声
    # reset_dof_pos_noise = 0.2  # 关节位置重置噪声
    # reset_dof_vel_noise = 0.0  # 关节速度重置噪声
    reset_position_noise = 0  # 位置重置噪声
    reset_dof_pos_noise = 0  # 关节位置重置噪声
    reset_dof_vel_noise = 0  # 关节速度重置噪声

    # 手中目标配置：在物体初始局部位置基础上，沿 DG5F 手 root 局部 +X 稍微推向抓取空间
    in_hand_local_offset = (0.025, 0.0, 0.0)

    # 奖励函数配置
    dist_reward_scale = -10.0  # 距离奖励缩放因子
    rot_reward_scale = 1.0  # 旋转奖励缩放因子
    rot_eps = 0.1  # 旋转奖励的epsilon值
    action_penalty_scale = -0.0002  # 动作惩罚缩放因子
    reach_goal_bonus = 250.0  # 达成目标奖励
    fall_penalty = -50.0  # 物体掉落惩罚
    fall_dist = 0.15  # 掉落距离阈值
    vel_obs_scale = 0.2  # 速度观测缩放因子
    success_tolerance = 0.1  # 成功容忍度
    max_consecutive_success = 0  # 最大连续成功次数
    av_factor = 0.1  # 平均因子
    act_moving_average = 1.0  # 动作移动平均因子
    force_torque_obs_scale = 10.0  # 力/力矩观测缩放因子

    # 目标物体可视化 marker：使用 tomato 作为目标姿态显示
    # goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "goal": sim_utils.UsdFileCfg(
    #             usd_path="/home/amlrobotics/hcy_ws/Tesollo_Delto_RL_main/source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/tomato.usd",
    #             scale=(1.0, 1.0, 1.0),
    #         )
    #     },
    # )

    # viewer camera, 近距离看手和物体
    # viewer = ViewerCfg(
    #     eye=(0.35, -0.75, 0.55),
    #     lookat=(0.10, 0.00, 0.20),
    #     origin_type="env",
    #     env_index=0,
    # )


@configclass
class TesolloDeltoRlOpenAIEnvCfg(TesolloDeltoRlEnvCfg):
    # env
    decimation = 3
    episode_length_s = 8.0

    action_space = 20
    observation_space = 42
    state_space = 179
    asymmetric_obs = True
    obs_type = "openai"

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.3,
            dynamic_friction=0.3,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = -50.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0

    events: EventCfg = EventCfg()

    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

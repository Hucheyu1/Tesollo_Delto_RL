from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# from isaaclab.utils.math import quat_from_euler_xyz


TESOLLO_CFG = ArticulationCfg(
    # prim_path = "/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/gpufree-data/Tesollo_Delto_RL/source/Tesollo_Delto_RL/Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/dg5f_right.usd",
        activate_contact_sensors=True,
        # 刚体属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1000.0,
        ),
        # 关节属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        # 关节驱动属性
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(0.0, 0.92388, 0.0, 0.38268),  # wxyz  水平向下倾斜45度
        # rot=(0.0, 0.86603, 0.0, 0.5),  #wxyz  水平向下倾斜30度
        # rot=(1., 0., 0., 0.),  #wxyz
        joint_pos={
            "rj_dg_1_1": 0.2,
            "rj_dg_2_1": -0.1,
            "rj_dg_3_1": 0.0,
            "rj_dg_4_1": 0.1,
            "rj_dg_5_1": 0.0,
            "rj_dg_1_2": -0.35,
            "rj_dg_1_3": 1.2,
            #    "rj_dg_1_4": 0.1,
            "rj_dg_1_4": 0.0,
            "rj_dg_(2|3|4)_2": 0.3,
            #    "rj_dg_(2|3|4)_2": 0.,
            "rj_dg_(2|3|4)_3": 1.0,
            "rj_dg_(2|3|4)_4": 1.2,
            #    "rj_dg_5_2": 0.0,
            "rj_dg_5_2": 0.2,
            "rj_dg_5_3": 1.3,
            "rj_dg_5_4": 1.57,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rj_dg_.*"],
            effort_limit=None,
            velocity_limit=None,
            effort_limit_sim=0.1,
            velocity_limit_sim=None,
            stiffness=2.0,
            damping=0.1,
            armature=None,
            friction=0.01,
            dynamic_friction=None,
            viscous_friction=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Pepper-bottle single-object rotation task for DG5F / Tesollo.

This file is intentionally modeled after ``TesolloDeltoVTDexEnv`` but removes the
multi-object VTDex source distribution and imports the pepper bottle as an Articulation.  The environment spawns exactly one USD
object:

    /home/amlrobotics/hcy_ws/Tesollo_Delto_RL_bak/source/Tesollo_Delto_RL/
    Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/peper_bottle_combined_v4.usd

Task objective:
    Rotate the pepper bottle on the table by ``target_yaw_delta_rad`` around the
    world/table Z axis while keeping it near the hand/table center.
    Robot/object/camera heights are expressed relative to ``table_top_z``.

Actor observation:
    DG5F proprioception + frozen VTDex RGB/tactile embedding, matching the
    VTDex-style environment.  The actor does not receive simulator object pose.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform

from .delto_cfg import TESOLLO_CFG
from .tesollo_delto_rl_env import TesolloDeltoRlEnv, rotation_distance, unscale
from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg
from .vtdex_encoder import (
    VTDEX_JOINT_MODEL_ID,
    VTDEX_VISION_MODEL_ID,
    VTDexPretrainedEncoder,
)


_VTDEx_ROOT = Path(__file__).resolve().parent / "vtdex_pretrained"
_PEPPER_BOTTLE_USD = (
    "/home/amlrobotics/hcy_ws/Tesollo_Delto_RL_bak/source/Tesollo_Delto_RL/"
    "Tesollo_Delto_RL/tasks/direct/tesollo_delto_rl/robots/peper_bottle_combined_v4.usd"
)
_BASE_ENV_CFG = TesolloDeltoRlEnvCfg()


@configclass
class TesolloDeltoPepperBottleRotateEnvCfg(TesolloDeltoRlEnvCfg):
    """Single pepper-bottle tabletop rotation task using an Articulation object.

    This is a simplified single-object version of the VTDex table task:
    - no VTDex object cycling;
    - no goal object collision in the workspace by default;
    - reward / termination focus on yaw rotation of the pepper bottle.
    """

    # Keep VTDex-like 60 Hz control: 120 Hz physics / decimation 2.
    decimation = 2
    episode_length_s = 10.0

    # Single repeated USD object can use replicated physics.  Increase num_envs
    # freely for PPO after verifying GPU memory.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=1.0, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
    )

    # -------------------------------------------------------------------------
    # Table-relative geometry
    # -------------------------------------------------------------------------
    # The user-provided DG5F right-hand pose was measured with the tabletop height
    # included.  Keep the Z coordinates expressed as table_top_z + offset so that
    # changing table_top_z later does not silently break the grasp geometry.
    table_top_z = 0.31
    pepper_root_z_above_table = 0
    right_hand_z_above_table = 0.65 - table_top_z

    # DG5F right-hand pose from the user's delto_peper configuration.
    # Keep activate_contact_sensors=True because this VTDex-style environment
    # still uses ContactSensor-based tactile observations.
    robot_cfg = TESOLLO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=TESOLLO_CFG.spawn.replace(
            activate_contact_sensors=True,
            rigid_props=TESOLLO_CFG.spawn.rigid_props.replace(max_depenetration_velocity=2.0),
            articulation_props=TESOLLO_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=TESOLLO_CFG.init_state.replace(
            pos=(-0.13, 0.022, table_top_z + right_hand_z_above_table),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),
            joint_pos={
                "rj_dg_(1|3)_1": 0.0,
                "rj_dg_1_2": -1.57,
                "rj_dg_1_3": -0.8,
                "rj_dg_1_4": 1.4,
                "rj_dg_2_1": -0.2,
                "rj_dg_4_1": 0.2,
                "rj_dg_(2|3|4)_2": 1.3,
                "rj_dg_(2|3|4)_3": 0.2,
                "rj_dg_(2|3|4)_4": 0.2,
                "rj_dg_5_1": 0.3,
                "rj_dg_5_2": 1.57,
                "rj_dg_5_3": 1.57,
                "rj_dg_5_4": 0.0,
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

    vtdex_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_[1-4]",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )
    vtdex_contact_threshold = 0.01

    # Table geometry.
    table_cfg = _BASE_ENV_CFG.object_cfg.replace(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.60),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.001,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=_BASE_ENV_CFG.object_cfg.init_state.replace(pos=(0.0, 0.0, 0.01)),
    )

    # The only manipulated training object.
    # The USD contains an internal RevoluteJoint, so it must be an ArticulationCfg,
    # not a RigidObjectCfg. If scale is wrong, tune spawn.scale here.
    object_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PEPPER_BOTTLE_USD,
            # Keep the original USD scale by default.  Change this if needed:
            # scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=2.0,
                max_linear_velocity=5.0,
                max_angular_velocity=720.0,
            ),
            # 关键：这个 USD 内部带 RevoluteJoint，因此必须作为 articulation 导入。
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.001,
            ),
            semantic_tags=[("class", "object")],
        ),
        # Bottle root height is table-relative.  With the default table_top_z=0.31
        # this evaluates to z=0.38, matching the previous VTDex-style tabletop setup.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.02, table_top_z + pepper_root_z_above_table),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        # 被动物体关节：不给 motor 力矩，只让 PhysX articulation 约束处理内部 RevoluteJoint。
        actuators={
            "passive_object_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=0.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # No GoalObject USD is spawned for this task.
    # The pepper-bottle USD contains an internal RevoluteJoint, and spawning a
    # second static / kinematic copy as GoalObject triggers PhysX errors such as:
    # "cannot create a joint between static bodies".  The target is represented
    # only by goal_pos / goal_rot; use debug frame markers for visualization.

    # DG5F pregrasp, ordered by actuated_joint_names.
    # Same initial joint pose as robot_cfg.init_state.joint_pos above, expanded in
    # the environment's layer-major actuated_joint_names order:
    # [1_1..5_1, 1_2..5_2, 1_3..5_3, 1_4..5_4].
    hand_position = [
        0.0,
        -0.2,
        0.0,
        0.2,
        0.3,
        -1.57,
        1.3,
        1.3,
        1.3,
        1.57,
        -0.8,
        0.2,
        0.2,
        0.2,
        1.57,
        1.4,
        0.2,
        0.2,
        0.2,
        0.0,
    ]

    hand_lower_limits = [
        -22,
        -20,
        -30,
        -32,
        0,
        -154,
        0,
        0,
        0,
        -15,
        -90,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    hand_upper_limits = [
        77,
        31,
        30,
        15,
        60,
        0,
        115,
        115,
        109,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
    ]

    hide_dg5f_tip_visuals = True

    # Dedicated RGB camera for frozen VTDex-style encoder.
    vtdex_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/VTDexCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, -0.4, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0,
            focus_distance=0.50,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=224,
        height=224,
        update_latest_camera_pose=True,
        debug_vis=False,
    )
    # Keep the camera table-relative as well.
    vtdex_camera_eye_local = (0.3, -0.3, table_top_z + 0.50)
    vtdex_camera_target_local = (0.0, 0.02, table_top_z + 0.20)

    # Actor: DG5F qpos(20) + qvel(20) + frozen representation.  Joint mode
    # uses the 384-D VT-JointPretrain representation; vision mode uses the
    # 512-D V-CLIP representation and is selected by ``--model vision``.
    observation_space = 424
    # Critic: inherited full simulator state plus 20 tactile values.  Keep this
    # as 104 to match the VTDex-style variant unless compute_full_state changes.
    state_space = 104
    asymmetric_obs = True
    obs_type = "vtdex"

    vtdex_model_mode = "joint"
    vtdex_repo_root = os.environ.get("TESOLLO_VTDEX_REPO_ROOT", str(_VTDEx_ROOT))
    vtdex_model_id = os.environ.get("TESOLLO_VTDEX_MODEL_ID", VTDEX_JOINT_MODEL_ID)
    vtdex_vision_repo_root = os.environ.get("TESOLLO_VTDEX_VISION_REPO_ROOT", str(_VTDEx_ROOT))
    vtdex_vision_model_id = VTDEX_VISION_MODEL_ID
    vtdex_embedding_dim = 384
    vtdex_vision_embedding_dim = 512
    vtdex_mask_tactile_input = False
    vtdex_tactile_body_names = (
        "rl_dg_5_4",
        "rl_dg_4_4",
        "rl_dg_3_4",
        "rl_dg_2_4",
        "rl_dg_1_4",
        "rl_dg_5_3",
        "rl_dg_4_3",
        "rl_dg_3_3",
        "rl_dg_2_3",
        "rl_dg_1_3",
        "rl_dg_5_2",
        "rl_dg_4_2",
        "rl_dg_3_2",
        "rl_dg_2_2",
        "rl_dg_1_2",
        "rl_dg_5_1",
        "rl_dg_4_1",
        "rl_dg_3_1",
        "rl_dg_2_1",
        "rl_dg_1_1",
    )
    fingertip_body_names = list(vtdex_tactile_body_names)
    vtdex_tactile_indices = tuple(range(20))

    # Reset / target.  This trains a pure rotation target: rotate the current
    # object yaw by target_yaw_delta_rad around table/world Z.
    fix_object_initial_pose = True
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
    target_yaw_delta_rad = 3.141592653589793  # 180 degrees

    # Rotation-focused reward / termination.
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.05
    success_tolerance = 0.1
    table_tilt_limit_deg = 60.0
    vel_reward_scale = 1.0
    fingertip_distance_reward_scale = 0.25
    action_scale = 1.0
    act_moving_average = 0.2

    # Default: no debug coordinate frames / goal mesh during training.
    debug_visualization = False


class TesolloDeltoPepperBottleRotateEnv(TesolloDeltoRlEnv):
    """Single-object tabletop rotation environment for the pepper bottle."""

    cfg: TesolloDeltoPepperBottleRotateEnvCfg

    def __init__(self, cfg: TesolloDeltoPepperBottleRotateEnvCfg, render_mode: str | None = None, **kwargs):
        if float(cfg.target_yaw_delta_rad) == 0.0:
            raise ValueError("target_yaw_delta_rad must be non-zero")
        if not 0.0 < float(cfg.table_tilt_limit_deg) < 90.0:
            raise ValueError("table_tilt_limit_deg must be between 0 and 90 degrees")
        if not Path(_PEPPER_BOTTLE_USD).is_file():
            raise FileNotFoundError(f"Pepper bottle USD not found: {_PEPPER_BOTTLE_USD}")

        super().__init__(cfg, render_mode, **kwargs)
        self._reward_fingertip_body_ids = torch.tensor(
            [self.hand.body_names.index(name) for name in self.cfg.vtdex_tactile_body_names[:5]],
            dtype=torch.long,
            device=self.device,
        )
        contact_body_ids, contact_body_names = self._vtdex_contact_sensor.find_bodies(
            list(self.cfg.vtdex_tactile_body_names), preserve_order=True
        )
        if contact_body_names != list(self.cfg.vtdex_tactile_body_names):
            raise RuntimeError(
                "VTDex ContactSensor body order does not match tactile token order: "
                f"expected={list(self.cfg.vtdex_tactile_body_names)}, resolved={contact_body_names}"
            )
        if len(contact_body_ids) != 20 or len(set(contact_body_ids)) != 20:
            raise RuntimeError(f"ContactSensor must resolve exactly 20 unique DG5F links; got ids={contact_body_ids}")
        self._vtdex_contact_body_ids = torch.tensor(contact_body_ids, dtype=torch.long, device=self.device)
        print(f"[INFO]: PepperBottleArticulationRotate tactile ContactSensor mapping: {contact_body_names}")

        encoder_repo_root = (
            self.cfg.vtdex_repo_root
            if self.cfg.vtdex_model_mode == "joint"
            else self.cfg.vtdex_vision_repo_root
        )
        encoder_model_id = (
            self.cfg.vtdex_model_id
            if self.cfg.vtdex_model_mode == "joint"
            else self.cfg.vtdex_vision_model_id
        )
        self.vtdex_encoder = VTDexPretrainedEncoder(
            model_mode=self.cfg.vtdex_model_mode,
            repo_root=encoder_repo_root,
            model_id=encoder_model_id,
            device=self.device,
            tactile_indices=tuple(self.cfg.vtdex_tactile_indices),
        )
        print(
            "[INFO]: PepperBottle frozen encoder: "
            f"mode={self.cfg.vtdex_model_mode}, model_id={encoder_model_id}, root={encoder_repo_root}"
        )
        if self.vtdex_encoder.embedding_dim != self.cfg.vtdex_embedding_dim:
            raise ValueError(
                "Configured VTDex embedding dimension does not match checkpoint: "
                f"{self.cfg.vtdex_embedding_dim} != {self.vtdex_encoder.embedding_dim}"
            )
        self.vtdex_embeddings = torch.zeros(
            (self.num_envs, self.cfg.vtdex_embedding_dim), dtype=torch.float32, device=self.device
        )
        self._configure_vtdex_camera_pose()
        self._hide_dg5f_tip_visuals()
        self._hide_goal_marker()

    def _setup_scene(self):
        print("[INFO]: PepperBottleArticulationRotate _setup_scene: spawning Robot/Object(Articulation)/Table only; NO GoalObject is created.")
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self._vtdex_camera = Camera(self.cfg.vtdex_camera)
        self._vtdex_contact_sensor = ContactSensor(self.cfg.vtdex_contact_sensor)

        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["vtdex_camera"] = self._vtdex_camera
        self.scene.sensors["vtdex_contact"] = self._vtdex_contact_sensor

        # 关键：必须 clone environments，否则 scene.env_origins 是 None
        self.scene.clone_environments(copy_from_source=False)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_vtdex_camera_pose(self):
        """Aim every tiled camera using environment-local eye/target points."""

        eye_local = torch.tensor(self.cfg.vtdex_camera_eye_local, dtype=torch.float32, device=self.device).view(1, 3)
        target_local = torch.tensor(self.cfg.vtdex_camera_target_local, dtype=torch.float32, device=self.device).view(
            1, 3
        )
        if torch.linalg.vector_norm(eye_local - target_local).item() < 1.0e-4:
            raise ValueError("vtdex_camera_eye_local and target must be different points")
        eyes_w = self.scene.env_origins + eye_local
        targets_w = self.scene.env_origins + target_local
        self._vtdex_camera.set_world_poses_from_view(eyes_w, targets_w)

    def _hide_dg5f_tip_visuals(self):
        """Hide DG5F detached silicone visuals without changing physics."""

        if not self.cfg.hide_dg5f_tip_visuals:
            return
        for env_path in self.scene.env_prim_paths:
            for finger_index in range(1, 6):
                visual_path = f"{env_path}/Robot/rl_dg_{finger_index}_tip/visuals"
                visual_prim = self.scene.stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    # Some USD variants do not include detached tip visuals.  Do
                    # not fail the pepper-bottle environment because of that.
                    continue
                UsdGeom.Imageable(visual_prim).MakeInvisible()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        rgb = self._vtdex_camera.data.output["rgb"]
        raw_tactile = self.fingertip_force_binary_results.to(dtype=torch.float32)
        policy_tactile = (
            raw_tactile
            if self.cfg.vtdex_model_mode == "joint" and not self.cfg.vtdex_mask_tactile_input
            else torch.zeros_like(raw_tactile)
        )
        self.vtdex_policy_tactile_input = policy_tactile
        self.vtdex_embeddings = self.vtdex_encoder(rgb, policy_tactile)
        self.extras.setdefault("log", {})["vtdex_policy_tactile_active_ratio"] = policy_tactile.mean()

        policy_obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.vtdex_embeddings,
            ),
            dim=-1,
        )
        critic_obs = self.compute_full_state()

        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"PepperBottleArticulationRotate policy observation has {policy_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.observation_space}"
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"PepperBottleArticulationRotate critic observation has {critic_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Map normalized actions to absolute position targets."""

        self.raw_actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = self.raw_actions.clone()
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        scaled_actions = torch.clamp(float(self.cfg.action_scale) * self.raw_actions, -1.0, 1.0)
        self.target_pos[:, self.actuated_dof_indices] = 0.5 * (scaled_actions + 1.0) * (upper - lower) + lower

    def _rotation_task_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return planar drift, rotation error, excessive tilt and table fall."""

        planar_drift = torch.linalg.vector_norm(self.object_pos[:, :2] - self.in_hand_pos[:, :2], dim=-1)
        rotation_error = rotation_distance(self.object_rot, self.goal_rot)
        object_up = quat_apply(self.object_rot, self.z_unit_tensor)
        tilt_cosine = object_up[:, 2]
        tilt_limit_cosine = torch.cos(
            torch.deg2rad(torch.tensor(float(self.cfg.table_tilt_limit_deg), device=self.device))
        )
        excessive_tilt = tilt_cosine <= tilt_limit_cosine
        below_table = self.object_pos[:, 2] < float(self.cfg.table_top_z) - 0.025
        return planar_drift, rotation_error, excessive_tilt, below_table

    def _get_rewards(self) -> torch.Tensor:
        """Rotation-focused reward for the single pepper bottle."""

        planar_drift, rotation_error, excessive_tilt, below_table = self._rotation_task_metrics()
        fingertip_pos_w = self.hand.data.body_pos_w.index_select(1, self._reward_fingertip_body_ids)
        fingertip_z = fingertip_pos_w[:, :, 2] - self.scene.env_origins[:, 2:3]
        fingertip_height_error = torch.abs(fingertip_z - self.object_pos[:, 2:3]).sum(dim=-1)

        distance_reward = planar_drift * float(self.cfg.dist_reward_scale)
        rotation_reward = float(self.cfg.rot_reward_scale) / (torch.abs(rotation_error) + float(self.cfg.rot_eps))
        # Encourage spinning around table normal.  If your desired axis is not Z,
        # replace index 2 with the corresponding angular-velocity component.
        velocity_reward = torch.clamp(self.object_angvel[:, 2], -10.0, 10.0) * float(self.cfg.vel_reward_scale)
        fingertip_reward = torch.exp(-10.0 * fingertip_height_error) * float(self.cfg.fingertip_distance_reward_scale)
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        reward = (
            distance_reward
            + rotation_reward
            + velocity_reward
            + fingertip_reward
            + action_penalty * float(self.cfg.action_penalty_scale)
        )

        success = rotation_error <= float(self.cfg.success_tolerance)
        displaced = planar_drift >= float(self.cfg.fall_dist)
        failed = displaced | excessive_tilt | below_table
        reward = torch.where(success, reward + float(self.cfg.reach_goal_bonus), reward)
        reward = torch.where(failed, reward + float(self.cfg.fall_penalty), reward)

        self.reset_goal_buf[:] = success
        self.successes[:] = torch.where(success, torch.ones_like(self.successes), self.successes)
        self.extras.setdefault("log", {}).update(
            {
                "pepper_rotation_error_rad": rotation_error.mean(),
                "pepper_planar_drift_m": planar_drift.mean(),
                "pepper_success_rate": success.float().mean(),
                "pepper_tilt_failure_rate": excessive_tilt.float().mean(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        planar_drift, rotation_error, excessive_tilt, below_table = self._rotation_task_metrics()
        success = rotation_error <= float(self.cfg.success_tolerance)
        failed = (planar_drift >= float(self.cfg.fall_dist)) | excessive_tilt | below_table
        terminated = success | failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _compute_tactile_observations(self):
        """Use VTDex-compatible net contact forces for the 20 tactile bits."""

        if not hasattr(self, "_vtdex_contact_body_ids"):
            raise RuntimeError("VTDex tactile body mapping has not been initialized")
        net_forces_w = self._vtdex_contact_sensor.data.net_forces_w
        if net_forces_w is None or net_forces_w.shape[:2] != (self.num_envs, self._vtdex_contact_sensor.num_bodies):
            raise RuntimeError(
                "Unexpected VTDex ContactSensor force tensor shape: "
                f"{None if net_forces_w is None else tuple(net_forces_w.shape)}"
            )
        tactile_forces_w = net_forces_w.index_select(1, self._vtdex_contact_body_ids)
        tactile_force_norms = torch.linalg.vector_norm(tactile_forces_w, dim=-1)
        self.fingertip_force_sensors = tactile_forces_w
        self.fingertip_force_binary_results = (tactile_force_norms > float(self.cfg.vtdex_contact_threshold)).to(
            dtype=torch.int32
        )
        if hasattr(self, "extras"):
            self.extras.setdefault("log", {})["vtdex_tactile_active_ratio"] = (
                self.fingertip_force_binary_results.float().mean()
            )
            self.extras["log"]["vtdex_tactile_force_mean_n"] = tactile_force_norms.mean()
            self.extras["log"]["vtdex_tactile_force_max_n"] = tactile_force_norms.max()


    def _reset_object_articulation_joints(self, env_ids: Sequence[int]) -> None:
        """Reset passive joints inside the pepper-bottle articulation to their default states."""

        data = getattr(self.object, "data", None)
        if data is None or not hasattr(data, "default_joint_pos"):
            return
        if data.default_joint_pos is None or data.default_joint_pos.shape[1] == 0:
            return

        joint_pos = data.default_joint_pos[env_ids].clone()
        if hasattr(data, "default_joint_vel") and data.default_joint_vel is not None:
            joint_vel = data.default_joint_vel[env_ids].clone()
        else:
            joint_vel = torch.zeros_like(joint_pos)

        # Isaac Lab Articulation supports write_joint_state_to_sim(position, velocity, ...).
        self.object.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_target_pose(self, env_ids: Sequence[int]):
        """Set target to current object pose rotated by target_yaw_delta_rad about world Z."""

        object_rot = self.object.data.root_quat_w[env_ids]
        target_delta = quat_from_angle_axis(
            torch.full(
                (len(env_ids),),
                float(self.cfg.target_yaw_delta_rad),
                dtype=torch.float32,
                device=self.device,
            ),
            self.z_unit_tensor[env_ids],
        )
        self.goal_rot[env_ids] = quat_mul(object_rot, target_delta)
        self.in_hand_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.goal_pos[env_ids] = self.in_hand_pos[env_ids]
        self.goal_pos[env_ids, 2] -= 0.03
        self.reset_goal_buf[env_ids] = False
        self._hide_goal_marker()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the bottle on the table with small XY noise and random yaw."""

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore[attr-defined]
        super()._reset_idx(env_ids)

        object_root_state = self.object.data.default_root_state[env_ids].clone()
        xy_noise = sample_uniform(
            -float(self.cfg.reset_position_noise),
            float(self.cfg.reset_position_noise),
            (len(env_ids), 2),
            device=self.device,
        )
        object_root_state[:, :2] += xy_noise
        yaw = sample_uniform(-torch.pi, torch.pi, (len(env_ids),), device=self.device)
        yaw_rotation = quat_from_angle_axis(yaw, self.z_unit_tensor[env_ids])
        object_root_state[:, 3:7] = quat_mul(object_root_state[:, 3:7], yaw_rotation)
        object_root_state[:, :3] += self.scene.env_origins[env_ids]
        object_root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids)
        self._reset_object_articulation_joints(env_ids)

        object_pos_env = object_root_state[:, :3] - self.scene.env_origins[env_ids]
        self.in_hand_pos[env_ids] = object_pos_env
        self.goal_pos[env_ids] = object_pos_env
        self.goal_pos[env_ids, 2] -= 0.03
        target_delta = quat_from_angle_axis(
            torch.full(
                (len(env_ids),),
                float(self.cfg.target_yaw_delta_rad),
                dtype=torch.float32,
                device=self.device,
            ),
            self.z_unit_tensor[env_ids],
        )
        self.goal_rot[env_ids] = quat_mul(object_root_state[:, 3:7], target_delta)
        self.reset_goal_buf[env_ids] = False
        self._hide_goal_marker()
        self._compute_intermediate_values()

    def _write_goal_object_pose(self, env_ids: Sequence[int]):
        """Compatibility no-op: this task does not spawn a GoalObject USD."""

        return

    def _hide_goal_object(self):
        """Compatibility no-op: this task does not spawn a GoalObject USD."""

        return

    def _hide_goal_marker(self):
        """Hide inherited goal marker if it exists; target is stored in goal_pos / goal_rot."""

        if hasattr(self, "goal_markers"):
            hidden_goal_pos_w = torch.full_like(self.goal_pos, -10.0)
            self.goal_markers.visualize(hidden_goal_pos_w, self.goal_rot)

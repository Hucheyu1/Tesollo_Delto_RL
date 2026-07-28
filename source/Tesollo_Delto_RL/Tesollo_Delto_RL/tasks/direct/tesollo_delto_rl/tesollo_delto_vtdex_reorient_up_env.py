# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""VTDexManip in-hand ``reorient_up`` task adapted to the DG5F right hand."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform

from .delto_cfg import TESOLLO_CFG
from .tesollo_delto_rl_env import TesolloDeltoRlEnv
from .tesollo_delto_vtdex_env import TesolloDeltoVTDexEnv, TesolloDeltoVTDexEnvCfg
from .vtdex_data import (
    DG5F_VTDEX_CAMERA_CLIPPING_RANGE,
    DG5F_VTDEX_CAMERA_EYE_LOCAL,
    DG5F_VTDEX_CAMERA_FOCAL_LENGTH,
    DG5F_VTDEX_CAMERA_FOCUS_DISTANCE,
    DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE,
    DG5F_VTDEX_CAMERA_RESOLUTION,
    DG5F_VTDEX_CAMERA_TARGET_LOCAL,
)


_VTDEX_ROOT = Path(__file__).resolve().parent / "vtdex_pretrained"
_VTDEX_OBJECT_ROOT = _VTDEX_ROOT / "assets" / "reorient_up"

# Exact train/seen object list and order from VTDexManip's reorient_up.yaml.
_VTDEX_REORIENT_UP_OBJECT_CODES = (
    "ddg-ycb_065-c_cups",
    "ddg-ycb_056_tennis_ball",
    "ddg-ycb_012_strawberry",
    "ddg-ycb_003_cracker_box",
    "ddg-ycb_013_apple",
    "ddg-ycb_077_rubiks_cube",
    "ddg-ycb_070-a_colored_wood_blocks",
    "ddg-ycb_010_potted_meat_can",
    "ddg-ycb_014_lemon",
    "ddg-gd_rubber_duck_poisson_001",
)


@configclass
class TesolloDeltoVTDexReorientUpEnvCfg(TesolloDeltoVTDexEnvCfg):
    """DG5F reproduction of VTDexManip's in-air ``reorient_up`` task."""

    # Keep one full ten-object cycle by default. A two-metre pitch prevents a
    # policy camera from seeing neighbouring hands even though this task has no
    # table or ground plane to occlude them.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10,
        env_spacing=2.0,
        replicate_physics=False,
    )

    # The source hand root is fixed and the object is unsupported in the hand.
    # Shadow Hand and DG5F use different palm axes, so copy the functional
    # geometry rather than the raw quaternion: this is DG5F's verified in-hand
    # pose, with the object centred inside the five-finger enclosure.
    robot_cfg = TESOLLO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=TESOLLO_CFG.spawn.replace(
            activate_contact_sensors=True,
            rigid_props=TESOLLO_CFG.spawn.rigid_props.replace(max_depenetration_velocity=2.0),
        ),
        init_state=TESOLLO_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.0, 0.8660254, 0.0),
        ),
    )
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.11, 0.00267, 0.36)),
    )
    goal_vtdex_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalObject",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.11, 0.01267, 0.35)),
    )

    # Reorient-up begins with an actual grasp, unlike the tabletop task's open
    # pregrasp. These values and ranges are local to this new task; Tomato and
    # Reorient Down remain unchanged.
    hand_position = [
        0.10,
        0.00,
        0.00,
        0.00,
        0.00,
        -1.70,
        0.50,
        0.50,
        0.50,
        0.00,
        0.70,
        0.70,
        0.70,
        0.70,
        1.57,
        0.30,
        1.00,
        1.00,
        1.00,
        1.57,
    ]
    hand_lower_limits = [
        -22,
        -24,
        -30,
        -35,
        0,
        -150,
        0,
        0,
        0,
        -24,
        0,
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
        50,
        35,
        30,
        24,
        60,
        0,
        115,
        112,
        109,
        35,
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

    # The source policy is 60 Hz and uses unsmoothed absolute joint targets.
    # A small DG5F-specific filter prevents one-step contact impulses while
    # retaining the same absolute-action semantics.
    act_moving_average = 0.2
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.0
    reset_position_noise = 0.01
    fix_object_initial_pose = True

    # Source reorient_up places an almost invisible goal copy 1 cm along +Y
    # and 1 cm down. Its displayed target is +90 degrees, while the implemented
    # reward succeeds near 180 degrees from the reset orientation. Both details
    # are retained explicitly for reproducibility.
    goal_displacement = (0.0, 0.01, -0.01)
    target_yaw_delta_rad = 1.5707963267948966
    success_rotation_rad = 3.0
    planar_fall_distance = 0.04
    vertical_fall_distance = 0.24
    table_tilt_limit_deg = 60.0

    dist_reward_scale = -10.0
    vel_reward_scale = 1.0
    fingertip_distance_reward_scale = 0.25
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0

    # Use the already validated DG5F side view requested for this project.
    # It looks along -Y at the unsupported object and has no debug geometry.
    vtdex_camera_eye_local = DG5F_VTDEX_CAMERA_EYE_LOCAL
    vtdex_camera_target_local = DG5F_VTDEX_CAMERA_TARGET_LOCAL
    vtdex_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/VTDexCamera",
        offset=CameraCfg.OffsetCfg(
            pos=DG5F_VTDEX_CAMERA_EYE_LOCAL,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DG5F_VTDEX_CAMERA_FOCAL_LENGTH,
            focus_distance=DG5F_VTDEX_CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE,
            clipping_range=DG5F_VTDEX_CAMERA_CLIPPING_RANGE,
        ),
        width=DG5F_VTDEX_CAMERA_RESOLUTION[0],
        height=DG5F_VTDEX_CAMERA_RESOLUTION[1],
        update_latest_camera_pose=True,
        debug_vis=False,
    )
    debug_visualization = False

    # Allow the same retrained DG5F VT-JointPretrain artifact override used by
    # the Tomato task, while remaining self-contained by default.
    vtdex_repo_root = os.environ.get("TESOLLO_VTDEX_REPO_ROOT", str(_VTDEX_ROOT))
    vtdex_model_id = os.environ.get(
        "TESOLLO_VTDEX_MODEL_ID",
        "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210",
    )


class TesolloDeltoVTDexReorientUpEnv(TesolloDeltoVTDexEnv):
    """Unsupported in-hand half-turn task with frozen VTDex RGB/touch input."""

    cfg: TesolloDeltoVTDexReorientUpEnvCfg

    def _setup_scene(self):
        # Deliberately no table or ground: gravity makes retaining the object
        # part of the source in-hand task.
        self._spawn_vtdex_objects()
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal_object = RigidObject(self.cfg.goal_vtdex_object_cfg)
        self._vtdex_camera = Camera(self.cfg.vtdex_camera)
        self._vtdex_contact_sensor = ContactSensor(self.cfg.vtdex_contact_sensor)

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_object"] = self.goal_object
        self.scene.sensors["vtdex_camera"] = self._vtdex_camera
        self.scene.sensors["vtdex_contact"] = self._vtdex_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_vtdex_objects(self):
        """Spawn the exact ten reorient-up train objects and tiny goal copies."""

        yellow = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(204.0 / 255.0, 204.0 / 255.0, 0.0),
            roughness=0.5,
            metallic=0.0,
        )
        active_rigid_props = sim_utils.RigidBodyPropertiesCfg(
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
        )
        goal_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        goal_offset = tuple(float(value) for value in self.cfg.goal_displacement)

        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            object_code = _VTDEX_REORIENT_UP_OBJECT_CODES[
                env_index % len(_VTDEX_REORIENT_UP_OBJECT_CODES)
            ]
            object_dir = _VTDEX_OBJECT_ROOT / object_code / "coacd"
            urdf_path = object_dir / "coacd_1.urdf"
            goal_urdf_path = object_dir / "coacd_goal.urdf"
            if not urdf_path.is_file():
                raise FileNotFoundError(f"Missing copied VTDex reorient-up object asset: {urdf_path}")
            if not goal_urdf_path.is_file():
                raise FileNotFoundError(f"Missing collision-free VTDex goal asset: {goal_urdf_path}")

            common = {
                "fix_base": False,
                "merge_fixed_joints": True,
                "joint_drive": None,
                "collider_type": "convex_hull",
                "visual_material": yellow,
                "make_instanceable": False,
            }
            object_spawn_cfg = sim_utils.UrdfFileCfg(
                **common,
                asset_path=str(urdf_path),
                scale=(0.04, 0.04, 0.04),
                semantic_tags=[("class", "object")],
                rigid_props=active_rigid_props,
            )
            goal_spawn_cfg = sim_utils.UrdfFileCfg(
                **common,
                asset_path=str(goal_urdf_path),
                scale=(0.001, 0.001, 0.001),
                semantic_tags=[("class", "goal")],
                rigid_props=goal_rigid_props,
            )
            object_position = tuple(self.cfg.object_cfg.init_state.pos)
            goal_position = tuple(
                object_position[axis] + goal_offset[axis] for axis in range(3)
            )
            object_spawn_cfg.func(
                f"{env_path}/Object",
                object_spawn_cfg,
                translation=object_position,
                orientation=tuple(self.cfg.object_cfg.init_state.rot),
            )
            goal_spawn_cfg.func(
                f"{env_path}/GoalObject",
                goal_spawn_cfg,
                translation=goal_position,
                orientation=tuple(self.cfg.goal_vtdex_object_cfg.init_state.rot),
            )

    @staticmethod
    def _relative_yaw(relative_quat: torch.Tensor) -> torch.Tensor:
        """Return wrapped ZYX yaw for Isaac Lab's WXYZ quaternion layout."""

        w, x, y, z = relative_quat.unbind(dim=-1)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(sin_yaw, cos_yaw)

    def _in_hand_metrics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return planar drift, vertical drift, reset-relative yaw and tilt."""

        planar_drift = torch.linalg.vector_norm(
            self.object_pos[:, :2] - self.goal_pos[:, :2],
            dim=-1,
        )
        vertical_drift = torch.abs(self.object_pos[:, 2] - self.goal_pos[:, 2])
        relative_quat = quat_mul(self.object_rot, quat_conjugate(self.initial_object_rot))
        yaw = self._relative_yaw(relative_quat)
        object_up = quat_apply(self.object_rot, self.z_unit_tensor)
        tilt_cosine = object_up[:, 2]
        return planar_drift, vertical_drift, yaw, tilt_cosine

    def _get_rewards(self) -> torch.Tensor:
        """Retain the source task's implemented yaw and retention reward."""

        planar_drift, vertical_drift, yaw, tilt_cosine = self._in_hand_metrics()
        fingertip_pos_w = self.hand.data.body_pos_w.index_select(
            1,
            self._reward_fingertip_body_ids,
        )
        fingertip_z = fingertip_pos_w[:, :, 2] - self.scene.env_origins[:, 2:3]
        fingertip_height_error = torch.abs(
            fingertip_z - self.object_pos[:, 2:3]
        ).sum(dim=-1)

        distance_reward = planar_drift * float(self.cfg.dist_reward_scale)
        velocity_reward = (
            torch.clamp(self.object_angvel[:, 2], -10.0, 10.0)
            * float(self.cfg.vel_reward_scale)
        )
        fingertip_reward = (
            torch.exp(-10.0 * fingertip_height_error)
            * float(self.cfg.fingertip_distance_reward_scale)
        )
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        reward = (
            distance_reward
            + torch.abs(yaw)
            + velocity_reward
            + fingertip_reward
            + action_penalty * float(self.cfg.action_penalty_scale)
        )

        success = torch.abs(yaw) > float(self.cfg.success_rotation_rad)
        tilt_limit_cosine = torch.cos(
            torch.deg2rad(
                torch.tensor(float(self.cfg.table_tilt_limit_deg), device=self.device)
            )
        )
        failed = (
            (planar_drift > float(self.cfg.planar_fall_distance))
            | (vertical_drift >= float(self.cfg.vertical_fall_distance))
            | (tilt_cosine <= tilt_limit_cosine)
        )
        reward = torch.where(
            success,
            reward + float(self.cfg.reach_goal_bonus),
            reward,
        )
        reward = torch.where(
            failed,
            reward + float(self.cfg.fall_penalty),
            reward,
        )

        self.reset_goal_buf[:] = success
        self.successes[:] = torch.where(
            success,
            torch.ones_like(self.successes),
            self.successes,
        )
        self.extras.setdefault("log", {}).update(
            {
                "reorient_up_abs_yaw_rad": torch.abs(yaw).mean(),
                "reorient_up_planar_drift_m": planar_drift.mean(),
                "reorient_up_vertical_drift_m": vertical_drift.mean(),
                "reorient_up_success_rate": success.float().mean(),
                "reorient_up_drop_or_tilt_rate": failed.float().mean(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        planar_drift, vertical_drift, yaw, tilt_cosine = self._in_hand_metrics()
        tilt_limit_cosine = torch.cos(
            torch.deg2rad(
                torch.tensor(float(self.cfg.table_tilt_limit_deg), device=self.device)
            )
        )
        success = torch.abs(yaw) > float(self.cfg.success_rotation_rad)
        failed = (
            (planar_drift > float(self.cfg.planar_fall_distance))
            | (vertical_drift >= float(self.cfg.vertical_fall_distance))
            | (tilt_cosine <= tilt_limit_cosine)
        )
        terminated = success | failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_target_pose(self, env_ids: Sequence[int]):
        """Record reset orientation and reproduce the source goal actor pose."""

        object_rot = self.object.data.root_quat_w[env_ids]
        if not hasattr(self, "initial_object_rot"):
            self.initial_object_rot = self.object.data.root_quat_w.clone()
        self.initial_object_rot[env_ids] = object_rot

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
        self.in_hand_pos[env_ids] = (
            self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        )
        goal_offset = torch.tensor(
            self.cfg.goal_displacement,
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_pos[env_ids] = self.in_hand_pos[env_ids] + goal_offset
        self.reset_goal_buf[env_ids] = False
        self._write_goal_object_pose(env_ids)
        self._hide_goal_marker()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the grasp, then apply source XY/yaw object randomization."""

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore

        # Skip the tabletop subclass reset while retaining the shared DG5F
        # joint/root/object initialization.
        TesolloDeltoRlEnv._reset_idx(self, env_ids)

        object_root_state = self.object.data.default_root_state[env_ids].clone()
        object_root_state[:, :2] += sample_uniform(
            -float(self.cfg.reset_position_noise),
            float(self.cfg.reset_position_noise),
            (len(env_ids), 2),
            device=self.device,
        )
        yaw = sample_uniform(-torch.pi, torch.pi, (len(env_ids),), device=self.device)
        yaw_rotation = quat_from_angle_axis(yaw, self.z_unit_tensor[env_ids])
        object_root_state[:, 3:7] = quat_mul(
            object_root_state[:, 3:7],
            yaw_rotation,
        )
        object_root_state[:, :3] += self.scene.env_origins[env_ids]
        object_root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids)

        object_pos_env = object_root_state[:, :3] - self.scene.env_origins[env_ids]
        self.in_hand_pos[env_ids] = object_pos_env
        self.initial_object_rot[env_ids] = object_root_state[:, 3:7]
        goal_offset = torch.tensor(
            self.cfg.goal_displacement,
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_pos[env_ids] = object_pos_env + goal_offset
        target_delta = quat_from_angle_axis(
            torch.full(
                (len(env_ids),),
                float(self.cfg.target_yaw_delta_rad),
                dtype=torch.float32,
                device=self.device,
            ),
            self.z_unit_tensor[env_ids],
        )
        self.goal_rot[env_ids] = quat_mul(
            object_root_state[:, 3:7],
            target_delta,
        )
        self.reset_goal_buf[env_ids] = False
        self._write_goal_object_pose(env_ids)
        self._hide_goal_marker()
        self._compute_intermediate_values()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Preserved DG5F tomato reorientation task using VTDexManip features."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

from .tesollo_delto_rl_env import TesolloDeltoRlEnv, unscale
from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg
from .delto_cfg import TESOLLO_CFG
from .vtdex_data import (
    DG5F_VTDEX_CAMERA_CLIPPING_RANGE,
    DG5F_VTDEX_CAMERA_EYE_LOCAL,
    DG5F_VTDEX_CAMERA_FOCAL_LENGTH,
    DG5F_VTDEX_CAMERA_FOCUS_DISTANCE,
    DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE,
    DG5F_VTDEX_CAMERA_RESOLUTION,
    DG5F_VTDEX_CAMERA_TARGET_LOCAL,
    DG5F_VTDEX_CONTACT_THRESHOLD_N,
    DG5F_VTDEX_TACTILE_BODY_NAMES,
    DG5F_VTDEX_TACTILE_INDICES,
    DG5F_VTDEX_TOMATO_MARKER_DIFFUSE_COLORS,
    DG5F_VTDEX_TOMATO_MARKER_EMISSIVE_COLORS,
    DG5F_VTDEX_TOMATO_MARKER_OFFSETS,
    DG5F_VTDEX_TOMATO_MARKER_RADIUS,
)
from .vtdex_encoder import VTDexJointEncoder
from .vtdex_markers import spawn_tomato_orientation_markers


_VTDEx_ROOT = Path(__file__).resolve().parent / "vtdex_pretrained"


@configclass
class TesolloDeltoVTDexTomatoEnvCfg(TesolloDeltoRlEnvCfg):
    """Original tomato task with privileged simulator state only for critic/reward."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)

    # VTDex uses net rigid-body contact forces, not articulation joint forces.
    # Contact reporting must be enabled when the DG5F USD is spawned.
    robot_cfg = TESOLLO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=TESOLLO_CFG.spawn.replace(activate_contact_sensors=True),
    )
    vtdex_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_[1-4]",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )
    # Match reorient_up/down's binary tactile threshold (Newtons).
    vtdex_contact_threshold = DG5F_VTDEX_CONTACT_THRESHOLD_N

    vtdex_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/VTDexCamera",
        offset=CameraCfg.OffsetCfg(
            # An exact look-at pose is assigned after scene initialization.
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
    # Environment-local eye and look-at point. The camera sits on the +Y side
    # and looks strictly along -Y at the nominal tomato center.
    vtdex_camera_eye_local = DG5F_VTDEX_CAMERA_EYE_LOCAL
    vtdex_camera_target_local = DG5F_VTDEX_CAMERA_TARGET_LOCAL

    # Two non-collinear colored dots make the otherwise near-spherical tomato
    # orientation observable. They are visual-only and must be reproduced on
    # the real tomato at the same object-local offsets.
    show_tomato_orientation_markers = True
    # Both markers lie on the +Y hemisphere so that the -Y-facing camera can
    # observe them, while their non-collinear offsets still disambiguate pose.
    tomato_orientation_marker_offsets = DG5F_VTDEX_TOMATO_MARKER_OFFSETS
    # Keep marker settings as plain Hydra-safe values. Nested config objects in
    # a tuple are converted to dictionaries during Hydra's round trip.
    tomato_orientation_marker_radius = DG5F_VTDEX_TOMATO_MARKER_RADIUS
    tomato_orientation_marker_diffuse_colors = DG5F_VTDEX_TOMATO_MARKER_DIFFUSE_COLORS
    tomato_orientation_marker_emissive_colors = DG5F_VTDEX_TOMATO_MARKER_EMISSIVE_COLORS

    # actor = qpos(20) + qvel(20) + target position in hand frame(3)
    #       + target quaternion in hand frame(4) + binary tactile(20)
    #       + previous action(20) + frozen VTDex CLS feature(384) = 471
    observation_space = 471
    # critic keeps the 84-dimensional simulator state plus 20 tactile values.
    state_space = 104
    asymmetric_obs = True
    obs_type = "vtdex"

    # Use the self-contained checkpoint copy shared with the upstream-scene
    # reproduction; this task no longer depends on an external checkout.
    vtdex_repo_root = os.environ.get("TESOLLO_VTDEX_REPO_ROOT", str(_VTDEx_ROOT))
    vtdex_model_id = os.environ.get(
        "TESOLLO_VTDEX_MODEL_ID",
        "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210",
    )
    vtdex_embedding_dim = 384
    # Match VTDex's layer-major token semantics: five fingers ordered
    # little/ring/middle/index/thumb at distal, then the same order at middle,
    # proximal and knuckle. DG5F's thumb-base link proxies the final Shadow-Hand
    # palm token. Keep this exact 20-channel order in the real robot node.
    vtdex_tactile_body_names = DG5F_VTDEX_TACTILE_BODY_NAMES
    fingertip_body_names = list(vtdex_tactile_body_names)
    vtdex_tactile_indices = DG5F_VTDEX_TACTILE_INDICES

    # Sample a reachable target position around the nominal in-hand point in
    # the hand-root frame. Orientation continues to use the base task's random
    # full-pose target sampling.
    randomize_goal_position = True
    goal_position_delta_range = ((-0.015, 0.015), (-0.015, 0.015), (-0.015, 0.015))
    use_fixed_goal_local_pos = False
    fixed_goal_local_pos = (0.0, 0.0, 0.0)
    # "all" samples full SO(3); "x", "y" or "z" restricts targets to one
    # rotation axis expressed in the hand-root frame. Keep this string-typed so
    # Isaac Lab's Hydra config updater can accept command-line axis overrides.
    goal_rotation_axis = "all"
    goal_rotation_angle_range = (-3.141592653589793, 3.141592653589793)
    require_position_for_success = True
    position_success_tolerance = 0.025

    # The goal tomato must never enter the policy camera; only the real tomato
    # is encoded. Debug coordinate frames still use the true target pose.
    debug_visualization = False


class TesolloDeltoVTDexTomatoEnv(TesolloDeltoRlEnv):
    """Actor observes VTDex features, never the simulator tomato pose."""

    cfg: TesolloDeltoVTDexTomatoEnvCfg

    def __init__(self, cfg: TesolloDeltoVTDexTomatoEnvCfg, render_mode: str | None = None, **kwargs):
        goal_rotation_axis = str(cfg.goal_rotation_axis).lower()
        if goal_rotation_axis in {"all", "none", "full"}:
            goal_rotation_axis = None
        elif goal_rotation_axis not in {"x", "y", "z"}:
            raise ValueError(
                'goal_rotation_axis must be one of "all", "x", "y", "z"; '
                f"got {cfg.goal_rotation_axis}"
            )
        goal_rotation_angle_range = tuple(float(angle) for angle in cfg.goal_rotation_angle_range)
        if len(goal_rotation_angle_range) != 2 or goal_rotation_angle_range[0] > goal_rotation_angle_range[1]:
            raise ValueError(
                "goal_rotation_angle_range must be an ordered (min, max) pair, "
                f"got {cfg.goal_rotation_angle_range}"
            )

        goal_position_ranges = tuple(
            tuple(float(bound) for bound in axis_range) for axis_range in cfg.goal_position_delta_range
        )
        if len(goal_position_ranges) != 3 or any(len(axis_range) != 2 for axis_range in goal_position_ranges):
            raise ValueError(
                "goal_position_delta_range must contain three (min, max) pairs, "
                f"got {cfg.goal_position_delta_range}"
            )
        if any(axis_range[0] > axis_range[1] for axis_range in goal_position_ranges):
            raise ValueError(
                "goal_position_delta_range bounds must be ordered, "
                f"got {cfg.goal_position_delta_range}"
            )
        fixed_goal_local_pos = tuple(float(value) for value in cfg.fixed_goal_local_pos)
        if len(fixed_goal_local_pos) != 3:
            raise ValueError(
                "fixed_goal_local_pos must contain exactly three values, "
                f"got {cfg.fixed_goal_local_pos}"
            )

        super().__init__(cfg, render_mode, **kwargs)
        self._goal_rotation_axis = goal_rotation_axis
        self._goal_rotation_angle_range = goal_rotation_angle_range
        self._goal_position_ranges = torch.tensor(
            goal_position_ranges, dtype=torch.float32, device=self.device
        )
        contact_body_ids, contact_body_names = self._vtdex_contact_sensor.find_bodies(
            list(self.cfg.vtdex_tactile_body_names), preserve_order=True
        )
        if contact_body_names != list(self.cfg.vtdex_tactile_body_names):
            raise RuntimeError(
                "VTDex ContactSensor body order does not match the configured tactile token order: "
                f"expected={list(self.cfg.vtdex_tactile_body_names)}, resolved={contact_body_names}"
            )
        if len(contact_body_ids) != 20 or len(set(contact_body_ids)) != 20:
            raise RuntimeError(
                "VTDex ContactSensor must resolve exactly 20 unique DG5F links; "
                f"got ids={contact_body_ids}"
            )
        self._vtdex_contact_body_ids = torch.tensor(
            contact_body_ids, dtype=torch.long, device=self.device
        )
        print(
            "[INFO]: VTDex tactile ContactSensor mapping (net contact force): "
            f"{contact_body_names}"
        )
        self._spawn_tomato_orientation_markers()
        self.vtdex_encoder = VTDexJointEncoder(
            repo_root=self.cfg.vtdex_repo_root,
            model_id=self.cfg.vtdex_model_id,
            device=self.device,
            tactile_indices=tuple(self.cfg.vtdex_tactile_indices),
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
        self._hide_goal_marker()

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._vtdex_camera = Camera(self.cfg.vtdex_camera)
        self._vtdex_contact_sensor = ContactSensor(self.cfg.vtdex_contact_sensor)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["vtdex_camera"] = self._vtdex_camera
        self.scene.sensors["vtdex_contact"] = self._vtdex_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_tomato_orientation_markers(self):
        """Attach dots to the visual surface of the actual tomato rigid prim."""

        if not self.cfg.show_tomato_orientation_markers:
            return
        spawn_tomato_orientation_markers(
            object_asset=self.object,
            num_envs=self.num_envs,
            marker_offsets=self.cfg.tomato_orientation_marker_offsets,
            marker_radius=self.cfg.tomato_orientation_marker_radius,
            diffuse_colors=self.cfg.tomato_orientation_marker_diffuse_colors,
            emissive_colors=self.cfg.tomato_orientation_marker_emissive_colors,
        )

    def _configure_vtdex_camera_pose(self):
        """Aim every tiled camera using environment-local eye/target points."""

        eye_local = torch.tensor(
            self.cfg.vtdex_camera_eye_local, dtype=torch.float32, device=self.device
        ).view(1, 3)
        target_local = torch.tensor(
            self.cfg.vtdex_camera_target_local, dtype=torch.float32, device=self.device
        ).view(1, 3)
        if torch.linalg.vector_norm(eye_local - target_local).item() < 1.0e-4:
            raise ValueError("vtdex_camera_eye_local and target must be different points")
        eyes_w = self.scene.env_origins + eye_local
        targets_w = self.scene.env_origins + target_local
        self._vtdex_camera.set_world_poses_from_view(eyes_w, targets_w)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        rgb = self._vtdex_camera.data.output["rgb"]
        tactile = self.fingertip_force_binary_results.to(dtype=torch.float32)
        self.vtdex_embeddings = self.vtdex_encoder(rgb, tactile)

        target_pos_hand = quat_apply(
            quat_conjugate(self.hand_base_rot),
            self.in_hand_pos - self.hand_base_pos,
        )
        target_rot_hand = quat_mul(quat_conjugate(self.hand_base_rot), self.goal_rot)
        policy_obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                target_pos_hand,
                target_rot_hand,
                tactile,
                self.actions,
                self.vtdex_embeddings,
            ),
            dim=-1,
        )

        # Simulator object pose/velocity is privileged and never enters policy_obs.
        critic_obs = self.compute_full_state()
        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"VTDex policy observation has {policy_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.observation_space}"
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"VTDex critic observation has {critic_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _compute_tactile_observations(self):
        """Use VTDex-compatible net contact forces for the 20 tactile bits."""

        if not hasattr(self, "_vtdex_contact_body_ids"):
            raise RuntimeError("VTDex tactile body mapping has not been initialized")

        net_forces_w = self._vtdex_contact_sensor.data.net_forces_w
        if net_forces_w is None or net_forces_w.shape[:2] != (
            self.num_envs,
            self._vtdex_contact_sensor.num_bodies,
        ):
            raise RuntimeError(
                "Unexpected VTDex ContactSensor force tensor shape: "
                f"{None if net_forces_w is None else tuple(net_forces_w.shape)}"
            )
        tactile_forces_w = net_forces_w.index_select(1, self._vtdex_contact_body_ids)
        tactile_force_norms = torch.linalg.vector_norm(tactile_forces_w, dim=-1)
        self.fingertip_force_sensors = tactile_forces_w
        self.fingertip_force_binary_results = (
            tactile_force_norms > float(self.cfg.vtdex_contact_threshold)
        ).to(dtype=torch.int32)
        if hasattr(self, "extras"):
            self.extras.setdefault("log", {})["vtdex_tactile_active_ratio"] = (
                self.fingertip_force_binary_results.float().mean()
            )
            self.extras["log"]["vtdex_tactile_force_mean_n"] = tactile_force_norms.mean()
            self.extras["log"]["vtdex_tactile_force_max_n"] = tactile_force_norms.max()

    def _reset_target_pose(self, env_ids: Sequence[int]):
        super()._reset_target_pose(env_ids)

        if self._goal_rotation_axis is not None:
            axis_tensor = getattr(self, f"{self._goal_rotation_axis}_unit_tensor")[env_ids]
            angle_min, angle_max = self._goal_rotation_angle_range
            goal_angles = angle_min + torch.rand(len(env_ids), device=self.device) * (angle_max - angle_min)
            goal_rot_local = quat_from_angle_axis(goal_angles, axis_tensor)
            self.goal_rot[env_ids] = quat_mul(self.hand_base_rot[env_ids], goal_rot_local)

        if self.cfg.use_fixed_goal_local_pos:
            goal_local_pos = torch.tensor(
                self.cfg.fixed_goal_local_pos, dtype=torch.float32, device=self.device
            ).view(1, 3).repeat(len(env_ids), 1)
        elif getattr(self.cfg, "randomize_goal_position", False):
            random_unit = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
            delta = self._goal_position_ranges[:, 0] + random_unit * (
                self._goal_position_ranges[:, 1] - self._goal_position_ranges[:, 0]
            )
            goal_local_pos = self.in_hand_local_pos[env_ids] + delta
        else:
            # ``_reset_idx`` has just placed this target at the initial object
            # position. On subsequent goal resets, retain the same position and
            # only resample orientation.
            self.goal_pos[env_ids] = self.in_hand_pos[env_ids]
            self._hide_goal_marker()
            return

        self.in_hand_pos[env_ids] = self.hand_base_pos[env_ids] + quat_apply(
            self.hand_base_rot[env_ids], goal_local_pos
        )
        self.goal_pos[env_ids] = self.in_hand_pos[env_ids]
        self._hide_goal_marker()

    def _hide_goal_marker(self):
        hidden_goal_pos_w = torch.full_like(self.goal_pos, -10.0)
        self.goal_markers.visualize(hidden_goal_pos_w, self.goal_rot)

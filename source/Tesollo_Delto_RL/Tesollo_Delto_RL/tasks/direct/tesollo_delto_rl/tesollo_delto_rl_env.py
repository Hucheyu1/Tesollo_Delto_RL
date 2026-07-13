# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .yolo_seg_image_estimator import axial_angle_error_features

if TYPE_CHECKING:
    from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg


class TesolloDeltoRlEnv(DirectRLEnv):
    cfg: TesolloDeltoRlEnvCfg

    def __init__(self, cfg: TesolloDeltoRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # number of hand dofs - 手部自由度数量
        self.num_hand_dofs = self.hand.num_joints
        # action buffer - 动作缓冲区
        self.action_scale = self.cfg.action_scale
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.raw_actions = torch.zeros((self.num_envs, self.cfg.action_space),dtype=torch.float,device=self.device)
        # position target buffers - 位置目标缓冲区
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # actuated joints - 受控关节
        self.actuated_dof_indices = []
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        # fingertip bodies - 手指体
        self.finger_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)
        print(f"Available joints: {self.hand.joint_names}")
        print(f"Available bodies: {self.hand.body_names}")

        # joint limits - 关节限制
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0].clone()
        self.hand_dof_upper_limits = joint_pos_limits[..., 1].clone()
        # default reset joint position buffer - 默认重置关节位置缓冲区
        self.default_hand_dof_pos = self.hand.data.default_joint_pos.clone()
        if getattr(self.cfg, "use_manual_joint_cfg", False):
            manual_pos = torch.tensor(self.cfg.hand_position,dtype=torch.float,device=self.device).view(1, -1).repeat(self.num_envs, 1)
            manual_upper = torch.deg2rad(torch.tensor(self.cfg.hand_upper_limits,dtype=torch.float,device=self.device)).view(1, -1).repeat(self.num_envs, 1)
            manual_lower = torch.deg2rad(torch.tensor(self.cfg.hand_lower_limits,dtype=torch.float,device=self.device,)).view(1, -1).repeat(self.num_envs, 1)
            # 防止初始手型超出手动限制
            manual_pos = torch.max(torch.min(manual_pos, manual_upper), manual_lower)
            for action_id, joint_id in enumerate(self.actuated_dof_indices):
                self.default_hand_dof_pos[:, joint_id] = manual_pos[:, action_id]
                self.hand_dof_lower_limits[:, joint_id] = manual_lower[:, action_id]
                self.hand_dof_upper_limits[:, joint_id] = manual_upper[:, action_id]      
        # 初始化
        self.target_pos = self.default_hand_dof_pos.clone()
        self.prev_targets[:] = self.default_hand_dof_pos
        self.cur_targets[:] = self.default_hand_dof_pos
        self.hand_dof_targets[:] = self.default_hand_dof_pos

        # goal reset buffer - 目标重置缓冲区
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---------------------------------------------------------------------
        # Build object / goal reference in hand-root local frame 在手部根坐标系中构建物体/目标参考
        # ---------------------------------------------------------------------
        # 环境局部坐标系中的手部默认姿态
        self.hand_base_pos = self.hand.data.default_root_state[:, 0:3].clone()
        self.hand_base_rot = self.hand.data.default_root_state[:, 3:7].clone()

        # 环境局部坐标系中的物体默认姿态
        object_default_pos_env = self.object.data.default_root_state[:, 0:3].clone()
        object_default_rot_env = self.object.data.default_root_state[:, 3:7].clone()

        # 物体相对于手部根部的位置，以手部根部坐标系表示
        self.object_local_pos = quat_apply(quat_conjugate(self.hand_base_rot),object_default_pos_env - self.hand_base_pos)

        # 物体相对于手部根部的默认旋转
        self.object_local_rot = quat_mul(quat_conjugate(self.hand_base_rot),object_default_rot_env)

        # 手内奖励目标位置（手部根部坐标系）
        in_hand_local_offset = torch.tensor(self.cfg.in_hand_local_offset,dtype=torch.float,device=self.device).repeat((self.num_envs, 1))

        self.in_hand_local_pos = self.object_local_pos + in_hand_local_offset

        # 当前手内目标和目标标记位置（环境局部坐标系）
        self.in_hand_pos = self.hand_base_pos + quat_apply(self.hand_base_rot, self.in_hand_local_pos)
        self.goal_pos = self.hand_base_pos + quat_apply(self.hand_base_rot, self.object_local_pos)

        # 目标旋转（环境/世界坐标系）
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        # goal marker 目标标记
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        # 坐标轴可视化
        if getattr(self.cfg, "debug_visualization", False):
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.prim_path = "/Visuals/debug_frames"
            frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
            self.frame_markers = VisualizationMarkers(frame_marker_cfg)

        # track successes 跟踪成功次数
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors 单位张量
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # 打印映射信息，用于调试
        print("==== Action joint mapping ====")
        for action_id, joint_id in enumerate(self.actuated_dof_indices):
            print(f"action[{action_id:02d}] -> joint_id={joint_id:02d}, name={self.hand.joint_names[joint_id]}")

        # YOLO is only initialized for the distillation student. The teacher,
        # rewards and termination logic continue to use simulator ground truth.
        self.yolo_student_estimator = None
        self.last_yolo_estimate = None
        self.last_yolo_target_angle_features = None
        self.last_yolo_observed_object_y = None
        self.last_yolo_object_y_angle = None
        self.last_yolo_goal_y_angle = None
        if getattr(self.cfg, "use_yolo_student_obs", False):
            from .yolo_seg_image_estimator import YoloSegImageEstimator

            self.yolo_student_estimator = YoloSegImageEstimator(
                model_path=self.cfg.yolo_model_path,
                num_envs=self.num_envs,
                device=self.device,
                class_id=self.cfg.yolo_class_id,
                confidence_threshold=self.cfg.yolo_confidence_threshold,
                iou_threshold=self.cfg.yolo_iou_threshold,
                inference_size=self.cfg.yolo_inference_size,
                min_mask_pixels=self.cfg.yolo_min_mask_pixels,
                position_gain=self.cfg.yolo_position_gain,
                min_anisotropy=self.cfg.yolo_min_anisotropy,
                min_visible_ratio=self.cfg.yolo_min_visible_ratio,
                max_angle_jump_rad=self.cfg.yolo_max_angle_jump,
            )
            configured_offset = getattr(self.cfg, "yolo_angle_offset_rad", None)
            initial_offset = 0.0 if configured_offset is None else float(configured_offset)
            self.yolo_angle_offset = torch.full(
                (self.num_envs,), initial_offset, dtype=torch.float32, device=self.device
            )
            self.yolo_angle_calibrated = torch.full(
                (self.num_envs,), configured_offset is not None, dtype=torch.bool, device=self.device
            )

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        if getattr(self.cfg, "use_yolo_student_obs", False):
            self._student_camera = TiledCamera(self.cfg.student_camera)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        if getattr(self.cfg, "use_yolo_student_obs", False):
            self.scene.sensors["student_camera"] = self._student_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # policy 输出 [-1, 1]，这里表示关节增量方向
        self.raw_actions = torch.clamp(actions, -1.0, 1.0)
        # self.actions 用于 observation 和 reward
        self.actions = self.raw_actions.clone()
        # 只更新可控 20 个关节(其实就是全关节)
        self.target_pos[:, self.actuated_dof_indices] = (self.target_pos[:, self.actuated_dof_indices]+ self.action_scale * self.raw_actions)
        # 限制在关节上下限内
        self.target_pos[:, self.actuated_dof_indices] = torch.clamp(
            self.target_pos[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

    def _apply_action(self) -> None:
        # 使用移动平均滤波器更新目标位置，结合当前目标和上一时刻目标
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.target_pos[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        # 对更新后的目标位置进行限制，确保在关节限制范围内
        self.cur_targets[:, self.actuated_dof_indices] = torch.clamp(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # 保存当前目标位置作为下一时刻的上一时刻目标
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        # 设置手部关节的目标位置
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _get_observations(self) -> dict:
        if self.cfg.obs_type in ("openai", "distill"):
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            raise RuntimeError(f"Unknown obs_type: {self.cfg.obs_type}")

        observations = {"policy": obs}

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()
            observations = {"policy": obs, "critic": states}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,                           # 总奖励值
            self.reset_goal_buf,                    # 目标重置缓冲区
            self.successes[:],                      # 成功次数数组
            self.consecutive_successes[:],          # 连续成功次数数组
        ) = compute_rewards(
            self.reset_buf,                         # 重置缓冲区
            self.reset_goal_buf,                    # 目标重置缓冲区
            self.successes,                         # 成功次数
            self.consecutive_successes,             # 连续成功次数
            self.max_episode_length,                # 最大回合长度
            self.object_pos,                        # 物体位置
            self.object_rot,                        # 物体旋转
            self.in_hand_pos,                       # 手部位置
            self.goal_rot,                          # 目标旋转
            self.cfg.dist_reward_scale,             # 距离奖励缩放因子
            self.cfg.rot_reward_scale,              # 旋转奖励缩放因子
            self.cfg.rot_eps,                       # 旋转epsilon值
            self.actions,                           # 动作
            self.cfg.action_penalty_scale,          # 动作惩罚缩放因子
            self.cfg.success_tolerance,             # 成功容差
            self.cfg.reach_goal_bonus,              # 达成目标奖励
            self.cfg.fall_dist,                     # 坠落距离
            self.cfg.fall_penalty,                  # 坠落惩罚
            self.cfg.av_factor,                     # 平均值因子
            bool(getattr(self.cfg, "y_axis_only", False)),
            self.hand_base_rot,
        )
        # 初始化extras中的log字典
        if "log" not in self.extras:
            self.extras["log"] = dict()
        # 记录连续成功次数的平均值到日志中
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        # 获取需要重置目标的环境ID
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # 计算目标物体与手部位置之间的欧几里得距离
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        # 如果配置了最大连续成功次数，则需要检查旋转距离
        if self.cfg.max_consecutive_success > 0:
            # 计算物体旋转与目标旋转之间的距离
            if getattr(self.cfg, "y_axis_only", False):
                rot_dist = axial_y_rotation_distance(self.object_rot, self.goal_rot, self.hand_base_rot)
            else:
                rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            # 如果旋转距离在容差范围内，则重置episode长度缓冲区
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            # 判断是否达到最大连续成功次数
            max_success_reached = self.successes >= self.cfg.max_consecutive_success
        # 判断是否达到最大episode长度
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 如果配置了最大连续成功次数，则超时条件还需要考虑是否达到最大连续成功次数
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached # type: ignore

        return out_of_reach, time_out

    def _reset_target_pose(self, env_ids):
        fixed_goal_rot = getattr(self.cfg, "fixed_goal_rot", None)
        if fixed_goal_rot is not None:
            goal_rot = torch.tensor(fixed_goal_rot, dtype=torch.float, device=self.device).view(1, 4)
            goal_rot = goal_rot / torch.linalg.vector_norm(goal_rot, dim=-1, keepdim=True).clamp_min(1e-8)
            self.goal_rot[env_ids] = goal_rot.repeat(len(env_ids), 1)
            self._update_goal_marker(env_ids)
            return

        fixed_goal_y_angle = getattr(self.cfg, "fixed_goal_y_angle_rad", None)
        if fixed_goal_y_angle is not None:
            goal_angles = torch.full((len(env_ids),), float(fixed_goal_y_angle), dtype=torch.float, device=self.device)
            goal_rot_local = quat_from_angle_axis(
                goal_angles,
                self.y_unit_tensor[env_ids],
            )
            self.goal_rot[env_ids] = quat_mul(self.hand_base_rot[env_ids], goal_rot_local)
            self._update_goal_marker(env_ids)
            return

        # 目标旋转在手部-根部局部坐标系中
        if getattr(self.cfg, "y_axis_only", False):
            goal_angles = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)[:, 0] * np.pi
            goal_rot_local = quat_from_angle_axis(
                goal_angles,
                self.y_unit_tensor[env_ids],
            )
            self.goal_rot[env_ids] = quat_mul(self.hand_base_rot[env_ids], goal_rot_local)
            self._update_goal_marker(env_ids)
            return

        # 生成均匀分布的随机数用于随机化旋转
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)

        # 根据随机数生成局部目标旋转
        goal_rot_local = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        # 将局部目标旋转转换为环境/世界坐标系
        # 通过四元数乘法将手部基础旋转与局部旋转相乘
        self.goal_rot[env_ids] = quat_mul(
            self.hand_base_rot[env_ids],
            goal_rot_local,
        )

        self._update_goal_marker(env_ids)

    def _update_goal_marker(self, env_ids):
        """Update goal marker position after assigning ``self.goal_rot``."""

        goal_marker_offset = torch.tensor(
            self.cfg.goal_marker_offset,
            dtype=torch.float,
            device=self.device,
        ).view(1, 3).repeat(len(env_ids), 1)

        # 世界/env Z 方向上方
        self.goal_pos[env_ids] = self.hand_base_pos[env_ids] + goal_marker_offset

        goal_pos_w = self.goal_pos + self.scene.env_origins
        if getattr(self.cfg, "use_yolo_student_obs", False) and getattr(self.cfg, "hide_goal_marker_from_yolo", False):
            # The marker uses the same tomato mesh and could be selected by
            # YOLO as a second object. Keep it out of all student cameras.
            goal_pos_w = torch.full_like(goal_pos_w, -10.0)
        self.goal_markers.visualize(goal_pos_w, self.goal_rot)

        # 重置目标缓冲区，表示目标已重置
        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # 计算手指尖数据
        # 获取手指身体在世界坐标系中的位置和旋转
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        # 将手指尖位置相对于环境原点进行偏移
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs,
            self.num_fingertips,
            3,
        )
        # 获取手指尖在世界坐标系中的速度
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]
        # 计算手指body的力传感器数据
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]
        force_norms = torch.norm(self.fingertip_force_sensors[:,:,:3], dim=2)
        self.fingertip_force_binary_results = (force_norms > self.cfg.contact_threshold).int() 

        # 获取手部关节位置和速度数据
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # 计算物体数据
        # 获取物体在世界坐标系中的位置和旋转
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        # 获取物体的线速度和角速度
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # 显示调试坐标系
        self._visualize_debug_frames()

    def compute_reduced_observations(self):
        if self.cfg.obs_type == "distill" and getattr(self.cfg, "use_yolo_student_obs", False):
            object_position_obs, target_angle_features = self._compute_yolo_student_object_observations()
            obs = torch.cat(
                (
                    unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    object_position_obs,
                    target_angle_features,
                    self.fingertip_force_binary_results,
                    self.actions,
                ),
                dim=-1,
            )
            return obs

        obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # self.goal_rot,
                self.fingertip_force_binary_results,  # 10
                self.actions,
            ),
            dim=-1,
        )
        return obs

    @torch.no_grad()
    def _compute_yolo_student_object_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized YOLO 2-D center and axial-angle error features."""

        if self.yolo_student_estimator is None:
            raise RuntimeError("YOLO student observations are enabled but the estimator is not initialized.")

        estimate = self.yolo_student_estimator.estimate(self._student_camera.data)

        object_y_angle = y_axis_twist_angle(self.object_rot, self.hand_base_rot)
        goal_y_angle = y_axis_twist_angle(self.goal_rot, self.hand_base_rot)
        angle_sign = float(self.cfg.yolo_angle_sign)

        if not self.yolo_angle_calibrated.any() and estimate.measurement_valid.any():
            # The camera convention is shared by all cloned environments, so
            # estimate one robust modulo-pi offset and retain it across resets.
            # For real deployment, set yolo_angle_offset_rad in the config and
            # this simulator-ground-truth calibration path is never used.
            candidates = torch.remainder(
                estimate.angle_image_rad[estimate.measurement_valid]
                - angle_sign * object_y_angle[estimate.measurement_valid],
                torch.pi,
            )
            shared_offset = 0.5 * torch.atan2(
                torch.sin(2.0 * candidates).mean(),
                torch.cos(2.0 * candidates).mean(),
            )
            shared_offset = torch.remainder(shared_offset, torch.pi)
            self.yolo_angle_offset[:] = shared_offset
            self.yolo_angle_calibrated[:] = True

        observed_object_y = angle_sign * torch.remainder(
            estimate.angle_image_rad - self.yolo_angle_offset,
            torch.pi,
        )
        # The PCA axis is modulo pi, so use double-angle features. This maps
        # equivalent errors theta and theta+pi to the same continuous vector.
        angle_features = axial_angle_error_features(observed_object_y, goal_y_angle)
        uncalibrated_features = torch.zeros_like(angle_features)
        uncalibrated_features[:, 1] = 1.0  # sin(0), cos(0)
        angle_features = torch.where(
            self.yolo_angle_calibrated.unsqueeze(-1),
            angle_features,
            uncalibrated_features,
        )
        self.last_yolo_estimate = estimate
        self.last_yolo_target_angle_features = angle_features.clone()
        self.last_yolo_observed_object_y = observed_object_y.clone()
        self.last_yolo_object_y_angle = object_y_angle.clone()
        self.last_yolo_goal_y_angle = goal_y_angle.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["yolo_position_valid"] = estimate.position_valid.float().mean()
        self.extras["log"]["yolo_angle_measurement_valid"] = estimate.measurement_valid.float().mean()
        self.extras["log"]["yolo_visible_ratio"] = estimate.visible_ratio.mean()
        self.extras["log"]["yolo_angle_offset_rad"] = self.yolo_angle_offset.mean()

        return estimate.position_image, angle_features

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                # self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.fingertip_force_binary_results, # 10
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        teacher_goal_rot = self.goal_rot
        if self.cfg.obs_type == "distill" and getattr(self.cfg, "y_axis_only", False):
            # PCA-based student observations cannot distinguish theta from
            # theta+pi. Present the teacher with the nearest equivalent goal
            # so its action target follows the same axial-angle semantics.
            teacher_goal_rot = nearest_axial_y_goal_rotation(
                self.object_rot,
                self.goal_rot,
                self.hand_base_rot,
                self.y_unit_tensor,
            )

        state_parts = [
            # hand
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.cfg.vel_obs_scale * self.hand_dof_vel,
            # object
            self.object_pos,
            self.object_rot,
            self.object_linvel,
            self.cfg.vel_obs_scale * self.object_angvel,
            # goal
            self.in_hand_pos,
            teacher_goal_rot,
            quat_mul(self.object_rot, quat_conjugate(teacher_goal_rot)),
        ]
        # The existing teacher checkpoint was trained with the original
        # 84-dimensional state, before 10 tactile values were added. Keep that
        # exact layout for distillation; other asymmetric tasks retain tactile.
        if not (
            self.cfg.obs_type == "distill"
            and not getattr(self.cfg, "distill_teacher_include_tactile", False)
        ):
            state_parts.append(self.fingertip_force_binary_results)
        state_parts.append(self.actions)
        states = torch.cat(state_parts, dim=-1)
        return states

    def _visualize_debug_frames(self):
        """显示手 root、object、goal 的坐标系。"""
        if not getattr(self.cfg, "debug_visualization", False):
            return
        if not hasattr(self, "frame_markers"):
            return

        # hand root frame
        hand_pos_w = self.hand.data.root_pos_w
        hand_rot_w = self.hand.data.root_quat_w

        # object frame
        object_pos_w = self.object.data.root_pos_w
        object_rot_w = self.object.data.root_quat_w

        # goal frame
        # self.goal_pos 是 env-local 坐标，所以需要加 env_origins 转成 world 坐标
        goal_pos_w = self.goal_pos + self.scene.env_origins
        goal_rot_w = self.goal_rot

        # 合并所有 frame
        frame_pos_w = torch.cat(
            (
                hand_pos_w,
                object_pos_w,
                goal_pos_w,
            ),
            dim=0,
        )

        frame_rot_w = torch.cat(
            (
                hand_rot_w,
                object_rot_w,
                goal_rot_w,
            ),
            dim=0,
        )

        self.frame_markers.visualize(frame_pos_w, frame_rot_w)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore

        super()._reset_idx(env_ids)

        # ---------------------------------------------------------------------
        # 1. Reset robot root pose: 放回 robot_cfg.init_state
        # ---------------------------------------------------------------------
        hand_root_state = self.hand.data.default_root_state[env_ids].clone()
        hand_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.hand.write_root_pose_to_sim(hand_root_state[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(hand_root_state[:, 7:], env_ids)

        # 更新 hand base buffer，后面 goal / in_hand_pos 还会用到
        self.hand_base_pos[env_ids] = self.hand.data.default_root_state[env_ids, 0:3]
        self.hand_base_rot[env_ids] = self.hand.data.default_root_state[env_ids, 3:7]

        # ---------------------------------------------------------------------
        # 2. Reset object root pose: 放回 object_cfg.init_state
        # ---------------------------------------------------------------------
        object_root_state = self.object.data.default_root_state[env_ids].clone()
        object_root_state[:, :3] += self.scene.env_origins[env_ids]
        if getattr(self.cfg, "y_axis_only", False):
            rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            y_noise = quat_from_angle_axis(
                rot_noise[:, 0] * 20.0 / 180.0 * np.pi,
                self.y_unit_tensor[env_ids],
            )
            object_root_state[:, 3:7] = quat_mul(object_root_state[:, 3:7], y_noise)
        else:
            # 如果你想完全固定初始姿态，就保持默认 rot，不随机
            rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            object_root_state[:, 3:7] = randomize_rotation(
                rot_noise[:, 0] * 20.0 / 180.0,
                rot_noise[:, 1] * 20.0 / 180.0,
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids],
            )
        object_root_state[:, 7:] = 0.0

        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids)

        # ---------------------------------------------------------------------
        # 3. Reset hand joints: 使用手动配置的默认手型 self.default_hand_dof_pos
        # ---------------------------------------------------------------------
        delta_max = self.hand_dof_upper_limits[env_ids] - self.default_hand_dof_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.default_hand_dof_pos[env_ids]

        u = sample_uniform(0.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * u

        dof_pos = self.default_hand_dof_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_pos = saturate(
            dof_pos,
            self.hand_dof_lower_limits[env_ids],
            self.hand_dof_upper_limits[env_ids],
        )

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        # ---------------------------------------------------------------------
        # 4. Reset delta-action buffers
        # ---------------------------------------------------------------------
        self.target_pos[env_ids] = dof_pos
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # ---------------------------------------------------------------------
        # 5. Reset reward target position
        # ---------------------------------------------------------------------
        # 这里直接把 in_hand_pos 设置为 object 默认位置的 env-local 坐标。
        # 这样 reward 里的 goal_dist = ||object_pos - in_hand_pos|| 初始接近 0。
        self.in_hand_pos[env_ids] = object_root_state[:, :3] - self.scene.env_origins[env_ids]

        # ---------------------------------------------------------------------
        # 6. Reset goal rotation
        # ---------------------------------------------------------------------
        self.successes[env_ids] = 0
        self._reset_target_pose(env_ids)

        self._compute_intermediate_values()

        if getattr(self, "yolo_student_estimator", None) is not None:
            self.yolo_student_estimator.reset(env_ids)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def y_axis_twist_angle(current_rot: torch.Tensor, reference_rot: torch.Tensor) -> torch.Tensor:
    """Return signed Y-axis swing-twist angle in ``[-pi, pi)``."""

    relative = quat_mul(quat_conjugate(reference_rot), current_rot)
    twist_norm = torch.sqrt(relative[:, 0] ** 2 + relative[:, 2] ** 2).clamp_min(1e-8)
    twist_w = relative[:, 0] / twist_norm
    twist_y = relative[:, 2] / twist_norm
    angle = 2.0 * torch.atan2(twist_y, twist_w)
    return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


@torch.jit.script
def axial_y_rotation_distance(
    object_rot: torch.Tensor,
    target_rot: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    """Absolute Y-axis angle error with a directionless period of pi."""

    object_angle = y_axis_twist_angle(object_rot, reference_rot)
    target_angle = y_axis_twist_angle(target_rot, reference_rot)
    half_period = np.pi / 2.0
    error = torch.remainder(object_angle - target_angle + half_period, np.pi) - half_period
    return torch.abs(error)


@torch.jit.script
def nearest_axial_y_goal_rotation(
    object_rot: torch.Tensor,
    target_rot: torch.Tensor,
    reference_rot: torch.Tensor,
    y_axis: torch.Tensor,
) -> torch.Tensor:
    """Return the target or its pi-equivalent nearest to the object angle."""

    object_angle = y_axis_twist_angle(object_rot, reference_rot)
    target_angle = y_axis_twist_angle(target_rot, reference_rot)
    half_period = np.pi / 2.0
    object_minus_target = torch.remainder(
        object_angle - target_angle + half_period,
        np.pi,
    ) - half_period
    nearest_target_angle = object_angle - object_minus_target
    target_local = quat_from_angle_axis(nearest_target_angle, y_axis)
    return quat_mul(reference_rot, target_local)


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    y_axis_only: bool,
    rotation_reference: torch.Tensor,
):
    # 计算物体到目标的距离和旋转差异
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    if y_axis_only:
        rot_dist = axial_y_rotation_distance(object_rot, target_rot, rotation_reference)
    else:
        rot_dist = rotation_distance(object_rot, target_rot)
    # 计算距离奖励和旋转奖励
    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    # 计算动作惩罚
    action_penalty = torch.sum(actions**2, dim=-1)
    # 综合奖励计算
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
    # 判断是否达成目标并更新目标重置标志
    goal_resets = torch.where(
        torch.abs(rot_dist) <= success_tolerance,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    # 更新成功次数
    successes = successes + goal_resets
    # 如果达成目标，增加奖励; 如果物体距离目标过远，施加坠落惩罚
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
    # 判断是否需要重置环境
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    # 计算需要重置的环境数量和已完成的连续成功次数
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    # 更新连续成功次数（使用滑动平均）
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes

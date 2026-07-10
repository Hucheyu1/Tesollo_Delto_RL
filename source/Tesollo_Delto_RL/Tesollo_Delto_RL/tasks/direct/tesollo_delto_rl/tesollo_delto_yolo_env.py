# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import save_images_to_file
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .tesollo_delto_rl_env import TesolloDeltoRlEnv
from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg

import os
import cv2
import numpy as np

@configclass
class TesolloDeltoYoloEnvCfg(TesolloDeltoRlEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)

    # camera
    # YOLO 数据采集需要目标在画面中足够大、且能看到手心中的物体。
    # 这里将相机放在手前上方，使用 world convention: 相机局部 +X 朝向目标、+Z 尽量朝上。
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.11, 0.25, 0.37),
            rot=(0.707106, 0, 0, -0.707106),
            convention="world",
        ),
        # data_types=["rgb", "depth", "semantic_segmentation"],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.45,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.5),
        ),
        width=640,
        height=480,
        update_latest_camera_pose=True,
        debug_vis=True,
    )

class TesolloDeltoYoloEnv(TesolloDeltoRlEnv):
    cfg: TesolloDeltoYoloEnvCfg

    def __init__(self, cfg: TesolloDeltoYoloEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.step_count = 0
        self.image_dir = "/root/gpufree-data/Tesollo_Delto_RL/datasets"

    def _setup_scene(self):
        # 添加手部、手内物体和目标物体
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # 克隆和复制（此环境中不需要过滤）
        self.scene.clone_environments(copy_from_source=False)
        # 将关节体添加到场景中 - 我们必须注册到场景以使用EventManager进行随机化
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        if self.cfg.obs_type in ("openai", "distill"):
            obs = super().compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = super().compute_full_observations()
        else:
            raise RuntimeError(f"Unknown obs_type: {self.cfg.obs_type}")

        observations = {"policy": obs}

        if self.cfg.asymmetric_obs:
            states = super().compute_full_state()
            observations = {"policy": obs, "critic": states}
        
        rgb = self._tiled_camera.data.output["rgb"]
        # rgb_img= self._preprocess_images(rgb)
        # self._save_images(rgb_img)
        rgb_np = rgb[0].cpu().numpy().astype('uint8')  # 取第一个环境的图像
        
        # 转换颜色空间从BGR到RGB（如果需要）
        rgb_display = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        
        # 显示图像
        cv2.imshow("RGB Camera View", rgb_display)
        cv2.waitKey(1)  # 等待1毫秒以更新显示

        return observations
    
    def _preprocess_images(self, rgb_img):
        """预处理输入图像。

        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
        """
        rgb_img = rgb_img / 255.0
        return rgb_img

    def _save_images(self, rgb_img: torch.Tensor):
        """将图像缓冲区写入文件。

        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)。
        """
        """只保存一个 env 的 RGB / depth / segmentation 图像。"""
        env_id = 0
        # 只取一个 env，保持 batch 维度: [1, H, W, C]
        rgb_one = rgb_img[env_id : env_id + 1]

        save_images_to_file(
            rgb_one,
            os.path.join(self.image_dir, f"rgb_env_step_{self.step_count}.png"),
        )

        self.step_count += 1
    
    def _visualize_debug_frames(self):
        """显示手 root、object、goal 的坐标系。"""
        if not getattr(self.cfg, "debug_visualization", False):
            return
        if not hasattr(self, "frame_markers"):
            return

        # hand root frame
        # hand_pos_w = self.hand.data.root_pos_w
        # hand_rot_w = self.hand.data.root_quat_w

        # object frame
        # object_pos_w = self.object.data.root_pos_w
        # object_rot_w = self.object.data.root_quat_w

        # goal frame
        # self.goal_pos 是 env-local 坐标，所以需要加 env_origins 转成 world 坐标
        goal_pos_w = self.goal_pos + self.scene.env_origins
        goal_rot_w = self.goal_rot

        # 合并所有 frame
        frame_pos_w = torch.cat(
            (
                # hand_pos_w,
                # object_pos_w,
                goal_pos_w,
            ),
            dim=0,
        )

        frame_rot_w = torch.cat(
            (
                # hand_rot_w,
                # object_rot_w,
                goal_rot_w,
            ),
            dim=0,
        )

        self.frame_markers.visualize(frame_pos_w, frame_rot_w)
    
    def _reset_target_pose(self, env_ids):
        # 目标旋转在手部-根部局部坐标系中
        # 只绕Y轴旋转，绕X和Z轴的角度固定为0
        # 生成均匀分布的随机数用于随机化Y轴旋转
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)

        # 根据随机数生成局部目标旋转（只绕Y轴旋转）
        # X和Z轴角度固定为0，只在Y轴上进行随机旋转
        goal_rot_local = quat_from_angle_axis(
            rand_floats[:, 0] * np.pi, 
            self.y_unit_tensor[env_ids]
        )

        # 将局部目标旋转转换为环境/世界坐标系
        # 通过四元数乘法将手部基础旋转与局部旋转相乘
        self.goal_rot[env_ids] = quat_mul(
            self.hand_base_rot[env_ids],
            goal_rot_local,
        )

        # 目标标记跟随物体的默认局部位置
        # 通过四元数应用将局部位置转换到世界坐标系
        # ---------------------------------------------------------------------
        goal_marker_offset = torch.tensor(
            self.cfg.goal_marker_offset,
            dtype=torch.float,
            device=self.device,
        ).view(1, 3).repeat(len(env_ids), 1)

        # 世界/env Z 方向上方
        self.goal_pos[env_ids] = self.hand_base_pos[env_ids] + goal_marker_offset

        goal_pos_w = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos_w, self.goal_rot)

        # 重置目标缓冲区，表示目标已重置
        self.reset_goal_buf[env_ids] = 0

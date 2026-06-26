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

from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .tesollo_delto_rl_env import TesolloDeltoRlEnv, unscale
from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg


@configclass
class TesolloDeltoRlVisionEnvCfg(TesolloDeltoRlEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=1.0, replicate_physics=True)

    # camera
    # YOLO 数据采集需要目标在画面中足够大、且能看到手心中的物体。
    # 这里将相机放在手前上方，使用 world convention: 相机局部 +X 朝向目标、+Z 尽量朝上。
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, -0.50, 0.70),
            rot=(0.508068, -0.218646, 0.135131, 0.822071),
            convention="world",
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.7,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),
        ),
        width=256,
        height=256,
        update_latest_camera_pose=True,
    )
    feature_extractor = FeatureExtractorCfg()

    # env
    observation_space = 118  # proprioception + goal keypoints + vision CNN embedding
    state_space = 111  # asymmetric states + vision CNN embedding


@configclass
class TesolloDeltoRlVisionEnvPlayCfg(TesolloDeltoRlVisionEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)


class TesolloDeltoRlVisionEnv(TesolloDeltoRlEnv):
    cfg: TesolloDeltoRlVisionEnvCfg

    def __init__(self, cfg: TesolloDeltoRlVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Use the log directory from the configuration
        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device, self.cfg.log_dir)
        self.embeddings = torch.zeros((self.num_envs, 27), dtype=torch.float32, device=self.device)
        # YOLO 数据集中只应该出现真实被抓物体；将 goal marker 隐藏到相机视野外，避免生成伪目标。
        self.hidden_goal_pos = torch.tensor((-10.0, -10.0, -10.0), dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self.goal_pos[:, :] = self.hidden_goal_pos
        self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)
        # keypoints buffer
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_image_observations(self):
        # generate ground truth keypoints for in-hand cube
        compute_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), out=self.gt_keypoints)

        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        rgb = self._tiled_camera.data.output["rgb"]

        # export_yolo_dataset.py 会关闭 CNN 训练/加载。此时环境只负责渲染图像和提供真值标签，
        # 直接返回零 embedding，避免旧 CNN 的 120x120 固定输入限制影响 YOLO 数据采集。
        if not self.cfg.feature_extractor.train and not self.cfg.feature_extractor.load_checkpoint:
            pose_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            self.embeddings.zero_()
        else:
            depth = self._tiled_camera.data.output["depth"]
            semantic = self._tiled_camera.data.output["semantic_segmentation"][..., :3]
            # train CNN to regress on keypoint positions
            pose_loss, embeddings = self.feature_extractor.step(rgb, depth, semantic, object_pose)
            self.embeddings = embeddings.clone().detach()
        # compute keypoints for goal cube
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1), out=self.goal_keypoints
        )

        obs = torch.cat(
            (
                self.embeddings,
                self.goal_keypoints.view(-1, 24),
            ),
            dim=-1,
        )

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["pose_loss"] = pose_loss

        return obs

    def _reset_target_pose(self, env_ids):
        super()._reset_target_pose(env_ids)
        if not hasattr(self, "hidden_goal_pos"):
            return
        # 父类会把 goal marker 放回手附近；视觉采集环境中再次隐藏，避免相机看到第二个物体。
        self.goal_pos[env_ids] = self.hidden_goal_pos[env_ids]
        self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot)

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        state = torch.cat((sim_states, self.embeddings), dim=-1)
        return state

    def _get_observations(self) -> dict:
        # proprioception observations
        state_obs = self._compute_proprio_observations()
        # vision observations from CMM
        image_obs = self._compute_image_observations()
        obs = torch.cat((state_obs, image_obs), dim=-1)
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
        return observations


@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """Computes positions of 8 corner keypoints of a cube.

    Args:
        pose: Position and orientation of the center of the cube. Shape is (N, 7)
        num_keypoints: Number of keypoints to compute. Default = 8
        size: Length of X, Y, Z dimensions of cube. Default = [0.06, 0.06, 0.06]
        out: Buffer to store keypoints. If None, a new buffer will be created.
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]  # type: ignore
        # express corner position in the world frame
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)  # type: ignore

    return out

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""FoundationPose 物体 6D 位姿估计封装。

FoundationPose 本身不是检测器：它需要 RGB、深度、相机内参、物体 mask 和物体 mesh。
这个模块只做输入/输出适配，便于同一套接口在 Isaac Lab 仿真和真机 RGB-D 相机上复用。
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class FoundationPoseEstimatorCfg:
    """FoundationPoseEstimator 的配置。"""

    mesh_path: str
    """物体网格文件路径。FoundationPose 通常需要 ``.obj``、``.ply`` 或 ``.stl``，不建议直接传 USD。"""

    foundationpose_root: str | None = None
    """FoundationPose 仓库路径；为 ``None`` 时读取环境变量 ``FOUNDATIONPOSE_ROOT``。"""

    device: str = "cuda:0"
    """FoundationPose 推理设备。"""

    est_refine_iter: int = 5
    """首帧 register 的 refine 迭代次数。"""

    track_refine_iter: int = 2
    """后续 track_one 的 refine 迭代次数。"""

    use_tracking: bool = True
    """首帧 register 成功后，后续是否调用 track_one。"""

    min_depth: float = 0.05
    """有效深度下限，单位 m。"""

    max_depth: float = 2.0
    """有效深度上限，单位 m。"""

    min_mask_pixels: int = 64
    """mask 内至少需要多少个有效像素才运行 FoundationPose。"""

    pose_is_object_to_camera: bool = True
    """FoundationPose 输出是否为物体到相机的变换矩阵。官方仓库默认是 True。"""

    debug: int = 0
    """FoundationPose debug 等级。"""

    debug_dir: str = "foundationpose_debug"
    """FoundationPose debug 输出目录。"""


@dataclass
class FoundationPoseEstimate:
    """批量 FoundationPose 位姿估计结果。"""

    valid: torch.Tensor
    confidence: torch.Tensor
    position_w: torch.Tensor
    quat_w: torch.Tensor
    pose_w: torch.Tensor
    pose_camera: torch.Tensor
    position_env: torch.Tensor | None = None


class FoundationPoseEstimator:
    """把 RGB-D + mask 输入转换为 FoundationPose 6D 位姿输出。"""

    def __init__(self, cfg: FoundationPoseEstimatorCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._foundationpose_ready = False
        self._mesh = None
        self._estimators: dict[int, Any] = {}
        self._registered: dict[int, bool] = {}

    @torch.no_grad()
    def estimate_from_tiled_camera(
        self,
        camera_data: Any,
        mask: torch.Tensor | None = None,
        env_origins: torch.Tensor | None = None,
        camera_quat_w: torch.Tensor | None = None,
        reset_tracking: bool = False,
    ) -> FoundationPoseEstimate:
        """直接从 Isaac Lab ``TiledCamera.data`` 估计位姿。

        Args:
            camera_data: Isaac Lab ``Camera.data`` 或 ``TiledCamera.data``。
            mask: 目标物体 mask，形状为 ``(N,H,W)`` 或 ``(N,H,W,1)``。如果为 ``None``，
                会尝试从 ``semantic_segmentation`` 中提取 tomato mask。
            env_origins: 可选环境原点，用于输出环境局部位置。
            camera_quat_w: 可选相机姿态。默认使用 ``camera_data.quat_w_ros``，即相机 +Z 朝前。
            reset_tracking: 是否强制重新 register，而不是沿用上一帧 track。
        """

        rgb = camera_data.output["rgb"]
        depth = camera_data.output.get("depth", None)
        if depth is None:
            depth = camera_data.output["distance_to_image_plane"]

        if mask is None:
            mask = mask_from_semantic(camera_data, semantic_label="tomato", foreground_fallback=True)

        quat_w = camera_quat_w if camera_quat_w is not None else camera_data.quat_w_ros
        return self.estimate(
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=camera_data.intrinsic_matrices,
            camera_pos_w=camera_data.pos_w,
            camera_quat_w=quat_w,
            env_origins=env_origins,
            reset_tracking=reset_tracking,
        )

    @torch.no_grad()
    def estimate(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
        env_origins: torch.Tensor | None = None,
        reset_tracking: bool = False,
    ) -> FoundationPoseEstimate:
        """从批量 RGB-D 图像和 mask 估计物体 6D 位姿。

        Args:
            rgb: RGB 图像，形状 ``(N,H,W,3)`` 或 ``(H,W,3)``。
            depth: 米制深度图，形状 ``(N,H,W,1)``、``(N,H,W)`` 或 ``(H,W)``。
            mask: bool/0-1 目标 mask，形状 ``(N,H,W,1)``、``(N,H,W)`` 或 ``(H,W)``。
            intrinsics: 相机内参，形状 ``(N,3,3)`` 或 ``(3,3)``。
            camera_pos_w: 世界系下相机位置，形状 ``(N,3)`` 或 ``(3,)``。
            camera_quat_w: 世界系下相机姿态，形状 ``(N,4)`` 或 ``(4,)``，四元数顺序为 ``wxyz``。
            env_origins: 可选环境原点，用于额外输出环境局部坐标。
            reset_tracking: 是否强制每个环境重新 register。
        """

        rgb_t = _ensure_batched_rgb(rgb)
        depth_t = _ensure_batched_depth(depth).to(device=rgb_t.device, dtype=torch.float32)
        mask_t = _ensure_batched_mask(mask).to(device=rgb_t.device)
        batch_size = rgb_t.shape[0]

        intrinsics_t = _expand_matrix(intrinsics, batch_size).to(device=rgb_t.device, dtype=torch.float32)
        camera_pos_t = _expand_vector(camera_pos_w, batch_size, 3).to(device=rgb_t.device, dtype=torch.float32)
        camera_quat_t = _expand_vector(camera_quat_w, batch_size, 4).to(device=rgb_t.device, dtype=torch.float32)

        valid = torch.zeros(batch_size, dtype=torch.bool, device=rgb_t.device)
        confidence = torch.zeros(batch_size, dtype=torch.float32, device=rgb_t.device)
        position_w = torch.zeros(batch_size, 3, dtype=torch.float32, device=rgb_t.device)
        quat_w = torch.zeros(batch_size, 4, dtype=torch.float32, device=rgb_t.device)
        quat_w[:, 0] = 1.0
        pose_w = torch.eye(4, dtype=torch.float32, device=rgb_t.device).repeat(batch_size, 1, 1)
        pose_camera = torch.eye(4, dtype=torch.float32, device=rgb_t.device).repeat(batch_size, 1, 1)

        for env_id in range(batch_size):
            mask_i = _valid_mask(mask_t[env_id], depth_t[env_id], self.cfg.min_depth, self.cfg.max_depth)
            if int(mask_i.sum().item()) < self.cfg.min_mask_pixels:
                self._registered[env_id] = False
                continue

            pose_camera_np = self._estimate_one(
                env_id=env_id,
                rgb=_to_numpy_uint8(rgb_t[env_id]),
                depth=_to_numpy_float32(depth_t[env_id]),
                mask=_to_numpy_mask(mask_i),
                intrinsic=_to_numpy_float32(intrinsics_t[env_id]),
                reset_tracking=reset_tracking,
            )
            if pose_camera_np is None:
                self._registered[env_id] = False
                continue

            if not self.cfg.pose_is_object_to_camera:
                pose_camera_np = np.linalg.inv(pose_camera_np)

            transform_world_camera = _transform_from_pos_quat(
                _to_numpy_float32(camera_pos_t[env_id]), _to_numpy_float32(camera_quat_t[env_id])
            )
            transform_world_object = transform_world_camera @ pose_camera_np
            pose_world_t = torch.as_tensor(transform_world_object, dtype=torch.float32, device=rgb_t.device)

            valid[env_id] = True
            confidence[env_id] = 1.0
            position_w[env_id] = pose_world_t[:3, 3]
            quat_w[env_id] = _matrix_to_quat_wxyz(pose_world_t[:3, :3])
            pose_w[env_id] = pose_world_t
            pose_camera[env_id] = torch.as_tensor(pose_camera_np, dtype=torch.float32, device=rgb_t.device)

        position_env = None
        if env_origins is not None:
            position_env = position_w - _expand_vector(env_origins, batch_size, 3).to(position_w.device)

        return FoundationPoseEstimate(
            valid=valid,
            confidence=confidence,
            position_w=position_w,
            quat_w=quat_w,
            pose_w=pose_w,
            pose_camera=pose_camera,
            position_env=position_env,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | None = None):
        """清除 tracking 状态；环境 reset 后应调用。"""

        if env_ids is None:
            self._registered.clear()
            return
        for env_id in env_ids:
            self._registered[int(env_id)] = False

    def _estimate_one(
        self,
        env_id: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsic: np.ndarray,
        reset_tracking: bool,
    ) -> np.ndarray | None:
        estimator = self._get_estimator(env_id)
        should_register = reset_tracking or not self.cfg.use_tracking or not self._registered.get(env_id, False)

        try:
            if should_register:
                pose = estimator.register(
                    K=intrinsic,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=self.cfg.est_refine_iter,
                )
            else:
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=intrinsic,
                    iteration=self.cfg.track_refine_iter,
                )
        except Exception:
            self._registered[env_id] = False
            raise

        if pose is None:
            return None

        pose_np = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        self._registered[env_id] = True
        return pose_np

    def _get_estimator(self, env_id: int):
        if env_id not in self._estimators:
            self._estimators[env_id] = self._create_foundationpose_model()
        return self._estimators[env_id]

    def _create_foundationpose_model(self):
        self._load_foundationpose_modules()

        import trimesh
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

        try:
            dr = importlib.import_module("nvdiffrast.torch")
        except ImportError:
            dr = getattr(importlib.import_module("estimater"), "dr", None)
            if dr is None:
                raise ImportError("FoundationPose requires nvdiffrast.torch, but it could not be imported.")

        if self._mesh is None:
            mesh = trimesh.load(self.cfg.mesh_path, force="mesh")
            # 有些格式会加载成 Scene；FoundationPose 需要单个 mesh。
            if hasattr(mesh, "geometry"):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            self._mesh = mesh

        mesh = self._mesh
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        debug_dir = Path(self.cfg.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        kwargs = dict(
            model_pts=np.asarray(mesh.vertices, dtype=np.float32),
            model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir=str(debug_dir),
            debug=self.cfg.debug,
        )
        try:
            return FoundationPose(**kwargs)
        except TypeError:
            kwargs.pop("debug_dir", None)
            kwargs.pop("debug", None)
            return FoundationPose(**kwargs)

    def _load_foundationpose_modules(self):
        if self._foundationpose_ready:
            return

        foundationpose_root = self.cfg.foundationpose_root or os.environ.get("FOUNDATIONPOSE_ROOT")
        if foundationpose_root:
            root = str(Path(foundationpose_root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)

        try:
            importlib.import_module("estimater")
        except ImportError as exc:
            raise ImportError(
                "未找到 FoundationPose。请先安装/clone NVLabs FoundationPose，并设置 "
                "FoundationPoseEstimatorCfg.foundationpose_root 或环境变量 FOUNDATIONPOSE_ROOT。"
            ) from exc

        mesh_path = Path(self.cfg.mesh_path).expanduser()
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"FoundationPose mesh 不存在: {mesh_path}. 请提供 tomato 的 .obj/.ply/.stl 网格文件。"
            )
        self.cfg.mesh_path = str(mesh_path.resolve())
        self._foundationpose_ready = True


def fixed_axis_twist_angle(
    quat_w: torch.Tensor,
    axis: tuple[float, float, float] | torch.Tensor = (0.0, 1.0, 0.0),
    reference_quat_w: torch.Tensor | None = None,
    *,
    degrees: bool = True,
) -> torch.Tensor:
    """Extract the signed twist angle around one fixed axis from ``wxyz`` quaternions.

    Args:
        quat_w: Current object orientation, shape ``(..., 4)``.
        axis: Unit axis expressed in the reference frame. For the current task,
            the hand-local fixed axis is normally ``(0, 1, 0)``.
        reference_quat_w: Optional reference/hand orientation, broadcastable to
            ``quat_w``. The relative rotation is ``conj(reference) * current``.
        degrees: Return degrees in ``[-180, 180)`` instead of radians in
            ``[-pi, pi)``.

    This swing-twist decomposition is preferable to selecting one Euler angle,
    because rotations around the other two axes do not get mixed into the
    requested fixed-axis component.
    """

    if quat_w.shape[-1] != 4:
        raise ValueError(f"quat_w 最后一维必须为 4，实际形状: {tuple(quat_w.shape)}")

    quat = quat_w.to(dtype=torch.float32)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    if reference_quat_w is not None:
        reference = reference_quat_w.to(device=quat.device, dtype=quat.dtype)
        if reference.shape[-1] != 4:
            raise ValueError(f"reference_quat_w 最后一维必须为 4，实际形状: {tuple(reference.shape)}")
        reference = reference / torch.linalg.vector_norm(reference, dim=-1, keepdim=True).clamp_min(1e-8)
        reference_conjugate = torch.cat((reference[..., :1], -reference[..., 1:]), dim=-1)
        quat = _quat_mul_wxyz(reference_conjugate, quat)

    axis_t = torch.as_tensor(axis, dtype=quat.dtype, device=quat.device)
    if axis_t.shape[-1] != 3:
        raise ValueError(f"axis 最后一维必须为 3，实际形状: {tuple(axis_t.shape)}")
    axis_t = axis_t / torch.linalg.vector_norm(axis_t, dim=-1, keepdim=True).clamp_min(1e-8)

    projected_length = torch.sum(quat[..., 1:] * axis_t, dim=-1, keepdim=True)
    twist_vector = projected_length * axis_t
    twist = torch.cat((quat[..., :1], twist_vector), dim=-1)
    twist = twist / torch.linalg.vector_norm(twist, dim=-1, keepdim=True).clamp_min(1e-8)

    signed_sin_half = torch.sum(twist[..., 1:] * axis_t, dim=-1)
    angle = 2.0 * torch.atan2(signed_sin_half, twist[..., 0])
    angle = torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.rad2deg(angle) if degrees else angle


def mask_from_semantic(
    camera_data: Any,
    semantic_label: str = "tomato",
    foreground_fallback: bool = False,
) -> torch.Tensor:
    """从 Isaac Lab semantic segmentation 输出中提取目标 mask。

    推荐把相机配置为 ``semantic_filter='class:tomato'`` 且
    ``colorize_semantic_segmentation=False``，这样 mask 会更干净。
    """

    semantic = camera_data.output.get("semantic_segmentation", None)
    if semantic is None:
        raise RuntimeError("camera_data.output 中没有 semantic_segmentation，无法为 FoundationPose 生成 mask。")

    if semantic.ndim == 4 and semantic.shape[-1] == 1:
        semantic_ids = semantic[..., 0]
        label_ids = _find_semantic_ids(camera_data.info.get("semantic_segmentation", {}), semantic_label)
        if label_ids:
            label_ids_t = torch.as_tensor(label_ids, dtype=semantic_ids.dtype, device=semantic_ids.device)
            return torch.isin(semantic_ids, label_ids_t)
        if foreground_fallback:
            return semantic_ids != 0
        return torch.zeros_like(semantic_ids, dtype=torch.bool)

    # 如果用户仍使用彩色 semantic，并且相机 semantic_filter 只保留目标类别，则非零像素可作为近似 mask。
    if semantic.ndim == 4 and semantic.shape[-1] >= 3:
        nonzero = semantic[..., :3].to(torch.int32).sum(dim=-1) > 0
        if foreground_fallback:
            return nonzero
        raise RuntimeError(
            "semantic_segmentation 当前是彩色输出，无法可靠映射 semantic id。"
            "请在 TiledCameraCfg 中设置 colorize_semantic_segmentation=False。"
        )

    if semantic.ndim == 3:
        if foreground_fallback:
            return semantic != 0
        return torch.zeros_like(semantic, dtype=torch.bool)

    raise RuntimeError(f"不支持的 semantic_segmentation 形状: {tuple(semantic.shape)}")


def _find_semantic_ids(info: Any, semantic_label: str) -> list[int]:
    ids: set[int] = set()

    def _walk(obj: Any, parent_key: Any = None):
        if isinstance(obj, dict):
            text = " ".join(str(v) for v in obj.values() if isinstance(v, str))
            if semantic_label in text and _is_int_like(parent_key):
                ids.add(int(parent_key))
            for key, value in obj.items():
                if _contains_label(value, semantic_label) and _is_int_like(key):
                    ids.add(int(key))
                _walk(value, key)
        elif isinstance(obj, list | tuple):
            for item in obj:
                _walk(item, parent_key)
        elif isinstance(obj, str) and semantic_label in obj and _is_int_like(parent_key):
            ids.add(int(parent_key))

    _walk(info)
    return sorted(ids)


def _contains_label(value: Any, semantic_label: str) -> bool:
    if isinstance(value, str):
        return semantic_label in value
    if isinstance(value, dict):
        return any(_contains_label(item, semantic_label) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_label(item, semantic_label) for item in value)
    return False


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _ensure_batched_rgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[-1] != 3:
        raise ValueError(f"RGB 图像最后一维应为 3 或 4，实际形状: {tuple(rgb.shape)}")
    return rgb


def _ensure_batched_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth 形状应为 (N,H,W) 或 (N,H,W,1)，实际形状: {tuple(depth.shape)}")
    return depth


def _ensure_batched_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"mask 形状应为 (N,H,W) 或 (N,H,W,1)，实际形状: {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _expand_matrix(matrix: torch.Tensor, batch_size: int) -> torch.Tensor:
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    if matrix.shape[0] != batch_size:
        raise ValueError(f"矩阵 batch 维度 {matrix.shape[0]} 与图像 batch {batch_size} 不一致。")
    return matrix


def _expand_vector(vector: torch.Tensor, batch_size: int, width: int) -> torch.Tensor:
    if vector.ndim == 1:
        vector = vector.unsqueeze(0).repeat(batch_size, 1)
    if vector.shape != (batch_size, width):
        raise ValueError(f"向量形状应为 {(batch_size, width)}，实际为 {tuple(vector.shape)}。")
    return vector


def _valid_mask(mask: torch.Tensor, depth: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    return mask & torch.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


def _to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))


def _to_numpy_mask(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(bool))


def _transform_from_pos_quat(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _quat_to_matrix_wxyz(quat_wxyz)
    transform[:3, 3] = position
    return transform


def _quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for broadcastable ``(..., 4)`` wxyz tensors."""

    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    scalar = left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True)
    vector = left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    quat = torch.empty(4, dtype=m.dtype, device=m.device)

    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        quat[0] = 0.25 * s
        quat[1] = (m[2, 1] - m[1, 2]) / s
        quat[2] = (m[0, 2] - m[2, 0]) / s
        quat[3] = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat[0] = (m[2, 1] - m[1, 2]) / s
        quat[1] = 0.25 * s
        quat[2] = (m[0, 1] + m[1, 0]) / s
        quat[3] = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat[0] = (m[0, 2] - m[2, 0]) / s
        quat[1] = (m[0, 1] + m[1, 0]) / s
        quat[2] = 0.25 * s
        quat[3] = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat[0] = (m[1, 0] - m[0, 1]) / s
        quat[1] = (m[0, 2] + m[2, 0]) / s
        quat[2] = (m[1, 2] + m[2, 1]) / s
        quat[3] = 0.25 * s

    return quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)

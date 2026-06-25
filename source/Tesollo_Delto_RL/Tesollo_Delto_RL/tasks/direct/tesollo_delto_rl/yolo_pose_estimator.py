# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""用于仿真到真机迁移实验的 YOLO 物体位姿估计工具。

这个模块刻意不依赖 Isaac Lab 环境类，只需要 RGB-D 图像、相机内参和相机外参。
在仿真中可以直接接 ``TiledCamera.data``；在真机上只要把真实相机的 RGB-D、内参
和手眼标定后的相机位姿传进来，也可以复用同一套 API。

默认路径是 YOLO bbox/segmentation + depth，输出物体中心位置；如果提供 YOLO-pose
模型和物体 3D 关键点模板，则会使用 OpenCV PnP 进一步估计完整 6D 姿态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class YoloPoseEstimatorCfg:
    """YoloPoseEstimator 的配置。"""

    model_path: str
    """Ultralytics YOLO 权重路径，例如 ``runs/tesollo_yolo/tomato_pose/weights/best.pt``。"""

    class_id: int | None = None
    """可选类别 id；为 ``None`` 时使用置信度最高的检测结果。"""

    confidence_threshold: float = 0.35
    """接受 YOLO 检测结果的最低置信度。"""

    min_depth: float = 0.05
    """有效深度下限，单位米。"""

    max_depth: float = 2.0
    """有效深度上限，单位米。"""

    min_mask_pixels: int = 20
    """bbox/mask 内至少需要多少个有效深度像素。"""

    depth_stat: str = "median"
    """深度聚合方式，支持 ``median`` 和 ``mean``。"""

    object_size: tuple[float, float, float] | None = None
    """可选 3D 包围盒尺寸；设置后使用 8 个盒角作为 PnP 关键点模板。"""

    keypoint_3d_object: tuple[tuple[float, float, float], ...] | None = None
    """可选物体系 3D 关键点模板；提供后优先于 ``object_size``。"""

    min_keypoints_for_pnp: int = 6
    """运行 PnP 前所需的最少可用关键点数量。"""

    min_keypoint_confidence: float = 0.2
    """YOLO 提供关键点置信度时，低于该阈值的关键点不参与 PnP。"""

    prefer_pnp_position: bool = True
    """PnP 成功时是否优先使用 PnP 平移作为 ``position_w``；否则使用深度中心。"""

    device: str = "cuda:0"
    """YOLO 模型推理设备。"""


@dataclass
class YoloPoseEstimate:
    """批量 YOLO 位姿估计结果。

    ``position_w`` 是世界坐标系下的物体中心；如果传入 ``env_origins``，则
    ``position_env`` 会给出每个环境局部坐标系下的位置。``quat_w`` 只有在
    YOLO-pose 关键点和 PnP 成功时才有效。
    """

    valid: torch.Tensor
    confidence: torch.Tensor
    position_w: torch.Tensor
    position_env: torch.Tensor | None = None
    quat_w: torch.Tensor | None = None
    bbox_xyxy: torch.Tensor | None = None
    keypoints_2d: torch.Tensor | None = None


class YoloPoseEstimator:
    """根据 YOLO 检测和 RGB-D 图像估计物体位置或完整 6D 姿态。"""

    def __init__(self, cfg: YoloPoseEstimatorCfg):
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloPoseEstimator requires the optional 'ultralytics' package. "
                "Install it in the Isaac Lab Python environment before using this helper."
            ) from exc

        self.model = YOLO(cfg.model_path)

    @torch.no_grad()
    def estimate_from_tiled_camera(
        self,
        camera_data: Any,
        env_origins: torch.Tensor | None = None,
        camera_quat_w: torch.Tensor | None = None,
    ) -> YoloPoseEstimate:
        """直接从 Isaac Lab ``TiledCamera.data`` 估计位姿。

        Args:
            camera_data: ``TiledCamera.data`` 或 ``Camera.data``。
            env_origins: 可选环境原点，形状为 ``(N, 3)``。
            camera_quat_w: 可选相机姿态；默认使用 ``camera_data.quat_w_ros``，
                与深度图的针孔相机约定一致，即相机 +Z 朝前。
        """

        rgb = camera_data.output["rgb"]
        depth = camera_data.output.get("depth", None)
        if depth is None:
            depth = camera_data.output["distance_to_image_plane"]

        quat_w = camera_quat_w if camera_quat_w is not None else camera_data.quat_w_ros
        return self.estimate(
            rgb=rgb,
            depth=depth,
            intrinsics=camera_data.intrinsic_matrices,
            camera_pos_w=camera_data.pos_w,
            camera_quat_w=quat_w,
            env_origins=env_origins,
        )

    @torch.no_grad()
    def estimate(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
        env_origins: torch.Tensor | None = None,
    ) -> YoloPoseEstimate:
        """从批量 RGB-D 图像估计物体中心或完整 6D 姿态。

        Args:
            rgb: RGB 图像，形状为 ``(N, H, W, 3)``，dtype 可为 uint8 或 float。
            depth: 米制深度图，形状为 ``(N, H, W)`` 或 ``(N, H, W, 1)``。
            intrinsics: 相机内参，形状为 ``(N, 3, 3)`` 或 ``(3, 3)``。
            camera_pos_w: 世界系下相机位置，形状为 ``(N, 3)``。
            camera_quat_w: 世界系下相机姿态，形状为 ``(N, 4)``；Isaac Lab 深度图
                建议使用 ``camera_data.quat_w_ros``。
            env_origins: 可选环境原点，用于额外输出环境局部坐标。
        """

        device = rgb.device
        rgb = _ensure_rgb_uint8(rgb)
        depth = _ensure_depth_shape(depth).to(device=device, dtype=torch.float32)
        intrinsics = _expand_intrinsics(intrinsics.to(device=device, dtype=torch.float32), rgb.shape[0])
        camera_pos_w = camera_pos_w.to(device=device, dtype=torch.float32)
        camera_quat_w = camera_quat_w.to(device=device, dtype=torch.float32)

        detections = self._run_yolo(rgb)
        num_images = rgb.shape[0]
        keypoint_3d_object = _get_keypoint_template(self.cfg, device)
        num_keypoints = _detected_keypoint_count(detections)
        valid = torch.zeros(num_images, dtype=torch.bool, device=device)
        confidence = torch.zeros(num_images, dtype=torch.float32, device=device)
        position_w = torch.full((num_images, 3), float("nan"), dtype=torch.float32, device=device)
        quat_w = torch.full((num_images, 4), float("nan"), dtype=torch.float32, device=device)
        bbox_xyxy = torch.full((num_images, 4), float("nan"), dtype=torch.float32, device=device)
        keypoints_2d = (
            torch.full((num_images, num_keypoints, 2), float("nan"), dtype=torch.float32, device=device)
            if num_keypoints > 0
            else None
        )

        for env_id, detection in enumerate(detections):
            if detection is None:
                continue

            pose_position_w = None
            pose_quat_w = None
            detected_keypoints = detection.get("keypoints_2d")
            detected_keypoint_conf = detection.get("keypoint_confidence")
            if detected_keypoints is not None and keypoints_2d is not None:
                keypoints_2d[env_id, : detected_keypoints.shape[0]] = detected_keypoints.to(device=device)

            # 若 YOLO-pose 给出了足够的 2D 关键点，则尝试用 PnP 估计完整位姿。
            if (
                keypoint_3d_object is not None
                and detected_keypoints is not None
                and detected_keypoints.shape[0] == keypoint_3d_object.shape[0]
            ):
                pnp_pose = _estimate_pnp_pose_world(
                    keypoints_2d=detected_keypoints.to(device=device),
                    keypoint_confidence=None
                    if detected_keypoint_conf is None
                    else detected_keypoint_conf.to(device=device),
                    keypoints_3d_object=keypoint_3d_object,
                    intrinsic=intrinsics[env_id],
                    camera_pos_w=camera_pos_w[env_id],
                    camera_quat_w=camera_quat_w[env_id],
                    cfg=self.cfg,
                )
                if pnp_pose is not None:
                    pose_position_w, pose_quat_w = pnp_pose

            # 深度路径提供稳健的位置估计，也作为 PnP 失败时的位置兜底。
            depth_position_w = None
            mask = detection["mask"].to(device=device)
            depth_i = depth[env_id]
            depth_mask = torch.isfinite(depth_i) & (depth_i > self.cfg.min_depth) & (depth_i < self.cfg.max_depth)
            valid_pixels = mask & depth_mask
            if int(valid_pixels.sum().item()) >= self.cfg.min_mask_pixels:
                ys, xs = torch.nonzero(valid_pixels, as_tuple=True)
                depth_values = depth_i[ys, xs]
                if self.cfg.depth_stat == "mean":
                    z = depth_values.mean()
                elif self.cfg.depth_stat == "median":
                    z = depth_values.median()
                else:
                    raise ValueError(f"Unsupported depth_stat: {self.cfg.depth_stat}")

                u = xs.to(dtype=torch.float32).mean()
                v = ys.to(dtype=torch.float32).mean()
                point_c = _pixel_to_camera_point(u, v, z, intrinsics[env_id])
                depth_position_w = _camera_to_world(point_c, camera_pos_w[env_id], camera_quat_w[env_id])

            if pose_position_w is None and depth_position_w is None:
                continue

            valid[env_id] = True
            confidence[env_id] = detection["confidence"]
            if pose_position_w is not None and (self.cfg.prefer_pnp_position or depth_position_w is None):
                position_w[env_id] = pose_position_w
            else:
                position_w[env_id] = depth_position_w
            if pose_quat_w is not None:
                quat_w[env_id] = pose_quat_w
            bbox_xyxy[env_id] = detection["bbox_xyxy"].to(device=device)

        position_env = None
        if env_origins is not None:
            position_env = position_w - env_origins.to(device=device, dtype=torch.float32)

        return YoloPoseEstimate(
            valid=valid,
            confidence=confidence,
            position_w=position_w,
            position_env=position_env,
            quat_w=quat_w,
            bbox_xyxy=bbox_xyxy,
            keypoints_2d=keypoints_2d,
        )

    def _run_yolo(self, rgb: torch.Tensor) -> list[dict[str, torch.Tensor] | None]:
        """批量运行 YOLO，并为每张图选择一个最可信的目标。"""

        rgb_np = list(rgb.detach().cpu().numpy())
        results = self.model.predict(rgb_np, conf=self.cfg.confidence_threshold, device=self.cfg.device, verbose=False)

        detections: list[dict[str, torch.Tensor] | None] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                detections.append(None)
                continue

            boxes_xyxy = torch.as_tensor(result.boxes.xyxy, dtype=torch.float32)
            confs = torch.as_tensor(result.boxes.conf, dtype=torch.float32)
            classes = torch.as_tensor(result.boxes.cls, dtype=torch.long)

            keep = confs >= self.cfg.confidence_threshold
            if self.cfg.class_id is not None:
                keep = keep & (classes == self.cfg.class_id)
            keep_ids = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            if keep_ids.numel() == 0:
                detections.append(None)
                continue

            best_local = torch.argmax(confs[keep_ids])
            best_id = keep_ids[best_local]
            bbox = boxes_xyxy[best_id]
            detection: dict[str, torch.Tensor] = {
                "confidence": confs[best_id],
                "bbox_xyxy": bbox,
            }

            # segmentation 模型优先使用 mask；普通 detection 模型则退化为 bbox mask。
            if result.masks is not None and result.masks.data is not None:
                masks = torch.as_tensor(result.masks.data, dtype=torch.float32)
                mask = masks[best_id] > 0.5
            else:
                height, width = rgb.shape[1], rgb.shape[2]
                mask = _bbox_to_mask(bbox, height, width)

            # Ultralytics 的 mask 可能是模型输入尺寸，这里统一缩放回原始相机图尺寸。
            if mask.shape != rgb.shape[1:3]:
                mask = torch.nn.functional.interpolate(
                    mask[None, None].float(), size=rgb.shape[1:3], mode="nearest"
                )[0, 0].bool()

            detection["mask"] = mask

            # YOLO-pose 模型会额外返回关键点；这些关键点稍后用于 PnP。
            if result.keypoints is not None and result.keypoints.xy is not None:
                keypoints_xy = torch.as_tensor(result.keypoints.xy, dtype=torch.float32)
                detection["keypoints_2d"] = keypoints_xy[best_id]
                keypoint_conf = getattr(result.keypoints, "conf", None)
                if keypoint_conf is not None:
                    detection["keypoint_confidence"] = torch.as_tensor(keypoint_conf, dtype=torch.float32)[best_id]

            detections.append(detection)

        return detections


def _detected_keypoint_count(detections: list[dict[str, torch.Tensor] | None]) -> int:
    """从 YOLO 输出中推断关键点数量，便于预分配批量张量。"""

    for detection in detections:
        if detection is not None and "keypoints_2d" in detection:
            return int(detection["keypoints_2d"].shape[0])
    return 0


def _get_keypoint_template(cfg: YoloPoseEstimatorCfg, device: torch.device) -> torch.Tensor | None:
    """根据配置生成物体系下的 3D 关键点模板。"""

    if cfg.keypoint_3d_object is not None:
        return torch.tensor(cfg.keypoint_3d_object, dtype=torch.float32, device=device)
    if cfg.object_size is not None:
        return _box_keypoints_3d(torch.tensor(cfg.object_size, dtype=torch.float32, device=device), device)
    return None


def _box_keypoints_3d(size: torch.Tensor, device: torch.device) -> torch.Tensor:
    """生成与导出脚本一致的 8 个 3D 盒角关键点。"""

    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return signs * size.view(1, 3) * 0.5


def _estimate_pnp_pose_world(
    keypoints_2d: torch.Tensor,
    keypoint_confidence: torch.Tensor | None,
    keypoints_3d_object: torch.Tensor,
    intrinsic: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    cfg: YoloPoseEstimatorCfg,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """使用 YOLO 2D 关键点和物体 3D 模板执行 PnP，并转成世界系位姿。"""

    valid_keypoints = torch.isfinite(keypoints_2d).all(dim=-1)
    if keypoint_confidence is not None:
        valid_keypoints = valid_keypoints & (keypoint_confidence >= cfg.min_keypoint_confidence)
    if int(valid_keypoints.sum().item()) < cfg.min_keypoints_for_pnp:
        return None

    keypoints_2d_np = keypoints_2d[valid_keypoints].detach().cpu().numpy()
    keypoints_3d_np = keypoints_3d_object[valid_keypoints].detach().cpu().numpy()
    intrinsic_np = intrinsic.detach().cpu().numpy()

    try:
        import cv2

        rvec, tvec = estimate_pose_from_yolo_keypoints_pnp(keypoints_2d_np, keypoints_3d_np, intrinsic_np)
        rot_obj_to_camera_np, _ = cv2.Rodrigues(rvec)
    except (ImportError, RuntimeError, ValueError):
        return None

    # OpenCV PnP 返回物体系到相机系的变换；再左乘相机到世界系旋转得到物体系到世界系。
    rot_obj_to_camera = torch.as_tensor(rot_obj_to_camera_np, dtype=torch.float32, device=keypoints_2d.device)
    rot_camera_to_world = _matrix_from_quat(camera_quat_w.unsqueeze(0)).squeeze(0)
    rot_obj_to_world = rot_camera_to_world @ rot_obj_to_camera
    quat_w = _quat_from_matrix(rot_obj_to_world.unsqueeze(0)).squeeze(0)

    # tvec 是物体原点在相机系下的位置，使用相机外参转换到世界系。
    point_c = torch.as_tensor(tvec.reshape(3), dtype=torch.float32, device=keypoints_2d.device)
    position_w = _camera_to_world(point_c, camera_pos_w, camera_quat_w)
    return position_w, quat_w


def estimate_pose_from_yolo_keypoints_pnp(
    keypoints_2d: np.ndarray,
    keypoints_3d_object: np.ndarray,
    intrinsic_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """使用 OpenCV PnP 从 YOLO 2D 关键点估计 6D 位姿。

    这是非球形物体的姿态估计路径。对于西红柿这类接近球形的物体，mask + depth
    的位置估计通常比完整姿态更稳定。

    Returns:
        相机坐标系下的 ``(rvec, tvec)``。
    """

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for PnP pose estimation.") from exc

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    if keypoints_2d.shape[0] != keypoints_3d_object.shape[0]:
        raise ValueError("2D and 3D keypoint counts must match for PnP.")
    if keypoints_2d.shape[0] < 4:
        raise ValueError("PnP requires at least 4 keypoints.")

    success, rvec, tvec = cv2.solvePnP(
        keypoints_3d_object.astype(np.float32),
        keypoints_2d.astype(np.float32),
        intrinsic_matrix.astype(np.float32),
        dist_coeffs.astype(np.float32),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed to estimate object pose.")
    return rvec, tvec


def _ensure_rgb_uint8(rgb: torch.Tensor) -> torch.Tensor:
    """把输入 RGB 统一成 uint8 格式，满足 Ultralytics 推理接口。"""

    if rgb.dtype == torch.uint8:
        return rgb[..., :3]
    rgb = rgb[..., :3]
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return rgb.clamp(0, 255).to(dtype=torch.uint8)


def _ensure_depth_shape(depth: torch.Tensor) -> torch.Tensor:
    """把深度图统一成 ``(N, H, W)``。"""

    if depth.dim() == 4 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.dim() == 3:
        return depth
    raise ValueError(f"Expected depth shape (N,H,W) or (N,H,W,1), got {tuple(depth.shape)}")


def _expand_intrinsics(intrinsics: torch.Tensor, batch_size: int) -> torch.Tensor:
    """把单个相机内参扩展成批量内参。"""

    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape[0] == 1 and batch_size > 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1)
    if intrinsics.shape[0] != batch_size:
        raise ValueError(f"Intrinsics batch size {intrinsics.shape[0]} does not match image batch {batch_size}.")
    return intrinsics


def _pixel_to_camera_point(u: torch.Tensor, v: torch.Tensor, z: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """根据像素坐标和深度反投影到相机坐标系。"""

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return torch.stack((x, y, z), dim=0)


def _camera_to_world(point_c: torch.Tensor, camera_pos_w: torch.Tensor, camera_quat_w: torch.Tensor) -> torch.Tensor:
    """把相机系 3D 点转换到世界系。"""

    return camera_pos_w + _quat_apply(camera_quat_w.unsqueeze(0), point_c.unsqueeze(0)).squeeze(0)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """使用 ``(w, x, y, z)`` 四元数旋转向量，避免真机侧额外依赖 Isaac Lab。"""

    q_vec = quat[..., 1:]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (quat[..., 0:1] * uv + uuv)


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """将 ``(w, x, y, z)`` 四元数转换为旋转矩阵。"""

    w, x, y, z = torch.unbind(quat, dim=-1)
    two_s = 2.0 / torch.clamp((quat * quat).sum(dim=-1), min=1.0e-12)
    return torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为 ``(w, x, y, z)`` 四元数。"""

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)) * 0.5
    qx = _copy_sign(torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 0.5, m21 - m12)
    qy = _copy_sign(torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) * 0.5, m02 - m20)
    qz = _copy_sign(torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) * 0.5, m10 - m01)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-12)
    return torch.where(quat[..., 0:1] < 0.0, -quat, quat)


def _copy_sign(magnitude: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """返回带有 ``sign`` 符号的 ``magnitude``。"""

    return magnitude * torch.where(sign < 0.0, -torch.ones_like(sign), torch.ones_like(sign))


def _bbox_to_mask(bbox_xyxy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """将 bbox 转成二值 mask，用于没有 segmentation 输出的检测模型。"""

    x1, y1, x2, y2 = bbox_xyxy.round().to(dtype=torch.long)
    x1 = int(torch.clamp(x1, 0, width - 1).item())
    x2 = int(torch.clamp(x2, 0, width - 1).item())
    y1 = int(torch.clamp(y1, 0, height - 1).item())
    y2 = int(torch.clamp(y2, 0, height - 1).item())
    mask = torch.zeros((height, width), dtype=torch.bool)
    if x2 >= x1 and y2 >= y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask

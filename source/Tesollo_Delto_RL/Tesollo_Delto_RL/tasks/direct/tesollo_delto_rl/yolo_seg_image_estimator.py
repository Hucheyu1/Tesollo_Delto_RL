"""Batched YOLO-seg 2-D center and image-axis estimation for distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass
class YoloSegImageEstimate:
    """Per-environment image observation used by the student policy."""

    # Normalized image coordinates in [-1, 1], with image center at (0, 0).
    position_image: torch.Tensor
    # PCA principal-axis angle in [0, pi); the axis is directionless.
    angle_image_rad: torch.Tensor
    position_valid: torch.Tensor
    angle_valid: torch.Tensor
    measurement_valid: torch.Tensor
    confidence: torch.Tensor
    mask_area: torch.Tensor
    # Binary selected YOLO mask, shaped (num_envs, height, width).
    mask_image: torch.Tensor
    anisotropy: torch.Tensor
    visible_ratio: torch.Tensor


class YoloSegImageEstimator:
    """Estimate filtered 2-D center and modulo-pi angle from YOLO masks."""

    def __init__(
        self,
        *,
        model_path: str,
        num_envs: int,
        device: str | torch.device,
        class_id: int = 0,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        inference_size: int = 640,
        min_mask_pixels: int = 64,
        position_gain: float = 0.65,
        min_anisotropy: float = 0.10,
        min_visible_ratio: float = 0.40,
        max_angle_jump_rad: float = 0.70,
        angle_gain: float = 0.55,
        velocity_gain: float = 0.20,
        velocity_decay: float = 0.90,
        reacquire_after: int = 5,
    ) -> None:
        model_path_obj = Path(model_path).expanduser()
        if not model_path_obj.is_file():
            raise FileNotFoundError(f"YOLO segmentation checkpoint does not exist: {model_path_obj}")

        # Import lazily so non-visual tasks do not require ultralytics.
        from ultralytics import YOLO

        self.model = YOLO(str(model_path_obj.resolve()))
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.class_id = int(class_id)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.inference_size = int(inference_size)
        self.min_mask_pixels = int(min_mask_pixels)
        self.position_gain = float(position_gain)
        self.min_anisotropy = float(min_anisotropy)
        self.min_visible_ratio = float(min_visible_ratio)
        self.max_angle_jump_rad = float(max_angle_jump_rad)
        self.angle_gain = float(angle_gain)
        self.velocity_gain = float(velocity_gain)
        self.velocity_decay = float(velocity_decay)
        self.reacquire_after = max(int(reacquire_after), 1)

        self._position = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._position_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._angle = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._angle_velocity = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._angle_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._angle_outlier_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reference_area = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def estimate(self, camera_data: Any) -> YoloSegImageEstimate:
        """Run batched RGB inference and return filtered image observations."""

        rgb = camera_data.output["rgb"][..., :3]
        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb_nchw.dtype == torch.uint8:
            rgb_nchw = rgb_nchw.to(dtype=torch.float32) / 255.0
        else:
            rgb_nchw = rgb_nchw.to(dtype=torch.float32)
            if float(rgb_nchw.detach().amax()) > 1.0:
                rgb_nchw = rgb_nchw / 255.0

        results = self.model.predict(
            source=rgb_nchw,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=4,
            imgsz=self.inference_size,
            device=str(self.device),
            verbose=False,
        )

        batch_size, height, width = rgb.shape[:3]
        if batch_size != self.num_envs:
            raise RuntimeError(f"YOLO batch {batch_size} does not match configured num_envs {self.num_envs}.")

        position_measurement = torch.zeros_like(self._position)
        angle_measurement = torch.zeros_like(self._angle)
        detection_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        raw_angle_valid = torch.zeros_like(detection_valid)
        confidence = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        mask_area = torch.zeros_like(confidence)
        mask_image = torch.zeros((self.num_envs, height, width), dtype=torch.bool, device=self.device)
        anisotropy = torch.zeros_like(confidence)

        for env_id, result in enumerate(results):
            selection = self._select_mask(result, height=height, width=width)
            if selection is None:
                continue
            mask, confidence_value = selection
            rows, columns = torch.where(mask)
            pixel_count = rows.numel()
            if pixel_count < self.min_mask_pixels:
                continue

            confidence[env_id] = confidence_value
            mask_area[env_id] = float(pixel_count)
            mask_image[env_id] = mask
            # Normalize around the optical image center. +X is right and +Y is
            # down, matching ordinary image coordinates and real-camera use.
            position_measurement[env_id, 0] = 2.0 * columns.to(torch.float32).mean() / max(width - 1, 1) - 1.0
            position_measurement[env_id, 1] = 2.0 * rows.to(torch.float32).mean() / max(height - 1, 1) - 1.0
            detection_valid[env_id] = True

            angle_value, anisotropy_value = _mask_principal_axis_from_points(rows, columns)
            angle_measurement[env_id] = angle_value
            anisotropy[env_id] = anisotropy_value
            raw_angle_valid[env_id] = anisotropy_value >= self.min_anisotropy

        self._reference_area = torch.maximum(self._reference_area, mask_area)
        visible_ratio = torch.where(
            self._reference_area > 0.0,
            mask_area / self._reference_area.clamp_min(1.0),
            torch.zeros_like(mask_area),
        ).clamp(0.0, 1.0)
        sufficiently_visible = visible_ratio >= self.min_visible_ratio
        position_valid = detection_valid & sufficiently_visible
        measurement_valid = raw_angle_valid & sufficiently_visible

        self._update_positions(position_measurement, position_valid)
        self._update_angles(angle_measurement, measurement_valid)

        return YoloSegImageEstimate(
            position_image=self._position.clone(),
            angle_image_rad=self._angle.clone(),
            position_valid=position_valid,
            angle_valid=self._angle_initialized.clone(),
            measurement_valid=measurement_valid,
            confidence=confidence,
            mask_area=mask_area,
            mask_image=mask_image,
            anisotropy=anisotropy,
            visible_ratio=visible_ratio,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Clear temporal state for reset environments."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._position[env_ids_t] = 0.0
        self._position_initialized[env_ids_t] = False
        self._angle[env_ids_t] = 0.0
        self._angle_velocity[env_ids_t] = 0.0
        self._angle_initialized[env_ids_t] = False
        self._angle_outlier_count[env_ids_t] = 0
        # Keep the largest clean mask area across episode resets. It provides
        # an amodal-area prior when the first frame of a new episode is already
        # occluded. A new estimator instance still starts from zero.

    def _select_mask(self, result: Any, *, height: int, width: int) -> tuple[torch.Tensor, float] | None:
        if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
            return None
        classes = result.boxes.cls.to(device=self.device, dtype=torch.long)
        confidences = result.boxes.conf.to(device=self.device)
        candidates = torch.where(classes == self.class_id)[0]
        if candidates.numel() == 0:
            return None
        selected = candidates[torch.argmax(confidences[candidates])]
        mask = result.masks.data[selected].to(self.device)
        if mask.shape != (height, width):
            mask = functional.interpolate(
                mask[None, None].to(torch.float32),
                size=(height, width),
                mode="nearest",
            )[0, 0]
        return mask > 0.5, float(confidences[selected])

    def _update_positions(self, measurement: torch.Tensor, valid: torch.Tensor) -> None:
        initialize = (~self._position_initialized) & valid
        self._position[initialize] = measurement[initialize]
        self._position_initialized[initialize] = True
        update = self._position_initialized & valid & ~initialize
        self._position[update] = (
            (1.0 - self.position_gain) * self._position[update]
            + self.position_gain * measurement[update]
        )

    def _update_angles(self, measurement: torch.Tensor, valid: torch.Tensor) -> None:
        predicted = torch.remainder(self._angle + self._angle_velocity, torch.pi)
        innovation = _signed_periodic_difference(measurement, predicted, period=torch.pi)
        initialized_and_valid = self._angle_initialized & valid
        accepted = initialized_and_valid & (innovation.abs() <= self.max_angle_jump_rad)
        outlier = initialized_and_valid & ~accepted
        self._angle_outlier_count[outlier] += 1
        self._angle_outlier_count[accepted | ~valid] = 0

        reacquire = outlier & (self._angle_outlier_count >= self.reacquire_after)
        accepted = accepted & ~reacquire
        if accepted.any():
            previous_angle = self._angle[accepted].clone()
            self._angle[accepted] = torch.remainder(
                predicted[accepted] + self.angle_gain * innovation[accepted], torch.pi
            )
            measured_velocity = _signed_periodic_difference(
                measurement[accepted], previous_angle, period=torch.pi
            )
            self._angle_velocity[accepted] = (
                (1.0 - self.velocity_gain) * self._angle_velocity[accepted]
                + self.velocity_gain * measured_velocity
            )

        initialize = ((~self._angle_initialized) & valid) | reacquire
        self._angle[initialize] = torch.remainder(measurement[initialize], torch.pi)
        self._angle_velocity[initialize] = 0.0
        self._angle_initialized[initialize] = True
        self._angle_outlier_count[initialize] = 0

        predicted_only = self._angle_initialized & ~accepted & ~initialize
        self._angle[predicted_only] = predicted[predicted_only]
        self._angle_velocity[predicted_only] *= self.velocity_decay


def _mask_principal_axis_from_points(
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return counter-clockwise image angle in ``[0, pi)`` and PCA anisotropy."""

    if rows.numel() < 5:
        zero = torch.zeros((), dtype=torch.float32, device=rows.device)
        return zero, zero
    points = torch.stack((columns.to(torch.float32), rows.to(torch.float32)), dim=-1)
    centered = points - points.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    primary_axis = eigenvectors[:, -1]
    angle = torch.remainder(torch.atan2(-primary_axis[1], primary_axis[0]), torch.pi)
    anisotropy = (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues.sum().clamp_min(1e-8)
    return angle, anisotropy.clamp(0.0, 1.0)


def _signed_periodic_difference(
    angle: torch.Tensor,
    reference: torch.Tensor,
    *,
    period: float | torch.Tensor,
) -> torch.Tensor:
    half_period = period / 2.0
    return torch.remainder(angle - reference + half_period, period) - half_period


def axial_angle_error_features(
    observed_angle_rad: torch.Tensor,
    target_angle_rad: torch.Tensor,
) -> torch.Tensor:
    """Encode a modulo-pi angle error as continuous double-angle features."""

    error = _signed_periodic_difference(
        observed_angle_rad,
        target_angle_rad,
        period=torch.pi,
    )
    return torch.stack((torch.sin(2.0 * error), torch.cos(2.0 * error)), dim=-1)

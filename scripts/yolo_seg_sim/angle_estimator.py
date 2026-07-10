"""Robust fixed-axis angle estimation from a segmentation-mask sequence.

The mask only describes the visible part of an object.  Consequently, the
single-frame PCA angle is treated as a noisy measurement rather than as the
final pose.  This module attaches shape-quality metrics to that measurement
and uses a period-aware temporal tracker to reject unreliable updates.

Angles use the usual mathematical image convention: zero points to the right
and positive angles rotate counter-clockwise.  Since a silhouette principal
axis has no direction, all angles are modulo 180 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskAngleMeasurement:
    """Single-frame PCA measurement and its reliability metrics."""

    angle_deg: float | None
    center: tuple[int, int] | None
    primary_axis: tuple[float, float] | None
    area: float
    anisotropy: float
    solidity: float
    contour: np.ndarray | None
    valid: bool
    reason: str


@dataclass(frozen=True)
class TrackedAngle:
    """Output of :class:`AxialAngleTracker` for one frame."""

    angle_deg: float | None
    measurement_deg: float | None
    predicted_deg: float | None
    accepted: bool
    quality: float
    visible_ratio: float
    status: str


def wrap_period(angle_deg: float, period_deg: float = 180.0) -> float:
    """Wrap an angle into ``[0, period_deg)``."""

    return float(angle_deg % period_deg)


def signed_periodic_difference(
    angle_deg: float,
    reference_deg: float,
    period_deg: float = 180.0,
) -> float:
    """Return the shortest signed ``angle-reference`` difference."""

    half_period = period_deg / 2.0
    return float((angle_deg - reference_deg + half_period) % period_deg - half_period)


def measure_mask_angle(mask: np.ndarray, min_area: float = 100.0) -> MaskAngleMeasurement:
    """Measure a visible mask's principal-axis angle using PCA.

    ``anisotropy`` is close to zero for a circle, where the principal axis is
    undefined, and increases as the shape becomes elongated. ``solidity`` is
    the mask area divided by its convex-hull area; a low value usually signals
    a strongly clipped or fragmented visible mask.
    """

    mask_u8 = _as_binary_mask(mask)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _invalid_measurement("no_contour")

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < min_area:
        return _invalid_measurement("area_too_small", area=contour_area, contour=contour)

    isolated_mask = np.zeros_like(mask_u8)
    cv2.drawContours(isolated_mask, [contour], -1, 255, thickness=cv2.FILLED)
    points_yx = np.argwhere(isolated_mask > 0)
    if len(points_yx) < 5:
        return _invalid_measurement("too_few_pixels", area=contour_area, contour=contour)

    points_xy = points_yx[:, [1, 0]].astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points_xy, mean=None)
    lambda_1, lambda_2 = (float(value) for value in eigenvalues[:, 0])
    eigenvalue_sum = lambda_1 + lambda_2
    anisotropy = (lambda_1 - lambda_2) / eigenvalue_sum if eigenvalue_sum > 0.0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / hull_area if hull_area > 0.0 else 0.0

    # OpenCV image coordinates have +Y down. Negating Y gives the usual
    # counter-clockwise mathematical angle used by the visualizer and tracker.
    primary_axis = eigenvectors[0]
    angle_deg = np.degrees(np.arctan2(-primary_axis[1], primary_axis[0]))
    angle_deg = wrap_period(float(angle_deg))
    center = (int(round(float(mean[0, 0]))), int(round(float(mean[0, 1]))))

    return MaskAngleMeasurement(
        angle_deg=angle_deg,
        center=center,
        primary_axis=(float(primary_axis[0]), float(primary_axis[1])),
        area=contour_area,
        anisotropy=float(np.clip(anisotropy, 0.0, 1.0)),
        solidity=float(np.clip(solidity, 0.0, 1.0)),
        contour=contour,
        valid=True,
        reason="ok",
    )


class AxialAngleTracker:
    """Quality-gated constant-velocity tracker for modulo-180 angles."""

    def __init__(
        self,
        *,
        period_deg: float = 180.0,
        min_detection_confidence: float = 0.5,
        min_anisotropy: float = 0.10,
        min_solidity: float = 0.82,
        min_visible_ratio: float = 0.45,
        max_innovation_deg: float = 35.0,
        max_speed_deg_per_frame: float = 30.0,
        angle_gain: float = 0.55,
        velocity_gain: float = 0.20,
        velocity_decay: float = 0.90,
        reacquire_after: int = 5,
        reference_area: float | None = None,
    ) -> None:
        self.period_deg = float(period_deg)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_anisotropy = float(min_anisotropy)
        self.min_solidity = float(min_solidity)
        self.min_visible_ratio = float(min_visible_ratio)
        self.max_innovation_deg = float(max_innovation_deg)
        self.max_speed_deg_per_frame = float(max_speed_deg_per_frame)
        self.angle_gain = float(angle_gain)
        self.velocity_gain = float(velocity_gain)
        self.velocity_decay = float(np.clip(velocity_decay, 0.0, 1.0))
        self.reacquire_after = max(int(reacquire_after), 1)
        self.reference_area = float(reference_area) if reference_area is not None else None

        self.angle_deg: float | None = None
        self.velocity_deg_per_frame = 0.0
        self.rejected_frames = 0
        self._strong_outlier_frames = 0

    def reset(self) -> None:
        """Reset temporal state while retaining a user-provided area prior."""

        self.angle_deg = None
        self.velocity_deg_per_frame = 0.0
        self.rejected_frames = 0
        self._strong_outlier_frames = 0

    def update(
        self,
        measurement: MaskAngleMeasurement | None,
        *,
        detection_confidence: float = 1.0,
        dt_frames: float = 1.0,
    ) -> TrackedAngle:
        """Update the tracker and return either a corrected or predicted angle."""

        dt_frames = max(float(dt_frames), 1e-6)
        predicted = None
        if self.angle_deg is not None:
            predicted = wrap_period(
                self.angle_deg + self.velocity_deg_per_frame * dt_frames,
                self.period_deg,
            )

        if measurement is None or not measurement.valid or measurement.angle_deg is None:
            status = "no_measurement" if measurement is None else measurement.reason
            self._strong_outlier_frames = 0
            return self._prediction_only(predicted, status)

        # The largest clean mask seen so far acts as an approximate amodal-area
        # prior. A fixed value can be supplied when it is known from simulation.
        if self.reference_area is None:
            self.reference_area = measurement.area
        else:
            self.reference_area = max(self.reference_area, measurement.area)
        visible_ratio = float(np.clip(measurement.area / max(self.reference_area, 1e-6), 0.0, 1.0))

        anisotropy_quality = min(measurement.anisotropy / max(self.min_anisotropy * 2.0, 1e-6), 1.0)
        solidity_quality = float(
            np.clip(
                (measurement.solidity - self.min_solidity) / max(1.0 - self.min_solidity, 1e-6),
                0.0,
                1.0,
            )
        )
        confidence_quality = float(np.clip(detection_confidence, 0.0, 1.0))
        quality = confidence_quality * anisotropy_quality * max(solidity_quality, 0.05) * visible_ratio

        rejection = self._quality_rejection_reason(
            measurement=measurement,
            detection_confidence=detection_confidence,
            visible_ratio=visible_ratio,
        )

        innovation = 0.0
        if rejection is None and predicted is not None:
            innovation = signed_periodic_difference(
                measurement.angle_deg,
                predicted,
                self.period_deg,
            )
            if abs(innovation) > self.max_innovation_deg:
                rejection = "innovation_too_large"

        if rejection == "innovation_too_large":
            self._strong_outlier_frames += 1
            if self._strong_outlier_frames >= self.reacquire_after:
                self.angle_deg = wrap_period(measurement.angle_deg, self.period_deg)
                self.velocity_deg_per_frame = 0.0
                self.rejected_frames = 0
                self._strong_outlier_frames = 0
                return TrackedAngle(
                    angle_deg=self.angle_deg,
                    measurement_deg=measurement.angle_deg,
                    predicted_deg=predicted,
                    accepted=True,
                    quality=quality,
                    visible_ratio=visible_ratio,
                    status="reacquired",
                )
        elif rejection is not None:
            self._strong_outlier_frames = 0

        if rejection is not None:
            result = self._prediction_only(predicted, rejection)
            return TrackedAngle(
                angle_deg=result.angle_deg,
                measurement_deg=measurement.angle_deg,
                predicted_deg=result.predicted_deg,
                accepted=False,
                quality=quality,
                visible_ratio=visible_ratio,
                status=rejection,
            )

        if self.angle_deg is None:
            self.angle_deg = wrap_period(measurement.angle_deg, self.period_deg)
            self.velocity_deg_per_frame = 0.0
            self.rejected_frames = 0
            self._strong_outlier_frames = 0
            return TrackedAngle(
                angle_deg=self.angle_deg,
                measurement_deg=measurement.angle_deg,
                predicted_deg=None,
                accepted=True,
                quality=quality,
                visible_ratio=visible_ratio,
                status="initialized",
            )

        assert predicted is not None
        adaptive_gain = float(np.clip(self.angle_gain * (0.5 + quality), 0.15, 0.85))
        corrected = wrap_period(predicted + adaptive_gain * innovation, self.period_deg)

        measured_velocity = signed_periodic_difference(
            measurement.angle_deg,
            self.angle_deg,
            self.period_deg,
        ) / dt_frames
        measured_velocity = float(
            np.clip(
                measured_velocity,
                -self.max_speed_deg_per_frame,
                self.max_speed_deg_per_frame,
            )
        )
        self.velocity_deg_per_frame = (
            (1.0 - self.velocity_gain) * self.velocity_deg_per_frame
            + self.velocity_gain * measured_velocity
        )
        self.angle_deg = corrected
        self.rejected_frames = 0
        self._strong_outlier_frames = 0

        return TrackedAngle(
            angle_deg=corrected,
            measurement_deg=measurement.angle_deg,
            predicted_deg=predicted,
            accepted=True,
            quality=quality,
            visible_ratio=visible_ratio,
            status="updated",
        )

    def _quality_rejection_reason(
        self,
        *,
        measurement: MaskAngleMeasurement,
        detection_confidence: float,
        visible_ratio: float,
    ) -> str | None:
        if detection_confidence < self.min_detection_confidence:
            return "low_detection_confidence"
        if measurement.anisotropy < self.min_anisotropy:
            return "axis_ambiguous"
        if measurement.solidity < self.min_solidity:
            return "mask_concave_or_fragmented"
        if visible_ratio < self.min_visible_ratio:
            return "object_occluded"
        return None

    def _prediction_only(self, predicted: float | None, status: str) -> TrackedAngle:
        if predicted is not None:
            self.angle_deg = predicted
            # Constant velocity is useful over a short occlusion, but indefinite
            # extrapolation quickly becomes worse than holding the latest pose.
            self.velocity_deg_per_frame *= self.velocity_decay
        self.rejected_frames += 1
        return TrackedAngle(
            angle_deg=predicted,
            measurement_deg=None,
            predicted_deg=predicted,
            accepted=False,
            quality=0.0,
            visible_ratio=0.0,
            status=status,
        )


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W) or (H, W, C), got {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _invalid_measurement(
    reason: str,
    *,
    area: float = 0.0,
    contour: np.ndarray | None = None,
) -> MaskAngleMeasurement:
    return MaskAngleMeasurement(
        angle_deg=None,
        center=None,
        primary_axis=None,
        area=area,
        anisotropy=0.0,
        solidity=0.0,
        contour=contour,
        valid=False,
        reason=reason,
    )

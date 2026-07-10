"""Run YOLO segmentation and robust axial-angle tracking on an image sequence."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from angle_estimator import AxialAngleTracker, MaskAngleMeasurement, TrackedAngle, measure_mask_angle


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR.parent.parent / "datasets" / "test_picture"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "best.pt"


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort ``step_2`` before ``step_10``."""

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a tomato's modulo-180 image angle with occlusion-aware temporal filtering."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="Input image directory.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO segmentation checkpoint.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--class-id", type=int, default=0, help="Target YOLO class ID.")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--min-anisotropy", type=float, default=0.10, help="Reject nearly circular PCA masks.")
    parser.add_argument("--min-solidity", type=float, default=0.82, help="Reject concave/fragmented masks.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.45, help="Reject masks below area prior.")
    parser.add_argument("--max-angle-jump", type=float, default=35.0, help="Maximum accepted innovation in degrees.")
    parser.add_argument(
        "--reacquire-after",
        type=int,
        default=5,
        help="Reinitialize after this many consecutive high-quality gated measurements.",
    )
    parser.add_argument(
        "--reference-area",
        type=float,
        default=None,
        help="Expected unoccluded mask area in pixels; otherwise learned from the sequence.",
    )
    parser.add_argument("--auto", action="store_true", help="Advance automatically instead of waiting for keys.")
    parser.add_argument("--delay", type=int, default=30, help="Display delay in ms when --auto is used.")
    parser.add_argument("--no-show", action="store_true", help="Disable the OpenCV window.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for visualized frames.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-frame quality/angle CSV report.")
    return parser.parse_args()


def read_images_from_directory(
    directory_path: Path,
    *,
    model_path: Path,
    conf: float = 0.5,
    iou: float = 0.7,
    class_id: int = 0,
    device: str | None = None,
    min_anisotropy: float = 0.10,
    min_solidity: float = 0.82,
    min_visible_ratio: float = 0.45,
    max_angle_jump: float = 35.0,
    reacquire_after: int = 5,
    reference_area: float | None = None,
    auto: bool = False,
    delay: int = 30,
    no_show: bool = False,
    output_dir: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Process an image directory in natural filename order."""

    directory_path = directory_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {model_path}")

    image_files = sorted(
        (path for path in directory_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_sort_key,
    )
    if not image_files:
        raise RuntimeError(f"No supported images found in: {directory_path}")

    # Keep ultralytics optional when importing this file for unit tests/tools.
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    tracker = AxialAngleTracker(
        min_detection_confidence=conf,
        min_anisotropy=min_anisotropy,
        min_solidity=min_solidity,
        min_visible_ratio=min_visible_ratio,
        max_innovation_deg=max_angle_jump,
        reacquire_after=reacquire_after,
        reference_area=reference_area,
    )

    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    if csv_path is not None:
        csv_path = csv_path.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[int, tuple[np.ndarray, dict[str, object]]] = {}
    current_index = 0
    max_processed_index = -1

    if not no_show:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 1200, 800)
        print("Keys: n/space=next, p=previous, q=quit")

    while 0 <= current_index < len(image_files):
        if current_index not in cache:
            # New frames must be processed in sequence so that tracker state is valid.
            if current_index != max_processed_index + 1:
                raise RuntimeError("Internal sequence error: uncached frames must be processed in order.")
            image_path = image_files[current_index]
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARN] Cannot read image: {image_path}")
                current_index += 1
                max_processed_index += 1
                continue

            started_at = time.perf_counter()
            predict_kwargs: dict[str, object] = {
                "source": image,
                "conf": conf,
                "iou": iou,
                "max_det": 10,
                "verbose": False,
            }
            if device is not None:
                predict_kwargs["device"] = device
            result = model.predict(**predict_kwargs)[0]
            inference_ms = (time.perf_counter() - started_at) * 1000.0

            selected = select_target_mask(result, image.shape[:2], class_id=class_id)
            if selected is None:
                measurement = None
                tracked = tracker.update(None, detection_confidence=0.0)
                mask = None
                detection_confidence = 0.0
            else:
                mask, detection_confidence = selected
                measurement = measure_mask_angle(mask)
                tracked = tracker.update(measurement, detection_confidence=detection_confidence)

            visualization = visualize_tracking(
                image=image,
                mask=mask,
                measurement=measurement,
                tracked=tracked,
                filename=image_path.name,
                index=current_index,
                total=len(image_files),
                detection_confidence=detection_confidence,
                inference_ms=inference_ms,
            )
            row = build_report_row(
                filename=image_path.name,
                measurement=measurement,
                tracked=tracked,
                detection_confidence=detection_confidence,
                inference_ms=inference_ms,
            )
            cache[current_index] = visualization, row
            max_processed_index = current_index

            if output_dir is not None:
                cv2.imwrite(str(output_dir / image_path.name), visualization)

            print(
                f"[{current_index + 1:04d}/{len(image_files):04d}] {image_path.name}: "
                f"angle={_format_angle(tracked.angle_deg)}, accepted={tracked.accepted}, "
                f"status={tracked.status}, visible={tracked.visible_ratio:.2f}"
            )

        visualization, _ = cache[current_index]
        if no_show:
            current_index += 1
            continue

        cv2.imshow("result", visualization)
        key = cv2.waitKey(max(delay, 1) if auto else 0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p") and not auto:
            current_index = max(current_index - 1, 0)
        else:
            current_index += 1

    if csv_path is not None:
        write_report(csv_path, [cache[index][1] for index in sorted(cache)])
        print(f"CSV report: {csv_path}")
    if not no_show:
        cv2.destroyAllWindows()


def select_target_mask(result, image_shape: tuple[int, int], class_id: int) -> tuple[np.ndarray, float] | None:
    """Select the highest-confidence target instance and resize its mask."""

    if result.masks is None or len(result.masks.data) == 0:
        return None

    masks = result.masks.data
    count = len(masks)
    confidences = np.ones(count, dtype=np.float32)
    classes = np.zeros(count, dtype=np.int64)
    if result.boxes is not None and len(result.boxes) == count:
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)

    candidates = np.flatnonzero(classes == class_id)
    if len(candidates) == 0:
        return None
    best_index = int(candidates[np.argmax(confidences[candidates])])
    mask = masks[best_index].detach().cpu().numpy()
    height, width = image_shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_u8 = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    return mask_u8, float(confidences[best_index])


def visualize_tracking(
    *,
    image: np.ndarray,
    mask: np.ndarray | None,
    measurement: MaskAngleMeasurement | None,
    tracked: TrackedAngle,
    filename: str,
    index: int,
    total: int,
    detection_confidence: float,
    inference_ms: float,
) -> np.ndarray:
    """Overlay mask, raw PCA measurement, filtered angle, and quality state."""

    canvas = image.copy()
    if mask is not None:
        overlay = np.zeros_like(canvas)
        overlay[mask > 0] = (180, 80, 20)
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.28, 0.0)

    if measurement is not None and measurement.contour is not None:
        cv2.drawContours(canvas, [measurement.contour], -1, (255, 160, 0), 2)

    center = measurement.center if measurement is not None else None
    if center is not None:
        cv2.circle(canvas, center, 4, (255, 255, 255), -1)
        if measurement is not None and measurement.angle_deg is not None:
            _draw_axis(canvas, center, measurement.angle_deg, (0, 255, 255), thickness=2)
        if tracked.angle_deg is not None:
            color = (0, 220, 0) if tracked.accepted else (0, 0, 255)
            _draw_axis(canvas, center, tracked.angle_deg, color, thickness=3)

    measurement_text = _format_angle(measurement.angle_deg if measurement is not None else None)
    tracked_text = _format_angle(tracked.angle_deg)
    anisotropy = measurement.anisotropy if measurement is not None else 0.0
    solidity = measurement.solidity if measurement is not None else 0.0
    area = measurement.area if measurement is not None else 0.0
    lines = [
        f"{filename} ({index + 1}/{total})",
        f"raw={measurement_text}  tracked={tracked_text}  status={tracked.status}",
        f"conf={detection_confidence:.2f}  visible={tracked.visible_ratio:.2f}  quality={tracked.quality:.2f}",
        f"anisotropy={anisotropy:.2f}  solidity={solidity:.2f}  area={area:.0f}  infer={inference_ms:.1f}ms",
        "yellow=raw PCA, green=accepted filter, red=prediction during rejection",
    ]
    _draw_text_panel(canvas, lines)
    return canvas


def build_report_row(
    *,
    filename: str,
    measurement: MaskAngleMeasurement | None,
    tracked: TrackedAngle,
    detection_confidence: float,
    inference_ms: float,
) -> dict[str, object]:
    return {
        "filename": filename,
        "raw_angle_deg": "" if measurement is None or measurement.angle_deg is None else measurement.angle_deg,
        "tracked_angle_deg": "" if tracked.angle_deg is None else tracked.angle_deg,
        "accepted": int(tracked.accepted),
        "status": tracked.status,
        "detection_confidence": detection_confidence,
        "visible_ratio": tracked.visible_ratio,
        "quality": tracked.quality,
        "mask_area": 0.0 if measurement is None else measurement.area,
        "anisotropy": 0.0 if measurement is None else measurement.anisotropy,
        "solidity": 0.0 if measurement is None else measurement.solidity,
        "inference_ms": inference_ms,
    }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _draw_axis(
    image: np.ndarray,
    center: tuple[int, int],
    angle_deg: float,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    length = int(min(image.shape[:2]) * 0.22)
    theta = np.radians(angle_deg)
    dx = int(round(length * np.cos(theta)))
    dy = int(round(-length * np.sin(theta)))
    point_1 = (center[0] - dx, center[1] - dy)
    point_2 = (center[0] + dx, center[1] + dy)
    cv2.line(image, point_1, point_2, color, thickness, cv2.LINE_AA)


def _draw_text_panel(image: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_height = 23
    panel_height = 12 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    for row, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 20 + row * line_height),
            font,
            font_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )


def _format_angle(angle: float | None) -> str:
    return "--" if angle is None else f"{angle:.1f}deg"


def main() -> None:
    args = parse_args()
    read_images_from_directory(
        args.images,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        class_id=args.class_id,
        device=args.device,
        min_anisotropy=args.min_anisotropy,
        min_solidity=args.min_solidity,
        min_visible_ratio=args.min_visible_ratio,
        max_angle_jump=args.max_angle_jump,
        reacquire_after=args.reacquire_after,
        reference_area=args.reference_area,
        auto=args.auto,
        delay=args.delay,
        no_show=args.no_show,
        output_dir=args.output_dir,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()

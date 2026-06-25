# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""从 Tesollo DG5F 视觉仿真中导出 YOLO 数据集。

脚本会保存 Isaac Lab tiled camera 的 RGB 图像，并把仿真物体的 3D 包围盒
投影到图像平面生成 YOLO 标签。普通 detection/segmentation 标签可用于
``yolo_pose_estimator.py`` 的 RGB-D 位置估计；使用 ``--label_type pose`` 时，
会额外导出 8 个 3D 盒角关键点，用于 YOLO-pose + PnP 的完整姿态估计。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export YOLO labels from the Tesollo DG5F vision simulation.")
parser.add_argument("--task", type=str, default="Tesollo-Delto-DG5F-Vision-Direct-v0", help="Vision task to run.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of simulated environments.")
parser.add_argument("--num_images", type=int, default=2000, help="Number of labeled images to export.")
parser.add_argument("--output_dir", type=str, default="datasets/tesollo_tomato_yolo", help="YOLO dataset directory.")
parser.add_argument("--class_name", type=str, default="tomato", help="YOLO class name.")
parser.add_argument("--class_id", type=int, default=0, help="YOLO class id.")
parser.add_argument("--train_ratio", type=float, default=0.9, help="Probability of assigning an image to train split.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for split/action sampling.")
parser.add_argument("--warmup_steps", type=int, default=10, help="Simulation steps before recording.")
parser.add_argument("--action_mode", choices=["zero", "random"], default="random", help="Action source while recording.")
parser.add_argument(
    "--label_type", choices=["detect", "segment", "pose"], default="detect", help="YOLO label format to write."
)
parser.add_argument("--object_size", type=float, nargs=3, default=(0.06, 0.06, 0.06), help="3D box size in meters.")
parser.add_argument("--min_box_pixels", type=float, default=4.0, help="Minimum projected bbox size in pixels.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 数据集导出必须打开相机；Hydra 参数保留给 Isaac Lab 环境解析。
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Tesollo_Delto_RL.tasks  # noqa: F401, E402


def main():
    """启动仿真、采集相机图像，并写出 YOLO 格式数据集。"""

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    output_dir = Path(args_cli.output_dir)
    _prepare_dataset_dirs(output_dir)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_vision_env_for_export(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    action_shape = env.action_space.shape
    num_saved = 0
    step_count = 0

    try:
        while simulation_app.is_running() and num_saved < args_cli.num_images:
            with torch.inference_mode():
                actions = _sample_actions(action_shape, device)
                env.step(actions)
                step_count += 1

                if step_count <= args_cli.warmup_steps:
                    continue

                # 使用仿真真值位姿生成标签：图像来自相机，标签来自物体 3D 包围盒投影。
                camera = env.unwrapped._tiled_camera
                labels = _compute_yolo_labels(
                    object_pos_env=env.unwrapped.object_pos,
                    object_quat_w=env.unwrapped.object_rot,
                    env_origins=env.unwrapped.scene.env_origins,
                    camera_pos_w=camera.data.pos_w,
                    camera_quat_w=camera.data.quat_w_ros,
                    intrinsics=camera.data.intrinsic_matrices,
                    image_height=camera.data.output["rgb"].shape[1],
                    image_width=camera.data.output["rgb"].shape[2],
                    object_size=torch.tensor(args_cli.object_size, dtype=torch.float32, device=device),
                )

                rgb = camera.data.output["rgb"]
                for env_id in range(rgb.shape[0]):
                    if num_saved >= args_cli.num_images:
                        break
                    label = labels[env_id]
                    if label is None:
                        continue

                    # 每个子环境对应一张图像；训练/验证划分用随机数保持简单可复现。
                    split = "train" if random.random() < args_cli.train_ratio else "val"
                    image_name = f"{num_saved:07d}.png"
                    _save_rgb_image(rgb[env_id], output_dir / "images" / split / image_name)
                    _save_yolo_label(label, output_dir / "labels" / split / image_name.replace(".png", ".txt"))
                    num_saved += 1

                if num_saved % 100 == 0 and num_saved > 0:
                    print(f"[INFO] Exported {num_saved}/{args_cli.num_images} images")
    finally:
        env.close()

    _write_dataset_yaml(output_dir, args_cli.class_name, args_cli.label_type)
    print(f"[INFO] YOLO dataset written to: {output_dir.resolve()}")
    print(f"[INFO] Dataset config: {(output_dir / 'dataset.yaml').resolve()}")


def _configure_vision_env_for_export(env_cfg):
    """导出数据时关闭 CNN 特征训练，只保留相机输出。"""

    if hasattr(env_cfg, "feature_extractor"):
        env_cfg.feature_extractor.train = False
        env_cfg.feature_extractor.load_checkpoint = False
        env_cfg.feature_extractor.write_image_to_file = False
    if hasattr(env_cfg, "tiled_camera"):
        env_cfg.tiled_camera.update_latest_camera_pose = True
        if "rgb" not in env_cfg.tiled_camera.data_types:
            env_cfg.tiled_camera.data_types.append("rgb")


def _sample_actions(action_shape: tuple[int, ...], device: str) -> torch.Tensor:
    """采样仿真动作，让物体/手在数据集中出现更丰富的相对姿态。"""

    if args_cli.action_mode == "zero":
        return torch.zeros(action_shape, device=device)
    return 2.0 * torch.rand(action_shape, device=device) - 1.0


def _compute_yolo_labels(
    object_pos_env: torch.Tensor,
    object_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    object_size: torch.Tensor,
) -> list[str | None]:
    """把每个环境中的物体 3D 包围盒投影为 YOLO 标签。

    这里使用仿真中的 ``object_pos`` 和 ``object_rot`` 真值生成标签，因此导出的
    YOLO-pose 关键点顺序必须和估计器里的 3D 模板顺序完全一致。
    """

    corners_obj = _box_corners(object_size, object_pos_env.device)
    corners_obj = corners_obj.unsqueeze(0).expand(object_pos_env.shape[0], -1, -1)

    flat_quat = object_quat_w[:, None, :].expand(-1, 8, -1).reshape(-1, 4)
    flat_corners = corners_obj.reshape(-1, 3)
    corners_env = object_pos_env[:, None, :] + quat_apply(flat_quat, flat_corners).reshape(-1, 8, 3)
    corners_w = corners_env + env_origins[:, None, :]

    corners_c = _world_to_camera_ros(corners_w, camera_pos_w, camera_quat_w)
    pixels, z = _project_camera_points(corners_c, intrinsics)

    labels: list[str | None] = []
    for env_id in range(object_pos_env.shape[0]):
        if torch.any(z[env_id] <= 1e-5):
            labels.append(None)
            continue

        u = pixels[env_id, :, 0]
        v = pixels[env_id, :, 1]
        x1 = torch.clamp(u.min(), 0.0, float(image_width - 1))
        y1 = torch.clamp(v.min(), 0.0, float(image_height - 1))
        x2 = torch.clamp(u.max(), 0.0, float(image_width - 1))
        y2 = torch.clamp(v.max(), 0.0, float(image_height - 1))

        if (x2 - x1) < args_cli.min_box_pixels or (y2 - y1) < args_cli.min_box_pixels:
            labels.append(None)
            continue

        labels.append(_format_yolo_label(x1, y1, x2, y2, pixels[env_id], z[env_id], image_width, image_height))

    return labels


def _box_corners(size: torch.Tensor, device: torch.device) -> torch.Tensor:
    """返回物体坐标系下 8 个包围盒角点，顺序需与 PnP 模板保持一致。"""

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
    return signs * size.to(device=device).view(1, 3) * 0.5


def _world_to_camera_ros(points_w: torch.Tensor, camera_pos_w: torch.Tensor, camera_quat_w: torch.Tensor) -> torch.Tensor:
    """把世界系 3D 点转换到相机 ROS 坐标系。"""

    rel = points_w - camera_pos_w[:, None, :]
    quat_cw = quat_conjugate(camera_quat_w)
    flat_quat = quat_cw[:, None, :].expand(-1, points_w.shape[1], -1).reshape(-1, 4)
    flat_rel = rel.reshape(-1, 3)
    return quat_apply(flat_quat, flat_rel).reshape_as(points_w)


def _project_camera_points(points_c: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """使用针孔相机内参把相机系 3D 点投影到像素坐标。"""

    fx = intrinsics[:, 0, 0].view(-1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1)

    x = points_c[..., 0]
    y = points_c[..., 1]
    z = points_c[..., 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return torch.stack((u, v), dim=-1), z


def _format_yolo_label(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    keypoints_px: torch.Tensor,
    keypoint_depth: torch.Tensor,
    image_width: int,
    image_height: int,
) -> str:
    """按 detection、segmentation 或 pose 格式写出一行 YOLO 标签。"""

    x1n = float(x1.item() / image_width)
    y1n = float(y1.item() / image_height)
    x2n = float(x2.item() / image_width)
    y2n = float(y2.item() / image_height)

    xc = (x1n + x2n) * 0.5
    yc = (y1n + y2n) * 0.5
    width = x2n - x1n
    height = y2n - y1n
    bbox = [xc, yc, width, height]

    if args_cli.label_type == "segment":
        polygon = [x1n, y1n, x2n, y1n, x2n, y2n, x1n, y2n]
        return f"{args_cli.class_id} " + " ".join(f"{v:.6f}" for v in polygon)

    if args_cli.label_type == "pose":
        keypoint_values: list[float] = []
        for point, z in zip(keypoints_px, keypoint_depth):
            u = float(point[0].item())
            v = float(point[1].item())
            inside_image = 0.0 <= u < image_width and 0.0 <= v < image_height
            # Ultralytics pose 标签中 visibility=2 表示可见，1 表示被裁剪/不可见但仍有坐标。
            visible = 2.0 if (inside_image and float(z.item()) > 1e-5) else 1.0
            keypoint_values.extend(
                [
                    min(max(u / image_width, 0.0), 1.0),
                    min(max(v / image_height, 0.0), 1.0),
                    visible,
                ]
            )
        return f"{args_cli.class_id} " + " ".join(f"{v:.6f}" for v in [*bbox, *keypoint_values])

    return f"{args_cli.class_id} " + " ".join(f"{v:.6f}" for v in bbox)


def _prepare_dataset_dirs(output_dir: Path):
    """创建 YOLO 标准目录结构。"""

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _save_rgb_image(rgb: torch.Tensor, path: Path):
    """保存 tiled camera 输出的 RGB 图像。"""

    array = rgb[..., :3].detach().cpu()
    if array.dtype != torch.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.clamp(0, 255).to(torch.uint8)
    Image.fromarray(array.numpy()).save(path)


def _save_yolo_label(label: str, path: Path):
    """保存一张图片对应的 YOLO 标签文件。"""

    path.write_text(label + os.linesep, encoding="utf-8")


def _write_dataset_yaml(output_dir: Path, class_name: str, label_type: str):
    """写出 Ultralytics 数据集配置；pose 任务需要额外声明关键点形状。"""

    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if label_type == "pose":
        lines.append("kpt_shape: [8, 3]")
    lines.extend(
        [
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    dataset_yaml = "\n".join(lines)
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


if __name__ == "__main__":
    main()
    simulation_app.close()

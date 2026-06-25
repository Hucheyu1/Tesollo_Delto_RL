# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""在导出的 Tesollo 数据集上训练 Ultralytics YOLO 模型。"""

from __future__ import annotations

import argparse


def main():
    """解析训练参数，并按任务类型选择 detection/pose/segmentation 模型。"""

    parser = argparse.ArgumentParser(description="Train YOLO on the exported Tesollo tomato dataset.")
    parser.add_argument("--data", type=str, default="datasets/tesollo_tomato_yolo/dataset.yaml", help="Dataset YAML.")
    parser.add_argument(
        "--task_type",
        choices=["detect", "pose", "segment"],
        default="detect",
        help="YOLO model family to train.",
    )
    parser.add_argument("--model", type=str, default=None, help="YOLO base model or checkpoint path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="Ultralytics device, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--project", type=str, default="runs/tesollo_yolo", help="Training output project dir.")
    parser.add_argument("--name", type=str, default=None, help="Training run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--exist_ok", action="store_true", default=False, help="Overwrite an existing run dir.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "This script requires the optional 'ultralytics' package. Install it with:\n"
            "  pip install ultralytics"
        ) from exc

    # 不同任务需要不同的 Ultralytics 预训练权重；用户也可以用 --model 显式覆盖。
    default_models = {
        "detect": "yolov8n.pt",
        "pose": "yolov8n-pose.pt",
        "segment": "yolov8n-seg.pt",
    }
    model_path = args.model or default_models[args.task_type]
    run_name = args.name or f"tomato_{args.task_type}"

    model = YOLO(model_path)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )
    if args.task_type == "pose":
        # 水平翻转会改变 3D 盒角关键点的身份编号，PnP 需要稳定编号，所以关闭它。
        train_kwargs["fliplr"] = 0.0

    model.train(**train_kwargs)

    print(f"[INFO] Training complete. Check: {args.project}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()

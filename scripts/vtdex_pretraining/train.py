# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""在 DG5F 仿真数据上训练与 VTDexJointEncoder 兼容的 VT-JointPretrain。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DG5FVTDexH5Dataset, discover_split


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (
    REPO_ROOT
    / "source"
    / "Tesollo_Delto_RL"
    / "Tesollo_Delto_RL"
    / "tasks"
    / "direct"
    / "tesollo_delto_rl"
)
DEFAULT_MODEL_ROOT = TASK_ROOT / "vtdex_pretrained"
DEFAULT_CONFIG = (
    DEFAULT_MODEL_ROOT
    / "model"
    / "vitac"
    / "model_and_config"
    / "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG5F VT-JointPretrain 单卡训练。")
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("datasets/dg5f_vtdex_pretrain")
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("outputs/vtdex_pretraining"),
        help="输出根目录；可直接作为 Tomato 配置的 vtdex_repo_root。",
    )
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--base_config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="可选：用官方或已有模型初始化；不指定则从头预训练。",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="恢复本脚本保存的训练 checkpoint（同时恢复 optimizer/epoch）。",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--effective_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    return parser.parse_args()


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state_dict.items()
    )


def _load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint 缺少 model_state_dict: {path}")
    return checkpoint


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    model_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "model_id": model_id,
        },
        path,
    )


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    update_lr,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    if is_train:
        optimizer.zero_grad(set_to_none=True)
        update_lr(epoch, 0.0)

    for batch_index, (image, tactile) in enumerate(loader):
        image = image.to(device=device, non_blocking=True)
        tactile = tactile.to(device=device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                loss, _, _, _ = model(
                    image,
                    tactile,
                    mask_ratio=float(model.mask_ratio),
                )
                # 参考实现的重建 loss 对 batch 求和；除以当前 batch，
                # 保留图像 196 patches 与 20×10 触觉权重的原始相对比例。
                loss = loss / image.shape[0]

            if is_train:
                normalized_loss = loss / accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(normalized_loss).backward()
                else:
                    normalized_loss.backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_lr(epoch, (batch_index + 1) / max(1, len(loader)))

        batch_size = image.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.effective_batch_size <= 0:
        raise ValueError("epochs、batch_size、effective_batch_size 必须为正数")
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError("effective_batch_size 必须能被 batch_size 整除")
    if not args.base_config.is_file():
        raise FileNotFoundError(f"模型配置不存在: {args.base_config}")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("请求 CUDA 训练，但当前 PyTorch 看不到 CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    train_files = discover_split(dataset_root, "train")
    val_files = discover_split(dataset_root, "val")
    if not train_files or not val_files:
        raise FileNotFoundError(
            f"{dataset_root} 必须同时包含 train/*.h5 和 val/*.h5；"
            "请分别运行两次采集命令。"
        )
    train_dataset = DG5FVTDexH5Dataset(
        train_files, max_samples=args.max_train_samples
    )
    val_dataset = DG5FVTDexH5Dataset(val_files, max_samples=args.max_val_samples)
    if len(train_dataset) < args.batch_size:
        raise ValueError(
            f"训练样本数 {len(train_dataset)} 小于 batch_size={args.batch_size}"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    with args.base_config.open(encoding="utf-8") as file:
        model_config = json.load(file)
    model_config["train_cfg"]["effective_bsz"] = int(args.effective_batch_size)
    model_config["dataset_cfg"]["max_epochs"] = int(args.epochs)
    model_config["dataset_cfg"]["warmup_epochs"] = min(
        int(model_config["dataset_cfg"]["warmup_epochs"]),
        max(1, args.epochs // 20),
    )
    if model_config["input_cfg"]["tactile_dim"] != 20:
        raise ValueError("当前 DG5F 数据协议只支持 tactile_dim=20")
    if not model_config["train_cfg"]["use_cls_token"]:
        raise ValueError("Tomato 下游任务要求 use_cls_token=true")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model_id or (
        f"vt20t-reall-tmr05-bin-ft-cls+dataset-DG5F-Sim-{timestamp}"
    )
    if "/" in model_id or "\\" in model_id:
        raise ValueError("model_id 不能包含路径分隔符")
    output_root = args.output_root.expanduser().resolve()
    checkpoint_dir = output_root / "checkpoints" / model_id
    export_dir = output_root / "model" / "vitac" / "model_and_config"
    export_dir.mkdir(parents=True, exist_ok=True)
    config_path = export_dir / f"{model_id}.json"
    exported_checkpoint_path = export_dir / f"{model_id}.pt"
    if args.resume is None and (
        config_path.exists()
        or exported_checkpoint_path.exists()
        or (checkpoint_dir.exists() and any(checkpoint_dir.iterdir()))
    ):
        raise FileExistsError(
            f"model_id={model_id!r} 已存在输出；请更换 model_id，"
            "或用 --resume 显式恢复，避免覆盖已有训练。"
        )
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(model_config, file, ensure_ascii=False, indent=2)

    model_repo_root = DEFAULT_MODEL_ROOT.resolve()
    sys.path.insert(0, str(model_repo_root))
    from model.vitac.vtt_reall import VTT_ReAll

    device = torch.device(args.device)
    model = VTT_ReAll(**model_config).to(device)
    optimizer, update_lr = model.configure_optimizer()
    start_epoch = 0
    if args.resume is not None:
        checkpoint = _load_checkpoint(args.resume.expanduser().resolve())
        model.load_state_dict(_clean_state_dict(checkpoint["model_state_dict"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"[INFO] 恢复训练: {args.resume}, start_epoch={start_epoch}")
    elif args.init_checkpoint is not None:
        checkpoint = _load_checkpoint(args.init_checkpoint.expanduser().resolve())
        model.load_state_dict(_clean_state_dict(checkpoint["model_state_dict"]), strict=True)
        print(f"[INFO] 用已有 VT-JointPretrain 初始化: {args.init_checkpoint}")
    else:
        print("[INFO] 从随机初始化开始训练 VT-JointPretrain")

    accumulation_steps = args.effective_batch_size // args.batch_size
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history_path = checkpoint_dir / "history.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch={args.batch_size}, accumulation={accumulation_steps}, amp={use_amp}"
    )
    print(f"[INFO] model_id={model_id}")

    best_val = float("inf")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
            epoch=epoch,
            update_lr=update_lr,
        )
        with torch.inference_mode():
            val_loss = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                accumulation_steps=1,
                use_amp=use_amp,
                scaler=None,
                epoch=epoch,
                update_lr=update_lr,
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_s": time.time() - start_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{epoch + 1:04d}/{args.epochs:04d}] "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={record['lr']:.3e}"
        )

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            _save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                model_id=model_id,
            )
        if is_best:
            _save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                model_id=model_id,
            )
            shutil.copy2(checkpoint_dir / "best.pt", exported_checkpoint_path)

    print("[INFO] 训练完成")
    print(f"[INFO] vtdex_repo_root={output_root}")
    print(f"[INFO] vtdex_model_id={model_id}")


if __name__ == "__main__":
    main()

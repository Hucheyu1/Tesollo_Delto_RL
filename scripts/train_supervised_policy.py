# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Train a supervised state-to-action policy from collected play datasets.

This script consumes datasets produced by ``scripts/collect_play_dataset.py``.
The default behavior-cloning target is:

    tensors["obs_policy"] -> tensors["action"]

You can also build a richer state by concatenating multiple collected fields:

    --input_keys hand_dof_pos,yolo_position_image,yolo_target_angle_features,tactile_binary,previous_action,goal_rot

All tensors are flattened from ``[T, N, ...]`` to ``[T*N, ...]`` before
training, while ``env_id`` / ``episode_id`` remain available in the dataset if
you later want trajectory-aware sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    datasets: list[str]
    output_dir: str
    run_name: str
    input_keys: list[str]
    target_key: str
    mask_key: str | None
    filter_valid_yolo: bool
    exclude_done: bool
    max_samples: int | None
    val_fraction: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    hidden_dims: list[int]
    activation: str
    dropout: float
    loss: str
    normalize_inputs: bool
    grad_clip: float | None
    seed: int
    device: str


class MLPPolicy(nn.Module):
    """MLP policy with input normalization embedded in the module."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        *,
        activation: str = "elu",
        dropout: float = 0.0,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)

        if input_mean is None:
            input_mean = torch.zeros(self.input_dim, dtype=torch.float32)
        if input_std is None:
            input_std = torch.ones(self.input_dim, dtype=torch.float32)
        self.register_buffer("input_mean", input_mean.to(dtype=torch.float32).view(1, -1))
        self.register_buffer("input_std", input_std.to(dtype=torch.float32).view(1, -1).clamp_min(1e-6))

        layers: list[nn.Module] = []
        last_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.input_mean) / self.input_std
        return self.net(obs)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_hidden_dims(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one hidden layer size.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("--hidden_dims values must be positive.")
    return dims


def _flatten_samples(tensor: torch.Tensor, sample_shape: tuple[int, int]) -> torch.Tensor:
    """Flatten [T, N, ...] into [T*N, feature_dim]."""

    if tensor.shape[:2] != sample_shape:
        raise ValueError(f"Expected leading shape {sample_shape}, got {tuple(tensor.shape)}.")
    flat = tensor.reshape(sample_shape[0] * sample_shape[1], *tensor.shape[2:])
    return flat.reshape(flat.shape[0], -1)


def _flatten_mask(tensor: torch.Tensor, sample_shape: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[:2] != sample_shape:
        raise ValueError(f"Expected mask leading shape {sample_shape}, got {tuple(tensor.shape)}.")
    mask = tensor.reshape(sample_shape[0] * sample_shape[1], *tensor.shape[2:])
    if mask.ndim > 1:
        mask = mask.reshape(mask.shape[0], -1).all(dim=-1)
    return mask.to(dtype=torch.bool)


def _load_dataset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "tensors" in data:
        tensors = data["tensors"]
        metadata = data.get("metadata", {})
    elif isinstance(data, dict):
        tensors = data
        metadata = {}
    else:
        raise ValueError(f"Unsupported dataset format in {path}. Expected a dict.")
    if not isinstance(tensors, dict):
        raise ValueError(f"Dataset {path} has no tensor dictionary.")
    return tensors, metadata


def _build_xy_from_dataset(
    *,
    path: Path,
    input_keys: list[str],
    target_key: str,
    mask_key: str | None,
    filter_valid_yolo: bool,
    exclude_done: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    tensors, metadata = _load_dataset(path)
    for key in input_keys + [target_key]:
        if key not in tensors:
            raise KeyError(f"Dataset {path} does not contain tensor key '{key}'.")

    first = tensors[input_keys[0]]
    if first.ndim < 2:
        raise ValueError(f"Tensor '{input_keys[0]}' in {path} must have at least [T, N] dimensions.")
    sample_shape = tuple(first.shape[:2])

    input_parts = []
    input_dims: dict[str, int] = {}
    for key in input_keys:
        part = _flatten_samples(tensors[key], sample_shape).to(dtype=torch.float32)
        input_parts.append(part)
        input_dims[key] = int(part.shape[-1])
    x = torch.cat(input_parts, dim=-1)
    y = _flatten_samples(tensors[target_key], sample_shape).to(dtype=torch.float32)

    mask = torch.ones(x.shape[0], dtype=torch.bool)
    if mask_key:
        if mask_key not in tensors:
            raise KeyError(f"Dataset {path} does not contain mask key '{mask_key}'.")
        mask &= _flatten_mask(tensors[mask_key], sample_shape)
    if filter_valid_yolo:
        if "valid_sample_mask" in tensors:
            mask &= _flatten_mask(tensors["valid_sample_mask"], sample_shape)
        elif "yolo_position_valid" in tensors and "yolo_measurement_valid" in tensors:
            mask &= _flatten_mask(tensors["yolo_position_valid"], sample_shape)
            mask &= _flatten_mask(tensors["yolo_measurement_valid"], sample_shape)
        else:
            print(f"[WARN] {path} has no YOLO validity tensors; --filter_valid_yolo is ignored for this file.")
    if exclude_done and "done" in tensors:
        mask &= ~_flatten_mask(tensors["done"], sample_shape)

    finite_mask = torch.isfinite(x).all(dim=-1) & torch.isfinite(y).all(dim=-1)
    mask &= finite_mask

    info = {
        "path": str(path),
        "metadata": metadata,
        "sample_shape": sample_shape,
        "num_raw_samples": int(x.shape[0]),
        "num_kept_samples": int(mask.sum().item()),
        "input_dims": input_dims,
        "target_dim": int(y.shape[-1]),
    }
    return x[mask], y[mask], mask, info


def _load_training_data(cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    infos: list[dict[str, Any]] = []
    input_dims_ref: dict[str, int] | None = None
    target_dim_ref: int | None = None

    for dataset_path in cfg.datasets:
        path = Path(dataset_path).expanduser()
        x, y, _, info = _build_xy_from_dataset(
            path=path,
            input_keys=cfg.input_keys,
            target_key=cfg.target_key,
            mask_key=cfg.mask_key,
            filter_valid_yolo=cfg.filter_valid_yolo,
            exclude_done=cfg.exclude_done,
        )
        if x.numel() == 0:
            print(f"[WARN] No samples kept from {path}.")
            continue
        if input_dims_ref is None:
            input_dims_ref = info["input_dims"]
            target_dim_ref = info["target_dim"]
        elif input_dims_ref != info["input_dims"] or target_dim_ref != info["target_dim"]:
            raise ValueError(
                f"Dataset {path} feature dimensions do not match previous datasets. "
                f"Expected input={input_dims_ref}, target={target_dim_ref}; got input={info['input_dims']}, "
                f"target={info['target_dim']}."
            )
        xs.append(x)
        ys.append(y)
        infos.append(info)
        print(f"[INFO] Loaded {path}: kept {info['num_kept_samples']}/{info['num_raw_samples']} samples.")

    if not xs:
        raise RuntimeError("No training samples were loaded.")
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    if cfg.max_samples is not None and cfg.max_samples > 0 and x_all.shape[0] > cfg.max_samples:
        generator = torch.Generator().manual_seed(cfg.seed)
        selected = torch.randperm(x_all.shape[0], generator=generator)[: cfg.max_samples]
        x_all = x_all[selected]
        y_all = y_all[selected]
        print(f"[INFO] Subsampled to {cfg.max_samples} samples.")
    return x_all, y_all, infos, input_dims_ref or {}


def _split_train_val(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset | None]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val_fraction must be in [0, 1).")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=generator)
    val_count = int(round(x.shape[0] * val_fraction))
    if x.shape[0] > 1:
        val_count = min(val_count, x.shape[0] - 1)
    else:
        val_count = 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_ds = TensorDataset(x[train_indices], y[train_indices])
    val_ds = TensorDataset(x[val_indices], y[val_indices]) if val_count > 0 else None
    return train_ds, val_ds


def _loss_fn(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if loss_name == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"Unsupported loss: {loss_name}")


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_name: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        batch_count = x.shape[0]
        total_loss += float(_loss_fn(pred, y, loss_name).item()) * batch_count
        total_mse += float(F.mse_loss(pred, y).item()) * batch_count
        total_mae += float(F.l1_loss(pred, y).item()) * batch_count
        total_count += batch_count
    return {
        "loss": total_loss / max(total_count, 1),
        "mse": total_mse / max(total_count, 1),
        "rmse": math.sqrt(total_mse / max(total_count, 1)),
        "mae": total_mae / max(total_count, 1),
    }


def _save_checkpoint(
    *,
    path: Path,
    model: MLPPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
    input_dims: dict[str, int],
    source_infos: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(cfg),
            "input_dim": model.input_dim,
            "action_dim": model.action_dim,
            "hidden_dims": model.hidden_dims,
            "activation": model.activation_name,
            "dropout": model.dropout,
            "input_keys": cfg.input_keys,
            "input_dims": input_dims,
            "target_key": cfg.target_key,
            "source_datasets": source_infos,
            "model_class": "MLPPolicy",
        },
        path,
    )


def _export_torchscript(model: MLPPolicy, output_path: Path, device: torch.device) -> None:
    model.eval()
    example = torch.zeros(1, model.input_dim, dtype=torch.float32, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(output_path))


def _make_run_dir(output_dir: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"supervised_policy_{timestamp}"
    run_dir = Path(output_dir).expanduser() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return str(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a supervised state-to-action policy.")
    parser.add_argument("datasets", nargs="+", help="One or more .pt datasets from scripts/collect_play_dataset.py.")
    parser.add_argument("--output_dir", "--output-dir", default="logs/supervised_policy", help="Root output folder.")
    parser.add_argument("--run_name", "--run-name", default=None, help="Optional run folder name.")
    parser.add_argument(
        "--input_keys",
        "--input-keys",
        default="obs_policy",
        help="Comma-separated tensor keys to concatenate as input features.",
    )
    parser.add_argument("--target_key", "--target-key", default="action", help="Tensor key used as target action.")
    parser.add_argument("--mask_key", "--mask-key", default=None, help="Optional boolean tensor key for sample filtering.")
    parser.add_argument(
        "--filter_valid_yolo",
        "--filter-valid-yolo",
        action="store_true",
        help="Keep only samples with valid YOLO position and angle measurements when those keys exist.",
    )
    parser.add_argument("--exclude_done", "--exclude-done", action="store_true", help="Drop samples whose done flag is true.")
    parser.add_argument("--max_samples", "--max-samples", type=int, default=None, help="Optional random subsample cap.")
    parser.add_argument("--val_fraction", "--val-fraction", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--hidden_dims", "--hidden-dims", default="256,256,128", help="Comma-separated hidden dims.")
    parser.add_argument("--activation", choices=("elu", "relu", "silu", "tanh", "gelu"), default="elu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=("mse", "smooth_l1", "l1"), default="mse")
    parser.add_argument("--no_normalize_inputs", "--no-normalize-inputs", action="store_true")
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=1.0, help="Set <=0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--save_every", "--save-every", type=int, default=0, help="Save periodic checkpoints every N epochs.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = TrainConfig(
        datasets=[str(Path(path).expanduser()) for path in args.datasets],
        output_dir=args.output_dir,
        run_name=args.run_name or "",
        input_keys=_parse_csv(args.input_keys),
        target_key=args.target_key,
        mask_key=args.mask_key,
        filter_valid_yolo=bool(args.filter_valid_yolo),
        exclude_done=bool(args.exclude_done),
        max_samples=args.max_samples,
        val_fraction=float(args.val_fraction),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=float(args.dropout),
        loss=args.loss,
        normalize_inputs=not bool(args.no_normalize_inputs),
        grad_clip=float(args.grad_clip) if args.grad_clip is not None and args.grad_clip > 0.0 else None,
        seed=int(args.seed),
        device=args.device,
    )
    if not cfg.input_keys:
        raise ValueError("--input_keys must contain at least one key.")

    run_dir = _make_run_dir(cfg.output_dir, args.run_name)
    cfg.run_name = run_dir.name
    print(f"[INFO] Output run dir: {run_dir}")
    print(f"[INFO] Input keys: {cfg.input_keys}")
    print(f"[INFO] Target key: {cfg.target_key}")

    x, y, source_infos, input_dims = _load_training_data(cfg)
    print(f"[INFO] Total samples: {x.shape[0]}")
    print(f"[INFO] Input dim: {x.shape[-1]}, action dim: {y.shape[-1]}")

    train_ds, val_ds = _split_train_val(x, y, val_fraction=cfg.val_fraction, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.device(cfg.device).type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.device(cfg.device).type == "cuda",
        )
        if val_ds is not None
        else None
    )

    if cfg.normalize_inputs:
        input_mean = train_ds.tensors[0].mean(dim=0)
        input_std = train_ds.tensors[0].std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        input_mean = torch.zeros(x.shape[-1])
        input_std = torch.ones(x.shape[-1])

    device = torch.device(cfg.device)
    model = MLPPolicy(
        input_dim=x.shape[-1],
        action_dim=y.shape[-1],
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable({"config": asdict(cfg), "source_datasets": source_infos, "input_dims": input_dims}), f, indent=2)

    best_metric = float("inf")
    best_epoch = -1
    last_metrics: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            pred = model(batch_x)
            loss = _loss_fn(pred, batch_y, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * batch_x.shape[0]
            train_count += batch_x.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)
        train_metrics = _evaluate(model, train_loader, device, cfg.loss)
        val_metrics = _evaluate(model, val_loader, device, cfg.loss) if val_loader is not None else train_metrics
        current_metric = val_metrics["mse"]
        last_metrics = {
            "train_loss_step": train_loss,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
        }

        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            _save_checkpoint(
                path=run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                metrics=last_metrics,
                input_dims=input_dims,
                source_infos=source_infos,
            )
            _export_torchscript(model, run_dir / "policy_jit.pt", device)

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            _save_checkpoint(
                path=run_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                metrics=last_metrics,
                input_dims=input_dims,
                source_infos=source_infos,
            )

        print(
            f"[INFO] epoch={epoch:04d}/{cfg.epochs} "
            f"train_mse={last_metrics['train_mse']:.6e} train_mae={last_metrics['train_mae']:.6e} "
            f"val_mse={last_metrics['val_mse']:.6e} val_mae={last_metrics['val_mae']:.6e} "
            f"{'*' if is_best else ''}"
        )

    _save_checkpoint(
        path=run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg.epochs,
        metrics=last_metrics,
        input_dims=input_dims,
        source_infos=source_infos,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_mse": best_metric,
        "last_metrics": last_metrics,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "torchscript_policy": str(run_dir / "policy_jit.pt"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print(f"[INFO] Best epoch: {best_epoch}, best val MSE: {best_metric:.6e}")
    print(f"[INFO] Saved best checkpoint: {run_dir / 'best.pt'}")
    print(f"[INFO] Saved TorchScript policy: {run_dir / 'policy_jit.pt'}")


if __name__ == "__main__":
    main()

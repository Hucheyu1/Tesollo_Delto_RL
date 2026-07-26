# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""验证新预训练模型可被 Tomato 使用，并输出 384 维 CLS 特征。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DG5FVTDexH5Dataset, discover_split


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE_ROOT = (
    REPO_ROOT
    / "source"
    / "Tesollo_Delto_RL"
    / "Tesollo_Delto_RL"
    / "tasks"
    / "direct"
    / "tesollo_delto_rl"
    / "vtdex_pretrained"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--vtdex_repo_root", type=Path, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model_dir = (
        args.vtdex_repo_root.expanduser().resolve()
        / "model"
        / "vitac"
        / "model_and_config"
    )
    config_path = model_dir / f"{args.model_id}.json"
    checkpoint_path = model_dir / f"{args.model_id}.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"模型必须同时包含 {config_path.name} 和 {checkpoint_path.name}"
        )
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    sys.path.insert(0, str(MODEL_CODE_ROOT))
    from model.vitac.vtt_reall import VTT_ReAll

    model = VTT_ReAll(**config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in checkpoint["model_state_dict"].items()
    )
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()

    val_files = discover_split(args.dataset_root.expanduser().resolve(), "val")
    dataset = DG5FVTDexH5Dataset(val_files, max_samples=8)
    image, tactile = next(iter(DataLoader(dataset, batch_size=min(8, len(dataset)))))
    with torch.inference_mode():
        features = model.get_representations(
            image.to(device), tactile.to(device), mode="cls"
        )
    if features.shape != (image.shape[0], 384):
        raise RuntimeError(f"CLS 特征尺寸错误: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise RuntimeError("CLS 特征包含 NaN/Inf")
    print(
        f"[OK] model_id={args.model_id}, batch={image.shape[0]}, "
        f"CLS={tuple(features.shape)}, mean={float(features.mean()):.6f}, "
        f"std={float(features.std()):.6f}"
    )


if __name__ == "__main__":
    main()

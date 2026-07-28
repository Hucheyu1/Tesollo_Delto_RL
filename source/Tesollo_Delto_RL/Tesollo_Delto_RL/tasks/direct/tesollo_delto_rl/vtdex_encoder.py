# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen VTDexManip visual-tactile representation adapter.

The original VTDexManip policy stores the frozen backbone outside PPO and learns
the downstream projection together with the actor.  This adapter exposes the
384-dimensional pretrained CLS token directly; the RSL-RL actor's first layer
therefore plays the role of the trainable downstream projection.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class VTDexJointEncoder(nn.Module):
    """Load and run VTDexManip's frozen joint RGB/binary-touch encoder."""

    def __init__(
        self,
        *,
        repo_root: str,
        model_id: str,
        device: str | torch.device,
        tactile_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.model_id = model_id
        self.device = torch.device(device)
        self.tactile_indices = tuple(int(index) for index in tactile_indices)

        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"VTDex model root does not exist: {self.repo_root}")

        config_path = self.repo_root / "model" / "vitac" / "model_and_config" / f"{model_id}.json"
        checkpoint_path = self.repo_root / "model" / "vitac" / "model_and_config" / f"{model_id}.pt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                "VTDex model files are missing. Expected both "
                f"{config_path} and {checkpoint_path}."
            )

        # Fine-tuning exports only ``model_and_config/*.json`` and ``*.pt``;
        # it intentionally does not duplicate VTDexManip's Python sources.
        # Load architecture code from a complete external checkout when one
        # was explicitly supplied, otherwise use the self-contained copy next
        # to this adapter. Model artifacts always come from ``repo_root``.
        external_code_file = self.repo_root / "model" / "vitac" / "vtt_reall.py"
        bundled_code_root = Path(__file__).resolve().parent / "vtdex_pretrained"
        bundled_code_file = bundled_code_root / "model" / "vitac" / "vtt_reall.py"
        if external_code_file.is_file():
            code_root = self.repo_root
        elif bundled_code_file.is_file():
            code_root = bundled_code_root
        else:
            raise FileNotFoundError(
                "VTDex architecture source is missing. Expected either "
                f"{external_code_file} or {bundled_code_file}."
            )

        # VTDexManip uses absolute imports rooted at its code tree (model.*).
        code_root_str = str(code_root)
        if code_root_str not in sys.path:
            sys.path.insert(0, code_root_str)
        from model.vitac.vtt_reall import VTT_ReAll

        with config_path.open(encoding="utf-8") as config_file:
            model_cfg = json.load(config_file)
        self.expected_tactile_dim = int(model_cfg["input_cfg"]["tactile_dim"])
        self.embedding_dim = int(model_cfg["encoder_decoder_cfg"]["encoder_embed_dim"])

        if len(self.tactile_indices) == 0:
            raise ValueError("vtdex_tactile_indices must contain at least one destination index")
        if len(set(self.tactile_indices)) != len(self.tactile_indices):
            raise ValueError("vtdex_tactile_indices must not contain duplicates")
        if min(self.tactile_indices) < 0 or max(self.tactile_indices) >= self.expected_tactile_dim:
            raise ValueError(
                f"vtdex_tactile_indices must be within [0, {self.expected_tactile_dim - 1}], "
                f"got {self.tactile_indices}"
            )

        backbone = VTT_ReAll(**model_cfg)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_state = checkpoint["model_state_dict"]
        model_state = backbone.state_dict()
        cleaned_state = OrderedDict(
            (key.replace("module.", ""), value)
            for key, value in checkpoint_state.items()
            if key.replace("module.", "") in model_state
        )
        backbone.load_state_dict(cleaned_state, strict=True)
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone.to(self.device)

        self.register_buffer(
            "image_mean",
            torch.tensor(
                (0.485, 0.456, 0.406), dtype=torch.float32, device=self.device
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                (0.229, 0.224, 0.225), dtype=torch.float32, device=self.device
            ).view(1, 3, 1, 1),
        )

    @torch.inference_mode()
    def forward(self, rgb: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        """Return one frozen joint representation per environment."""

        if rgb.ndim != 4 or rgb.shape[-1] < 3:
            raise ValueError(f"Expected RGB tensor [N,H,W,C>=3], got {tuple(rgb.shape)}")
        if tactile.ndim != 2 or tactile.shape[1] != len(self.tactile_indices):
            raise ValueError(
                f"Expected tactile tensor [N,{len(self.tactile_indices)}], got {tuple(tactile.shape)}"
            )
        if rgb.shape[0] != tactile.shape[0]:
            raise ValueError(
                f"RGB/tactile batch sizes must match, got {rgb.shape[0]} and {tactile.shape[0]}"
            )

        image = rgb[..., :3].permute(0, 3, 1, 2).contiguous()
        image = image.to(device=self.device, dtype=torch.float32)
        # Isaac Lab's ``rgb`` output and ordinary camera frames are uint8.
        # Floating-point callers are expected to provide data in [0, 1]; avoid
        # inspecting tensor values here because that would synchronize the GPU
        # on every control step.
        if not rgb.is_floating_point():
            image = image / 255.0
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = (image - self.image_mean) / self.image_std

        padded_tactile = torch.zeros(
            (tactile.shape[0], self.expected_tactile_dim),
            dtype=torch.float32,
            device=self.device,
        )
        padded_tactile[:, self.tactile_indices] = tactile.to(device=self.device, dtype=torch.float32)

        features = self.backbone.get_representations(image, padded_tactile, mode="cls")
        return features.detach()

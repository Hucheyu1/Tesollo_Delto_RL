# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""使用多模态历史观测训练 target_pos 回归策略。

默认映射关系：

    历史输入：
        hand_dof_pos
        tactile_binary
        yolo_position_image
        yolo_angle_image_rad
        yolo_mask_pixels
        goal_pos
        goal_rot

    监督目标：
        target_pos

数据集中的逐步 Tensor 应采用 ``[T, N, ...]`` 布局：

    T: 采集时间步数
    N: 并行环境数量

历史窗口严格按照 ``env_id + episode_id + episode_step`` 构造，不会跨环境、
不会跨 episode,  也不会把 reset 前后的帧拼到同一个窗口中。

网络结构：

    每帧二值 YOLO mask --CNN--> mask embedding
    每帧数值状态 ----------------> vector features
                                  |
                                  v
                    拼接为每帧多模态特征
                                  |
                                  v
                         单向 GRU 历史编码
                                  |
                                  v
                         MLP 回归 target_pos

使用单向 GRU, 因此部署时只需要当前帧和过去帧，不使用未来信息。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# 配置与数据结构
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    datasets: list[str]
    output_dir: str
    run_name: str

    # 输入与目标字段。
    vector_input_keys: list[str]
    mask_key: str
    target_key: str

    # 历史窗口。
    history_len: int
    history_stride: int
    mask_valid_mode: str

    # 数据划分与采样。
    split_mode: str
    val_fraction: float
    max_samples: int | None

    # 训练参数。
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    activation: str
    dropout: float
    loss: str
    normalize_vectors: bool
    grad_clip: float | None

    # 网络参数。
    mask_embedding_dim: int
    frame_embedding_dim: int
    gru_hidden_dim: int
    gru_layers: int
    head_hidden_dims: list[int]

    seed: int
    device: str
    num_workers: int
    save_every: int


@dataclass
class SourceData:
    """一个 .pt 数据文件在内存中的标准化表示。"""

    path: str
    metadata: dict[str, Any]

    # [T, N, vector_dim]
    vector_features: torch.Tensor

    # [T, N, 1, mask_height, mask_width]
    mask_pixels: torch.Tensor

    # [T, N, target_dim]
    target: torch.Tensor

    # [T, N]
    env_id: torch.Tensor
    episode_id: torch.Tensor
    episode_step: torch.Tensor

    # 每个原始字段摊平后的维数，用于部署时恢复输入顺序。
    vector_input_dims: dict[str, int]

    # 数据文件中实际使用的 target key。
    resolved_target_key: str


@dataclass(frozen=True)
class WindowIndex:
    """一个合法历史窗口的索引。"""

    source_id: int
    env_index: int
    end_time: int
    group_id: int


# -----------------------------------------------------------------------------
# 模型
# -----------------------------------------------------------------------------


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
    raise ValueError(f"不支持的激活函数: {name}")


class MaskEncoder(nn.Module):
    """将每一帧二值 YOLO mask 编码成低维特征。"""

    def __init__(self, output_dim: int, activation: str = "elu") -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("mask_embedding_dim 必须大于 0。")

        self.output_dim = int(output_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            _make_activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _make_activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _make_activation(activation),
            # 无论原 mask 是 48x64 还是其他尺寸，统一为 2x2 特征图。
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, self.output_dim),
            _make_activation(activation),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.conv(mask)


class HistoryTargetPolicy(nn.Module):
    """CNN + GRU 多模态历史策略，输出物理单位下的 target_pos。"""

    def __init__(
        self,
        *,
        vector_dim: int,
        target_dim: int,
        mask_embedding_dim: int,
        frame_embedding_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        head_hidden_dims: list[int],
        activation: str,
        dropout: float,
        vector_mean: torch.Tensor | None = None,
        vector_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if vector_dim <= 0:
            raise ValueError("vector_dim 必须大于 0。")
        if target_dim <= 0:
            raise ValueError("target_dim 必须大于 0。")
        if frame_embedding_dim <= 0 or gru_hidden_dim <= 0 or gru_layers <= 0:
            raise ValueError("frame_embedding_dim、gru_hidden_dim、gru_layers 必须大于 0。")

        self.vector_dim = int(vector_dim)
        self.target_dim = int(target_dim)
        self.mask_embedding_dim = int(mask_embedding_dim)
        self.frame_embedding_dim = int(frame_embedding_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_layers = int(gru_layers)
        self.head_hidden_dims = list(head_hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)

        if vector_mean is None:
            vector_mean = torch.zeros(self.vector_dim, dtype=torch.float32)
        if vector_std is None:
            vector_std = torch.ones(self.vector_dim, dtype=torch.float32)

        # [1, 1, vector_dim]，可直接广播到 [B, L, vector_dim]。
        self.register_buffer(
            "vector_mean",
            vector_mean.to(dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "vector_std",
            vector_std.to(dtype=torch.float32).view(1, 1, -1).clamp_min(1e-6),
        )

        self.mask_encoder = MaskEncoder(mask_embedding_dim, activation=activation)

        frame_input_dim = self.vector_dim + self.mask_embedding_dim
        frame_layers: list[nn.Module] = [
            nn.Linear(frame_input_dim, self.frame_embedding_dim),
            _make_activation(activation),
        ]
        if dropout > 0.0:
            frame_layers.append(nn.Dropout(dropout))
        self.frame_encoder = nn.Sequential(*frame_layers)

        # 单向 GRU，保证训练与真机在线部署都是因果的。
        self.temporal_encoder = nn.GRU(
            input_size=self.frame_embedding_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=dropout if self.gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        head_layers: list[nn.Module] = []
        last_dim = self.gru_hidden_dim
        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.Linear(last_dim, hidden_dim))
            head_layers.append(_make_activation(activation))
            if dropout > 0.0:
                head_layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        head_layers.append(nn.Linear(last_dim, self.target_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        vector_history: torch.Tensor,
        mask_history: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            vector_history: [B, L, vector_dim]
            mask_history: [B, L, 1, H, W]

        Returns:
            target_pos: [B, target_dim]
        """

        batch_size = vector_history.shape[0]
        history_len = vector_history.shape[1]

        vector_history = (vector_history - self.vector_mean) / self.vector_std

        # 将 B 和 L 合并，逐帧运行共享 CNN。
        masks_flat = mask_history.reshape(
            batch_size * history_len,
            mask_history.shape[2],
            mask_history.shape[3],
            mask_history.shape[4],
        )
        mask_features = self.mask_encoder(masks_flat)
        mask_features = mask_features.reshape(batch_size, history_len, -1)

        frame_features = torch.cat((vector_history, mask_features), dim=-1)
        frame_features = self.frame_encoder(frame_features)

        sequence_output, _ = self.temporal_encoder(frame_features)
        latest_feature = sequence_output[:, -1]
        return self.head(latest_feature)


# -----------------------------------------------------------------------------
# 数据读取与历史窗口构造
# -----------------------------------------------------------------------------


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in values):
        raise ValueError("隐藏层维数必须全部大于 0。")
    return values


def _load_dataset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "tensors" in data:
        tensors = data["tensors"]
        metadata = data.get("metadata", {})
    elif isinstance(data, dict):
        tensors = data
        metadata = {}
    else:
        raise ValueError(f"{path} 的格式不受支持，预期为 dict。")

    if not isinstance(tensors, dict):
        raise ValueError(f"{path} 中不存在 Tensor 字典。")

    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"数据字段 '{key}' 不是 torch.Tensor，而是 {type(value)!r}。")

    return tensors, metadata


def _resolve_target_key(tensors: dict[str, torch.Tensor], requested_key: str) -> str:
    """兼容 target_pos / targets_pos 两种命名。"""

    if requested_key in tensors:
        return requested_key

    aliases = {
        "target_pos": "targets_pos",
        "targets_pos": "target_pos",
    }
    alias = aliases.get(requested_key)
    if alias is not None and alias in tensors:
        print(f"[WARN] 数据中没有 '{requested_key}'，自动使用别名 '{alias}'。")
        return alias

    raise KeyError(
        f"数据中不存在目标字段 '{requested_key}'。"
        f"可用字段为: {sorted(tensors.keys())}"
    )


def _check_leading_shape(
    tensor: torch.Tensor,
    expected: tuple[int, int],
    *,
    key: str,
    path: Path,
) -> None:
    if tensor.ndim < 2 or tuple(tensor.shape[:2]) != expected:
        raise ValueError(
            f"{path} 中字段 '{key}' 的前两维应为 {expected}，"
            f"实际为 {tuple(tensor.shape)}。"
        )


def _load_source_data(
    *,
    path: Path,
    vector_input_keys: list[str],
    mask_key: str,
    target_key: str,
) -> SourceData:
    tensors, metadata = _load_dataset(path)

    required_index_keys = ["env_id", "episode_id", "episode_step"]
    for key in vector_input_keys + [mask_key] + required_index_keys:
        if key not in tensors:
            raise KeyError(
                f"{path} 中缺少字段 '{key}'。可用字段为: {sorted(tensors.keys())}"
            )

    resolved_target_key = _resolve_target_key(tensors, target_key)

    first = tensors[vector_input_keys[0]]
    if first.ndim < 2:
        raise ValueError(f"{path} 中字段 '{vector_input_keys[0]}' 至少需要 [T, N] 两维。")

    time_steps, num_envs = int(first.shape[0]), int(first.shape[1])
    sample_shape = (time_steps, num_envs)

    vector_parts: list[torch.Tensor] = []
    vector_input_dims: dict[str, int] = {}

    for key in vector_input_keys:
        tensor = tensors[key]
        _check_leading_shape(tensor, sample_shape, key=key, path=path)
        flattened = tensor.reshape(time_steps, num_envs, -1).to(dtype=torch.float32)
        vector_parts.append(flattened)
        vector_input_dims[key] = int(flattened.shape[-1])

    vector_features = torch.cat(vector_parts, dim=-1).contiguous()

    mask = tensors[mask_key]
    _check_leading_shape(mask, sample_shape, key=mask_key, path=path)
    if mask.ndim == 4:
        # [T, N, H, W] -> [T, N, 1, H, W]
        mask = mask.unsqueeze(2)
    elif mask.ndim == 5 and mask.shape[2] == 1:
        pass
    else:
        raise ValueError(
            f"{path} 中 '{mask_key}' 应为 [T,N,H,W] 或 [T,N,1,H,W]，"
            f"实际为 {tuple(mask.shape)}。"
        )
    mask = (mask > 0).to(dtype=torch.float32).contiguous()

    target = tensors[resolved_target_key]
    _check_leading_shape(target, sample_shape, key=resolved_target_key, path=path)
    target = target.reshape(time_steps, num_envs, -1).to(dtype=torch.float32).contiguous()

    env_id = tensors["env_id"]
    episode_id = tensors["episode_id"]
    episode_step = tensors["episode_step"]
    for key, tensor in (
        ("env_id", env_id),
        ("episode_id", episode_id),
        ("episode_step", episode_step),
    ):
        _check_leading_shape(tensor, sample_shape, key=key, path=path)
        if tensor.ndim > 2:
            tensor = tensor.reshape(time_steps, num_envs, -1)
            if tensor.shape[-1] != 1:
                raise ValueError(f"{path} 中 '{key}' 应为 [T,N] 或 [T,N,1]。")

    env_id = env_id.reshape(time_steps, num_envs).to(dtype=torch.long).contiguous()
    episode_id = episode_id.reshape(time_steps, num_envs).to(dtype=torch.long).contiguous()
    episode_step = episode_step.reshape(time_steps, num_envs).to(dtype=torch.long).contiguous()

    return SourceData(
        path=str(path),
        metadata=metadata,
        vector_features=vector_features,
        mask_pixels=mask,
        target=target,
        env_id=env_id,
        episode_id=episode_id,
        episode_step=episode_step,
        vector_input_dims=vector_input_dims,
        resolved_target_key=resolved_target_key,
    )


def _window_time_indices(end_time: int, history_len: int, history_stride: int) -> torch.Tensor:
    start_time = end_time - (history_len - 1) * history_stride
    return torch.arange(start_time, end_time + 1, history_stride, dtype=torch.long)


def _window_mask_is_valid(
    mask_pixels: torch.Tensor,
    *,
    env_index: int,
    time_indices: torch.Tensor,
    mode: Literal["none", "current", "all"],
) -> bool:
    if mode == "none":
        return True

    if mode == "current":
        return bool(mask_pixels[time_indices[-1], env_index].any().item())

    if mode == "all":
        per_frame_valid = mask_pixels[time_indices, env_index].flatten(start_dim=1).any(dim=1)
        return bool(per_frame_valid.all().item())

    raise ValueError(f"未知 mask_valid_mode: {mode}")


def _build_window_indices(
    sources: list[SourceData],
    *,
    history_len: int,
    history_stride: int,
    mask_valid_mode: Literal["none", "current", "all"],
) -> list[WindowIndex]:
    """按照连续 episode 段构造历史窗口。"""

    windows: list[WindowIndex] = []
    next_group_id = 0
    required_span = (history_len - 1) * history_stride

    for source_id, source in enumerate(sources):
        time_steps, num_envs = source.episode_id.shape

        for env_index in range(num_envs):
            segment_start = 0

            # t==time_steps 是哨兵，用于处理最后一个连续片段。
            for t in range(1, time_steps + 1):
                is_contiguous = False
                if t < time_steps:
                    same_env = source.env_id[t, env_index] == source.env_id[t - 1, env_index]
                    same_episode = source.episode_id[t, env_index] == source.episode_id[t - 1, env_index]
                    next_step = source.episode_step[t, env_index] == source.episode_step[t - 1, env_index] + 1
                    is_contiguous = bool((same_env & same_episode & next_step).item())

                if is_contiguous:
                    continue

                segment_end = t - 1
                first_valid_end = segment_start + required_span

                if first_valid_end <= segment_end:
                    group_id = next_group_id
                    next_group_id += 1

                    for end_time in range(first_valid_end, segment_end + 1):
                        time_indices = _window_time_indices(
                            end_time,
                            history_len,
                            history_stride,
                        )

                        if not _window_mask_is_valid(
                            source.mask_pixels,
                            env_index=env_index,
                            time_indices=time_indices,
                            mode=mask_valid_mode,
                        ):
                            continue

                        vector_window = source.vector_features[time_indices, env_index]
                        target = source.target[end_time, env_index]
                        finite = torch.isfinite(vector_window).all() and torch.isfinite(target).all()
                        if not bool(finite.item()):
                            continue

                        windows.append(
                            WindowIndex(
                                source_id=source_id,
                                env_index=env_index,
                                end_time=end_time,
                                group_id=group_id,
                            )
                        )

                segment_start = t

    return windows


class HistoryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """按索引动态切片历史帧，避免提前复制出巨大的历史 mask Tensor。"""

    def __init__(
        self,
        *,
        sources: list[SourceData],
        windows: list[WindowIndex],
        history_len: int,
        history_stride: int,
    ) -> None:
        self.sources = sources
        self.windows = windows
        self.history_len = int(history_len)
        self.history_stride = int(history_stride)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        source = self.sources[window.source_id]
        time_indices = _window_time_indices(
            window.end_time,
            self.history_len,
            self.history_stride,
        )

        vector_history = source.vector_features[time_indices, window.env_index]
        mask_history = source.mask_pixels[time_indices, window.env_index]
        target = source.target[window.end_time, window.env_index]

        return vector_history, mask_history, target


def _subsample_windows(
    windows: list[WindowIndex],
    *,
    max_samples: int | None,
    seed: int,
) -> list[WindowIndex]:
    if max_samples is None or max_samples <= 0 or len(windows) <= max_samples:
        return windows

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(windows)), max_samples)
    selected_indices.sort()
    return [windows[index] for index in selected_indices]


def _split_windows(
    windows: list[WindowIndex],
    *,
    val_fraction: float,
    seed: int,
    split_mode: Literal["episode", "sample"],
) -> tuple[list[WindowIndex], list[WindowIndex]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction 必须位于 [0, 1)。")
    if not windows:
        raise RuntimeError("没有可用于训练的历史窗口。")
    if val_fraction == 0.0 or len(windows) == 1:
        return windows, []

    rng = random.Random(seed)

    if split_mode == "sample":
        indices = list(range(len(windows)))
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        val_count = min(max(val_count, 1), len(indices) - 1)
        val_indices = set(indices[:val_count])
        train_windows = [window for i, window in enumerate(windows) if i not in val_indices]
        val_windows = [window for i, window in enumerate(windows) if i in val_indices]
        return train_windows, val_windows

    if split_mode != "episode":
        raise ValueError(f"未知 split_mode: {split_mode}")

    # 按 episode/连续轨迹分组，避免高度重叠的历史窗口同时出现在训练集和验证集。
    group_ids = sorted({window.group_id for window in windows})
    if len(group_ids) == 1:
        print("[WARN] 只有一个 episode group，无法按 episode 划分；改用按样本划分。")
        return _split_windows(
            windows,
            val_fraction=val_fraction,
            seed=seed,
            split_mode="sample",
        )

    rng.shuffle(group_ids)
    val_group_count = int(round(len(group_ids) * val_fraction))
    val_group_count = min(max(val_group_count, 1), len(group_ids) - 1)
    val_groups = set(group_ids[:val_group_count])

    train_windows = [window for window in windows if window.group_id not in val_groups]
    val_windows = [window for window in windows if window.group_id in val_groups]
    return train_windows, val_windows


@torch.no_grad()
def _compute_vector_statistics(
    dataset: HistoryWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只统计数值输入；二值 mask 保持 0/1，不参与此归一化。"""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    total_sum: torch.Tensor | None = None
    total_sq_sum: torch.Tensor | None = None
    total_count = 0

    for vector_history, _, _ in loader:
        # [B, L, D] -> [B*L, D]
        values = vector_history.reshape(-1, vector_history.shape[-1]).to(dtype=torch.float64)
        batch_sum = values.sum(dim=0)
        batch_sq_sum = (values * values).sum(dim=0)

        if total_sum is None:
            total_sum = batch_sum
            total_sq_sum = batch_sq_sum
        else:
            total_sum += batch_sum
            assert total_sq_sum is not None
            total_sq_sum += batch_sq_sum
        total_count += values.shape[0]

    if total_sum is None or total_sq_sum is None or total_count == 0:
        raise RuntimeError("无法计算输入归一化统计量：训练集为空。")

    mean = total_sum / total_count
    variance = total_sq_sum / total_count - mean * mean
    std = torch.sqrt(variance.clamp_min(1e-12))
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32).clamp_min(1e-6)


# -----------------------------------------------------------------------------
# 训练与保存
# -----------------------------------------------------------------------------


def _loss_fn(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if loss_name == "l1":
        return F.l1_loss(prediction, target)
    raise ValueError(f"不支持的损失函数: {loss_name}")


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    for vector_history, mask_history, target in loader:
        vector_history = vector_history.to(device, non_blocking=True)
        mask_history = mask_history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        prediction = model(vector_history, mask_history)
        batch_count = target.shape[0]

        total_loss += float(_loss_fn(prediction, target, loss_name).item()) * batch_count
        total_mse += float(F.mse_loss(prediction, target).item()) * batch_count
        total_mae += float(F.l1_loss(prediction, target).item()) * batch_count
        total_count += batch_count

    mean_mse = total_mse / max(total_count, 1)
    return {
        "loss": total_loss / max(total_count, 1),
        "mse": mean_mse,
        "rmse": math.sqrt(mean_mse),
        "mae": total_mae / max(total_count, 1),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _make_run_dir(output_dir: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"target_pos_history_{timestamp}"
    run_dir = Path(output_dir).expanduser() / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _source_infos(sources: list[SourceData]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for source in sources:
        infos.append(
            {
                "path": source.path,
                "metadata": source.metadata,
                "sample_shape": list(source.episode_id.shape),
                "vector_dim": int(source.vector_features.shape[-1]),
                "mask_shape": list(source.mask_pixels.shape[-2:]),
                "target_dim": int(source.target.shape[-1]),
                "resolved_target_key": source.resolved_target_key,
                "vector_input_dims": source.vector_input_dims,
            }
        )
    return infos


def _save_checkpoint(
    *,
    path: Path,
    model: HistoryTargetPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
    vector_input_dims: dict[str, int],
    mask_height: int,
    mask_width: int,
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
            "model_class": "HistoryTargetPolicy",
            "vector_dim": model.vector_dim,
            "target_dim": model.target_dim,
            "mask_embedding_dim": model.mask_embedding_dim,
            "frame_embedding_dim": model.frame_embedding_dim,
            "gru_hidden_dim": model.gru_hidden_dim,
            "gru_layers": model.gru_layers,
            "head_hidden_dims": model.head_hidden_dims,
            "activation": model.activation_name,
            "dropout": model.dropout,
            "vector_input_keys": cfg.vector_input_keys,
            "vector_input_dims": vector_input_dims,
            "mask_key": cfg.mask_key,
            "mask_height": mask_height,
            "mask_width": mask_width,
            "history_len": cfg.history_len,
            "history_stride": cfg.history_stride,
            "target_key": cfg.target_key,
            "source_datasets": source_infos,
        },
        path,
    )


def _export_torchscript(
    model: HistoryTargetPolicy,
    *,
    output_path: Path,
    device: torch.device,
    history_len: int,
    vector_dim: int,
    mask_height: int,
    mask_width: int,
) -> None:
    model.eval()
    example_vector = torch.zeros(
        1,
        history_len,
        vector_dim,
        dtype=torch.float32,
        device=device,
    )
    example_mask = torch.zeros(
        1,
        history_len,
        1,
        mask_height,
        mask_width,
        dtype=torch.float32,
        device=device,
    )
    traced = torch.jit.trace(model, (example_vector, example_mask))
    traced.save(str(output_path))


# -----------------------------------------------------------------------------
# 命令行
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用状态、YOLO mask 和历史帧训练 target_pos 回归策略。"
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="一个或多个采集得到的 .pt 数据集。",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="logs/target_pos_history_policy",
        help="训练输出根目录。",
    )
    parser.add_argument("--run_name", "--run-name", default=None)

    parser.add_argument(
        "--vector_input_keys",
        "--vector-input-keys",
        default=(
            "hand_dof_pos,tactile_binary,yolo_position_image,"
            "yolo_angle_image_rad,goal_pos,goal_rot"
        ),
        help=(
            "每帧要拼接的数值字段。yolo_mask_pixels 单独由 CNN 处理，"
            "不要写进此参数。"
        ),
    )
    parser.add_argument(
        "--mask_key",
        "--mask-key",
        default="yolo_mask_pixels",
        help="YOLO 二值 mask 字段。",
    )
    parser.add_argument(
        "--target_key",
        "--target-key",
        default="target_pos",
        help="监督目标字段；自动兼容 targets_pos 别名。",
    )

    parser.add_argument(
        "--history_len",
        "--history-len",
        type=int,
        default=10,
        help="每个训练样本包含的历史帧数量；1 表示单帧。",
    )
    parser.add_argument(
        "--history_stride",
        "--history-stride",
        type=int,
        default=1,
        help="历史帧之间的时间步间隔。",
    )
    parser.add_argument(
        "--mask_valid_mode",
        "--mask-valid-mode",
        choices=("none", "current", "all"),
        default="none",
        help=(
            "none: 不因空 mask 删除样本；current: 当前帧 mask 必须非空；"
            "all: 整个历史窗口每帧 mask 都必须非空。"
        ),
    )

    parser.add_argument(
        "--split_mode",
        "--split-mode",
        choices=("episode", "sample"),
        default="episode",
        help="默认按 episode 划分训练/验证集，防止历史窗口泄漏。",
    )
    parser.add_argument("--val_fraction", "--val-fraction", type=float, default=0.1)
    parser.add_argument("--max_samples", "--max-samples", type=int, default=None)

    parser.add_argument("--batch_size", "--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--activation",
        choices=("elu", "relu", "silu", "tanh", "gelu"),
        default="elu",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=("mse", "smooth_l1", "l1"), default="smooth_l1")
    parser.add_argument(
        "--no_normalize_vectors",
        "--no-normalize-vectors",
        action="store_true",
        help="关闭数值状态标准化；二值 mask 无论如何都不会做均值方差标准化。",
    )
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=1.0)

    parser.add_argument("--mask_embedding_dim", "--mask-embedding-dim", type=int, default=64)
    parser.add_argument("--frame_embedding_dim", "--frame-embedding-dim", type=int, default=128)
    parser.add_argument("--gru_hidden_dim", "--gru-hidden-dim", type=int, default=256)
    parser.add_argument("--gru_layers", "--gru-layers", type=int, default=1)
    parser.add_argument(
        "--head_hidden_dims",
        "--head-hidden-dims",
        default="256,128",
        help="GRU 后回归头的隐藏层维数。",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--save_every", "--save-every", type=int, default=0)
    return parser


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def main() -> None:
    args = _build_arg_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vector_input_keys = _parse_csv(args.vector_input_keys)
    if not vector_input_keys:
        raise ValueError("vector_input_keys 不能为空。")
    if args.mask_key in vector_input_keys:
        raise ValueError(
            f"mask 字段 '{args.mask_key}' 不应放入 vector_input_keys；"
            "它会由 CNN 单独编码。"
        )
    if args.history_len <= 0 or args.history_stride <= 0:
        raise ValueError("history_len 和 history_stride 必须大于 0。")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动改用 CPU。")
        device = torch.device("cpu")

    cfg = TrainConfig(
        datasets=[str(Path(path).expanduser()) for path in args.datasets],
        output_dir=args.output_dir,
        run_name=args.run_name or "",
        vector_input_keys=vector_input_keys,
        mask_key=args.mask_key,
        target_key=args.target_key,
        history_len=int(args.history_len),
        history_stride=int(args.history_stride),
        mask_valid_mode=args.mask_valid_mode,
        split_mode=args.split_mode,
        val_fraction=float(args.val_fraction),
        max_samples=args.max_samples,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        activation=args.activation,
        dropout=float(args.dropout),
        loss=args.loss,
        normalize_vectors=not bool(args.no_normalize_vectors),
        grad_clip=(
            float(args.grad_clip)
            if args.grad_clip is not None and args.grad_clip > 0.0
            else None
        ),
        mask_embedding_dim=int(args.mask_embedding_dim),
        frame_embedding_dim=int(args.frame_embedding_dim),
        gru_hidden_dim=int(args.gru_hidden_dim),
        gru_layers=int(args.gru_layers),
        head_hidden_dims=_parse_int_csv(args.head_hidden_dims),
        seed=int(args.seed),
        device=str(device),
        num_workers=int(args.num_workers),
        save_every=int(args.save_every),
    )

    run_dir = _make_run_dir(cfg.output_dir, args.run_name)
    cfg.run_name = run_dir.name

    print(f"[INFO] 输出目录: {run_dir}")
    print(f"[INFO] 数值输入字段: {cfg.vector_input_keys}")
    print(f"[INFO] mask 字段: {cfg.mask_key}")
    print(f"[INFO] 目标字段: {cfg.target_key}")
    print(
        f"[INFO] 历史窗口: len={cfg.history_len}, "
        f"stride={cfg.history_stride}"
    )

    sources: list[SourceData] = []
    vector_input_dims_ref: dict[str, int] | None = None
    vector_dim_ref: int | None = None
    target_dim_ref: int | None = None
    mask_shape_ref: tuple[int, int] | None = None

    for dataset_path in cfg.datasets:
        path = Path(dataset_path).expanduser()
        source = _load_source_data(
            path=path,
            vector_input_keys=cfg.vector_input_keys,
            mask_key=cfg.mask_key,
            target_key=cfg.target_key,
        )

        vector_dim = int(source.vector_features.shape[-1])
        target_dim = int(source.target.shape[-1])
        mask_shape = tuple(int(v) for v in source.mask_pixels.shape[-2:])

        if vector_input_dims_ref is None:
            vector_input_dims_ref = source.vector_input_dims
            vector_dim_ref = vector_dim
            target_dim_ref = target_dim
            mask_shape_ref = mask_shape
        else:
            if source.vector_input_dims != vector_input_dims_ref:
                raise ValueError(
                    f"{path} 的输入字段维数 {source.vector_input_dims} 与前面的"
                    f"数据集 {vector_input_dims_ref} 不一致。"
                )
            if vector_dim != vector_dim_ref or target_dim != target_dim_ref:
                raise ValueError(
                    f"{path} 的 vector_dim/target_dim=({vector_dim},{target_dim})，"
                    f"预期为 ({vector_dim_ref},{target_dim_ref})。"
                )
            if mask_shape != mask_shape_ref:
                raise ValueError(
                    f"{path} 的 mask 尺寸 {mask_shape} 与预期 {mask_shape_ref} 不一致。"
                )

        sources.append(source)
        print(
            f"[INFO] 加载 {path}: T={source.episode_id.shape[0]}, "
            f"N={source.episode_id.shape[1]}, vector_dim={vector_dim}, "
            f"mask={mask_shape}, target_dim={target_dim}, "
            f"target_key={source.resolved_target_key}"
        )

    assert vector_input_dims_ref is not None
    assert vector_dim_ref is not None
    assert target_dim_ref is not None
    assert mask_shape_ref is not None

    windows = _build_window_indices(
        sources,
        history_len=cfg.history_len,
        history_stride=cfg.history_stride,
        mask_valid_mode=cfg.mask_valid_mode,  # type: ignore[arg-type]
    )
    print(f"[INFO] 合法历史窗口总数: {len(windows)}")

    windows = _subsample_windows(
        windows,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    if cfg.max_samples is not None:
        print(f"[INFO] 采样后历史窗口数: {len(windows)}")

    train_windows, val_windows = _split_windows(
        windows,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        split_mode=cfg.split_mode,  # type: ignore[arg-type]
    )
    print(f"[INFO] 训练窗口: {len(train_windows)}")
    print(f"[INFO] 验证窗口: {len(val_windows)}")

    train_dataset = HistoryWindowDataset(
        sources=sources,
        windows=train_windows,
        history_len=cfg.history_len,
        history_stride=cfg.history_stride,
    )
    val_dataset = (
        HistoryWindowDataset(
            sources=sources,
            windows=val_windows,
            history_len=cfg.history_len,
            history_stride=cfg.history_stride,
        )
        if val_windows
        else None
    )

    if cfg.normalize_vectors:
        print("[INFO] 正在计算训练集数值输入均值和标准差……")
        vector_mean, vector_std = _compute_vector_statistics(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
    else:
        vector_mean = torch.zeros(vector_dim_ref, dtype=torch.float32)
        vector_std = torch.ones(vector_dim_ref, dtype=torch.float32)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    model = HistoryTargetPolicy(
        vector_dim=vector_dim_ref,
        target_dim=target_dim_ref,
        mask_embedding_dim=cfg.mask_embedding_dim,
        frame_embedding_dim=cfg.frame_embedding_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        gru_layers=cfg.gru_layers,
        head_hidden_dims=cfg.head_hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        vector_mean=vector_mean,
        vector_std=vector_std,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    source_infos = _source_infos(sources)
    config_payload = {
        "config": asdict(cfg),
        "vector_input_dims": vector_input_dims_ref,
        "vector_dim": vector_dim_ref,
        "target_dim": target_dim_ref,
        "mask_height": mask_shape_ref[0],
        "mask_width": mask_shape_ref[1],
        "history_order": "oldest_to_newest",
        "target_alignment": "target_pos at the newest history frame",
        "source_datasets": source_infos,
        "vector_mean": vector_mean,
        "vector_std": vector_std,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(config_payload), file, indent=2, ensure_ascii=False)

    best_metric = float("inf")
    best_epoch = -1
    last_metrics: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for vector_history, mask_history, target in train_loader:
            vector_history = vector_history.to(device, non_blocking=True)
            mask_history = mask_history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            prediction = model(vector_history, mask_history)
            loss = _loss_fn(prediction, target, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_count = target.shape[0]
            train_loss_sum += float(loss.item()) * batch_count
            train_count += batch_count

        train_step_loss = train_loss_sum / max(train_count, 1)
        train_metrics = _evaluate(model, train_loader, device, cfg.loss)
        val_metrics = (
            _evaluate(model, val_loader, device, cfg.loss)
            if val_loader is not None
            else train_metrics
        )

        current_metric = val_metrics["mse"]
        last_metrics = {
            "train_step_loss": train_step_loss,
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
                vector_input_dims=vector_input_dims_ref,
                mask_height=mask_shape_ref[0],
                mask_width=mask_shape_ref[1],
                source_infos=source_infos,
            )
            _export_torchscript(
                model,
                output_path=run_dir / "policy_jit.pt",
                device=device,
                history_len=cfg.history_len,
                vector_dim=vector_dim_ref,
                mask_height=mask_shape_ref[0],
                mask_width=mask_shape_ref[1],
            )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            _save_checkpoint(
                path=run_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch,
                metrics=last_metrics,
                vector_input_dims=vector_input_dims_ref,
                mask_height=mask_shape_ref[0],
                mask_width=mask_shape_ref[1],
                source_infos=source_infos,
            )

        print(
            f"[INFO] epoch={epoch:04d}/{cfg.epochs} "
            f"train_mse={last_metrics['train_mse']:.6e} "
            f"train_mae={last_metrics['train_mae']:.6e} "
            f"val_mse={last_metrics['val_mse']:.6e} "
            f"val_mae={last_metrics['val_mae']:.6e} "
            f"{'*' if is_best else ''}"
        )

    _save_checkpoint(
        path=run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg.epochs,
        metrics=last_metrics,
        vector_input_dims=vector_input_dims_ref,
        mask_height=mask_shape_ref[0],
        mask_width=mask_shape_ref[1],
        source_infos=source_infos,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mse": best_metric,
        "last_metrics": last_metrics,
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "torchscript_policy": str(run_dir / "policy_jit.pt"),
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(summary), file, indent=2, ensure_ascii=False)

    print(f"[INFO] 最佳 epoch: {best_epoch}")
    print(f"[INFO] 最佳验证 MSE: {best_metric:.6e}")
    print(f"[INFO] 最佳 checkpoint: {run_dir / 'best.pt'}")
    print(f"[INFO] TorchScript: {run_dir / 'policy_jit.pt'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utils
# ============================================================

def make_cols(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(dim)]


def read_csv_auto(csv_path: str | Path) -> pd.DataFrame:
    """
    自动兼容逗号 CSV / tab CSV。
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_history_suffix(policy_name: str, use_history_stack: bool) -> str:
    """
    使用历史帧时，自动给策略名追加 _history。

    supervised_action_policy_no_vision + --use_history_stack
    -> supervised_action_policy_no_vision_history
    """
    if use_history_stack and not policy_name.endswith("_history"):
        return policy_name + "_history"
    return policy_name


def default_policy_name_from_obs_mode(obs_mode: str) -> str:
    if obs_mode == "full":
        return "supervised_action_policy_full"
    if obs_mode == "no_vision":
        return "supervised_action_policy_no_vision"
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


@dataclass
class DatasetInfo:
    csv_paths: list[str]
    obs_mode: str
    use_history_stack: bool
    history_len: int
    history_include_previous_action: bool
    input_dim: int
    output_dim: int
    input_cols: list[str]
    target_cols: list[str]
    target_mode: str
    num_total_valid: int
    num_train: int
    num_val: int
    num_trajectories: int
    train_trajectories: list[str]
    val_trajectories: list[str]


# ============================================================
# Dataset
# ============================================================

class ActionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


# ============================================================
# Model
# ============================================================

class ActionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 20,
        hidden_dims=(256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormalizedPolicy(nn.Module):
    """
    部署用模型。

    输入 raw obs，内部自动 normalize；输出 raw target/action。
    """
    def __init__(
        self,
        model: nn.Module,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std
        y_norm = self.model(x_norm)
        y = y_norm * self.target_std + self.target_mean
        return y


# ============================================================
# CSV loading
# ============================================================

def resolve_csv_paths(args: argparse.Namespace) -> list[Path]:
    if args.csvs is not None and len(args.csvs) > 0:
        csv_paths = [Path(p) for p in args.csvs]
    else:
        csv_dir = Path(args.csv_dir)
        csv_paths = sorted(csv_dir.glob(args.csv_pattern))

    csv_paths = [p for p in csv_paths if p.exists() and p.is_file()]
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found. csv_dir={args.csv_dir}, "
            f"pattern={args.csv_pattern}, csvs={args.csvs}"
        )
    return csv_paths


def load_multi_csv(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    csv_paths = resolve_csv_paths(args)
    dfs = []
    csv_path_strs = []

    for csv_path in csv_paths:
        print(f"[LOAD CSV] {csv_path}")
        df = read_csv_auto(csv_path)
        dataset_name = csv_path.stem

        df["dataset_name"] = dataset_name
        df["source_csv"] = csv_path.name

        if "source_file" not in df.columns:
            df["source_file"] = csv_path.name

        # 多个 CSV 里的 file_index 会重复，所以必须构造全局轨迹 ID。
        if "file_index" in df.columns:
            df["global_traj_id"] = (
                df["dataset_name"].astype(str) + "::file_" + df["file_index"].astype(str)
            )
        else:
            df["file_index"] = 0
            df["global_traj_id"] = (
                df["dataset_name"].astype(str) + "::" + df["source_file"].astype(str)
            )

        dfs.append(df)
        csv_path_strs.append(str(csv_path))
        print(f"           rows={len(df)}, cols={len(df.columns)}")

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    print("")
    print("[DATA] merged rows:", len(merged))
    print("[DATA] merged cols:", len(merged.columns))
    print("[DATA] num csv:", len(csv_path_strs))
    print("[DATA] num trajectories:", merged["global_traj_id"].nunique())

    return merged, csv_path_strs


# ============================================================
# Input / Target building
# ============================================================

def build_previous_actions(df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
    """
    构造 previous_action。

    训练阶段必须按 global_traj_id 分组，避免多个 CSV / pkl 之间串帧。
    """
    missing = [c for c in action_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少 actions 列，无法构造 previous_action: {missing}")

    prev = df.groupby("global_traj_id")[action_cols].shift(1)
    prev = prev.fillna(0.0)
    prev.columns = make_cols("previous_action", 20)
    return prev


def build_target(df: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    action_cols = make_cols("actions", 20)
    joint_cols = make_cols("tesollo_joints_state", 20)

    missing_action = [c for c in action_cols if c not in df.columns]
    if missing_action:
        raise KeyError(f"CSV 缺少 actions 列: {missing_action}")

    if target_mode == "actions":
        target = df[action_cols].copy()
        target.columns = action_cols
        return target

    if target_mode == "action_delta":
        prev_action = df.groupby("global_traj_id")[action_cols].shift(1).fillna(0.0)
        target_np = df[action_cols].to_numpy(dtype=np.float32) - prev_action.to_numpy(dtype=np.float32)
        return pd.DataFrame(target_np, columns=action_cols)

    if target_mode == "next_actions":
        target = df.groupby("global_traj_id")[action_cols].shift(-1)
        target.columns = action_cols
        return target

    if target_mode == "next_joint_delta":
        missing_joint = [c for c in joint_cols if c not in df.columns]
        if missing_joint:
            raise KeyError(f"CSV 缺少关节列，无法构造 next_joint_delta: {missing_joint}")
        next_joint = df.groupby("global_traj_id")[joint_cols].shift(-1)
        current_joint = df[joint_cols]
        target_np = next_joint.to_numpy(dtype=np.float32) - current_joint.to_numpy(dtype=np.float32)
        return pd.DataFrame(target_np, columns=action_cols)

    raise ValueError(f"Unknown target_mode: {target_mode}")


def get_single_frame_base_cols(obs_mode: str) -> tuple[list[str], int]:
    ball_cols = make_cols("ball_center", 4)
    tactile_cols = make_cols("tactile_data", 13)
    joint_cols = make_cols("tesollo_joints_state", 20)

    if obs_mode == "full":
        return ball_cols + tactile_cols + joint_cols, 37
    if obs_mode == "no_vision":
        return tactile_cols + joint_cols, 33
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def build_history_stacked_features(
    df: pd.DataFrame,
    base_cols: list[str],
    group_col: str,
    history_len: int,
) -> pd.DataFrame:
    """
    历史堆叠输入。

    history_len=10 时：
        [t-9, t-8, ..., t-1, t]

    轨迹开头不足 history_len 的部分不补齐，直接保留 NaN。
    后续会通过 finite_mask 自动过滤掉这些样本。

    例如 history_len=10:
        每条轨迹的前 9 帧都会被舍弃。
    """
    if history_len <= 0:
        raise ValueError(f"history_len must be > 0, got {history_len}")

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少历史堆叠需要的列: {missing}")

    if group_col not in df.columns:
        raise KeyError(f"CSV 缺少分组列: {group_col}")

    parts = []

    # 从旧到新排列：
    # history_len=10 时：
    # lag=9 -> t-9
    # lag=8 -> t-8
    # ...
    # lag=0 -> t
    for lag in range(history_len - 1, -1, -1):
        shifted = df.groupby(group_col)[base_cols].shift(lag)

        # 不再 bfill，也不 fillna。
        # 轨迹开头不足 history_len 的样本会保留 NaN，
        # 后续 finite_mask 会把它们过滤掉。
        shifted.columns = [f"hist_{lag:02d}_{c}" for c in base_cols]

        parts.append(shifted)

    history_df = pd.concat(parts, axis=1)
    return history_df


def build_input_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], int]:
    """
    保留旧功能，同时新增历史帧输入。

    旧功能：
        full      = ball(4) + tactile(13) + joint(20) + previous_action(20) = 57
        no_vision = tactile(13) + joint(20) + previous_action(20) = 53

    新功能 --use_history_stack:
        full      = history_len * [ball(4) + tactile(13) + joint(20)]
        no_vision = history_len * [tactile(13) + joint(20)]

    可选 --history_include_previous_action: 
        在历史堆叠后额外拼接 previous_action(20)。
    """
    action_cols = make_cols("actions", 20)
    base_cols, single_step_dim = get_single_frame_base_cols(args.obs_mode)

    required_cols = list(base_cols) + action_cols
    missing = [c for c in sorted(set(required_cols)) if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少这些列: {missing}")

    prev_action_df = build_previous_actions(df, action_cols)

    if args.use_history_stack:
        history_df = build_history_stacked_features(
            df=df,
            base_cols=base_cols,
            group_col="global_traj_id",
            history_len=args.history_len,
        )

        if args.history_include_previous_action:
            input_df = pd.concat([history_df, prev_action_df], axis=1)
            expected_input_dim = args.history_len * single_step_dim + 20
        else:
            input_df = history_df
            expected_input_dim = args.history_len * single_step_dim
    else:
        raw_df = df[base_cols].copy()
        input_df = pd.concat([raw_df, prev_action_df], axis=1)
        expected_input_dim = single_step_dim + 20

    input_cols = list(input_df.columns)
    return input_df, input_cols, int(expected_input_dim)


def split_train_val_by_trajectory(
    x_all: np.ndarray,
    y_all: np.ndarray,
    traj_ids_all: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique_trajs = np.unique(traj_ids_all)

    if len(unique_trajs) >= 2:
        shuffled = unique_trajs.copy()
        rng.shuffle(shuffled)

        num_val = max(1, int(len(shuffled) * val_ratio))
        num_val = min(len(shuffled) - 1, num_val)

        val_trajs = set(shuffled[:num_val].tolist())
        train_trajs = set(shuffled[num_val:].tolist())

        train_mask = np.array([traj in train_trajs for traj in traj_ids_all])
        val_mask = np.array([traj in val_trajs for traj in traj_ids_all])

        train_traj_list = sorted(list(train_trajs))
        val_traj_list = sorted(list(val_trajs))
    else:
        print("[WARN] only one trajectory found, fallback to random row split")
        indices = np.arange(len(x_all))
        rng.shuffle(indices)

        num_val = max(1, int(len(indices) * val_ratio))
        num_val = min(len(indices) - 1, num_val)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        train_mask = np.zeros(len(x_all), dtype=bool)
        val_mask = np.zeros(len(x_all), dtype=bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True

        train_traj_list = unique_trajs.astype(str).tolist()
        val_traj_list = unique_trajs.astype(str).tolist()

    return (
        x_all[train_mask],
        y_all[train_mask],
        x_all[val_mask],
        y_all[val_mask],
        train_traj_list,
        val_traj_list,
    )


def load_dataset_from_multi_csv(args: argparse.Namespace):
    df, csv_path_strs = load_multi_csv(args)

    joint_cols = make_cols("tesollo_joints_state", 20)
    action_cols = make_cols("actions", 20)

    input_df, input_cols, expected_input_dim = build_input_dataframe(df, args)
    target_df = build_target(df, args.target_mode)

    # next_joint_delta 需要 joint_cols，这里额外检查一下。
    if args.target_mode == "next_joint_delta":
        missing_joint = [c for c in joint_cols if c not in df.columns]
        if missing_joint:
            raise KeyError(f"CSV 缺少关节列: {missing_joint}")

    x_all = input_df.to_numpy(dtype=np.float32)
    y_all = target_df.to_numpy(dtype=np.float32)

    finite_mask = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all).all(axis=1)

    df_valid = df.loc[finite_mask].copy()
    x_all = x_all[finite_mask]
    y_all = y_all[finite_mask]

    if len(x_all) == 0:
        raise RuntimeError("没有有效样本，请检查 CSV、target_mode 或 NaN/Inf。")

    traj_ids_all = df_valid["global_traj_id"].astype(str).to_numpy()
    unique_trajs = np.unique(traj_ids_all)

    print("")
    print("[DATA] obs_mode:", args.obs_mode)
    print("[DATA] use_history_stack:", args.use_history_stack)
    print("[DATA] history_len:", args.history_len)
    print("[DATA] history_include_previous_action:", args.history_include_previous_action)
    print("[DATA] valid samples:", len(x_all))
    print("[DATA] input dim:", x_all.shape[1])
    print("[DATA] expected input dim:", expected_input_dim)
    print("[DATA] output dim:", y_all.shape[1])
    print("[DATA] valid trajectories:", len(unique_trajs))

    if x_all.shape[1] != expected_input_dim:
        raise RuntimeError(f"输入维度错误，期望 {expected_input_dim}，实际 {x_all.shape[1]}")
    if y_all.shape[1] != 20:
        raise RuntimeError(f"输出维度错误，期望 20，实际 {y_all.shape[1]}")

    x_train, y_train, x_val, y_val, train_traj_list, val_traj_list = split_train_val_by_trajectory(
        x_all=x_all,
        y_all=y_all,
        traj_ids_all=traj_ids_all,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(f"训练集或验证集为空: train={len(x_train)}, val={len(x_val)}")

    print("")
    print("[SPLIT] train trajectories:", len(train_traj_list))
    print("[SPLIT] val trajectories:", len(val_traj_list))
    print("[SPLIT] train samples:", len(x_train))
    print("[SPLIT] val samples:", len(x_val))
    print("[SPLIT] example train traj:", train_traj_list[:5])
    print("[SPLIT] example val traj:", val_traj_list[:5])

    info = DatasetInfo(
        csv_paths=csv_path_strs,
        obs_mode=args.obs_mode,
        use_history_stack=bool(args.use_history_stack),
        history_len=int(args.history_len),
        history_include_previous_action=bool(args.history_include_previous_action),
        input_dim=int(expected_input_dim),
        output_dim=20,
        input_cols=input_cols,
        target_cols=action_cols,
        target_mode=args.target_mode,
        num_total_valid=int(len(x_all)),
        num_train=int(len(x_train)),
        num_val=int(len(x_val)),
        num_trajectories=int(len(unique_trajs)),
        train_trajectories=train_traj_list,
        val_trajectories=val_traj_list,
    )

    return x_train, y_train, x_val, y_val, info


# ============================================================
# Augmentation
# ============================================================

def build_input_group_indices(input_cols: list[str]) -> dict[str, list[int]]:
    groups = {
        "ball": [],
        "tactile": [],
        "joint": [],
        "previous_action": [],
    }

    for i, col in enumerate(input_cols):
        if "ball_center_" in col:
            groups["ball"].append(i)
        elif "tactile_data_" in col:
            groups["tactile"].append(i)
        elif "tesollo_joints_state_" in col:
            groups["joint"].append(i)
        elif "previous_action_" in col:
            groups["previous_action"].append(i)

    return groups


def augment_input_norm(
    x: torch.Tensor,
    args: argparse.Namespace,
    group_indices: dict[str, list[int]],
) -> torch.Tensor:
    if not args.use_augmentation:
        return x

    x = x.clone()

    if args.input_noise_std > 0.0:
        x = x + args.input_noise_std * torch.randn_like(x)

    def add_noise(group_name: str, noise_std: float):
        idx_list = group_indices.get(group_name, [])
        if len(idx_list) == 0 or noise_std <= 0.0:
            return
        idx = torch.as_tensor(idx_list, dtype=torch.long, device=x.device)
        x[:, idx] = x[:, idx] + noise_std * torch.randn_like(x[:, idx])

    add_noise("ball", args.ball_noise_std)
    add_noise("tactile", args.tactile_noise_std)
    add_noise("joint", args.joint_noise_std)
    add_noise("previous_action", args.prev_action_noise_std)

    def dropout_group(group_name: str, prob: float):
        idx_list = group_indices.get(group_name, [])
        if len(idx_list) == 0 or prob <= 0.0:
            return
        idx = torch.as_tensor(idx_list, dtype=torch.long, device=x.device)
        mask = torch.rand(x.shape[0], 1, device=x.device) < prob
        x[:, idx] = torch.where(mask, torch.zeros_like(x[:, idx]), x[:, idx])

    dropout_group("ball", args.ball_dropout_prob)
    dropout_group("tactile", args.tactile_dropout_prob)

    return x


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_count = 0
    mae_raw_sum = 0.0
    rmse_raw_sum = 0.0
    max_abs_err = 0.0

    target_mean_cpu = target_mean.cpu()
    target_std_cpu = target_std.cpu()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        pred_norm = model(batch_x)
        loss = loss_fn(pred_norm, batch_y)

        pred_raw = pred_norm.cpu() * target_std_cpu + target_mean_cpu
        target_raw = batch_y.cpu() * target_std_cpu + target_mean_cpu

        err = pred_raw - target_raw
        abs_err = torch.abs(err)

        batch_size = batch_x.shape[0]
        total_loss += loss.item() * batch_size
        mae_raw_sum += abs_err.mean().item() * batch_size
        rmse_raw_sum += torch.sqrt(torch.mean(err ** 2)).item() * batch_size
        max_abs_err = max(max_abs_err, abs_err.max().item())
        total_count += batch_size

    return {
        "loss": total_loss / max(1, total_count),
        "mae_raw": mae_raw_sum / max(1, total_count),
        "rmse_raw": rmse_raw_sum / max(1, total_count),
        "max_abs_err": max_abs_err,
    }


# ============================================================
# Train
# ============================================================

def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.policy_name:
        policy_name = args.policy_name
    else:
        policy_name = default_policy_name_from_obs_mode(args.obs_mode)

    policy_name = add_history_suffix(policy_name, args.use_history_stack)
    args.policy_name = policy_name

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path(args.log_root) / args.policy_name

    output_dir = output_root / args.target_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    args.output_root = str(output_root)
    args.output_dir = str(output_dir)

    print("[INFO] policy_name:", args.policy_name)
    print("[INFO] output root:", output_root)
    print("[INFO] output dir :", output_dir)

    return output_dir


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, use CPU instead.")
        return torch.device("cpu")
    return torch.device(device_str)


def train(args: argparse.Namespace):
    set_seed(args.seed)
    output_dir = resolve_output_dir(args)
    device = resolve_device(args.device)

    print("[INFO] device:", device)
    print("[INFO] obs_mode:", args.obs_mode)
    print("[INFO] target_mode:", args.target_mode)
    print("[INFO] use_history_stack:", args.use_history_stack)
    print("[INFO] history_len:", args.history_len)
    print("[INFO] history_include_previous_action:", args.history_include_previous_action)

    # -------------------------
    # 1. Load data
    # -------------------------
    x_train, y_train, x_val, y_val, info = load_dataset_from_multi_csv(args)

    # -------------------------
    # 2. Normalize
    # -------------------------
    input_mean = torch.as_tensor(x_train.mean(axis=0), dtype=torch.float32)
    input_std = torch.as_tensor(x_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)

    target_mean = torch.as_tensor(y_train.mean(axis=0), dtype=torch.float32)
    target_std = torch.as_tensor(y_train.std(axis=0), dtype=torch.float32).clamp_min(args.norm_eps)

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    x_train_norm = (x_train_t - input_mean) / input_std
    y_train_norm = (y_train_t - target_mean) / target_std
    x_val_norm = (x_val_t - input_mean) / input_std
    y_val_norm = (y_val_t - target_mean) / target_std

    train_dataset = ActionDataset(x_train_norm.numpy(), y_train_norm.numpy())
    val_dataset = ActionDataset(x_val_norm.numpy(), y_val_norm.numpy())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # -------------------------
    # 3. Model
    # -------------------------
    hidden_dims = tuple(args.hidden_dims)
    model = ActionMLP(
        input_dim=info.input_dim,
        output_dim=info.output_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    group_indices = build_input_group_indices(info.input_cols)

    # -------------------------
    # 4. Train loop
    # -------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    early_stop_counter = 0
    log_rows = []

    print("")
    print("[TRAIN] start supervised policy")
    print("[TRAIN] input_dim:", info.input_dim)
    print("[TRAIN] output_dim:", info.output_dim)
    print("[TRAIN] hidden_dims:", hidden_dims)
    print("[TRAIN] dropout:", args.dropout)
    print("[TRAIN] weight_decay:", args.weight_decay)
    print("[TRAIN] augmentation:", args.use_augmentation)
    print("[TRAIN] group_indices size:", {k: len(v) for k, v in group_indices.items()})
    print("")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            batch_x_aug = augment_input_norm(batch_x, args, group_indices)
            pred = model(batch_x_aug)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            target_mean=target_mean,
            target_std=target_std,
        )

        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss - args.early_stop_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_raw": val_metrics["mae_raw"],
            "val_rmse_raw": val_metrics["rmse_raw"],
            "val_max_abs_err": val_metrics["max_abs_err"],
            "lr": current_lr,
            "best_val_loss": best_val_loss,
            "early_stop_counter": early_stop_counter,
        }
        log_rows.append(row)

        if epoch == 1 or epoch % args.log_interval == 0 or improved:
            mark = "*" if improved else " "
            print(
                f"[TRAIN]{mark} epoch={epoch:04d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae_raw={val_metrics['mae_raw']:.6f} "
                f"val_rmse_raw={val_metrics['rmse_raw']:.6f} "
                f"val_max_abs_err={val_metrics['max_abs_err']:.6f} "
                f"lr={current_lr:.2e} "
                f"early={early_stop_counter}/{args.early_stop_patience}"
            )

        if early_stop_counter >= args.early_stop_patience:
            print("")
            print(
                f"[EARLY STOP] epoch={epoch}, "
                f"best_epoch={best_epoch}, "
                f"best_val_loss={best_val_loss:.6f}"
            )
            break

    # -------------------------
    # 5. Restore best model
    # -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    model_cpu = model.cpu().eval()

    # -------------------------
    # 6. Save checkpoint
    # -------------------------
    ckpt_path = output_dir / f"supervised_{args.target_mode}_policy.ckpt"

    checkpoint = {
        "model_state_dict": model_cpu.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "dataset_info": asdict(info),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }

    torch.save(checkpoint, ckpt_path)

    # -------------------------
    # 7. Save deployable TorchScript model
    # -------------------------
    deploy_policy = NormalizedPolicy(
        model=model_cpu,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    ).cpu().eval()

    jit_path = output_dir / f"supervised_{args.target_mode}_policy_jit.pt"
    example_input = torch.zeros(1, info.input_dim, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(deploy_policy, example_input)
    traced.save(jit_path)

    # -------------------------
    # 8. Save metadata and train log
    # -------------------------
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "dataset_info": asdict(info),
        "checkpoint": str(ckpt_path),
        "jit": str(jit_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "obs_mode": args.obs_mode,
        "target_mode": args.target_mode,
        "use_history_stack": bool(args.use_history_stack),
        "history_len": int(args.history_len),
        "history_include_previous_action": bool(args.history_include_previous_action),
        "input_dim": info.input_dim,
        "output_dim": info.output_dim,
        "input_order": info.input_cols,
        "target_order": info.target_cols,
        "note": (
            "Supervised policy. Old single-frame input is preserved when use_history_stack=False. "
            "History-stacked input is enabled only when use_history_stack=True. "
            "The JIT model internally normalizes input and denormalizes output."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    log_path = output_dir / "train_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print("")
    print("[DONE] saved checkpoint:", ckpt_path)
    print("[DONE] saved jit model:", jit_path)
    print("[DONE] saved metadata:", metadata_path)
    print("[DONE] saved train log:", log_path)
    print("[DONE] best_epoch:", best_epoch)
    print("[DONE] best_val_loss:", best_val_loss)


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # 多 CSV 输入
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/data/csv",
        help="包含多个 CSV 的文件夹",
    )
    parser.add_argument(
        "--csv_pattern",
        type=str,
        default="replay_data_*.csv",
        help="从 csv_dir 里匹配哪些 CSV",
    )
    parser.add_argument(
        "--csvs",
        nargs="*",
        default=None,
        help="也可以手动指定多个 CSV 路径。指定后会忽略 csv_dir。",
    )

    # 输入模式
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="no_vision",
        choices=["full", "no_vision"],
        help=(
            "full: 使用 ball_center + tactile_data + tesollo_joints_state；"
            "no_vision: 不使用 ball_center。"
        ),
    )
    parser.add_argument(
        "--use_history_stack",
        action="store_true",
        help="启用历史帧堆叠输入。启用后 policy_name 会自动追加 _history。",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=5,
        help="历史堆叠长度。只有 --use_history_stack 时生效。",
    )
    parser.add_argument(
        "--history_include_previous_action",
        action="store_true",
        help="历史堆叠输入后额外拼接 previous_action(20)。默认不拼接。",
    )

    # 输出目录
    parser.add_argument(
        "--log_root",
        type=str,
        default="/home/amlrobotics/hcy_ws/delto_walnut_hcy/logs",
        help="所有策略日志的根目录。",
    )
    parser.add_argument(
        "--policy_name",
        type=str,
        default="",
        help=(
            "策略名称。为空时根据 obs_mode 自动设置。"
            "如果使用 --use_history_stack，会自动追加 _history 后缀。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="模型输出根目录。若指定，则实际保存到 output_dir/target_mode。",
    )

    # Target
    parser.add_argument(
        "--target_mode",
        type=str,
        default="actions",
        choices=["actions", "action_delta", "next_actions", "next_joint_delta"],
        help=(
            "actions: 当前观测 -> 当前动作；"
            "action_delta: 当前观测 -> 当前动作 - 上一帧动作；"
            "next_actions: 当前观测 -> 下一帧动作；"
            "next_joint_delta: 当前观测 -> 下一帧关节角 - 当前关节角。"
        ),
    )

    # Model
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)

    # Train
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--huber_beta", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Split
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # Normalization
    parser.add_argument("--norm_eps", type=float, default=1e-6)

    # Augmentation
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--input_noise_std", type=float, default=0.01)
    parser.add_argument("--ball_noise_std", type=float, default=0.03)
    parser.add_argument("--tactile_noise_std", type=float, default=0.02)
    parser.add_argument("--joint_noise_std", type=float, default=0.01)
    parser.add_argument("--prev_action_noise_std", type=float, default=0.01)
    parser.add_argument("--ball_dropout_prob", type=float, default=0.02)
    parser.add_argument("--tactile_dropout_prob", type=float, default=0.02)

    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=60)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

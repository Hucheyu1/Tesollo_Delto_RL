# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""用 Direct-v0 已训练策略采集 DG5F 视觉-触觉预训练数据。

策略仍读取其训练时的原任务观测（自动兼容历史 84 维和当前 94 维）；
本脚本只动态附加与 VTDex Tomato 下游任务一致的 224x224 RGB 相机、
20 路 link 接触传感器和番茄方向标记。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="采集 DG5F VTDex 预训练 HDF5 数据集。")
parser.add_argument(
    "--task",
    type=str,
    default="Tesollo-Delto-DG5F-Direct-v0",
    help="必须使用保留原策略观测的 DG5F Direct 任务。",
)
parser.add_argument("--num_envs", type=int, default=16, help="并行环境数量。")
parser.add_argument("--num_steps", type=int, default=5000, help="策略控制步数。")
parser.add_argument("--warmup_steps", type=int, default=8, help="正式采集前的相机预热步数。")
parser.add_argument(
    "--sample_stride",
    type=int,
    default=1,
    help="每隔多少个控制步保存一次；1 表示每步保存。",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="datasets/dg5f_vtdex_pretrain",
    help="数据集根目录，文件保存到其 train/ 或 val/ 子目录。",
)
parser.add_argument(
    "--split", choices=("train", "val"), default="train", help="本次 rollout 所属划分。"
)
parser.add_argument(
    "--dataset_run_name",
    type=str,
    default=None,
    help="输出文件前缀；默认包含任务名和时间。",
)
parser.add_argument(
    "--max_samples_per_file",
    type=int,
    default=100000,
    help="单个 HDF5 shard 的最大样本数。",
)
parser.add_argument(
    "--flush_steps",
    type=int,
    default=16,
    help="累计多少个采样步后从 CPU buffer 写入 HDF5。",
)
parser.add_argument(
    "--contact_threshold",
    type=float,
    default=0.01,
    help="20 路触觉二值化阈值，单位 N；应与 Tomato 任务一致。",
)
parser.add_argument(
    "--no_tomato_markers",
    action="store_true",
    help="不添加两个番茄方向点（不建议，会增大与下游任务的视觉域差异）。",
)
parser.add_argument(
    "--keep_goal_marker_visible",
    action="store_true",
    help="让目标番茄 marker 出现在图像中（不建议）。",
)
parser.add_argument(
    "--randomize_object_initial_pose",
    action="store_true",
    help="保留 Direct-v0 的物体初始姿态随机化；默认固定起点。",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Tesollo_Delto_RL.tasks  # noqa: F401, E402
from Tesollo_Delto_RL.tasks.direct.tesollo_delto_rl.vtdex_data import (  # noqa: E402
    DG5F_VTDEX_CAMERA_CLIPPING_RANGE,
    DG5F_VTDEX_CAMERA_EYE_LOCAL,
    DG5F_VTDEX_CAMERA_FOCAL_LENGTH,
    DG5F_VTDEX_CAMERA_FOCUS_DISTANCE,
    DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE,
    DG5F_VTDEX_CAMERA_RESOLUTION,
    DG5F_VTDEX_CAMERA_TARGET_LOCAL,
    DG5F_VTDEX_TACTILE_BODY_NAMES,
    DG5F_VTDEX_TOMATO_MARKER_DIFFUSE_COLORS,
    DG5F_VTDEX_TOMATO_MARKER_EMISSIVE_COLORS,
    DG5F_VTDEX_TOMATO_MARKER_OFFSETS,
    DG5F_VTDEX_TOMATO_MARKER_RADIUS,
)


RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _rgb_to_uint8(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb[..., :3]
    if rgb.dtype == torch.uint8:
        return rgb
    return torch.clamp(rgb * 255.0, 0.0, 255.0).to(dtype=torch.uint8)


class H5ShardWriter:
    """可增量写入并按样本数自动切分的 HDF5 writer。"""

    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        max_samples: int,
        metadata_dict: dict[str, Any],
    ) -> None:
        self.output_dir = output_dir
        self.run_name = run_name
        self.max_samples = int(max_samples)
        self.metadata = metadata_dict
        self.part = 0
        self.file: h5py.File | None = None
        self.datasets: dict[str, h5py.Dataset] = {}
        self.count = 0
        self.paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open(self, batch: dict[str, np.ndarray]) -> None:
        path = self.output_dir / f"{self.run_name}_part{self.part:04d}.h5"
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有数据集 shard: {path}")
        self.file = h5py.File(path, "w")
        self.file.attrs["format"] = "dg5f-vtdex-pretrain-v1"
        self.file.attrs["metadata_json"] = json.dumps(
            _jsonable(self.metadata), ensure_ascii=False
        )
        self.datasets = {}
        for key, array in batch.items():
            compression = "lzf" if key == "rgb" else None
            chunk_rows = max(1, min(64, self.max_samples, array.shape[0]))
            self.datasets[key] = self.file.create_dataset(
                key,
                shape=(0, *array.shape[1:]),
                maxshape=(None, *array.shape[1:]),
                chunks=(chunk_rows, *array.shape[1:]),
                dtype=array.dtype,
                compression=compression,
            )
        self.count = 0
        self.paths.append(path)

    def append(self, batch: dict[str, np.ndarray]) -> None:
        if not batch:
            return
        lengths = {array.shape[0] for array in batch.values()}
        if len(lengths) != 1:
            raise ValueError(f"HDF5 batch 第一维不一致: {sorted(lengths)}")
        offset = 0
        total = lengths.pop()
        while offset < total:
            if self.file is None:
                self._open({key: value[offset:] for key, value in batch.items()})
            room = self.max_samples - self.count
            take = min(room, total - offset)
            for key, array in batch.items():
                dataset = self.datasets[key]
                dataset.resize(self.count + take, axis=0)
                dataset[self.count : self.count + take] = array[offset : offset + take]
            self.count += take
            offset += take
            if self.count >= self.max_samples:
                self.close_current()
                self.part += 1

    def close_current(self) -> None:
        if self.file is not None:
            self.file.attrs["num_samples"] = self.count
            self.file.flush()
            self.file.close()
            self.file = None
            self.datasets = {}

    def close(self) -> None:
        self.close_current()


def _inject_collection_sensors(env_cfg: DirectRLEnvCfg) -> None:
    env_cfg.enable_vtdex_collection_sensors = True
    # 数据图像中禁止出现手/object/goal 的调试坐标轴。
    env_cfg.debug_visualization = False
    env_cfg.robot_cfg = env_cfg.robot_cfg.replace(
        spawn=env_cfg.robot_cfg.spawn.replace(activate_contact_sensors=True)
    )
    env_cfg.vtdex_collection_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_[1-4]",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )
    env_cfg.vtdex_collection_tactile_body_names = DG5F_VTDEX_TACTILE_BODY_NAMES
    env_cfg.vtdex_collection_contact_threshold = float(args_cli.contact_threshold)
    env_cfg.vtdex_collection_camera = CameraCfg(
        prim_path="/World/envs/env_.*/VTDexCollectionCamera",
        offset=CameraCfg.OffsetCfg(
            pos=DG5F_VTDEX_CAMERA_EYE_LOCAL,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DG5F_VTDEX_CAMERA_FOCAL_LENGTH,
            focus_distance=DG5F_VTDEX_CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=DG5F_VTDEX_CAMERA_HORIZONTAL_APERTURE,
            clipping_range=DG5F_VTDEX_CAMERA_CLIPPING_RANGE,
        ),
        width=DG5F_VTDEX_CAMERA_RESOLUTION[0],
        height=DG5F_VTDEX_CAMERA_RESOLUTION[1],
        update_latest_camera_pose=True,
        debug_vis=False,
    )
    env_cfg.vtdex_collection_camera_eye_local = DG5F_VTDEX_CAMERA_EYE_LOCAL
    env_cfg.vtdex_collection_camera_target_local = DG5F_VTDEX_CAMERA_TARGET_LOCAL
    env_cfg.vtdex_collection_show_tomato_markers = not args_cli.no_tomato_markers
    env_cfg.vtdex_collection_tomato_marker_offsets = DG5F_VTDEX_TOMATO_MARKER_OFFSETS
    env_cfg.vtdex_collection_tomato_marker_radius = DG5F_VTDEX_TOMATO_MARKER_RADIUS
    env_cfg.vtdex_collection_tomato_marker_diffuse_colors = (
        DG5F_VTDEX_TOMATO_MARKER_DIFFUSE_COLORS
    )
    env_cfg.vtdex_collection_tomato_marker_emissive_colors = (
        DG5F_VTDEX_TOMATO_MARKER_EMISSIVE_COLORS
    )
    env_cfg.hide_goal_marker_from_vtdex_collection = not args_cli.keep_goal_marker_visible
    env_cfg.fix_object_initial_pose = not args_cli.randomize_object_initial_pose


def _configure_policy_observation_from_checkpoint(
    env_cfg: DirectRLEnvCfg, checkpoint_path: str
) -> int:
    """兼容仓库内早期 84 维与当前 94 维 Direct 策略。"""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError(
            "采集 checkpoint 必须是 Direct-v0 OnPolicyRunner 输出，"
            "且包含 actor_state_dict"
        )
    first_weight = actor_state.get("mlp.0.weight")
    if first_weight is None or first_weight.ndim != 2:
        raise ValueError("无法从 checkpoint 的 mlp.0.weight 推断策略输入维度")
    input_dim = int(first_weight.shape[1])
    if input_dim == 84:
        env_cfg.full_policy_include_tactile = False
        env_cfg.observation_space = 84
        print("[INFO] 检测到早期 84 维 Direct 策略：策略观测不含原任务 10 路触觉")
    elif input_dim == 94:
        env_cfg.full_policy_include_tactile = True
        env_cfg.observation_space = 94
        print("[INFO] 检测到当前 94 维 Direct 策略：策略观测包含原任务 10 路触觉")
    else:
        raise ValueError(
            f"Direct-v0 checkpoint 输入维度应为 84 或 94，实际为 {input_dim}"
        )
    return input_dim


def _to_numpy_batch(
    *,
    base_env,
    action: torch.Tensor,
    reward: torch.Tensor,
    dones: torch.Tensor,
    global_step: int,
    episode_id: torch.Tensor,
    episode_step: torch.Tensor,
) -> dict[str, np.ndarray]:
    rgb = _rgb_to_uint8(
        base_env._vtdex_collection_camera.data.output["rgb"]
    ).detach().cpu()
    tactile = base_env.vtdex_collection_tactile_binary.detach().cpu()
    tactile_force = base_env.vtdex_collection_tactile_force_norms.detach().cpu()
    num_envs = base_env.num_envs
    return {
        "rgb": rgb.numpy(),
        "tactile_binary": tactile.numpy(),
        "tactile_force_norms": tactile_force.numpy().astype(np.float32, copy=False),
        "env_id": np.arange(num_envs, dtype=np.int32),
        "episode_id": episode_id.detach().cpu().numpy().astype(np.int64, copy=False),
        "episode_step": episode_step.detach().cpu().numpy().astype(np.int32, copy=False),
        "global_step": np.full((num_envs,), global_step, dtype=np.int64),
        "action": action.detach().cpu().numpy().astype(np.float32, copy=False),
        "reward": reward.detach().cpu().numpy().astype(np.float32, copy=False),
        "done": dones.detach().cpu().numpy().astype(np.uint8, copy=False),
        "object_pos": base_env.object_pos.detach().cpu().numpy().astype(np.float32, copy=False),
        "object_rot_wxyz": base_env.object_rot.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
    }


def _merge_step_batches(step_batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([batch[key] for batch in step_batches], axis=0)
        for key in step_batches[0]
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.task != "Tesollo-Delto-DG5F-Direct-v0":
        raise ValueError(
            "本采集器只允许 Tesollo-Delto-DG5F-Direct-v0，"
            "以保证策略输入和数据协议不被其他任务悄悄改变。"
        )
    if args_cli.num_steps <= 0 or args_cli.num_envs <= 0:
        raise ValueError("--num_steps 和 --num_envs 必须为正数")
    if args_cli.sample_stride <= 0 or args_cli.flush_steps <= 0:
        raise ValueError("--sample_stride 和 --flush_steps 必须为正数")
    if args_cli.max_samples_per_file <= 0:
        raise ValueError("--max_samples_per_file 必须为正数")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _inject_collection_sensors(env_cfg)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, RSL_RL_VERSION)
    env_cfg.log_dir = os.path.dirname(resume_path)
    policy_input_dim = _configure_policy_observation_from_checkpoint(
        env_cfg, resume_path
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", args_cli.task)
    run_name = args_cli.dataset_run_name or f"{safe_task}_{timestamp}"
    output_dir = Path(args_cli.dataset_root).expanduser().resolve() / args_cli.split
    metadata_dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "format": "dg5f-vtdex-pretrain-v1",
        "task": args_cli.task,
        "policy_checkpoint": str(resume_path),
        "policy_input_dim": policy_input_dim,
        "split": args_cli.split,
        "num_envs": int(args_cli.num_envs),
        "num_steps_requested": int(args_cli.num_steps),
        "sample_stride": int(args_cli.sample_stride),
        "step_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        "camera_eye_local": DG5F_VTDEX_CAMERA_EYE_LOCAL,
        "camera_target_local": DG5F_VTDEX_CAMERA_TARGET_LOCAL,
        "camera_resolution_wh": DG5F_VTDEX_CAMERA_RESOLUTION,
        "tactile_body_names": DG5F_VTDEX_TACTILE_BODY_NAMES,
        "contact_threshold_n": float(args_cli.contact_threshold),
        "tomato_orientation_markers": not args_cli.no_tomato_markers,
        "goal_marker_visible": bool(args_cli.keep_goal_marker_visible),
        "object_initial_pose_randomized": bool(args_cli.randomize_object_initial_pose),
        "image_channel_order": "RGB",
        "quaternion_order": "wxyz",
    }
    writer = H5ShardWriter(
        output_dir,
        run_name,
        args_cli.max_samples_per_file,
        metadata_dict,
    )

    print(f"[INFO] 加载 Direct-v0 策略: {resume_path}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(
        f"[INFO] 计划采集 {args_cli.num_steps} steps × {args_cli.num_envs} envs，"
        f"stride={args_cli.sample_stride}"
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    if not hasattr(base_env, "_vtdex_collection_camera"):
        raise RuntimeError("Direct-v0 未创建 VTDex 采集相机")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"不支持的 runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)
    policy_nn = None
    if version.parse(RSL_RL_VERSION) < version.parse("4.0.0"):
        policy_nn = (
            runner.alg.policy
            if hasattr(runner.alg, "policy")
            else runner.alg.actor_critic
        )

    obs = env.get_observations()
    for warmup_step in range(args_cli.warmup_steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)
        print(f"[INFO] 相机预热 {warmup_step + 1}/{args_cli.warmup_steps}")

    with torch.inference_mode():
        obs, _ = env.reset()
        reset_dones = torch.ones(
            base_env.num_envs, dtype=torch.bool, device=base_env.device
        )
        if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
            policy.reset(reset_dones)
        elif policy_nn is not None:
            policy_nn.reset(reset_dones)

    episode_id = torch.zeros(
        base_env.num_envs, dtype=torch.long, device=base_env.device
    )
    episode_step = base_env.episode_length_buf.detach().clone().to(dtype=torch.long)
    pending: list[dict[str, np.ndarray]] = []
    recorded_samples = 0

    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                print("[WARN] Isaac Sim 提前停止")
                break
            with torch.inference_mode():
                action = policy(obs)
                sampled_batch = None
                if step % args_cli.sample_stride == 0:
                    # 在执行 action 前固定当前 RGB/触觉；随后补上该 transition 的
                    # reward/done，避免自动 reset 后把下一回合画面配给上一回合索引。
                    sampled_batch = _to_numpy_batch(
                        base_env=base_env,
                        action=action,
                        reward=torch.zeros(
                            base_env.num_envs,
                            dtype=torch.float32,
                            device=base_env.device,
                        ),
                        dones=torch.zeros(
                            base_env.num_envs,
                            dtype=torch.bool,
                            device=base_env.device,
                        ),
                        global_step=step,
                        episode_id=episode_id,
                        episode_step=episode_step,
                    )
                obs, reward, dones, _ = env.step(action)
                if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
                    policy.reset(dones)
                elif policy_nn is not None:
                    policy_nn.reset(dones)

            if sampled_batch is not None:
                sampled_batch["reward"] = (
                    reward.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                sampled_batch["done"] = (
                    dones.detach().cpu().numpy().astype(np.uint8, copy=False)
                )
                pending.append(sampled_batch)
                recorded_samples += base_env.num_envs
                if len(pending) >= args_cli.flush_steps:
                    writer.append(_merge_step_batches(pending))
                    pending.clear()

            episode_step += 1
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                episode_id[done_ids] += 1
                episode_step[done_ids] = 0

            if (step + 1) % 100 == 0 or step + 1 == args_cli.num_steps:
                print(
                    f"[INFO] rollout {step + 1}/{args_cli.num_steps}，"
                    f"已缓存/写入 {recorded_samples} samples"
                )
        if pending:
            writer.append(_merge_step_batches(pending))
            pending.clear()
    finally:
        writer.close()
        env.close()

    manifest_path = output_dir / f"{run_name}_manifest.json"
    manifest = {
        **metadata_dict,
        "num_samples_recorded": recorded_samples,
        "shards": [str(path) for path in writer.paths],
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(manifest), file, ensure_ascii=False, indent=2)
    print(f"[INFO] 采集完成: {recorded_samples} samples")
    print(f"[INFO] manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Collect play-time supervised-learning data from an RSL-RL checkpoint.

The saved dataset is meant for behavior cloning / supervised imitation:

    obs_policy[t, env] -> action[t, env]

For distillation tasks, ``obs_policy`` already contains the student-side YOLO
2-D center and angle features. The script also saves richer diagnostics such as
joint angles, object/goal pose, per-episode initial pose, tactile values and
the last YOLO mask estimate metadata.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# Reuse the RSL-RL CLI helpers shipped next to scripts/rsl_rl/play.py.
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Collect supervised play data from an RSL-RL policy.")
parser.add_argument("--num_steps", "--num-steps", type=int, default=1000, help="Number of play steps to record.")
parser.add_argument(
    "--output_dir",
    "--output-dir",
    type=str,
    default="datasets/play_supervised",
    help="Directory where the .pt dataset and .json metadata will be written.",
)
parser.add_argument(
    "--output_file",
    "--output-file",
    type=str,
    default=None,
    help="Optional output .pt file name. Defaults to '<task>_<timestamp>.pt'.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--only_valid_yolo",
    "--only-valid-yolo",
    action="store_true",
    default=False,
    help="Also save a flattened valid_sample_mask requiring YOLO position and angle measurements to be valid.",
)
parser.add_argument(
    "--save_yolo_mask_pixels",
    "--save-yolo-mask-pixels",
    action="store_true",
    default=False,
    help="Save selected YOLO binary mask pixels as yolo_mask_pixels. Disabled by default because it can be large.",
)
parser.add_argument(
    "--yolo_mask_size",
    "--yolo-mask-size",
    type=int,
    nargs=2,
    default=(64, 48),
    metavar=("WIDTH", "HEIGHT"),
    help="Saved yolo_mask_pixels size. Use 640 480 for full resolution; default is 64 48.",
)
parser.add_argument(
    "--goal_y_angle_deg",
    "--goal-y-angle-deg",
    type=float,
    default=None,
    help="Fix the play target to a hand-local Y-axis angle in degrees.",
)
parser.add_argument(
    "--goal_y_angle_rad",
    "--goal-y-angle-rad",
    type=float,
    default=None,
    help="Fix the play target to a hand-local Y-axis angle in radians. Takes precedence over degrees.",
)
parser.add_argument(
    "--goal_rot",
    "--goal-rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Fix self.goal_rot directly as a world/env quaternion in w x y z order. Takes precedence over angle.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Distillation observations need RTX camera rendering because YOLO runs on the
# configured student camera.
if args_cli.task is not None and "Distill" in args_cli.task:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has launched."""

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Tesollo_Delto_RL.tasks  # noqa: F401, E402

installed_version = metadata.version("rsl-rl-lib")


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor, clone it and move it to CPU for serialization."""

    return value.detach().clone().cpu()


def _clip_action(actions: torch.Tensor, clip_actions: float | None) -> torch.Tensor:
    if clip_actions is None:
        return actions
    return torch.clamp(actions, -clip_actions, clip_actions)


def _append_step(buffers: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        buffers.setdefault(key, []).append(_clone_cpu(value))


def _stack_buffers(buffers: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack(values, dim=0) for key, values in buffers.items() if values}


def _safe_get_tensor(obj: Any, name: str) -> torch.Tensor | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _capture_episode_initial_state(base_env) -> dict[str, torch.Tensor]:
    """Capture the current reset state used as per-episode initial pose."""

    return {
        "episode_initial_object_pos": base_env.object_pos.detach().clone(),
        "episode_initial_object_rot": base_env.object_rot.detach().clone(),
        "episode_initial_goal_pos": base_env.goal_pos.detach().clone(),
        "episode_initial_goal_rot": base_env.goal_rot.detach().clone(),
        "episode_initial_hand_dof_pos": base_env.hand_dof_pos.detach().clone(),
        "episode_initial_hand_base_pos": base_env.hand_base_pos.detach().clone(),
        "episode_initial_hand_base_rot": base_env.hand_base_rot.detach().clone(),
    }


def _update_episode_initial_state(
    initial_state: dict[str, torch.Tensor],
    new_initial_state: dict[str, torch.Tensor],
    done_ids: torch.Tensor,
) -> None:
    for key, value in initial_state.items():
        value[done_ids] = new_initial_state[key][done_ids]


def _resize_mask_pixels(mask: torch.Tensor, mask_size: tuple[int, int]) -> torch.Tensor:
    """Resize binary masks to (width, height) and store as uint8 0/1 pixels."""

    width, height = mask_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid --yolo_mask_size {mask_size}; width and height must be positive.")
    mask_nchw = mask.to(dtype=torch.float32).unsqueeze(1)
    resized = functional.interpolate(mask_nchw, size=(height, width), mode="nearest")
    return resized[:, 0].to(dtype=torch.uint8)


def _collect_yolo_tensors(
    base_env,
    *,
    save_mask_pixels: bool,
    mask_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Collect the YOLO estimate cached by the latest observation computation."""

    estimate = getattr(base_env, "last_yolo_estimate", None)
    if estimate is None:
        return {}

    values: dict[str, torch.Tensor] = {
        "yolo_position_image": estimate.position_image,
        "yolo_angle_image_rad": estimate.angle_image_rad,
        "yolo_position_valid": estimate.position_valid,
        "yolo_angle_valid": estimate.angle_valid,
        "yolo_measurement_valid": estimate.measurement_valid,
        "yolo_confidence": estimate.confidence,
        "yolo_mask_area": estimate.mask_area,
        "yolo_anisotropy": estimate.anisotropy,
        "yolo_visible_ratio": estimate.visible_ratio,
    }
    if save_mask_pixels:
        values["yolo_mask_pixels"] = _resize_mask_pixels(estimate.mask_image, mask_size)

    optional_names = {
        "yolo_target_angle_features": "last_yolo_target_angle_features",
        "yolo_observed_object_y": "last_yolo_observed_object_y",
        "yolo_object_y_angle_gt": "last_yolo_object_y_angle",
        "yolo_goal_y_angle_gt": "last_yolo_goal_y_angle",
        "yolo_angle_offset_rad": "yolo_angle_offset",
        "yolo_angle_calibrated": "yolo_angle_calibrated",
    }
    for out_name, attr_name in optional_names.items():
        tensor = _safe_get_tensor(base_env, attr_name)
        if tensor is not None:
            values[out_name] = tensor
    return values


def _collect_pre_step_tensors(
    *,
    base_env,
    obs,
    policy_action: torch.Tensor,
    action: torch.Tensor,
    global_step: int,
    episode_id: torch.Tensor,
    episode_step: torch.Tensor,
    initial_state: dict[str, torch.Tensor],
    save_yolo_mask_pixels: bool,
    yolo_mask_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Collect state at time t paired with action_t."""

    num_envs = base_env.num_envs
    device = base_env.device
    values: dict[str, torch.Tensor] = {
        "global_step": torch.full((num_envs,), global_step, dtype=torch.long, device=device),
        "env_id": torch.arange(num_envs, dtype=torch.long, device=device),
        "episode_id": episode_id,
        "episode_step": episode_step,
        # Recommended supervised learning pair: obs_policy -> action.
        "obs_policy": obs["policy"],
        "policy_action": policy_action,
        "action": action,
        "env_action": action,
        # Hand state.
        "hand_dof_pos": base_env.hand_dof_pos,
        "hand_dof_vel": base_env.hand_dof_vel,
        "hand_dof_targets": base_env.hand_dof_targets,
        "prev_targets": base_env.prev_targets,
        "cur_targets": base_env.cur_targets,
        "previous_action": base_env.actions,
        "tactile_binary": base_env.fingertip_force_binary_results,
        # Object and target state.
        "object_pos": base_env.object_pos,
        "object_rot": base_env.object_rot,
        "object_linvel": base_env.object_linvel,
        "object_angvel": base_env.object_angvel,
        "goal_pos": base_env.goal_pos,
        "goal_rot": base_env.goal_rot,
        "in_hand_pos": base_env.in_hand_pos,
        "hand_base_pos": base_env.hand_base_pos,
        "hand_base_rot": base_env.hand_base_rot,
    }

    if "critic" in obs.keys():
        values["obs_critic"] = obs["critic"]

    values.update(initial_state)
    values.update(
        _collect_yolo_tensors(
            base_env,
            save_mask_pixels=save_yolo_mask_pixels,
            mask_size=yolo_mask_size,
        )
    )
    return values


def _metadata_to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_metadata_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _metadata_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return str(value)


def _make_output_path(task_name: str) -> Path:
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.output_file:
        output_path = Path(args_cli.output_file).expanduser()
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)
        output_path = output_dir / f"{safe_task_name}_{timestamp}.pt"
    if output_path.suffix != ".pt":
        output_path = output_path.with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect play-time state/action data."""

    if args_cli.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.goal_rot is not None:
        env_cfg.fixed_goal_rot = tuple(args_cli.goal_rot)
        print(f"[INFO] Fixed collection goal quaternion wxyz: {env_cfg.fixed_goal_rot}")
    elif args_cli.goal_y_angle_rad is not None:
        env_cfg.fixed_goal_y_angle_rad = args_cli.goal_y_angle_rad
        print(f"[INFO] Fixed collection goal Y angle: {env_cfg.fixed_goal_y_angle_rad:.6f} rad")
    elif args_cli.goal_y_angle_deg is not None:
        env_cfg.fixed_goal_y_angle_rad = math.radians(args_cli.goal_y_angle_deg)
        print(
            f"[INFO] Fixed collection goal Y angle: {args_cli.goal_y_angle_deg:.3f} deg "
            f"({env_cfg.fixed_goal_y_angle_rad:.6f} rad)"
        )

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(f"No published pre-trained checkpoint is available for task '{train_task_name}'.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading checkpoint: {resume_path}")
    print(f"[INFO] Collecting {args_cli.num_steps} steps x {env_cfg.scene.num_envs} envs.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        if hasattr(runner.alg, "policy"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    obs = env.get_observations()
    buffers: dict[str, list[torch.Tensor]] = {}
    outcome_buffers: dict[str, list[torch.Tensor]] = {}
    episode_id = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    episode_step = base_env.episode_length_buf.detach().clone().to(dtype=torch.long)
    initial_state = _capture_episode_initial_state(base_env)

    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            print("[WARN] Simulation app stopped before reaching requested num_steps.")
            break

        with torch.inference_mode():
            policy_action = policy(obs)
            action = _clip_action(policy_action, agent_cfg.clip_actions)

            pre_step_values = _collect_pre_step_tensors(
                base_env=base_env,
                obs=obs,
                policy_action=policy_action,
                action=action,
                global_step=step,
                episode_id=episode_id,
                episode_step=episode_step,
                initial_state=initial_state,
                save_yolo_mask_pixels=bool(args_cli.save_yolo_mask_pixels),
                yolo_mask_size=tuple(args_cli.yolo_mask_size),
            )
            _append_step(buffers, pre_step_values)
            obs, rewards, dones, extras = env.step(policy_action)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)

        _append_step(
            outcome_buffers,
            {
                "reward": rewards,
                "done": dones.to(dtype=torch.bool),
                "terminated": base_env.reset_terminated.to(dtype=torch.bool),
                "time_out": base_env.reset_time_outs.to(dtype=torch.bool),
            },
        )

        episode_step += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            episode_id[done_ids] += 1
            episode_step[done_ids] = 0
            _update_episode_initial_state(initial_state, _capture_episode_initial_state(base_env), done_ids)

        if (step + 1) % 100 == 0 or step + 1 == args_cli.num_steps:
            print(f"[INFO] Collected {step + 1}/{args_cli.num_steps} steps.")

    tensors = _stack_buffers(buffers)
    tensors.update(_stack_buffers(outcome_buffers))

    if args_cli.only_valid_yolo and "yolo_position_valid" in tensors and "yolo_measurement_valid" in tensors:
        tensors["valid_sample_mask"] = tensors["yolo_position_valid"] & tensors["yolo_measurement_valid"]

    metadata_dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_steps_recorded": int(next(iter(tensors.values())).shape[0]) if tensors else 0,
        "num_envs": int(base_env.num_envs),
        "step_dt": float(base_env.step_dt),
        "rsl_rl_version": installed_version,
        "policy_input_key": "obs_policy",
        "policy_target_key": "action",
        "tensor_layout": "All per-step tensors are shaped [T, N, ...]. Use reshape(-1, dim) for supervised learning.",
        "distill_policy_obs_layout": {
            "joint_pos_normalized": [0, 20],
            "yolo_center_xy": [20, 22],
            "target_angle_features_sin2_cos2": [22, 24],
            "tactile_binary": [24, 34],
            "previous_action": [34, 54],
        },
        "actuated_joint_names": getattr(base_env.cfg, "actuated_joint_names", None),
        "hand_joint_names": getattr(base_env, "hand", None).joint_names if hasattr(base_env, "hand") else None,
        "object_usd_path": getattr(getattr(base_env.cfg.object_cfg, "spawn", None), "usd_path", None),
        "goal_y_angle_rad": getattr(base_env.cfg, "fixed_goal_y_angle_rad", None),
        "goal_rot": getattr(base_env.cfg, "fixed_goal_rot", None),
        "only_valid_yolo_requested": bool(args_cli.only_valid_yolo),
        "save_yolo_mask_pixels": bool(args_cli.save_yolo_mask_pixels),
        "yolo_mask_pixels_size_wh": list(args_cli.yolo_mask_size),
        "yolo_mask_pixels_note": (
            "Binary uint8 0/1 mask pixels shaped [T, N, H, W]. "
            "Use --input_keys yolo_mask_pixels in scripts/train_supervised_policy.py to flatten it as input."
        ),
    }

    output_path = _make_output_path(task_name)
    dataset = {"metadata": metadata_dict, "tensors": tensors}
    torch.save(dataset, output_path)
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_metadata_to_jsonable(metadata_dict), f, indent=2, ensure_ascii=False)

    total_samples = 0
    if "obs_policy" in tensors:
        total_samples = int(tensors["obs_policy"].shape[0] * tensors["obs_policy"].shape[1])
    print(f"[INFO] Saved dataset: {output_path}")
    print(f"[INFO] Saved metadata: {json_path}")
    print(f"[INFO] Total supervised samples: {total_samples}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

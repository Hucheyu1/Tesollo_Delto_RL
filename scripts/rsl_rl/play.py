# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import vtdex_model_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--mask_vtdex_tactile",
    "--mask-vtdex-tactile",
    action="store_true",
    default=False,
    help=(
        "Set every VTDex tactile channel presented to the actor/encoder to zero while preserving "
        "physical contacts and raw tactile metrics. Intended for tactile-ablation evaluation."
    ),
)
parser.add_argument(
    "--max_steps",
    "--max-steps",
    type=int,
    default=0,
    metavar="STEPS",
    help="Stop play cleanly after this many policy steps (0 runs until the simulator closes).",
)
parser.add_argument(
    "--load_exported_policy",
    "--load-exported-policy",
    action="store_true",
    default=False,
    help=(
        "Load --checkpoint as an exported TorchScript policy.pt with torch.jit.load(). "
        "This skips RSL-RL runner.load(), critic loading, optimizer loading, and policy re-export."
    ),
)
parser.add_argument(
    "--video_view",
    "--video-view",
    choices=("camera", "viewer"),
    default="camera",
    help=(
        "Viewpoint used by --video. 'camera' records from the configured student camera pose when available; "
        "'viewer' keeps the task's default viewer pose."
    ),
)
parser.add_argument(
    "--video_resolution",
    "--video-resolution",
    type=int,
    nargs=2,
    default=None,
    metavar=("WIDTH", "HEIGHT"),
    help=(
        "Override the recorded viewer/video resolution. Example: --video_resolution 1920 1080. "
        "This affects the RecordVideo render output, not the policy camera tensor."
    ),
)
parser.add_argument(
    "--video_distance_scale",
    "--video-distance-scale",
    type=float,
    default=1.0,
    help=(
        "When --video_view camera is used, multiply the vector from lookat to eye by this value. "
        "Use >1.0 to move the recording view farther away and include a larger scene."
    ),
)
parser.add_argument(
    "--video_eye_offset",
    "--video-eye-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help=(
        "Additional env-local offset added to the recording viewer eye after distance scaling. "
        "Useful for shifting the recorded view left/right/up/down without changing the policy camera."
    ),
)
parser.add_argument(
    "--video_lookat_offset",
    "--video-lookat-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help=(
        "Additional env-local offset added to the recording viewer lookat point. "
        "Useful for centering the object/hand in a wider recorded video."
    ),
)
parser.add_argument(
    "--hide_video_markers",
    "--hide-video-markers",
    action="store_true",
    default=False,
    help="Do not force goal/object coordinate-frame markers on while recording play video.",
)
parser.add_argument(
    "--show_video_goal_mesh",
    "--show-video-goal-mesh",
    action="store_true",
    default=False,
    help=(
        "Also show the target tomato mesh in play videos. For YOLO distillation policies this can change the "
        "student observation, so the safer default is to show only coordinate-frame markers."
    ),
)
parser.add_argument(
    "--goal_y_angle_deg",
    "--goal-y-angle-deg",
    type=float,
    default=None,
    help="Fix the play target to a hand-local Y-axis angle in degrees. Useful for distillation tests.",
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
vtdex_model_args.add_vtdex_model_arg(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.max_steps < 0:
    parser.error("--max_steps must be non-negative")
if args_cli.video_resolution is not None and (args_cli.video_resolution[0] <= 0 or args_cli.video_resolution[1] <= 0):
    parser.error("--video_resolution WIDTH HEIGHT must be positive")
if args_cli.video_distance_scale <= 0.0:
    parser.error("--video_distance_scale must be positive")
# Visual tasks require RTX camera rendering even without video output.
if args_cli.video or (
    args_cli.task is not None and any(tag in args_cli.task for tag in ("Distill", "VTDex"))
):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Tesollo_Delto_RL.tasks  # noqa: F401


def _as_tuple3(value) -> tuple[float, float, float]:
    return tuple(float(x) for x in value)


def _tuple3_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _tuple3_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _tuple3_scale(a: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _configure_video_camera_view(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Use the configured camera position as the viewport recording view."""

    camera_cfg = getattr(env_cfg, "student_camera", None)
    if camera_cfg is None:
        camera_cfg = getattr(env_cfg, "tiled_camera", None)
    if camera_cfg is None:
        camera_cfg = getattr(env_cfg, "vtdex_camera", None)
    if camera_cfg is None or not hasattr(camera_cfg, "offset"):
        print("[INFO] --video_view camera requested, but this task has no student/tiled camera cfg. Keeping viewer pose.")
        return

    if hasattr(env_cfg, "vtdex_camera_eye_local"):
        camera_pos = _as_tuple3(env_cfg.vtdex_camera_eye_local)
        camera_lookat = _as_tuple3(env_cfg.vtdex_camera_target_local)
    else:
        camera_pos = _as_tuple3(camera_cfg.offset.pos)
        object_cfg = getattr(env_cfg, "object_cfg", None)
        object_init_state = getattr(object_cfg, "init_state", None)
        camera_lookat = _as_tuple3(getattr(object_init_state, "pos", (0.10, 0.0, 0.50)))

    # Keep the policy/sensor camera unchanged. Only move the viewer used by RecordVideo.
    lookat_offset = _as_tuple3(args_cli.video_lookat_offset)
    eye_offset = _as_tuple3(args_cli.video_eye_offset)
    camera_lookat = _tuple3_add(camera_lookat, lookat_offset)
    view_vector = _tuple3_sub(camera_pos, camera_lookat)
    camera_pos = _tuple3_add(
        camera_lookat,
        _tuple3_scale(view_vector, float(args_cli.video_distance_scale)),
    )
    camera_pos = _tuple3_add(camera_pos, eye_offset)

    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = camera_pos
    env_cfg.viewer.lookat = camera_lookat
    if args_cli.video_resolution is not None:
        env_cfg.viewer.resolution = (int(args_cli.video_resolution[0]), int(args_cli.video_resolution[1]))
    elif hasattr(camera_cfg, "width") and hasattr(camera_cfg, "height"):
        env_cfg.viewer.resolution = (int(camera_cfg.width), int(camera_cfg.height))

    print(
        "[INFO] Recording play video from configured camera view: "
        f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}, resolution={env_cfg.viewer.resolution}, "
        f"distance_scale={args_cli.video_distance_scale}"
    )


def _apply_video_resolution_override(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Apply --video_resolution even when --video_view viewer keeps the default viewer pose."""

    if args_cli.video_resolution is None:
        return
    env_cfg.viewer.resolution = (int(args_cli.video_resolution[0]), int(args_cli.video_resolution[1]))
    print(f"[INFO] Recording play video resolution overridden to {env_cfg.viewer.resolution}")


class _ExportedJitPolicy:
    """Adapter so an exported TorchScript actor behaves like an RSL-RL inference policy.

    RSL-RL/Isaac-Lab environments may return either a plain tensor or a TensorDict-like
    object with keys such as "policy" and "critic".  The exported policy.pt is a
    TorchScript actor and expects only the policy observation tensor, so this adapter
    extracts that tensor only for the exported-policy path.  The normal checkpoint path
    still uses runner.get_inference_policy() unchanged.
    """

    def __init__(self, module: torch.jit.ScriptModule):
        self.module = module

    @staticmethod
    def _extract_policy_obs(obs):
        if isinstance(obs, torch.Tensor):
            return obs

        # TensorDict and dict both support key lookup.  Prefer the actor/policy
        # observation and ignore privileged critic/state entries.
        for key in ("policy", "obs", "student", "actor"):
            try:
                if key in obs:
                    value = obs[key]
                    if isinstance(value, torch.Tensor):
                        return value
            except Exception:
                pass

        # Some wrappers expose a get() method but not reliable membership checks.
        get_fn = getattr(obs, "get", None)
        if callable(get_fn):
            for key in ("policy", "obs", "student", "actor"):
                try:
                    value = get_fn(key)
                except Exception:
                    continue
                if isinstance(value, torch.Tensor):
                    return value

        # Last-resort diagnostic: show available keys to make shape/key mistakes
        # obvious instead of letting TorchScript fail with a TensorDict dispatch error.
        try:
            keys = list(obs.keys())
        except Exception:
            keys = f"unavailable for type {type(obs)!r}"
        raise TypeError(
            "Exported TorchScript policy expects a torch.Tensor observation, but play.py received "
            f"{type(obs)!r}. Could not find a tensor under keys policy/obs/student/actor; keys={keys}."
        )

    def __call__(self, obs):
        policy_obs = self._extract_policy_obs(obs)
        return self.module(policy_obs)

    def reset(self, dones=None):
        # Exported feed-forward actor has no recurrent state. Keep this method so
        # the original play loop can call policy.reset(dones) unchanged.
        return None


def _is_exported_policy_path(path: str) -> bool:
    normalized = os.path.normpath(str(path))
    return normalized.endswith(os.path.join("exported", "policy.pt")) or os.path.basename(normalized) in (
        "policy.pt",
        "policy.jit.pt",
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    vtdex_model_args.apply_vtdex_model_selection(env_cfg, agent_cfg, args_cli.model)
    if args_cli.mask_vtdex_tactile:
        if not hasattr(env_cfg, "vtdex_mask_tactile_input"):
            raise ValueError(
                "--mask_vtdex_tactile requires a VTDex task that implements "
                "the vtdex_mask_tactile_input configuration option"
            )
        env_cfg.vtdex_mask_tactile_input = True
        if args_cli.model == "vision":
            print(
                "[INFO] VTDex tactile ablation flag is redundant in vision mode: V-CLIP never receives "
                "touch; the Tomato actor's touch placeholders remain zero."
            )
        else:
            print(
                "[INFO] VTDex tactile ablation enabled: actor/encoder receives 20 zero touch channels; "
                "physics and raw tactile metrics remain active."
            )
    if args_cli.video:
        if args_cli.video_view == "camera":
            _configure_video_camera_view(env_cfg)
        _apply_video_resolution_override(env_cfg)
        if not args_cli.hide_video_markers:
            if hasattr(env_cfg, "debug_visualization"):
                env_cfg.debug_visualization = True
            print("[INFO] Play video frame markers enabled: tomato/object frame and goal frame.")
        if args_cli.show_video_goal_mesh and hasattr(env_cfg, "hide_goal_marker_from_yolo"):
            env_cfg.hide_goal_marker_from_yolo = False
            print("[INFO] Play video goal tomato mesh marker enabled.")
    if args_cli.goal_rot is not None:
        env_cfg.fixed_goal_rot = tuple(args_cli.goal_rot)
        print(f"[INFO] Fixed play goal quaternion wxyz: {env_cfg.fixed_goal_rot}")
    elif args_cli.goal_y_angle_rad is not None:
        env_cfg.fixed_goal_y_angle_rad = args_cli.goal_y_angle_rad
        print(f"[INFO] Fixed play goal Y angle: {env_cfg.fixed_goal_y_angle_rad:.6f} rad")
    elif args_cli.goal_y_angle_deg is not None:
        env_cfg.fixed_goal_y_angle_rad = math.radians(args_cli.goal_y_angle_deg)
        print(
            f"[INFO] Fixed play goal Y angle: {args_cli.goal_y_angle_deg:.3f} deg "
            f"({env_cfg.fixed_goal_y_angle_rad:.6f} rad)"
        )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    load_exported_policy = bool(args_cli.load_exported_policy) or _is_exported_policy_path(resume_path)

    if load_exported_policy:
        print(f"[INFO]: Loading exported TorchScript policy from: {resume_path}")
        policy_module = torch.jit.load(resume_path, map_location=env.unwrapped.device)
        policy_module.eval()
        policy = _ExportedJitPolicy(policy_module)
        # Keep the rest of the original play loop unchanged.  Exported policies are
        # inference-only, so there is no runner, critic, optimizer, or re-export.
        print("[INFO]: Exported policy loaded; skipped RSL-RL runner.load(), critic loading, and policy export.")
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    evaluation_reward_sum = 0.0
    evaluation_sample_count = 0
    evaluation_terminal_count = 0
    evaluation_success_count = 0.0
    evaluation_tactile_ratio_sum = 0.0
    evaluation_policy_tactile_ratio_sum = 0.0
    evaluation_tactile_steps = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            if args_cli.max_steps > 0:
                evaluation_reward_sum += float(rewards.sum().item())
                evaluation_sample_count += int(rewards.numel())
                evaluation_terminal_count += int(dones.sum().item())
                log_metrics = extras.get("log", {}) if isinstance(extras, dict) else {}
                success_rate = log_metrics.get("table_success_rate")
                if success_rate is None:
                    success_rate = log_metrics.get("reorient_up_success_rate")
                if success_rate is not None:
                    evaluation_success_count += float(success_rate) * int(env.unwrapped.num_envs)
                tactile_ratio = log_metrics.get("vtdex_tactile_active_ratio")
                policy_tactile_ratio = log_metrics.get("vtdex_policy_tactile_active_ratio")
                if tactile_ratio is not None and policy_tactile_ratio is not None:
                    evaluation_tactile_ratio_sum += float(tactile_ratio)
                    evaluation_policy_tactile_ratio_sum += float(policy_tactile_ratio)
                    evaluation_tactile_steps += 1
            # reset recurrent states for episodes that have terminated
            if load_exported_policy:
                policy.reset(dones)
            elif version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.max_steps > 0:
        mean_reward = evaluation_reward_sum / max(evaluation_sample_count, 1)
        success_per_terminal = evaluation_success_count / max(evaluation_terminal_count, 1)
        print(
            "[EVAL] "
            f"steps={timestep}, env_samples={evaluation_sample_count}, "
            f"mean_reward={mean_reward:.6f}, terminals={evaluation_terminal_count}, "
            f"successes={evaluation_success_count:.1f}, "
            f"success_per_terminal={success_per_terminal:.6f}"
        )
        if evaluation_tactile_steps > 0:
            print(
                "[EVAL] "
                f"raw_tactile_active_ratio="
                f"{evaluation_tactile_ratio_sum / evaluation_tactile_steps:.6f}, "
                f"policy_tactile_active_ratio="
                f"{evaluation_policy_tactile_ratio_sum / evaluation_tactile_steps:.6f}"
            )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
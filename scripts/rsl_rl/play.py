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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
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

    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = camera_pos
    env_cfg.viewer.lookat = camera_lookat
    if hasattr(camera_cfg, "width") and hasattr(camera_cfg, "height"):
        env_cfg.viewer.resolution = (int(camera_cfg.width), int(camera_cfg.height))

    print(
        "[INFO] Recording play video from configured camera view: "
        f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}, resolution={env_cfg.viewer.resolution}"
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
    if args_cli.video:
        if args_cli.video_view == "camera":
            _configure_video_camera_view(env_cfg)
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
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

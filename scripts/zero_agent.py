# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--camera_view",
    action="store_true",
    help="Align the interactive viewer with the task's VTDex policy camera.",
)
parser.add_argument(
    "--save_camera_frame",
    type=str,
    default=None,
    metavar="PATH",
    help="Save environment 0's exact VTDex policy RGB input as a PNG.",
)
parser.add_argument(
    "--save_camera_every",
    type=int,
    default=0,
    metavar="STEPS",
    help="Also save the VTDex policy RGB input every N simulation steps (0 disables it).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.save_camera_every < 0:
    parser.error("--save_camera_every must be non-negative")
if args_cli.save_camera_every > 0 and args_cli.save_camera_frame is None:
    parser.error("--save_camera_every requires --save_camera_frame PATH")

# TiledCamera rendering must be enabled before Isaac Sim is launched, including
# for a headless one-frame capture.
if (
    args_cli.camera_view
    or args_cli.save_camera_frame is not None
    or (args_cli.task is not None and "VTDex" in args_cli.task)
):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Tesollo_Delto_RL.tasks  # noqa: F401


def _configure_vtdex_viewer(env_cfg) -> None:
    """Align the viewport with environment 0's VTDex camera look-at pose."""

    if not hasattr(env_cfg, "vtdex_camera_eye_local") or not hasattr(env_cfg, "vtdex_camera_target_local"):
        raise ValueError("--camera_view requires a task configured with a VTDex camera")

    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(float(value) for value in env_cfg.vtdex_camera_eye_local)
    env_cfg.viewer.lookat = tuple(float(value) for value in env_cfg.vtdex_camera_target_local)
    camera_cfg = env_cfg.vtdex_camera
    env_cfg.viewer.resolution = (int(camera_cfg.width), int(camera_cfg.height))
    print(
        "[INFO]: Viewer aligned with VTDex camera: "
        f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}"
    )


def _camera_frame_path(base_path: Path, step: int | None = None) -> Path:
    """Return the requested single-frame path or a numbered sequence path."""

    path = base_path if base_path.suffix else base_path.with_suffix(".png")
    if step is None:
        return path
    return path.with_name(f"{path.stem}_step{step:06d}{path.suffix}")


def _save_vtdex_camera_frame(env, output_path: Path, step: int | None = None) -> None:
    """Save the exact RGB tensor consumed by the VTDex encoder for env 0."""

    try:
        camera = env.unwrapped.scene.sensors["vtdex_camera"]
    except KeyError as exc:
        raise ValueError("Camera capture requires a task with scene sensor 'vtdex_camera'") from exc

    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    rgb = rgb.astype(np.uint8)

    frame_path = _camera_frame_path(output_path, step)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(frame_path)
    print(f"[INFO]: Saved VTDex policy camera frame: {frame_path.resolve()}")


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.camera_view:
        _configure_vtdex_viewer(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    camera_output_path = Path(args_cli.save_camera_frame).expanduser() if args_cli.save_camera_frame else None
    if camera_output_path is not None:
        _save_vtdex_camera_frame(env, camera_output_path)
    # simulate environment
    step = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            step += 1
            if camera_output_path is not None and args_cli.save_camera_every > 0:
                if step % args_cli.save_camera_every == 0:
                    _save_vtdex_camera_frame(env, camera_output_path, step=step)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

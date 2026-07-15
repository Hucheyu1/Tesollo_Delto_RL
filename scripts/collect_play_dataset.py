# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""采集状态和 YOLO 数据。

本脚本 YOLO 只作为独立的数据采集模块运行。

保存字段：
    global_step            [T, N]
    env_id                 [T, N]
    episode_id             [T, N]
    episode_step           [T, N]
    hand_dof_pos           [T, N, num_dofs]
    tactile_binary         [T, N, tactile_dim]
    goal_pos               [T, N, 3]
    goal_rot               [T, N, 4]
    cur_targets            [T, N, num_dofs]
    prev_targets           [T, N, num_dofs]
    yolo_position_image    [T, N, 2]
    yolo_angle_image_rad   [T, N]
    yolo_mask_pixels       [T, N, mask_height, mask_width]

接入方式：
1. 在创建环境前，动态向普通 task 的 env_cfg 注入 student_camera 和 YOLO 参数；
2. 令环境按照其已有的可选入口创建 TiledCamera 和 YoloSegImageEstimator;
3. 不调用蒸馏 observation, 而是在采集循环中直接调用 estimator.estimate(camera.data);
4. 只把滤波中心、滤波轴角和缩放后的二值 mask 写入数据集。

要求：
- 环境类需要包含用户所给代码中的可选 YOLO 入口：
  use_yolo_student_obs、student_camera、yolo_student_estimator、_student_camera。
- task 包目录内需要存在 yolo_seg_image_estimator.py, 并且返回值包含 mask_image。
- best.pt 必须是 Ultralytics YOLO segmentation 模型。
"""

from __future__ import annotations

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

# 复用 scripts/rsl_rl/play.py 同目录中的 RSL-RL 命令行参数。
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402


# -----------------------------------------------------------------------------
# 命令行参数
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="在普通 RSL-RL task 中附加 YOLO-seg 相机并采集精简数据集。"
)

# 数据采集参数。
parser.add_argument("--num_steps", "--num-steps", type=int, default=1000, help="正式记录的控制步数。")
parser.add_argument(
    "--warmup_steps",
    "--warmup-steps",
    type=int,
    default=5,
    help="正式记录前的相机/YOLO 预热步数；预热数据不保存。",
)
parser.add_argument(
    "--no_reset_after_warmup",
    "--no-reset-after-warmup",
    action="store_true",
    default=False,
    help="默认会在 YOLO 预热后 reset 一次，让正式采集从固定初始位姿开始；打开该项则保留旧行为。",
)
parser.add_argument(
    "--output_dir",
    "--output-dir",
    type=str,
    default="datasets/play_yolo_minimal",
    help="保存 .pt 数据集和 .json 元数据的目录。",
)
parser.add_argument(
    "--output_file",
    "--output-file",
    type=str,
    default=None,
    help="可选输出文件名；默认使用 <task>_<timestamp>.pt。",
)
parser.add_argument("--num_envs", type=int, default=None, help="并行仿真环境数量。")
parser.add_argument("--task", type=str, default=None, help="Isaac Lab task 名称。")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent 配置入口。",
)
parser.add_argument("--seed", type=int, default=None, help="环境随机种子。")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="使用官方发布的预训练 checkpoint。",
)

# YOLO 模型参数。
parser.add_argument(
    "--yolo_model_path",
    "--yolo-model-path",
    type=str,
    required=True,
    help="YOLO segmentation best.pt 的路径。",
)
parser.add_argument("--yolo_class_id", type=int, default=0, help="需要选择的目标类别 ID。")
parser.add_argument("--yolo_confidence", type=float, default=0.5, help="YOLO 置信度阈值。")
parser.add_argument("--yolo_iou", type=float, default=0.7, help="YOLO NMS IoU 阈值。")
parser.add_argument("--yolo_inference_size", type=int, default=640, help="YOLO 推理输入尺寸。")
parser.add_argument("--yolo_min_mask_pixels", type=int, default=64, help="接受 mask 的最小像素数。")
parser.add_argument("--yolo_position_gain", type=float, default=0.65, help="二维中心 EMA 更新增益。")
parser.add_argument("--yolo_min_anisotropy", type=float, default=0.10, help="PCA 轴角最小各向异性。")
parser.add_argument("--yolo_min_visible_ratio", type=float, default=0.40, help="最小可见面积比例。")
parser.add_argument(
    "--yolo_max_angle_jump",
    type=float,
    default=0.70,
    help="允许的单步最大轴角创新，单位 rad。",
)

# 保存 mask 的尺寸。命令行顺序是 WIDTH HEIGHT；Tensor 中仍是 H, W。
parser.add_argument(
    "--yolo_mask_size",
    "--yolo-mask-size",
    type=int,
    nargs=2,
    default=(64, 48),
    metavar=("WIDTH", "HEIGHT"),
    help="保存的二值 mask 尺寸，顺序 WIDTH HEIGHT。",
)

# 采集相机参数，默认值来自用户给出的 distill camera 配置。
parser.add_argument(
    "--camera_pos",
    type=float,
    nargs=3,
    default=(0.11, 0.25, 0.37),
    metavar=("X", "Y", "Z"),
    help="相机在每个 env 中的位置。",
)
parser.add_argument(
    "--camera_rot",
    type=float,
    nargs=4,
    default=(0.707106, 0.0, 0.0, -0.707106),
    metavar=("W", "X", "Y", "Z"),
    help="相机 world convention 四元数，顺序 wxyz。",
)
parser.add_argument("--camera_width", type=int, default=640, help="相机图像宽度。")
parser.add_argument("--camera_height", type=int, default=480, help="相机图像高度。")
parser.add_argument("--camera_focal_length", type=float, default=18.0, help="相机焦距。")
parser.add_argument("--camera_focus_distance", type=float, default=0.45, help="相机焦点距离。")
parser.add_argument("--camera_horizontal_aperture", type=float, default=20.955, help="相机水平光圈。")
parser.add_argument(
    "--camera_clipping_range",
    type=float,
    nargs=2,
    default=(0.01, 1.5),
    metavar=("NEAR", "FAR"),
    help="相机裁剪范围。",
)
parser.add_argument(
    "--show_camera_frustum",
    action="store_true",
    default=False,
    help="显示相机视锥，仅调试时使用。",
)
parser.add_argument(
    "--keep_goal_marker_visible",
    action="store_true",
    default=False,
    help="不隐藏目标 tomato marker；默认隐藏，避免 YOLO 选中目标 marker。",
)

# 可选固定目标姿态。
parser.add_argument("--goal_y_angle_deg", type=float, default=None, help="固定手局部 Y 轴目标角，单位 degree。")
parser.add_argument("--goal_y_angle_rad", type=float, default=None, help="固定手局部 Y 轴目标角，单位 rad。")
parser.add_argument(
    "--goal_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="直接固定 goal_rot，wxyz；优先级最高。",
)

# 可选固定番茄初始位姿。采集脚本默认固定 object_cfg.init_state 中的初始位姿，
# 这样不同 env、不同 episode 的番茄起点一致；如需保留训练时随机初始姿态，可显式打开随机。
parser.add_argument(
    "--randomize_object_initial_pose",
    "--randomize-object-initial-pose",
    action="store_true",
    default=False,
    help="采集时保留环境原本的番茄初始姿态随机化；默认关闭随机化，让初始位姿一致。",
)
parser.add_argument(
    "--object_pos",
    "--object-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="可选：固定番茄 env-local 初始位置；默认使用 object_cfg.init_state.pos。",
)
parser.add_argument(
    "--object_y_angle_deg",
    "--object-y-angle-deg",
    type=float,
    default=None,
    help="可选：固定番茄初始 Y 轴角，单位 degree。",
)
parser.add_argument(
    "--object_y_angle_rad",
    "--object-y-angle-rad",
    type=float,
    default=None,
    help="可选：固定番茄初始 Y 轴角，单位 rad。",
)
parser.add_argument(
    "--object_rot",
    "--object-rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="可选：直接固定番茄初始四元数 wxyz；优先级高于 object_y_angle。",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 本脚本无论使用什么 task 都必须创建 RTX 相机。
args_cli.enable_cameras = True

# 只把未解析参数留给 Hydra。
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Sim 启动后再导入仿真相关模块
# -----------------------------------------------------------------------------
import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Tesollo_Delto_RL.tasks  # noqa: F401, E402

installed_version = metadata.version("rsl-rl-lib")


# -----------------------------------------------------------------------------
# 通用工具函数
# -----------------------------------------------------------------------------
def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    """复制张量并搬到 CPU，防止环境 buffer 后续原地修改历史数据。"""

    return value.detach().clone().cpu()


def _append_step(buffers: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]) -> None:
    """把一个采集步中的所有字段追加到时间序列 buffer。"""

    for key, value in values.items():
        buffers.setdefault(key, []).append(_clone_cpu(value))


def _stack_buffers(buffers: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """把逐步 list 堆叠为 [T, N, ...] Tensor。"""

    return {key: torch.stack(values, dim=0) for key, values in buffers.items() if values}


def _resize_mask_pixels(mask: torch.Tensor, mask_size: tuple[int, int]) -> torch.Tensor:
    """将 [N,H,W] bool mask 缩放并转换为 uint8 0/1。"""

    width, height = mask_size
    if width <= 0 or height <= 0:
        raise ValueError(f"--yolo_mask_size 非法: {mask_size}")

    mask_nchw = mask.to(dtype=torch.float32).unsqueeze(1)
    resized = functional.interpolate(mask_nchw, size=(height, width), mode="nearest")
    return resized[:, 0].to(dtype=torch.uint8)


def _metadata_to_jsonable(value: Any) -> Any:
    """把 metadata 中的对象转换为 JSON 可以写入的类型。"""

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
    """生成 .pt 输出路径。"""

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


# -----------------------------------------------------------------------------
# 向普通 task 配置动态注入 YOLO 相机和估计器参数
# -----------------------------------------------------------------------------
def _inject_yolo_collection_config(env_cfg: Any) -> None:
    """在 gym.make() 前向普通 task 的配置中补齐 YOLO 所需属性。

    用户给出的环境类会检查 cfg.use_yolo_student_obs：
    - 在 _setup_scene() 中创建并注册 self._student_camera；
    - 在 __init__() 中创建 self.yolo_student_estimator。

    对于 obs_type='full' 或 obs_type='openai' 的普通 task，该标志不会把 YOLO
    拼进策略 observation，因此 checkpoint 的输入维度保持原样。
    """

    model_path = Path(args_cli.yolo_model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO segmentation 模型不存在: {model_path}")

    if args_cli.camera_width <= 0 or args_cli.camera_height <= 0:
        raise ValueError("camera_width 和 camera_height 必须为正数。")

    # 让用户所给环境类进入可选相机/估计器创建分支。
    env_cfg.use_yolo_student_obs = True

    # 只采集 RGB；中心和 PCA 轴角均从分割 mask 得到。
    env_cfg.student_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/CollectionYoloCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in args_cli.camera_pos),
            rot=tuple(float(v) for v in args_cli.camera_rot),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args_cli.camera_focal_length),
            focus_distance=float(args_cli.camera_focus_distance),
            horizontal_aperture=float(args_cli.camera_horizontal_aperture),
            clipping_range=tuple(float(v) for v in args_cli.camera_clipping_range),
        ),
        width=int(args_cli.camera_width),
        height=int(args_cli.camera_height),
        update_latest_camera_pose=True,
        debug_vis=bool(args_cli.show_camera_frustum),
    )

    # YoloSegImageEstimator 构造参数。
    env_cfg.yolo_model_path = str(model_path)
    env_cfg.yolo_class_id = int(args_cli.yolo_class_id)
    env_cfg.yolo_confidence_threshold = float(args_cli.yolo_confidence)
    env_cfg.yolo_iou_threshold = float(args_cli.yolo_iou)
    env_cfg.yolo_inference_size = int(args_cli.yolo_inference_size)
    env_cfg.yolo_min_mask_pixels = int(args_cli.yolo_min_mask_pixels)
    env_cfg.yolo_position_gain = float(args_cli.yolo_position_gain)
    env_cfg.yolo_min_anisotropy = float(args_cli.yolo_min_anisotropy)
    env_cfg.yolo_min_visible_ratio = float(args_cli.yolo_min_visible_ratio)
    env_cfg.yolo_max_angle_jump = float(args_cli.yolo_max_angle_jump)

    # 环境 __init__ 中会访问这些属性。普通采集不使用角度标定特征，
    # 但仍设置默认值以保证配置完整。
    env_cfg.yolo_angle_sign = 1.0
    env_cfg.yolo_angle_offset_rad = None

    # 默认把使用相同 tomato mesh 的目标 marker 移出相机视野，避免误检。
    env_cfg.hide_goal_marker_from_yolo = not bool(args_cli.keep_goal_marker_visible)

    print("[INFO] 已向普通 task 动态注入 YOLO 采集配置：")
    print(f"       model={env_cfg.yolo_model_path}")
    print(f"       camera={args_cli.camera_width}x{args_cli.camera_height}")
    print(f"       num_envs={env_cfg.scene.num_envs}")
    print(f"       hide_goal_marker={env_cfg.hide_goal_marker_from_yolo}")


def _validate_yolo_runtime(base_env: Any) -> None:
    """确认环境确实按注入配置创建了相机和 YOLO 估计器。"""

    estimator = getattr(base_env, "yolo_student_estimator", None)
    camera = getattr(base_env, "_student_camera", None)

    if estimator is None or camera is None:
        raise RuntimeError(
            "环境没有创建 YOLO 采集模块。请确认当前 task 使用的是用户提供的 "
            "TesolloDeltoRlEnv 环境类，并保留其中 use_yolo_student_obs、"
            "_student_camera 和 yolo_student_estimator 的可选初始化代码。"
        )


def _estimate_yolo_for_collection(base_env: Any):
    """直接从采集相机运行 YOLO，不依赖 task observation 中是否包含 YOLO。"""

    estimator = getattr(base_env, "yolo_student_estimator", None)
    camera = getattr(base_env, "_student_camera", None)
    if estimator is None or camera is None:
        raise RuntimeError("YOLO estimator 或 collection camera 未初始化。")

    estimate = estimator.estimate(camera.data)

    # 采集脚本保存 mask，因此估计器返回值必须含有 mask_image。
    if not isinstance(getattr(estimate, "mask_image", None), torch.Tensor):
        raise RuntimeError(
            "YoloSegImageEstimate 缺少 mask_image。请使用包含 mask_image 字段的 "
            "yolo_seg_image_estimator.py。"
        )

    return estimate


# -----------------------------------------------------------------------------
# 单步数据采集
# -----------------------------------------------------------------------------
def _collect_step_tensors(
    *,
    base_env: Any,
    yolo_estimate: Any,
    global_step: int,
    episode_id: torch.Tensor,
    episode_step: torch.Tensor,
    yolo_mask_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """只收集用户指定的状态字段和三个 YOLO 字段。"""

    num_envs = int(base_env.num_envs)
    device = base_env.device

    return {
        # 时间和 episode 标识。
        "global_step": torch.full((num_envs,), global_step, dtype=torch.long, device=device),
        "env_id": torch.arange(num_envs, dtype=torch.long, device=device),
        "episode_id": episode_id,
        "episode_step": episode_step,

        # 用户指定的灵巧手、触觉和目标数据。
        "hand_dof_pos": base_env.hand_dof_pos,
        "tactile_binary": base_env.fingertip_force_binary_results,
        "goal_pos": base_env.goal_pos,
        "goal_rot": base_env.goal_rot,
        "cur_targets": base_env.cur_targets,
        "target_pos": base_env.target_pos,

        # YOLO 部分只保存以下三个字段。
        # position_image、angle_image_rad 已由估计器完成时序滤波。
        "yolo_position_image": yolo_estimate.position_image,
        "yolo_angle_image_rad": yolo_estimate.angle_image_rad,
        "yolo_mask_pixels": _resize_mask_pixels(
            yolo_estimate.mask_image,
            yolo_mask_size,
        ),
    }


# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """加载 checkpoint，在普通 task 中附加 YOLO 并采集精简数据。"""

    if args_cli.task is None:
        raise ValueError("必须通过 --task 指定任务。")
    if args_cli.num_steps <= 0:
        raise ValueError("--num_steps 必须大于 0。")
    if args_cli.warmup_steps < 0:
        raise ValueError("--warmup_steps 不能小于 0。")

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # 应用 RSL-RL 命令行覆盖项。
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)

    # 固定目标姿态是可选的。
    if args_cli.goal_rot is not None:
        env_cfg.fixed_goal_rot = tuple(float(v) for v in args_cli.goal_rot)
        print(f"[INFO] 固定目标四元数 wxyz: {env_cfg.fixed_goal_rot}")
    elif args_cli.goal_y_angle_rad is not None:
        env_cfg.fixed_goal_y_angle_rad = float(args_cli.goal_y_angle_rad)
        print(f"[INFO] 固定目标 Y 角: {env_cfg.fixed_goal_y_angle_rad:.6f} rad")
    elif args_cli.goal_y_angle_deg is not None:
        env_cfg.fixed_goal_y_angle_rad = math.radians(float(args_cli.goal_y_angle_deg))
        print(
            f"[INFO] 固定目标 Y 角: {args_cli.goal_y_angle_deg:.3f} deg "
            f"({env_cfg.fixed_goal_y_angle_rad:.6f} rad)"
        )

    # 采集数据默认固定番茄初始位姿，避免同一数据集中 episode 起点不一致。
    env_cfg.fix_object_initial_pose = not bool(args_cli.randomize_object_initial_pose)
    if args_cli.object_pos is not None:
        env_cfg.fixed_object_pos = tuple(float(v) for v in args_cli.object_pos)
        print(f"[INFO] 固定番茄初始位置 env-local xyz: {env_cfg.fixed_object_pos}")
    if args_cli.object_rot is not None:
        env_cfg.fixed_object_rot = tuple(float(v) for v in args_cli.object_rot)
        print(f"[INFO] 固定番茄初始四元数 wxyz: {env_cfg.fixed_object_rot}")
    elif args_cli.object_y_angle_rad is not None:
        env_cfg.fixed_object_y_angle_rad = float(args_cli.object_y_angle_rad)
        print(f"[INFO] 固定番茄初始 Y 角: {env_cfg.fixed_object_y_angle_rad:.6f} rad")
    elif args_cli.object_y_angle_deg is not None:
        env_cfg.fixed_object_y_angle_rad = math.radians(float(args_cli.object_y_angle_deg))
        print(
            f"[INFO] 固定番茄初始 Y 角: {args_cli.object_y_angle_deg:.3f} deg "
            f"({env_cfg.fixed_object_y_angle_rad:.6f} rad)"
        )
    if env_cfg.fix_object_initial_pose:
        print("[INFO] 番茄初始位姿随机化已关闭：每次 reset 使用同一个初始位姿。")
    else:
        print("[INFO] 番茄初始位姿随机化已开启：采集会保留环境原本 reset 随机性。")

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 重点：在创建环境前补入 YOLO 相机与模型参数。
    _inject_yolo_collection_config(env_cfg)

    # 查找 checkpoint。
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(f"任务 {train_task_name!r} 没有可用的已发布 checkpoint。")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    env_cfg.log_dir = os.path.dirname(resume_path)

    print(f"[INFO] 加载 checkpoint: {resume_path}")
    print(f"[INFO] 正式采集: {args_cli.num_steps} steps x {env_cfg.scene.num_envs} envs")

    # 创建原 task。策略 observation 的定义仍由原 task 决定。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    _validate_yolo_runtime(base_env)

    # 创建并加载 RSL-RL runner。
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
    else:
        raise ValueError(f"不支持的 runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    # 兼容旧版 rsl_rl 的网络 reset 接口。
    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

    obs = env.get_observations()

    # ------------------------------------------------------------------
    # 相机与 YOLO 预热
    # ------------------------------------------------------------------
    for warmup_step in range(args_cli.warmup_steps):
        if not simulation_app.is_running():
            raise RuntimeError("Isaac Sim 在 YOLO 预热阶段提前停止。")

        with torch.inference_mode():
            policy_action = policy(obs)
            obs, _, dones, _ = env.step(policy_action)

            # 用预热帧初始化参考 mask 面积、中心和角度滤波状态。
            _estimate_yolo_for_collection(base_env)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)

        print(f"[INFO] YOLO warmup {warmup_step + 1}/{args_cli.warmup_steps}")

    if args_cli.warmup_steps > 0 and not args_cli.no_reset_after_warmup:
        with torch.inference_mode():
            obs, _ = env.reset()
            reset_dones = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(reset_dones)
            elif policy_nn is not None:
                policy_nn.reset(reset_dones)
        print("[INFO] YOLO 预热完成后已 reset，正式采集将从固定初始位姿开始。")

    # 正式记录开始时，从当前环境状态重新定义采集 episode 编号。
    buffers: dict[str, list[torch.Tensor]] = {}
    episode_id = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    episode_step = base_env.episode_length_buf.detach().clone().to(dtype=torch.long)

    # ------------------------------------------------------------------
    # 正式采集循环
    # ------------------------------------------------------------------
    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            print("[WARN] Isaac Sim 提前停止，未达到请求的 num_steps。")
            break

        with torch.inference_mode():
            # 当前普通 task 的策略推理；YOLO 不参与其 observation。
            policy_action = policy(obs)

            # 对当前相机帧独立运行 YOLO，并取得滤波后的中心和轴角。
            yolo_estimate = _estimate_yolo_for_collection(base_env)

            # 在执行本步动作前记录当前状态。
            step_values = _collect_step_tensors(
                base_env=base_env,
                yolo_estimate=yolo_estimate,
                global_step=step,
                episode_id=episode_id,
                episode_step=episode_step,
                yolo_mask_size=tuple(args_cli.yolo_mask_size),
            )
            _append_step(buffers, step_values)

            # 执行原策略动作。
            obs, _, dones, _ = env.step(policy_action)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)

        # 更新 episode 索引。即使不保存 done，也需要它防止历史序列跨 episode。
        episode_step += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            episode_id[done_ids] += 1
            episode_step[done_ids] = 0

        if (step + 1) % 100 == 0 or step + 1 == args_cli.num_steps:
            print(f"[INFO] 已采集 {step + 1}/{args_cli.num_steps} steps")

    tensors = _stack_buffers(buffers)

    # 明确检查最终字段，避免以后误加入不需要的数据。
    expected_keys = {
        "global_step",
        "env_id",
        "episode_id",
        "episode_step",
        "hand_dof_pos",
        "tactile_binary",
        "goal_pos",
        "goal_rot",
        "cur_targets",
        "target_pos",
        "yolo_position_image",
        "yolo_angle_image_rad",
        "yolo_mask_pixels",
    }
    actual_keys = set(tensors.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"保存字段与预期不一致。missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    metadata_dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_steps_recorded": int(tensors["global_step"].shape[0]),
        "num_envs": int(base_env.num_envs),
        "total_samples": int(tensors["global_step"].numel()),
        "step_dt": float(base_env.step_dt),
        "rsl_rl_version": installed_version,
        "tensor_layout": "所有字段均为 [T, N, ...]；mask 为 [T, N, H, W]。",
        "saved_tensor_keys": sorted(expected_keys),
        "warmup_steps": int(args_cli.warmup_steps),
        "reset_after_warmup": bool(args_cli.warmup_steps > 0 and not args_cli.no_reset_after_warmup),
        "actuated_joint_names": getattr(base_env.cfg, "actuated_joint_names", None),
        "hand_joint_names": base_env.hand.joint_names if hasattr(base_env, "hand") else None,
        "yolo_model_path": str(Path(args_cli.yolo_model_path).expanduser().resolve()),
        "yolo_class_id": int(args_cli.yolo_class_id),
        "yolo_mask_pixels_size_wh": list(args_cli.yolo_mask_size),
        "camera_position": list(args_cli.camera_pos),
        "camera_rotation_wxyz": list(args_cli.camera_rot),
        "camera_resolution_wh": [args_cli.camera_width, args_cli.camera_height],
        "fix_object_initial_pose": bool(getattr(env_cfg, "fix_object_initial_pose", False)),
        "fixed_object_pos": getattr(env_cfg, "fixed_object_pos", None),
        "fixed_object_rot": getattr(env_cfg, "fixed_object_rot", None),
        "fixed_object_y_angle_rad": getattr(env_cfg, "fixed_object_y_angle_rad", None),
        "yolo_position_note": "滤波二维中心，范围 [-1,1]，图像中心为 (0,0)，+Y 向下。",
        "yolo_angle_note": "滤波无方向轴角，范围 [0,pi)，theta 与 theta+pi 等价。",
        "yolo_mask_note": "当前帧二值分割 mask，uint8 0/1；不是时序滤波后的 mask。",
    }

    output_path = _make_output_path(task_name)
    dataset = {"metadata": metadata_dict, "tensors": tensors}
    torch.save(dataset, output_path)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(_metadata_to_jsonable(metadata_dict), file, indent=2, ensure_ascii=False)

    print(f"[INFO] 数据集已保存: {output_path}")
    print(f"[INFO] 元数据已保存: {json_path}")
    print(f"[INFO] 总样本数: {metadata_dict['total_samples']}")
    print("[INFO] 保存字段及形状：")
    for key, tensor in tensors.items():
        print(f"       {key:24s} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

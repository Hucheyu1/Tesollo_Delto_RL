from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
import types

import torch

from isaaclab.app import AppLauncher

# 复用 Isaac Lab / RSL-RL 的命令行参数
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # noqa: E402


parser = argparse.ArgumentParser(
    description="Play target_pos history policy with YOLO in Isaac Lab."
)

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--policy_jit", type=str, required=True)

parser.add_argument("--history_len", type=int, default=10)
parser.add_argument("--history_stride", type=int, default=1)

parser.add_argument("--yolo_model_path", type=str, required=True)
parser.add_argument("--yolo_mask_size", type=int, nargs=2, default=(64, 48))

parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--camera_pos", type=float, nargs=3, default=(0.11, 0.25, 0.37))
parser.add_argument(
    "--camera_rot",
    type=float,
    nargs=4,
    default=(0.707106, 0.0, 0.0, -0.707106),
)

parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument(
    "--no_reset_after_warmup",
    "--no-reset-after-warmup",
    action="store_true",
    default=False,
    help="默认会在 YOLO warmup 后 reset 一次，让正式 play 从固定初始位姿开始；打开该项则保留旧行为。",
)

parser.add_argument(
    "--randomize_object_initial_pose",
    "--randomize-object-initial-pose",
    action="store_true",
    default=False,
    help="play 时保留环境原本的番茄初始姿态随机化；默认关闭随机化，让初始位姿一致。",
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

parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the environment/agent config entry point used by hydra_task_config.",
)

parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video during play.",
)

parser.add_argument(
    "--video_length",
    "--video-length",
    type=int,
    default=500,
    help="Length of the recorded video in simulation steps.",
)

parser.add_argument(
    "--video_dir",
    "--video-dir",
    type=str,
    default="./logs/supervised_policy/videos",
    help="Directory where recorded videos will be saved.",
)

parser.add_argument(
    "--supervised_action_mode",
    "--supervised-action-mode",
    choices=("absolute", "delta"),
    default="absolute",
    help=(
        "How to feed supervised policy output into env.step(). "
        "'absolute': policy output is target_pos in rad and env.step(target_pos_pred). "
        "'delta': convert target_pos_pred to normalized delta action before env.step()."
    ),
)

parser.add_argument(
    "--debug_action",
    "--debug-action",
    action="store_true",
    default=False,
    help="Print action/target statistics during play.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# YOLO 和视频录制都需要相机。即使 --headless，也必须打开 enable_cameras。
args_cli.enable_cameras = True

# 清理 argv，交给 Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import Tesollo_Delto_RL.tasks  # noqa: F401, E402


def _inject_yolo_cfg(env_cfg):
    """给普通 task 动态补充 YOLO 相机和估计器配置。"""

    env_cfg.scene.num_envs = args_cli.num_envs

    # play 测试默认固定番茄初始位姿，和 collect_play_dataset.py 的默认采集分布对齐。
    env_cfg.fix_object_initial_pose = not bool(args_cli.randomize_object_initial_pose)
    if args_cli.object_pos is not None:
        env_cfg.fixed_object_pos = tuple(float(v) for v in args_cli.object_pos)
    if args_cli.object_rot is not None:
        env_cfg.fixed_object_rot = tuple(float(v) for v in args_cli.object_rot)
    elif args_cli.object_y_angle_rad is not None:
        env_cfg.fixed_object_y_angle_rad = float(args_cli.object_y_angle_rad)
    elif args_cli.object_y_angle_deg is not None:
        env_cfg.fixed_object_y_angle_rad = math.radians(float(args_cli.object_y_angle_deg))

    # 开启你环境类中预留的 YOLO camera 创建逻辑
    env_cfg.use_yolo_student_obs = True

    # YOLO 只用于采集/部署输入，不改变普通 policy observation
    env_cfg.student_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/StudentCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(args_cli.camera_pos),
            rot=tuple(args_cli.camera_rot),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.45,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.5),
        ),
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        update_latest_camera_pose=True,
        debug_vis=False,
    )

    env_cfg.yolo_model_path = args_cli.yolo_model_path
    env_cfg.yolo_class_id = 0
    env_cfg.yolo_confidence_threshold = 0.5
    env_cfg.yolo_iou_threshold = 0.7
    env_cfg.yolo_inference_size = 640
    env_cfg.yolo_min_mask_pixels = 64
    env_cfg.yolo_position_gain = 0.65
    env_cfg.yolo_min_anisotropy = 0.10
    env_cfg.yolo_min_visible_ratio = 0.40
    env_cfg.yolo_max_angle_jump = 0.70

    # 如果环境里有目标 marker，建议部署视觉时隐藏，避免 YOLO 检测到目标番茄
    env_cfg.hide_goal_marker_from_yolo = True

    # 给环境 cfg 也写入动作模式。
    # 即使环境源码暂时没有 action_mode 字段，Python config 通常也允许动态属性；
    # 真正的行为兼容由下面的 monkey patch 保证。
    env_cfg.action_mode = args_cli.supervised_action_mode
    env_cfg.use_absolute_target_action = args_cli.supervised_action_mode == "absolute"

    if env_cfg.fix_object_initial_pose:
        print("[INFO] 番茄初始位姿随机化已关闭：play 每次 reset 使用同一个初始位姿。")
    else:
        print("[INFO] 番茄初始位姿随机化已开启：play 会保留环境原本 reset 随机性。")
    if args_cli.object_pos is not None:
        print(f"[INFO] 固定番茄初始位置 env-local xyz: {env_cfg.fixed_object_pos}")
    if args_cli.object_rot is not None:
        print(f"[INFO] 固定番茄初始四元数 wxyz: {env_cfg.fixed_object_rot}")
    elif getattr(env_cfg, "fixed_object_y_angle_rad", None) is not None:
        print(f"[INFO] 固定番茄初始 Y 角: {env_cfg.fixed_object_y_angle_rad:.6f} rad")

    return env_cfg


def _resize_mask(mask: torch.Tensor, mask_size: tuple[int, int]) -> torch.Tensor:
    """将 [N,H,W] bool mask resize 成 [N,1,h,w] float，用于模型输入。"""

    import torch.nn.functional as F

    width, height = mask_size

    mask_f = mask.to(dtype=torch.float32).unsqueeze(1)
    mask_f = F.interpolate(
        mask_f,
        size=(height, width),
        mode="nearest",
    )
    return mask_f


def _make_vector_frame(base_env, yolo_estimate: object) -> torch.Tensor:
    """构造单帧数值输入，顺序必须和训练时完全一致。"""

    yolo_angle = yolo_estimate.angle_image_rad.unsqueeze(-1)

    vector = torch.cat(
        [
            base_env.hand_dof_pos.to(dtype=torch.float32),
            base_env.fingertip_force_binary_results.to(dtype=torch.float32),
            yolo_estimate.position_image.to(dtype=torch.float32),
            yolo_angle.to(dtype=torch.float32),
            base_env.goal_pos.to(dtype=torch.float32),
            base_env.goal_rot.to(dtype=torch.float32),
        ],
        dim=-1,
    )

    return vector


def _push_history(
    vector_hist: torch.Tensor,
    mask_hist: torch.Tensor,
    vector_frame: torch.Tensor,
    mask_frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """滑动更新历史 buffer。"""

    vector_hist[:, :-1] = vector_hist[:, 1:].clone()
    vector_hist[:, -1] = vector_frame

    mask_hist[:, :-1] = mask_hist[:, 1:].clone()
    mask_hist[:, -1] = mask_frame

    return vector_hist, mask_hist


def _reset_history_envs(
    vector_hist: torch.Tensor,
    mask_hist: torch.Tensor,
    env_ids: torch.Tensor,
    vector_frame: torch.Tensor,
    mask_frame: torch.Tensor,
) -> None:
    """环境 reset 后，用当前帧重复填满该环境历史，避免跨 episode 记忆污染。"""

    if env_ids.numel() == 0:
        return

    vector_hist[env_ids] = vector_frame[env_ids].unsqueeze(1).repeat(
        1,
        vector_hist.shape[1],
        1,
    )

    mask_hist[env_ids] = mask_frame[env_ids].unsqueeze(1).repeat(
        1,
        mask_hist.shape[1],
        1,
        1,
        1,
    )


def _get_action_scale(base_env) -> float:
    """兼容 self.action_scale 或 cfg.action_scale 两种写法。"""

    action_scale = getattr(base_env, "action_scale", None)
    if action_scale is None:
        action_scale = getattr(base_env.cfg, "action_scale", 1.0)
    return max(float(action_scale), 1e-8)


def _joint_limits(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    """返回可控关节的 lower / upper limits。"""

    lower = base_env.hand_dof_lower_limits[:, base_env.actuated_dof_indices]
    upper = base_env.hand_dof_upper_limits[:, base_env.actuated_dof_indices]
    return lower, upper


def _make_warmup_action(base_env, action_mode: str) -> torch.Tensor:
    """生成 warmup 阶段的 action。

    delta 模式：
        用 0 action，让 target_pos 保持不变。

    absolute 模式：
        不能用全 0，因为全 0 会被解释成绝对关节角目标。
        因此用当前 target_pos 作为 warmup action，使手基本保持当前目标。
    """

    if action_mode == "absolute":
        return base_env.target_pos[:, base_env.actuated_dof_indices].clone()

    return torch.zeros(
        base_env.num_envs,
        base_env.cfg.action_space,
        dtype=torch.float32,
        device=base_env.device,
    )


def _make_env_action_from_target_pos(
    base_env,
    target_pos_pred: torch.Tensor,
    action_mode: str,
) -> torch.Tensor:
    """把监督模型输出的 target_pos_pred 转成 env.step() 需要的输入。

    absolute:
        env.step(target_pos_pred)

    delta:
        env.step((target_pos_pred - current_target) / action_scale)
    """

    lower, upper = _joint_limits(base_env)

    # 网络输出的是 target_pos，先限制到关节范围。
    target_pos_pred = torch.clamp(target_pos_pred, lower, upper)

    if action_mode == "absolute":
        return target_pos_pred

    if action_mode == "delta":
        current_target = base_env.target_pos[:, base_env.actuated_dof_indices]

        env_action = (
            target_pos_pred - current_target
        ) / _get_action_scale(base_env)

        return torch.clamp(env_action, -1.0, 1.0)

    raise RuntimeError(
        f"Unknown supervised_action_mode={action_mode}. "
        "Expected 'absolute' or 'delta'."
    )


def _patch_supervised_action_mode(base_env, action_mode: str) -> None:
    """把环境临时改成兼容 delta / absolute 两种 action 模式。

    这个 patch 只对当前 play 脚本运行有效，不会永久修改环境源码。

    delta:
        actions 是 [-1,1] 增量动作。
        target_pos += action_scale * actions

    absolute:
        actions 直接是 target_pos，单位 rad。
        target_pos = actions

    注意：
        absolute 模式下，self.actions 仍保存“等效 delta action”，
        这样原来依赖 self.actions 的 observation / reward / log 更兼容。
    """

    if action_mode not in ("absolute", "delta"):
        raise RuntimeError(
            f"Unknown supervised_action_mode={action_mode}. "
            "Expected 'absolute' or 'delta'."
        )

    def _pre_physics_step_compatible(self, actions: torch.Tensor) -> None:
        # 每次 step 时从 cfg 中再读一次，方便后续临时调试切换。
        mode = getattr(self.cfg, "action_mode", action_mode)
        if getattr(self.cfg, "use_absolute_target_action", False):
            mode = "absolute"

        if mode not in ("absolute", "delta"):
            raise RuntimeError(
                f"Unknown action_mode={mode}. Expected 'absolute' or 'delta'."
            )

        expected_dim = len(self.actuated_dof_indices)
        if actions.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Action last dim should be {expected_dim}, "
                f"but got shape {tuple(actions.shape)}."
            )

        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]

        # ----------------------------------------------------------
        # absolute target_pos 模式
        # ----------------------------------------------------------
        if mode == "absolute":
            previous_target = self.target_pos[:, self.actuated_dof_indices].clone()

            target = actions.to(
                device=self.target_pos.device,
                dtype=torch.float32,
            )
            target = torch.clamp(target, lower, upper)

            # 直接写入 target_pos。后续 _apply_action() 会继续执行：
            # target_pos -> cur_targets -> hand.set_joint_position_target(...)
            self.target_pos[:, self.actuated_dof_indices] = target

            # raw_actions / absolute_actions 用于调试记录。
            self.raw_actions = target.clone()
            self.absolute_actions = target.clone()

            # 为了兼容原来的 reward、obs 和日志，这里把 self.actions
            # 设成等效 delta action，而不是绝对关节角。
            action_scale = getattr(self, "action_scale", None)
            if action_scale is None:
                action_scale = getattr(self.cfg, "action_scale", 1.0)
            action_scale = max(float(action_scale), 1e-8)

            equivalent_delta_action = (target - previous_target) / action_scale

            self.actions = torch.clamp(
                equivalent_delta_action,
                -1.0,
                1.0,
            )

            return

        # ----------------------------------------------------------
        # delta action 模式：保持原 PPO 语义
        # ----------------------------------------------------------
        self.raw_actions = torch.clamp(
            actions.to(device=self.target_pos.device, dtype=torch.float32),
            -1.0,
            1.0,
        )
        self.actions = self.raw_actions.clone()

        action_scale = getattr(self, "action_scale", None)
        if action_scale is None:
            action_scale = getattr(self.cfg, "action_scale", 1.0)
        action_scale = float(action_scale)

        self.target_pos[:, self.actuated_dof_indices] = (
            self.target_pos[:, self.actuated_dof_indices]
            + action_scale * self.raw_actions
        )

        self.target_pos[:, self.actuated_dof_indices] = torch.clamp(
            self.target_pos[:, self.actuated_dof_indices],
            lower,
            upper,
        )

        # 也保存一下当前绝对目标，便于调试。
        self.absolute_actions = self.target_pos[:, self.actuated_dof_indices].clone()

    # 把兼容函数绑定成 base_env 的实例方法。
    base_env._pre_physics_step = types.MethodType(
        _pre_physics_step_compatible,
        base_env,
    )

    base_env.cfg.action_mode = action_mode
    base_env.cfg.use_absolute_target_action = action_mode == "absolute"

    print(f"[INFO] Patched env._pre_physics_step to supervised action_mode={action_mode}.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg = _inject_yolo_cfg(env_cfg)

    # ---------------------------------------------------------------------
    # 创建 Isaac Lab 环境
    # 如果要录视频，render_mode 必须是 "rgb_array"。
    # ---------------------------------------------------------------------
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # ---------------------------------------------------------------------
    # 视频录制：必须放在 RslRlVecEnvWrapper 之前
    # ---------------------------------------------------------------------
    if args_cli.video:
        video_folder = os.path.abspath(args_cli.video_dir)
        os.makedirs(video_folder, exist_ok=True)
        video_start_step = (
            args_cli.warmup_steps
            if args_cli.warmup_steps > 0 and not args_cli.no_reset_after_warmup
            else 0
        )

        print(f"[INFO] Recording video to: {video_folder}")
        print(f"[INFO] Video length: {args_cli.video_length} steps")
        print(f"[INFO] Video starts at env step: {video_start_step}")

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == video_start_step,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # ---------------------------------------------------------------------
    # 包装成 RSL-RL VecEnv 接口
    #
    # absolute 模式下 target_pos_pred 是 rad 关节角，不是 [-1,1] action。
    # 因此不要用 1.0 裁剪。这里用 100.0 避免老版本 wrapper 不支持 None。
    # delta 模式保持原来的 clip_actions=1.0。
    # ---------------------------------------------------------------------
    wrapper_clip_actions = 100.0 if args_cli.supervised_action_mode == "absolute" else 1.0

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=wrapper_clip_actions,
    )

    base_env = env.unwrapped
    device = torch.device(base_env.device)

    # 在第一次 env.step() 之前 patch，使当前脚本同时支持 absolute / delta。
    _patch_supervised_action_mode(
        base_env,
        args_cli.supervised_action_mode,
    )

    if getattr(base_env, "yolo_student_estimator", None) is None:
        raise RuntimeError(
            "YOLO estimator was not created. "
            "Please check env_cfg.use_yolo_student_obs and student_camera."
        )

    policy = torch.jit.load(
        str(Path(args_cli.policy_jit).expanduser()),
        map_location=device,
    )
    policy.eval()

    obs = env.get_observations()

    # 先走几个 warmup step，让相机/渲染/YOLO 有稳定输入。
    # 注意 absolute 模式不能用全 0 作为 warmup，否则全 0 会被解释成绝对关节角。
    warmup_action = _make_warmup_action(
        base_env,
        args_cli.supervised_action_mode,
    )

    for _ in range(args_cli.warmup_steps):
        obs, rewards, dones, extras = env.step(warmup_action)

    if args_cli.warmup_steps > 0 and not args_cli.no_reset_after_warmup:
        obs, _ = env.reset()
        print("[INFO] YOLO warmup 完成后已 reset，正式 play 将从固定初始位姿开始。")

    # 计算第一帧 YOLO 和数值输入
    with torch.inference_mode():
        yolo_estimate = base_env.yolo_student_estimator.estimate(
            base_env._student_camera.data
        )

        vector_frame = _make_vector_frame(
            base_env,
            yolo_estimate,
        )

        mask_frame = _resize_mask(
            yolo_estimate.mask_image,
            tuple(args_cli.yolo_mask_size),
        )

    history_len = args_cli.history_len

    # 初始化历史：用当前帧重复填满
    vector_hist = vector_frame.unsqueeze(1).repeat(
        1,
        history_len,
        1,
    ).contiguous()

    mask_hist = mask_frame.unsqueeze(1).repeat(
        1,
        history_len,
        1,
        1,
        1,
    ).contiguous()

    print("[INFO] Start playing target_pos policy.")
    print(f"[INFO] supervised_action_mode: {args_cli.supervised_action_mode}")
    print(f"[INFO] wrapper_clip_actions: {wrapper_clip_actions}")
    print(f"[INFO] vector_history shape: {tuple(vector_hist.shape)}")
    print(f"[INFO] mask_history shape: {tuple(mask_hist.shape)}")

    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # ----------------------------------------------------------
            # 1. 当前帧视觉估计
            # ----------------------------------------------------------
            yolo_estimate = base_env.yolo_student_estimator.estimate(
                base_env._student_camera.data
            )

            vector_frame = _make_vector_frame(
                base_env,
                yolo_estimate,
            )

            mask_frame = _resize_mask(
                yolo_estimate.mask_image,
                tuple(args_cli.yolo_mask_size),
            )

            # ----------------------------------------------------------
            # 2. 更新历史 buffer
            # ----------------------------------------------------------
            vector_hist, mask_hist = _push_history(
                vector_hist,
                mask_hist,
                vector_frame,
                mask_frame,
            )

            # ----------------------------------------------------------
            # 3. 网络输出 target_pos
            # ----------------------------------------------------------
            target_pos_pred = policy(
                vector_hist,
                mask_hist,
            )

            # 防止输出维度异常
            expected_dim = len(base_env.actuated_dof_indices)
            if target_pos_pred.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"Policy output dim {target_pos_pred.shape[-1]} does not match "
                    f"action dim {expected_dim}."
                )

            # ----------------------------------------------------------
            # 4. 根据 supervised_action_mode 生成 env.step() 输入
            # ----------------------------------------------------------
            env_action = _make_env_action_from_target_pos(
                base_env,
                target_pos_pred,
                args_cli.supervised_action_mode,
            )

            if args_cli.debug_action and step_count % 30 == 0:
                lower, upper = _joint_limits(base_env)
                target_pos_clamped = torch.clamp(target_pos_pred, lower, upper)
                print(
                    "[DEBUG] "
                    f"step={step_count} "
                    f"mode={args_cli.supervised_action_mode} "
                    f"target_min={target_pos_clamped.min().item():+.4f} "
                    f"target_max={target_pos_clamped.max().item():+.4f} "
                    f"env_action_min={env_action.min().item():+.4f} "
                    f"env_action_max={env_action.max().item():+.4f}"
                )

            # ----------------------------------------------------------
            # 5. 推进一步仿真
            # ----------------------------------------------------------
            obs, rewards, dones, extras = env.step(env_action)

            step_count += 1

            # ----------------------------------------------------------
            # 6. reset 的环境清空历史
            # ----------------------------------------------------------
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

            if done_ids.numel() > 0:
                # reset 后重新估计一帧，填满历史
                yolo_estimate = base_env.yolo_student_estimator.estimate(
                    base_env._student_camera.data
                )

                vector_frame = _make_vector_frame(
                    base_env,
                    yolo_estimate,
                )

                mask_frame = _resize_mask(
                    yolo_estimate.mask_image,
                    tuple(args_cli.yolo_mask_size),
                )

                _reset_history_envs(
                    vector_hist,
                    mask_hist,
                    done_ids,
                    vector_frame,
                    mask_frame,
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

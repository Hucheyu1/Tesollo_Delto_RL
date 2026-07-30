# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""VTDexManip bottle-cap rotation task adapted to the DG5F right hand."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .tesollo_delto_rl_env import TesolloDeltoRlEnv, unscale
from .tesollo_delto_vtdex_env import TesolloDeltoVTDexEnv, TesolloDeltoVTDexEnvCfg

_VTDEX_ROOT = Path(__file__).resolve().parent / "vtdex_pretrained"
_BOTTLE_ASSET_ROOT = _VTDEX_ROOT / "assets" / "bottle_cap"
_BOTTLE_CODES = (
    "core-bottle-2722bec1947151b86e22e2d2f64c8cef",
    "core-bottle-91235f7d65aec958ca972daa503b3095",
    "core-bottle-add17b35bc4665e6f33a685c2506bbe6",
    "core-bottle-b8767b71c2216dcad317c475f024f3b8",
    "core-bottle-cf0a733f9a63f4f5664b3b9b23ddfcbc",
    "core-bottle-cf7a79435eb5b1bdb0be98650cd7fb6f",
    "core-bottle-ed55f39e04668bf9837048966ef3fcb9",
    "core-bottle-fda8d8820e4d166bd7134844380eaeb0",
    "core-bottle-77a2242bf4ea8f9fc02fe00a7187a6a9",
    "core-bottle-d8021dc9fc9109b130612f5c0ef21eb8",
)
# Values copied from VTDexManip's obj_init_height.pickle and
# hand_init_height.pickle in exactly the order above.
_BOTTLE_ROOT_RISES = (
    0.0841214,
    0.0833685,
    0.0830363,
    0.0898106,
    0.0848081,
    0.0891602,
    0.0880805,
    0.0890446,
    0.0884919,
    0.0879818,
)
_SOURCE_HAND_HEIGHTS = (
    0.2182428,
    0.2167370,
    0.2160726,
    0.2296192,
    0.2196169,
    0.2283204,
    0.2261592,
    0.2280892,
    0.2269838,
    0.2259636,
)
_DOWN_CFG = TesolloDeltoVTDexEnvCfg()
_BOTTLE_FINGER_ACTUATOR_CFG = _DOWN_CFG.robot_cfg.actuators["fingers"].replace(
    # The shared DG5F limit (0.1 N m) is suitable for free-space motion but is
    # far below the 0.7245--2.3722 N m Shadow-Hand limits used by VTDexManip.
    # Keep the source-like 0.8 N m override local to this contact-heavy task.
    effort_limit_sim=2.0
)


@configclass
class TesolloDeltoVTDexBottleCapEnvCfg(TesolloDeltoVTDexEnvCfg):
    """DG5F reproduction of VTDexManip's ``bottle_cap-vt_all_cls`` task."""

    # The source task runs for 500 policy steps at 60 Hz.
    episode_length_s = 500.0 / 60.0
    # Reuse Reorient Down's verified palm-down root orientation and X/Y
    # alignment. Z is initialized per bottle in _reset_idx so the clearance
    # follows the source task's ten object-specific hand heights.
    robot_cfg = _DOWN_CFG.robot_cfg.replace(
        init_state=_DOWN_CFG.robot_cfg.init_state.replace(
            # Bring the palm 2 cm closer than the previous hover-dominated
            # setup while retaining clearance for all ten cap geometries.
            pos=(-0.13, 0.022, 0.55),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),
            joint_pos={
                "rj_dg_(1|3)_1": 0.0,
                "rj_dg_1_2": -1.57,
                "rj_dg_1_3": -0.8,
                "rj_dg_1_4": 1.4,
                "rj_dg_2_1": -0.2,
                "rj_dg_4_1": 0.2,
                "rj_dg_(2|3|4)_2": 1.3,
                "rj_dg_(2|3|4)_3": 0.2,
                "rj_dg_(2|3|4)_4": 0.2,
                "rj_dg_5_1": 0.3,
                "rj_dg_5_2": 1.57,
                "rj_dg_5_3": 1.57,
                "rj_dg_5_4": 0.0,
            },
        ),
        actuators={"fingers": _BOTTLE_FINGER_ACTUATOR_CFG},
    )

    # Cap-specific multi-finger pregrasp. Reorient Down intentionally starts
    # open above a table object; a bottle cap instead needs a light enclosure
    # so exploration can generate tangential torque from the first episode.
    # Values follow the layer-major ``actuated_joint_names`` order.
    hand_position = [
        # 第一关节：rj_dg_1_1 ~ rj_dg_5_1
        0.0,
        -0.2,
        0.0,
        0.2,
        0.3,
        # 第二关节：rj_dg_1_2 ~ rj_dg_5_2
        -1.57,
        1.3,
        1.3,
        1.3,
        1.57,
        # 第三关节：rj_dg_1_3 ~ rj_dg_5_3
        -0.8,
        0.2,
        0.2,
        0.2,
        1.57,
        # 第四关节：rj_dg_1_4 ~ rj_dg_5_4
        1.4,
        0.2,
        0.2,
        0.2,
        0.0,
    ]
    # Bottle-cap-specific limits. Keep the verified Reorient Down ranges as
    # the baseline, then add a small amount of room for closing, releasing and
    # regrasping the cap. The DG5F asset's physical limits remain the final
    # hard constraint.
    hand_lower_limits = list(_DOWN_CFG.hand_lower_limits)
    # Little-finger second joint: retain some outward motion around its 90 deg
    # pregrasp while expanding the inherited -15 deg lower range slightly.
    hand_lower_limits[9] = -25
    # Permit slight extension at the four non-thumb middle joints and at all
    # five distal joints.
    hand_lower_limits[11:15] = [-10, -10, -10, -10]
    hand_lower_limits[15:20] = [-10, -10, -10, -10, -10]

    hand_upper_limits = list(_DOWN_CFG.hand_upper_limits)
    # Give ring abduction and little-finger base motion another 5--10 degrees.
    hand_upper_limits[3] = 20
    hand_upper_limits[4] = 55
    # Increase index/middle/ring proximal flexion from 90 to 100 degrees.
    hand_upper_limits[6:9] = [100, 100, 100]
    # The current cap pregrasp requests rj_dg_5_2=1.57 rad (90 deg);
    # this is its physical upper limit and must not be expanded further.
    hand_upper_limits[9] = 90

    # Each copied URDF contains a fixed bottle_body and one passive revolute
    # bottle_cap_joint. Assets are spawned manually per environment, then
    # wrapped by this common articulation view.
    bottle_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Bottle",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.02, 0.395),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"bottle_cap_joint": 0.0},
            joint_vel={"bottle_cap_joint": 0.0},
        ),
        actuators={},
    )
    bottle_cap_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Bottle/bottle_cap",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
        # One cap body against the same 20 DG5F links used as VTDex tactile
        # tokens. This reproduces match_contacts2() instead of accepting a
        # hand contact with the table or bottle body.
        filter_prim_paths_expr=[
            f"/World/envs/env_.*/Robot/{body_name}" for body_name in _DOWN_CFG.vtdex_tactile_body_names
        ],
    )

    # Keep Reorient Down's camera eye and raise only its look-at point from the
    # tabletop object to the bottle-cap band.
    vtdex_camera_eye_local = _DOWN_CFG.vtdex_camera_eye_local
    vtdex_camera_target_local = (0.0, 0.02, 0.46)

    # Default joint-mode actor layout is qpos(20) + qvel(20) + VTDex CLS(384);
    # ``--model vision`` changes only the feature/observation dimension to 512.
    # The critic sees qpos/qvel, cap angle/velocity, 20 raw tactile bits and the
    # previous action in either mode.
    observation_space = 424
    state_space = 82

    # Preserve the source task's reward weights, contact gate and success
    # thresholds while testing the DG5F-specific physical corrections.
    min_cap_contacts = 1
    cap_contact_threshold = 0.01
    cap_position_reward_scale = 0.5
    cap_velocity_reward_scale = 1.0
    fingertip_height_reward_scale = 0.5
    bottle_action_penalty_scale = 0.0
    cap_success_angle = 6.0
    cap_reset_angle = 6.15
    cap_success_bonus = 5.0
    cap_joint_upper_limit = 6.28
    # Ignore sub-0.1 mrad solver jitter when deriving velocity from the actual
    # angle displacement between two 60 Hz policy steps.
    cap_progress_deadband_rad = 1.0e-4

    # VTDexManip uses 1.0. Use a task-local 0.5 compromise: twice the previous
    # response while the remaining low-pass action still limits impact spikes.
    act_moving_average = 0.8
    # Keep bottle_cap.yaml's reset distribution around the Reorient Down
    # pregrasp: the fixed bottle pose has no noise, while the hand root and
    # joints receive the source task's small reset perturbations.
    reset_position_noise = 0.0
    hand_root_position_noise = 0.01
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.1
    debug_visualization = False


class TesolloDeltoVTDexBottleCapEnv(TesolloDeltoVTDexEnv):
    """Rotate a passive bottle-cap joint through one positive revolution."""

    cfg: TesolloDeltoVTDexBottleCapEnvCfg

    def __init__(self, cfg: TesolloDeltoVTDexBottleCapEnvCfg, render_mode: str | None = None, **kwargs):
        if float(cfg.cap_progress_deadband_rad) < 0.0:
            raise ValueError("cap_progress_deadband_rad must be non-negative")
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_bottle_indices()
        self._compute_intermediate_values()
        # PhysX joint velocity can contain large contact/limit impulses even
        # when the cap angle does not move. Reward the finite-difference
        # velocity instead and synchronize this buffer on every reset.
        self._previous_cap_joint_pos = self.cap_joint_pos.clone()
        if self._bottle_cap_contact_sensor.num_bodies != 1:
            raise RuntimeError(
                "Bottle-cap ContactSensor must resolve exactly one body; "
                f"got {self._bottle_cap_contact_sensor.num_bodies}"
            )
        force_matrix_w = self._bottle_cap_contact_sensor.data.force_matrix_w
        if force_matrix_w is None or force_matrix_w.shape[2] != len(self.cfg.vtdex_tactile_body_names):
            raise RuntimeError(
                "Bottle-cap contact filter must resolve the 20 configured DG5F tactile links; "
                f"got shape={None if force_matrix_w is None else tuple(force_matrix_w.shape)}"
            )
        print(
            "[INFO]: Bottle-cap contact filter: "
            f"cap body={self.object.body_names[self._bottle_cap_body_id]}, "
            f"DG5F links={force_matrix_w.shape[2]}"
        )
        print(
            "[INFO]: Bottle-cap DG5F tuning: "
            f"effort_limit={self.cfg.robot_cfg.actuators['fingers'].effort_limit_sim} N m, "
            f"action_moving_average={self.cfg.act_moving_average}, "
            f"hand_root_z={self.cfg.robot_cfg.init_state.pos[2]} m"
        )

    def _setup_scene(self):
        self._spawn_bottle_articulations()
        self.hand = Articulation(self.cfg.robot_cfg)
        self.bottle = Articulation(self.cfg.bottle_cfg)
        # The shared base class treats the manipulated asset through
        # self.object; Articulation exposes the same root-state interface plus
        # the cap joint state required here.
        self.object = self.bottle
        self.table = RigidObject(self.cfg.table_cfg)
        self._vtdex_camera = Camera(self.cfg.vtdex_camera)
        self._vtdex_contact_sensor = ContactSensor(self.cfg.vtdex_contact_sensor)
        self._bottle_cap_contact_sensor = ContactSensor(self.cfg.bottle_cap_contact_sensor)

        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["bottle"] = self.bottle
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["vtdex_camera"] = self._vtdex_camera
        self.scene.sensors["vtdex_contact"] = self._vtdex_contact_sensor
        self.scene.sensors["bottle_cap_contact"] = self._bottle_cap_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_bottle_articulations(self):
        """Spawn the exact ten fixed-base bottle articulations from VTDexManip."""

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=2.0,
            max_linear_velocity=5.0,
            max_angular_velocity=720.0,
        )
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            fix_root_link=True,
        )
        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            object_index = env_index % len(_BOTTLE_CODES)
            bottle_code = _BOTTLE_CODES[object_index]
            urdf_path = _BOTTLE_ASSET_ROOT / bottle_code / f"{bottle_code}.urdf"
            if not urdf_path.is_file():
                raise FileNotFoundError(f"Missing copied VTDex bottle asset: {urdf_path}")

            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=str(urdf_path),
                # Keep the generated USD layer name a valid USD identifier as
                # well; upstream object codes contain hyphens.
                usd_file_name=f"bottle_{object_index}.usd",
                fix_base=True,
                # The original Isaac Gym loader used density=500 together
                # with override_com/override_inertia. The copied upstream
                # URDFs contained malformed partial inertial tags, so the
                # Isaac-Sim-compatible copies omit them and let the importer
                # compute valid mass properties from this same density.
                link_density=500.0,
                merge_fixed_joints=False,
                joint_drive=None,
                collider_type="convex_hull",
                self_collision=False,
                make_instanceable=False,
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                articulation_props=articulation_props,
                semantic_tags=[("class", "bottle")],
            )
            spawn_cfg.func(
                f"{env_path}/Bottle",
                spawn_cfg,
                translation=(
                    0.0,
                    0.02,
                    float(self.cfg.table_top_z) + _BOTTLE_ROOT_RISES[object_index],
                ),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

    def _resolve_bottle_indices(self):
        if not hasattr(self, "_bottle_cap_joint_id"):
            try:
                self._bottle_cap_joint_id = self.object.joint_names.index("bottle_cap_joint")
                self._bottle_cap_body_id = self.object.body_names.index("bottle_cap")
            except ValueError as exc:
                raise RuntimeError(
                    "Copied bottle URDF must expose body 'bottle_cap' and joint 'bottle_cap_joint'; "
                    f"bodies={self.object.body_names}, joints={self.object.joint_names}"
                ) from exc

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        self._resolve_bottle_indices()
        self.cap_joint_pos = self.object.data.joint_pos[:, self._bottle_cap_joint_id]
        self.cap_joint_vel = self.object.data.joint_vel[:, self._bottle_cap_joint_id]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        rgb = self._vtdex_camera.data.output["rgb"]
        raw_tactile = self.fingertip_force_binary_results.to(dtype=torch.float32)
        policy_tactile = (
            raw_tactile
            if self.cfg.vtdex_model_mode == "joint" and not self.cfg.vtdex_mask_tactile_input
            else torch.zeros_like(raw_tactile)
        )
        self.vtdex_policy_tactile_input = policy_tactile
        self.vtdex_embeddings = self.vtdex_encoder(rgb, policy_tactile)
        self.extras.setdefault("log", {})["vtdex_policy_tactile_active_ratio"] = policy_tactile.mean()

        policy_obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.vtdex_embeddings,
            ),
            dim=-1,
        )
        critic_obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.cap_joint_pos.unsqueeze(-1) / float(self.cfg.cap_joint_upper_limit),
                self.cfg.vel_obs_scale * self.cap_joint_vel.unsqueeze(-1),
                raw_tactile,
                self.actions,
            ),
            dim=-1,
        )
        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Bottle-cap policy observation has {policy_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.observation_space}"
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"Bottle-cap critic observation has {critic_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _cap_contact_counts(self) -> torch.Tensor:
        force_matrix_w = self._bottle_cap_contact_sensor.data.force_matrix_w
        if force_matrix_w is None:
            raise RuntimeError("Bottle-cap filtered contact forces are unavailable")
        force_norms = torch.linalg.vector_norm(force_matrix_w[:, 0], dim=-1)
        return (force_norms > float(self.cfg.cap_contact_threshold)).sum(dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        cap_pos_reward = torch.minimum(
            self.cap_joint_pos,
            torch.tensor(7.0, dtype=torch.float32, device=self.device),
        )
        cap_angle_delta = self.cap_joint_pos - self._previous_cap_joint_pos
        cap_angle_delta = torch.where(
            torch.abs(cap_angle_delta) < float(self.cfg.cap_progress_deadband_rad),
            torch.zeros_like(cap_angle_delta),
            cap_angle_delta,
        )
        cap_progress_velocity = torch.clamp(
            cap_angle_delta / float(self.step_dt),
            -10.0,
            10.0,
        )
        cap_velocity_reward = cap_progress_velocity
        cap_contact_counts = self._cap_contact_counts()
        valid_cap_contact = cap_contact_counts >= int(self.cfg.min_cap_contacts)
        # Preserve the source asymmetry: positive opening velocity needs valid
        # cap contact, while reverse rotation remains penalized.
        cap_velocity_reward = torch.where(
            cap_velocity_reward > 0.0,
            torch.where(valid_cap_contact, cap_velocity_reward, torch.zeros_like(cap_velocity_reward)),
            cap_velocity_reward,
        )

        fingertip_pos_w = self.hand.data.body_pos_w.index_select(1, self._reward_fingertip_body_ids)
        fingertip_z = fingertip_pos_w[:, :, 2] - self.scene.env_origins[:, 2:3]
        source_hand_heights = torch.tensor(_SOURCE_HAND_HEIGHTS, dtype=torch.float32, device=self.device)
        target_height = (
            float(self.cfg.table_top_z)
            + source_hand_heights[torch.arange(self.num_envs, device=self.device) % len(_SOURCE_HAND_HEIGHTS)]
            - 0.07
        )
        fingertip_height_error = torch.abs(fingertip_z - target_height.unsqueeze(-1)).sum(dim=-1)
        fingertip_height_reward = torch.exp(-10.0 * fingertip_height_error)
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)

        reward = (
            cap_pos_reward * float(self.cfg.cap_position_reward_scale)
            + cap_velocity_reward * float(self.cfg.cap_velocity_reward_scale)
            + fingertip_height_reward * float(self.cfg.fingertip_height_reward_scale)
            + action_penalty * float(self.cfg.bottle_action_penalty_scale)
        )
        success = self.cap_joint_pos > float(self.cfg.cap_success_angle)
        self.successes[:] = torch.where(success, torch.ones_like(self.successes), self.successes)
        reward = torch.where(
            self.successes > 0.0,
            reward + float(self.cfg.cap_success_bonus),
            reward,
        )
        self.extras.setdefault("log", {}).update(
            {
                "bottle_cap_angle_rad": self.cap_joint_pos.mean(),
                "bottle_cap_angle_deg": torch.rad2deg(self.cap_joint_pos).mean(),
                "bottle_cap_angle_max_rad": self.cap_joint_pos.max(),
                # Keep the established key for the effective velocity that is
                # now used by the reward, and expose raw PhysX velocity
                # separately for detecting contact/limit impulses.
                "bottle_cap_velocity_rad_s": cap_progress_velocity.mean(),
                "bottle_cap_reward_velocity_rad_s": cap_velocity_reward.mean(),
                "bottle_cap_raw_joint_velocity_rad_s": self.cap_joint_vel.mean(),
                "bottle_cap_angle_delta_abs_rad": torch.abs(cap_angle_delta).mean(),
                "bottle_cap_contact_count": cap_contact_counts.float().mean(),
                "bottle_cap_position_reward_term": (
                    cap_pos_reward * float(self.cfg.cap_position_reward_scale)
                ).mean(),
                "bottle_cap_velocity_reward_term": (
                    cap_velocity_reward * float(self.cfg.cap_velocity_reward_scale)
                ).mean(),
                "bottle_cap_height_reward_term": (
                    fingertip_height_reward * float(self.cfg.fingertip_height_reward_scale)
                ).mean(),
                "bottle_cap_positive_rotation_ratio": (
                    (cap_progress_velocity > 0.0) & valid_cap_contact
                ).float().mean(),
                "bottle_cap_success_rate": success.float().mean(),
            }
        )
        self._previous_cap_joint_pos.copy_(self.cap_joint_pos)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terminated = self.cap_joint_pos > float(self.cfg.cap_reset_angle)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_target_pose(self, env_ids: Sequence[int]):
        # Bottle-cap success is a scalar joint target; hide the inherited
        # orientation marker and keep its buffers in a benign state.
        self.goal_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.goal_rot[env_ids] = self.object.data.root_quat_w[env_ids]
        self.reset_goal_buf[env_ids] = False
        self._hide_goal_marker()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore

        # Skip Reorient Down's free-object/goal-object reset while retaining
        # the shared DG5F joint/action reset.
        TesolloDeltoRlEnv._reset_idx(self, env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        object_indices = env_ids_tensor % len(_BOTTLE_CODES)

        hand_root_state = self.hand.data.default_root_state[env_ids_tensor].clone()
        source_hand_heights = torch.tensor(_SOURCE_HAND_HEIGHTS, dtype=torch.float32, device=self.device)
        # Treat robot_cfg.init_state.pos[2] as the hand-height control exposed
        # to the user. Before the configured reset noise, environment 0 uses
        # that value exactly; the other nine environments retain only
        # VTDexManip's bottle-specific relative height differences.
        hand_root_state[:, 2] = (
            float(self.cfg.robot_cfg.init_state.pos[2])
            + source_hand_heights[object_indices]
            - float(_SOURCE_HAND_HEIGHTS[0])
        )
        hand_root_state[:, :3] += sample_uniform(
            -float(self.cfg.hand_root_position_noise),
            float(self.cfg.hand_root_position_noise),
            (len(env_ids_tensor), 3),
            device=self.device,
        )
        hand_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        hand_root_state[:, 7:] = 0.0
        self.hand.write_root_pose_to_sim(hand_root_state[:, :7], env_ids_tensor)
        self.hand.write_root_velocity_to_sim(hand_root_state[:, 7:], env_ids_tensor)
        self.hand_base_pos[env_ids_tensor] = hand_root_state[:, :3] - self.scene.env_origins[env_ids_tensor]
        self.hand_base_rot[env_ids_tensor] = hand_root_state[:, 3:7]

        bottle_root_state = self.object.data.default_root_state[env_ids_tensor].clone()
        xy_noise = sample_uniform(
            -float(self.cfg.reset_position_noise),
            float(self.cfg.reset_position_noise),
            (len(env_ids_tensor), 2),
            device=self.device,
        )
        bottle_root_state[:, 0] = xy_noise[:, 0]
        bottle_root_state[:, 1] = 0.02 + xy_noise[:, 1]
        bottle_rises = torch.tensor(_BOTTLE_ROOT_RISES, dtype=torch.float32, device=self.device)
        bottle_root_state[:, 2] = float(self.cfg.table_top_z) + bottle_rises[object_indices]
        bottle_root_state[:, 3:7] = 0.0
        bottle_root_state[:, 3] = 1.0
        bottle_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        bottle_root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(bottle_root_state[:, :7], env_ids_tensor)
        self.object.write_root_velocity_to_sim(bottle_root_state[:, 7:], env_ids_tensor)

        self._resolve_bottle_indices()
        joint_pos = self.object.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = self.object.data.default_joint_vel[env_ids_tensor].clone()
        joint_pos[:, self._bottle_cap_joint_id] = 0.0
        joint_vel[:, self._bottle_cap_joint_id] = 0.0
        self.object.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)

        self.in_hand_pos[env_ids_tensor] = bottle_root_state[:, :3] - self.scene.env_origins[env_ids_tensor]
        self.goal_pos[env_ids_tensor] = self.in_hand_pos[env_ids_tensor]
        self.goal_rot[env_ids_tensor] = bottle_root_state[:, 3:7]
        self.reset_goal_buf[env_ids_tensor] = False
        self._hide_goal_marker()
        self._compute_intermediate_values()
        if hasattr(self, "_previous_cap_joint_pos"):
            # Prevent the successful/timeout reset from appearing as one large
            # negative angle displacement on the first step of a new episode.
            self._previous_cap_joint_pos[env_ids_tensor] = self.cap_joint_pos[env_ids_tensor]

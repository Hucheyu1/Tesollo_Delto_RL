# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Pepper-bottle cap-unscrewing task for the DG5F / Tesollo hand.

This task combines two existing environments:

* :mod:`tesollo_delto_pepper_bottle_rotate_env` supplies the verified pepper
  bottle USD, tabletop geometry, DG5F pose, camera, tactile tokens and frozen
  VTDex encoder. The bottle must be loaded as an articulation-capable object.
* :mod:`tesollo_delto_vtdex_bottle_cap_env` supplies the scalar passive-joint
  objective, filtered cap-contact gate, asymmetric actor/critic observations,
  reward and reset structure used for bottle-cap rotation.

The pepper-bottle root is fixed.  Therefore the policy can only solve the task
by rotating the bottle's internal cap revolute joint instead of spinning or
moving the entire bottle.

Before training, check the startup log.  The code automatically resolves a
single cap joint/body using configured names, ``cap``/``lid`` keywords, or the
single-joint/two-body topology.  Set ``pepper_cap_joint_name`` and
``pepper_cap_body_name`` explicitly if the USD uses ambiguous names.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .tesollo_delto_pepper_bottle_rotate_env import (
    TesolloDeltoPepperBottleRotateEnv,
    TesolloDeltoPepperBottleRotateEnvCfg,
)
from .tesollo_delto_rl_env import TesolloDeltoRlEnv, unscale


_PEPPER_ROTATE_CFG = TesolloDeltoPepperBottleRotateEnvCfg()
_PEPPER_FINGER_ACTUATOR_CFG = _PEPPER_ROTATE_CFG.robot_cfg.actuators["fingers"].replace(
    # Cap twisting is contact-heavy.  The 0.1 N m setting used by the tabletop
    # rotation task is often too weak to create sustained tangential torque.
    effort_limit_sim=2.0,
)


@configclass
class TesolloDeltoPepperBottleCapEnvCfg(TesolloDeltoPepperBottleRotateEnvCfg):
    """Unscrew the passive cap joint of the single pepper-bottle articulation."""

    # VTDex bottle-cap reference: 500 policy steps at 60 Hz.
    episode_length_s = 500.0 / 60.0

    # Keep the geometry already verified for the pepper-bottle tabletop task,
    # but provide more torque for cap contact.
    robot_cfg = _PEPPER_ROTATE_CFG.robot_cfg.replace(
        actuators={"fingers": _PEPPER_FINGER_ACTUATOR_CFG},
    )

    # The bottle body must not be a second solution path.  Fix its articulation
    # root, disable gravity, and activate object-side contact reporting.  Its
    # internal revolute joint remains passive through the inherited zero-drive
    # actuator configuration.
    object_cfg = _PEPPER_ROTATE_CFG.object_cfg.replace(
        spawn=_PEPPER_ROTATE_CFG.object_cfg.spawn.replace(
            activate_contact_sensors=True,
            rigid_props=_PEPPER_ROTATE_CFG.object_cfg.spawn.rigid_props.replace(
                disable_gravity=True,
            ),
            articulation_props=_PEPPER_ROTATE_CFG.object_cfg.spawn.articulation_props.replace(
                fix_root_link=True,
            ),
        ),
    )

    # Do not create a ContactSensor on /World/envs/env_.*/Object/.*.
    # Some pepper USD exports do not propagate contact-reporter API to every
    # articulated child body even when activate_contact_sensors=True, which makes
    # Isaac Lab fail during sensor initialization. This task instead reuses the
    # inherited robot-side VTDex tactile stream and uses cap-body proximity for
    # cap-specific shaping.

    # Leave these empty for automatic resolution.  Explicit names are strongly
    # recommended after the first successful startup, for example:
    # pepper_cap_joint_name = "bottle_cap_joint"
    # pepper_cap_body_name = "bottle_cap"
    pepper_cap_joint_name = ""
    pepper_cap_body_name = ""
    pepper_cap_name_keywords = ("cap", "lid")

    # Aim the image encoder at the cap band.  Tune this one value if the pepper
    # USD's local origin/scale differs from the current tabletop asset.
    table_top_z = 0.31
    cap_camera_z_above_table = 0.20
    vtdex_camera_target_local = (0.0, 0.02, table_top_z + cap_camera_z_above_table)

    # Actor: normalized qpos(20) + qvel(20) + frozen VTDex representation.
    # The inherited joint/vision selector changes 384-D/424-D to 512-D/552-D.
    # Critic: qpos(20) + qvel(20) + cap progress/velocity(2)
    #         + raw tactile(20) + action(20) = 82.
    observation_space = 424
    state_space = 82
    asymmetric_obs = True

    # Positive means that increasing USD joint position opens the cap.  Change
    # to -1.0 if the cap visibly turns in the desired direction while logged
    # ``pepper_cap_progress_rad`` becomes negative.
    cap_open_direction = 1.0
    cap_initial_angle = 0.0
    cap_joint_normalization = 6.28
    cap_success_angle = 6.0
    cap_reset_angle = 6.15
    cap_progress_reward_clip = 7.0

    # Contact-gated cap reward, following the VTDex bottle-cap task.
    min_cap_contacts = 1
    cap_contact_threshold = 0.01
    cap_position_reward_scale = 0.5
    cap_velocity_reward_scale = 1.0
    fingertip_proximity_reward_scale = 0.5
    fingertip_proximity_gain = 12.0
    fingertip_proximity_count = 3
    cap_contact_reward_scale = 0.1
    cap_action_penalty_scale = -0.0002
    cap_success_bonus = 5.0

    # Preserve the pepper-specific pregrasp.  The base reset applies joint
    # perturbations around cfg.hand_position; root noise is handled below.
    act_moving_average = 0.8
    reset_position_noise = 0.0
    hand_root_position_noise = 0.005
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.1
    debug_visualization = False


class TesolloDeltoPepperBottleCapEnv(TesolloDeltoPepperBottleRotateEnv):
    """Rotate the pepper bottle's passive cap joint through one revolution."""

    cfg: TesolloDeltoPepperBottleCapEnvCfg

    def __init__(
        self,
        cfg: TesolloDeltoPepperBottleCapEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        if float(cfg.cap_open_direction) == 0.0:
            raise ValueError("cap_open_direction must be non-zero")
        if float(cfg.cap_success_angle) <= 0.0:
            raise ValueError("cap_success_angle must be positive")
        if float(cfg.cap_reset_angle) <= float(cfg.cap_success_angle):
            raise ValueError("cap_reset_angle must be greater than cap_success_angle")
        if int(cfg.fingertip_proximity_count) < 1:
            raise ValueError("fingertip_proximity_count must be at least one")

        # This initializes the pepper asset, DG5F, VTDex camera/encoder and the
        # inherited 20-link robot-side tactile stream. This file deliberately
        # does not install an Object/.* ContactSensor, because that path caused
        # ContactSensor initialization failures for the articulated pepper USD.
        super().__init__(cfg, render_mode, **kwargs)

        self._ensure_reward_fingertip_body_ids()
        self._resolve_pepper_cap_indices()

        if not hasattr(self.object, "joint_names") or len(self.object.joint_names) == 0:
            raise RuntimeError(
                "Pepper-cap task requires the pepper bottle to be loaded as an Articulation "
                "with at least one internal joint. Check the parent rotate environment: "
                "Object must not be a RigidObject."
            )

        # The reset path normally creates this buffer. Initialize it here as a
        # fallback for Isaac Lab versions that do not reset during construction.
        if not hasattr(self, "cap_joint_start_pos"):
            self.cap_joint_start_pos = self.object.data.joint_pos[:, self._pepper_cap_joint_id].clone()

        print(
            "[INFO]: PepperBottleCap task mapping: "
            f"joint='{self._pepper_cap_joint_name}', "
            f"body='{self._pepper_cap_body_name}', "
            "contact_source=robot_vtdex_tactile, "
            f"open_direction={float(self.cfg.cap_open_direction):+.0f}"
        )
        print(
            "[INFO]: PepperBottleCap DG5F tuning: "
            f"effort_limit={self.cfg.robot_cfg.actuators['fingers'].effort_limit_sim} N m, "
            f"action_moving_average={self.cfg.act_moving_average}, "
            f"hand_root_z={self.cfg.robot_cfg.init_state.pos[2]} m"
        )

    def _setup_scene(self):
        """Create the pepper scene without any Object/.* ContactSensor."""

        super()._setup_scene()
        print(
            "[INFO]: PepperBottleCap _setup_scene: using inherited robot-side tactile sensor; "
            "NO /World/envs/env_.*/Object/.* ContactSensor is created.",
            flush=True,
        )

    def _ensure_reward_fingertip_body_ids(self) -> None:
        """Create fingertip-body indices if the parent rotate env did not."""

        if hasattr(self, "_reward_fingertip_body_ids"):
            return

        configured_names = list(getattr(self.cfg, "vtdex_tactile_body_names", ()))
        fingertip_names = configured_names[:5]
        if not fingertip_names:
            fingertip_names = [name for name in self.hand.body_names if name.endswith("_4")][:5]

        missing = [name for name in fingertip_names if name not in self.hand.body_names]
        if missing:
            raise RuntimeError(
                "Cannot build fingertip proximity mapping; missing robot bodies="
                f"{missing}, available bodies={self.hand.body_names}"
            )
        if len(fingertip_names) == 0:
            raise RuntimeError("Cannot build fingertip proximity mapping: no fingertip bodies were resolved")

        self._reward_fingertip_body_ids = torch.tensor(
            [self.hand.body_names.index(name) for name in fingertip_names],
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _resolve_named_item(
        names: Sequence[str],
        configured_name: str,
        keywords: Sequence[str],
        item_kind: str,
        allow_single_fallback: bool,
    ) -> str:
        """Resolve an exact or unambiguous keyword-matched body/joint name."""

        available = list(names)
        if configured_name:
            if configured_name not in available:
                raise RuntimeError(
                    f"Configured pepper cap {item_kind} '{configured_name}' was not found; "
                    f"available {item_kind}s={available}"
                )
            return configured_name

        lowered_keywords = tuple(keyword.lower() for keyword in keywords)
        candidates = [
            name
            for name in available
            if any(keyword in name.lower() for keyword in lowered_keywords)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if allow_single_fallback and len(available) == 1:
            return available[0]

        raise RuntimeError(
            f"Cannot uniquely resolve pepper cap {item_kind}. "
            f"keyword candidates={candidates}, available {item_kind}s={available}. "
            f"Set cfg.pepper_cap_{item_kind}_name explicitly."
        )

    def _resolve_pepper_cap_indices(self) -> None:
        """Resolve the passive cap joint and its rigid body in the pepper USD."""

        if hasattr(self, "_pepper_cap_joint_id"):
            return

        keywords = tuple(self.cfg.pepper_cap_name_keywords)
        joint_name = self._resolve_named_item(
            self.object.joint_names,
            str(self.cfg.pepper_cap_joint_name),
            keywords,
            "joint",
            allow_single_fallback=True,
        )

        try:
            body_name = self._resolve_named_item(
                self.object.body_names,
                str(self.cfg.pepper_cap_body_name),
                keywords,
                "body",
                allow_single_fallback=False,
            )
        except RuntimeError:
            # A common two-link articulation is [fixed bottle body, cap body].
            # Use the child link only when this topology is unambiguous.
            if not self.cfg.pepper_cap_body_name and len(self.object.body_names) == 2:
                body_name = self.object.body_names[1]
            else:
                raise

        self._pepper_cap_joint_name = joint_name
        self._pepper_cap_body_name = body_name
        self._pepper_cap_joint_id = self.object.joint_names.index(joint_name)
        self._pepper_cap_body_id = self.object.body_names.index(body_name)

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        self._resolve_pepper_cap_indices()

        self.cap_joint_pos = self.object.data.joint_pos[:, self._pepper_cap_joint_id]
        self.cap_joint_vel = self.object.data.joint_vel[:, self._pepper_cap_joint_id]

        if not hasattr(self, "cap_joint_start_pos") or self.cap_joint_start_pos.shape != self.cap_joint_pos.shape:
            self.cap_joint_start_pos = torch.full_like(self.cap_joint_pos, float(self.cfg.cap_initial_angle))

        direction = 1.0 if float(self.cfg.cap_open_direction) > 0.0 else -1.0
        self.cap_joint_progress = direction * (self.cap_joint_pos - self.cap_joint_start_pos)
        self.cap_joint_opening_velocity = direction * self.cap_joint_vel

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Use VTDex actor features and privileged cap state for the critic."""

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

        normalized_hand_pos = unscale(
            self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        policy_obs = torch.cat(
            (
                normalized_hand_pos,
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.vtdex_embeddings,
            ),
            dim=-1,
        )
        critic_obs = torch.cat(
            (
                normalized_hand_pos,
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.cap_joint_progress.unsqueeze(-1) / float(self.cfg.cap_joint_normalization),
                self.cfg.vel_obs_scale * self.cap_joint_opening_velocity.unsqueeze(-1),
                raw_tactile,
                self.actions,
            ),
            dim=-1,
        )

        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Pepper-cap policy observation has {policy_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.observation_space}"
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"Pepper-cap critic observation has {critic_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _cap_contact_counts(self) -> torch.Tensor:
        """Count active robot tactile links as a stable contact gate.

        The earlier object-side filtered ContactSensor used prim_path
        ``/World/envs/env_.*/Object/.*``. On this articulated pepper USD, Isaac
        Lab may fail before the simulation starts because not every Object child
        has contact-reporter API. The inherited VTDex tactile stream already
        monitors the configured DG5F links, so we use it as the contact gate and
        keep cap specificity through the proximity reward to the resolved cap body.
        """

        tactile = self.fingertip_force_binary_results
        if tactile.ndim != 2:
            raise RuntimeError(
                "Expected fingertip_force_binary_results with shape [N, tactile_dim]; "
                f"got {tuple(tactile.shape)}"
            )
        return tactile.to(dtype=torch.float32).sum(dim=-1)

    def _fingertip_proximity_reward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reward the nearest fingertips for staying around the actual cap body."""

        fingertip_pos_w = self.hand.data.body_pos_w.index_select(1, self._reward_fingertip_body_ids)
        cap_pos_w = self.object.data.body_pos_w[:, self._pepper_cap_body_id]
        fingertip_distances = torch.linalg.vector_norm(
            fingertip_pos_w - cap_pos_w.unsqueeze(1),
            dim=-1,
        )
        count = min(int(self.cfg.fingertip_proximity_count), fingertip_distances.shape[1])
        nearest_mean_distance = torch.topk(
            fingertip_distances,
            k=count,
            dim=-1,
            largest=False,
        ).values.mean(dim=-1)
        reward = torch.exp(-float(self.cfg.fingertip_proximity_gain) * nearest_mean_distance)
        return reward, nearest_mean_distance

    def _get_rewards(self) -> torch.Tensor:
        """Reward contact-supported positive rotation of the pepper cap joint."""

        cap_contact_counts = self._cap_contact_counts()
        valid_cap_contact = cap_contact_counts >= int(self.cfg.min_cap_contacts)

        cap_position_reward = torch.clamp(
            self.cap_joint_progress,
            min=-float(self.cfg.cap_progress_reward_clip),
            max=float(self.cfg.cap_progress_reward_clip),
        )
        cap_velocity_reward = torch.clamp(self.cap_joint_opening_velocity, -10.0, 10.0)
        # Positive opening motion is rewarded only while a configured DG5F link
        # touches the cap.  Reverse motion remains negative even without contact.
        cap_velocity_reward = torch.where(
            cap_velocity_reward > 0.0,
            torch.where(valid_cap_contact, cap_velocity_reward, torch.zeros_like(cap_velocity_reward)),
            cap_velocity_reward,
        )

        fingertip_proximity_reward, nearest_tip_distance = self._fingertip_proximity_reward()
        contact_reward = torch.clamp(
            cap_contact_counts.to(dtype=torch.float32),
            max=float(len(self.cfg.vtdex_tactile_body_names)),
        )
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)

        reward = (
            cap_position_reward * float(self.cfg.cap_position_reward_scale)
            + cap_velocity_reward * float(self.cfg.cap_velocity_reward_scale)
            + fingertip_proximity_reward * float(self.cfg.fingertip_proximity_reward_scale)
            + contact_reward * float(self.cfg.cap_contact_reward_scale)
            + action_penalty * float(self.cfg.cap_action_penalty_scale)
        )

        success = self.cap_joint_progress >= float(self.cfg.cap_success_angle)
        self.successes[:] = torch.where(success, torch.ones_like(self.successes), self.successes)
        reward = torch.where(
            self.successes > 0.0,
            reward + float(self.cfg.cap_success_bonus),
            reward,
        )

        self.extras.setdefault("log", {}).update(
            {
                "pepper_cap_joint_position_rad": self.cap_joint_pos.mean(),
                "pepper_cap_progress_rad": self.cap_joint_progress.mean(),
                "pepper_cap_progress_deg": torch.rad2deg(self.cap_joint_progress).mean(),
                "pepper_cap_progress_max_rad": self.cap_joint_progress.max(),
                "pepper_cap_opening_velocity_rad_s": self.cap_joint_opening_velocity.mean(),
                "pepper_cap_contact_count": cap_contact_counts.float().mean(),
                "pepper_cap_valid_contact_ratio": valid_cap_contact.float().mean(),
                "pepper_cap_nearest_tip_distance_m": nearest_tip_distance.mean(),
                "pepper_cap_positive_rotation_ratio": (
                    (self.cap_joint_opening_velocity > 0.0) & valid_cap_contact
                ).float().mean(),
                "pepper_cap_success_rate": success.float().mean(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terminated = self.cap_joint_progress >= float(self.cfg.cap_reset_angle)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_target_pose(self, env_ids: Sequence[int]):
        """Bottle-cap success is a scalar joint target; hide pose markers."""

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.goal_pos[env_ids_tensor] = (
            self.object.data.root_pos_w[env_ids_tensor] - self.scene.env_origins[env_ids_tensor]
        )
        self.goal_rot[env_ids_tensor] = self.object.data.root_quat_w[env_ids_tensor]
        self.reset_goal_buf[env_ids_tensor] = False
        self._hide_goal_marker()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the fixed pepper bottle, passive cap joint and DG5F pregrasp."""

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore[attr-defined]
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # Skip the inherited free-object rotation reset while retaining the
        # shared DG5F joint/action/history reset.
        TesolloDeltoRlEnv._reset_idx(self, env_ids_tensor)

        hand_root_state = self.hand.data.default_root_state[env_ids_tensor].clone()
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
        self.hand_base_pos[env_ids_tensor] = (
            hand_root_state[:, :3] - self.scene.env_origins[env_ids_tensor]
        )
        self.hand_base_rot[env_ids_tensor] = hand_root_state[:, 3:7]

        object_root_state = self.object.data.default_root_state[env_ids_tensor].clone()
        object_root_state[:, :2] += sample_uniform(
            -float(self.cfg.reset_position_noise),
            float(self.cfg.reset_position_noise),
            (len(env_ids_tensor), 2),
            device=self.device,
        )
        object_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        object_root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids_tensor)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids_tensor)

        self._resolve_pepper_cap_indices()
        if self.object.data.default_joint_pos.shape[1] == 0:
            raise RuntimeError("Pepper bottle articulation has no internal joint to use as a cap")
        joint_pos = self.object.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = self.object.data.default_joint_vel[env_ids_tensor].clone()
        joint_pos[:, self._pepper_cap_joint_id] = float(self.cfg.cap_initial_angle)
        joint_vel[:, self._pepper_cap_joint_id] = 0.0
        self.object.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)

        if not hasattr(self, "cap_joint_start_pos") or self.cap_joint_start_pos.shape[0] != self.num_envs:
            self.cap_joint_start_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cap_joint_start_pos[env_ids_tensor] = float(self.cfg.cap_initial_angle)

        object_pos_env = object_root_state[:, :3] - self.scene.env_origins[env_ids_tensor]
        self.in_hand_pos[env_ids_tensor] = object_pos_env
        self.goal_pos[env_ids_tensor] = object_pos_env
        self.goal_rot[env_ids_tensor] = object_root_state[:, 3:7]
        self.reset_goal_buf[env_ids_tensor] = False
        self._hide_goal_marker()
        self._compute_intermediate_values()

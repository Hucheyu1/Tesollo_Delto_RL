# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""VTDexManip table reorientation task adapted to the DG5F right hand."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform

from .tesollo_delto_rl_env import TesolloDeltoRlEnv, rotation_distance, unscale
from .tesollo_delto_rl_env_cfg import TesolloDeltoRlEnvCfg
from .delto_cfg import TESOLLO_CFG
from .vtdex_encoder import VTDexJointEncoder


_VTDEx_ROOT = Path(__file__).resolve().parent / "vtdex_pretrained"
_VTDEx_OBJECT_ROOT = _VTDEx_ROOT / "assets" / "reorient_up"
_VTDEx_OBJECT_CODES = (
    "ddg-ycb_013_apple",
    "ddg-ycb_077_rubiks_cube",
    "ddg-ycb_070-a_colored_wood_blocks",
    "grab-doorknob",
    "ddg-ycb_010_potted_meat_can",
    "ddg-ycb_065-a_cups",
    "ddg-ycb_072-a_toy_airplane",
    "ddg-gd_rubber_duck_poisson_001",
    "ddg-ycb_018_plum",
    "ddg-ycb_002_master_chef_can",
)
_BASE_ENV_CFG = TesolloDeltoRlEnvCfg()


@configclass
class TesolloDeltoVTDexEnvCfg(TesolloDeltoRlEnvCfg):
    """DG5F reproduction of VTDexManip's ``reorient_down-vt_all_cls`` task."""

    # VTDexManip advances its policy at 60 Hz for a maximum of 600 steps.
    # Isaac Lab runs two 120 Hz physics steps per policy action to retain a
    # useful contact solve rate while matching the original control frequency.
    decimation = 2
    episode_length_s = 10.0
    # The source task cycles through ten different objects, so environments
    # must own independent physics trees instead of cloning env_0.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=0.75, replicate_physics=False
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        # VTDex explicitly assigns 0.8 friction to every manipulated-object
        # collision shape. The table below overrides this default to 1.0.
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
    )

    # VTDex uses net rigid-body contact forces, not articulation joint forces.
    # Contact reporting must be enabled when the DG5F USD is spawned.
    robot_cfg = TESOLLO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=TESOLLO_CFG.spawn.replace(
            activate_contact_sensors=True,
            # The shared robot asset permits 1000 m/s depenetration for legacy
            # tasks. That turns a small initial overlap into a several-hundred
            # Newton contact spike in this 120 Hz table task.
            rigid_props=TESOLLO_CFG.spawn.rigid_props.replace(max_depenetration_velocity=2.0),
        ),
        # Place DG5F directly above the tabletop object with its palm normal
        # pointing straight down. The X alignment puts the object below the
        # palm rather than beyond the fingertips; the Z offset preserves a
        # collision-free gap for all ten source objects.
        init_state=TESOLLO_CFG.init_state.replace(
            pos=(-0.080, 0.01733, 0.470),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),
        ),
    )
    vtdex_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rl_dg_[1-5]_[1-4]",
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )
    # Match reorient_up/down's binary tactile threshold (Newtons).
    vtdex_contact_threshold = 0.01

    # Exact VTDexManip reorient_down geometry in a globally translated frame:
    # source table top/object/goal z = 0.60/0.67/0.64 m; here they are
    # 0.31/0.38/0.35 m. All relative offsets and dimensions are unchanged.
    table_top_z = 0.31
    table_cfg = _BASE_ENV_CFG.object_cfg.replace(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.60),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                # URDF collision meshes are instanced by Isaac's importer, so
                # configure the contact envelope on the non-instanced table.
                # The positive rest offset keeps rendered surfaces separated.
                contact_offset=0.004,
                rest_offset=0.001,
            ),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=_BASE_ENV_CFG.object_cfg.init_state.replace(pos=(0.0, 0.0, 0.01)),
    )
    # These assets are spawned per environment in _setup_scene so the exact
    # ten-object source distribution can be retained.
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.02, 0.38)),
    )
    goal_vtdex_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalObject",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.02, 0.35)),
    )

    # Task-local DG5F pregrasp (radians), ordered layer-major as
    # ``actuated_joint_names``. VTDexManip resets Shadow Hand to zero, which is
    # already a useful open pose for that morphology. Here DG5F starts with a
    # lightly opened thumb and only 0.3 rad of four-finger flexion: the object
    # remains visibly below the palm while all fingers retain room to close.
    # Keep this override local so the tomato task retains its independently
    # trained initial pose.
    hand_position = [
        0.10, 0.00, 0.00, 0.00, 0.00,
        -0.80, 0.30, 0.30, 0.30, 0.00,
        -0.50, 0.30, 0.30, 0.30, 0.40,
        0.00, 0.30, 0.30, 0.30, 0.40,
    ]
    # Map Shadow Hand's useful actuator ranges onto the nearest DG5F joints.
    # Four-finger abduction is about +/-20 degrees and flexion is 0..90
    # degrees in the source MJCF. DG5F's asymmetric physical limits are
    # respected (ring abduction +15, little abduction -15..+20), while its
    # wider 109..115 degree proximal ranges are deliberately excluded because
    # they curl the fingertips back toward the palm instead of around the
    # tabletop object. Thumb opposition keeps the DG5F-specific -150..0 range;
    # its third joint retains the physical +/-90 degree range so the policy can
    # move from this open initial thumb into opposition.
    hand_lower_limits = [
        -22, -20, -20, -20, 0,
        -150, 0, 0, 0, -15,
        -90, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]
    hand_upper_limits = [
        60, 20, 20, 15, 45,
        0, 90, 90, 90, 20,
        90, 90, 90, 90, 90,
        90, 90, 90, 90, 90,
    ]
    # DG5F's five silicone tip meshes are fixed visual children rather than
    # tactile articulation bodies. They can lag behind moving links in rendered
    # camera frames, so hide only those visuals; _4 link collisions and tactile
    # observations remain enabled.
    hide_dg5f_tip_visuals = True

    # A dedicated policy camera keeps the frozen encoder input independent of
    # the interactive viewer and supports the scene's batched regex prim path.
    vtdex_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/VTDexCamera",
        offset=CameraCfg.OffsetCfg(
            # An exact look-at pose is assigned after scene initialization.
            pos=(0.3, 0.3, 0.71),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # Approximately the original Isaac Gym camera's 45 degree HFOV.
            focal_length=25.0,
            focus_distance=0.50,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=224,
        height=224,
        update_latest_camera_pose=True,
        debug_vis=False,
    )
    # Exact source reorient_down camera after translating the full scene down
    # by 0.29 m: source eye/target=(0.3,0.3,1.0)/(0,0,0.65).
    vtdex_camera_eye_local = (0.3, 0.3, 0.71)
    vtdex_camera_target_local = (0.0, 0.0, 0.36)

    # Original policy state is proprioception only. Shadow Hand's 48 values
    # become DG5F qpos(20) + qvel(20); RGB and 20 touch bits are fused by the
    # frozen joint encoder into one 384-D CLS representation.
    observation_space = 424
    # critic keeps the 84-dimensional simulator state plus 20 tactile values.
    state_space = 104
    asymmetric_obs = True
    obs_type = "vtdex"

    # Self-contained copy under this project; no external VTDexManip checkout
    # is needed at training or deployment time.
    vtdex_repo_root = str(_VTDEx_ROOT)
    vtdex_model_id = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
    vtdex_embedding_dim = 384
    # Match VTDex's layer-major token semantics: five fingers ordered
    # little/ring/middle/index/thumb at distal, then the same order at middle,
    # proximal and knuckle. DG5F's thumb-base link proxies the final Shadow-Hand
    # palm token. Keep this exact 20-channel order in the real robot node.
    vtdex_tactile_body_names = (
        "rl_dg_5_4", "rl_dg_4_4", "rl_dg_3_4", "rl_dg_2_4", "rl_dg_1_4",
        "rl_dg_5_3", "rl_dg_4_3", "rl_dg_3_3", "rl_dg_2_3", "rl_dg_1_3",
        "rl_dg_5_2", "rl_dg_4_2", "rl_dg_3_2", "rl_dg_2_2", "rl_dg_1_2",
        "rl_dg_5_1", "rl_dg_4_1", "rl_dg_3_1", "rl_dg_2_1", "rl_dg_1_1",
    )
    fingertip_body_names = list(vtdex_tactile_body_names)
    vtdex_tactile_indices = tuple(range(20))

    # reorient_down resets the object with random table yaw and commands a
    # fixed half turn about the table normal relative to that initial yaw.
    fix_object_initial_pose = True
    reset_position_noise = 0.01
    # Independent Shadow-Hand joint noise can put the shorter DG5F fingers
    # inside an object before the first physics step. Object position/yaw still
    # provide reset diversity, while the hand starts from its verified pregrasp.
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
    target_yaw_delta_rad = 3.141592653589793

    # Reward and termination values copied from reorient_down.yaml.
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_penalty = 0.0
    fall_dist = 0.05
    success_tolerance = 0.1
    table_tilt_limit_deg = 45.0
    vel_reward_scale = 1.0
    fingertip_distance_reward_scale = 0.25
    action_scale = 1.0
    # Smooth position targets at 60 Hz. With 1.0, the first PPO action can move
    # a DG5F joint by tens of degrees in a single control interval.
    act_moving_average = 0.2

    # Keep the tiny source goal mesh visible exactly as in VTDexManip.
    debug_visualization = False


class TesolloDeltoVTDexEnv(TesolloDeltoRlEnv):
    """Tabletop half-turn task whose actor never receives simulator object pose."""

    cfg: TesolloDeltoVTDexEnvCfg

    def __init__(self, cfg: TesolloDeltoVTDexEnvCfg, render_mode: str | None = None, **kwargs):
        if float(cfg.target_yaw_delta_rad) == 0.0:
            raise ValueError("target_yaw_delta_rad must be non-zero")
        if not 0.0 < float(cfg.table_tilt_limit_deg) < 90.0:
            raise ValueError("table_tilt_limit_deg must be between 0 and 90 degrees")

        super().__init__(cfg, render_mode, **kwargs)
        self._reward_fingertip_body_ids = torch.tensor(
            [self.hand.body_names.index(name) for name in self.cfg.vtdex_tactile_body_names[:5]],
            dtype=torch.long,
            device=self.device,
        )
        contact_body_ids, contact_body_names = self._vtdex_contact_sensor.find_bodies(
            list(self.cfg.vtdex_tactile_body_names), preserve_order=True
        )
        if contact_body_names != list(self.cfg.vtdex_tactile_body_names):
            raise RuntimeError(
                "VTDex ContactSensor body order does not match the configured tactile token order: "
                f"expected={list(self.cfg.vtdex_tactile_body_names)}, resolved={contact_body_names}"
            )
        if len(contact_body_ids) != 20 or len(set(contact_body_ids)) != 20:
            raise RuntimeError(
                "VTDex ContactSensor must resolve exactly 20 unique DG5F links; "
                f"got ids={contact_body_ids}"
            )
        self._vtdex_contact_body_ids = torch.tensor(
            contact_body_ids, dtype=torch.long, device=self.device
        )
        print(
            "[INFO]: VTDex tactile ContactSensor mapping (net contact force): "
            f"{contact_body_names}"
        )
        self.vtdex_encoder = VTDexJointEncoder(
            repo_root=self.cfg.vtdex_repo_root,
            model_id=self.cfg.vtdex_model_id,
            device=self.device,
            tactile_indices=tuple(self.cfg.vtdex_tactile_indices),
        )
        if self.vtdex_encoder.embedding_dim != self.cfg.vtdex_embedding_dim:
            raise ValueError(
                "Configured VTDex embedding dimension does not match checkpoint: "
                f"{self.cfg.vtdex_embedding_dim} != {self.vtdex_encoder.embedding_dim}"
            )
        self.vtdex_embeddings = torch.zeros(
            (self.num_envs, self.cfg.vtdex_embedding_dim), dtype=torch.float32, device=self.device
        )
        self._configure_vtdex_camera_pose()
        self._hide_dg5f_tip_visuals()
        self._hide_goal_marker()

    def _setup_scene(self):
        self._spawn_vtdex_objects()
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal_object = RigidObject(self.cfg.goal_vtdex_object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self._vtdex_camera = Camera(self.cfg.vtdex_camera)
        self._vtdex_contact_sensor = ContactSensor(self.cfg.vtdex_contact_sensor)

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_object"] = self.goal_object
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["vtdex_camera"] = self._vtdex_camera
        self.scene.sensors["vtdex_contact"] = self._vtdex_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_vtdex_objects(self):
        """Spawn the source task's heterogeneous active and goal URDFs."""

        yellow = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(204.0 / 255.0, 204.0 / 255.0, 0.0),
            roughness=0.5,
            metallic=0.0,
        )
        active_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            # Keep overlap recovery bounded. The previous 1000 m/s setting was
            # the main source of non-physical contact impulses.
            max_depenetration_velocity=2.0,
            max_linear_velocity=5.0,
            max_angular_velocity=720.0,
        )
        goal_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        for env_index, env_path in enumerate(self.scene.env_prim_paths):
            object_code = _VTDEx_OBJECT_CODES[env_index % len(_VTDEx_OBJECT_CODES)]
            urdf_path = _VTDEx_OBJECT_ROOT / object_code / "coacd" / "coacd_1.urdf"
            # Derived from coacd_1.urdf by removing only <collision> elements.
            # This reproduces Isaac Gym's separate goal collision group without
            # relying on collision-property edits below instanced mesh prims.
            goal_urdf_path = _VTDEx_OBJECT_ROOT / object_code / "coacd" / "coacd_goal.urdf"
            if not urdf_path.is_file():
                raise FileNotFoundError(f"Missing copied VTDex object asset: {urdf_path}")
            if not goal_urdf_path.is_file():
                raise FileNotFoundError(f"Missing derived VTDex goal asset: {goal_urdf_path}")

            common = {
                "fix_base": False,
                "merge_fixed_joints": True,
                "joint_drive": None,
                "collider_type": "convex_hull",
                "visual_material": yellow,
                # Collision overrides must reach the imported mesh prims:
                # active objects collide, while the tiny goal actors do not.
                "make_instanceable": False,
            }
            object_spawn_cfg = sim_utils.UrdfFileCfg(
                **common,
                asset_path=str(urdf_path),
                scale=(0.05, 0.05, 0.05),
                semantic_tags=[("class", "object")],
                rigid_props=active_rigid_props,
            )
            goal_spawn_cfg = sim_utils.UrdfFileCfg(
                **common,
                asset_path=str(goal_urdf_path),
                scale=(0.005, 0.005, 0.005),
                semantic_tags=[("class", "goal")],
                rigid_props=goal_rigid_props,
            )
            object_spawn_cfg.func(
                f"{env_path}/Object",
                object_spawn_cfg,
                translation=tuple(self.cfg.object_cfg.init_state.pos),
                orientation=tuple(self.cfg.object_cfg.init_state.rot),
            )
            goal_spawn_cfg.func(
                f"{env_path}/GoalObject",
                goal_spawn_cfg,
                translation=tuple(self.cfg.goal_vtdex_object_cfg.init_state.pos),
                orientation=tuple(self.cfg.goal_vtdex_object_cfg.init_state.rot),
            )

    def _configure_vtdex_camera_pose(self):
        """Aim every tiled camera using environment-local eye/target points."""

        eye_local = torch.tensor(
            self.cfg.vtdex_camera_eye_local, dtype=torch.float32, device=self.device
        ).view(1, 3)
        target_local = torch.tensor(
            self.cfg.vtdex_camera_target_local, dtype=torch.float32, device=self.device
        ).view(1, 3)
        if torch.linalg.vector_norm(eye_local - target_local).item() < 1.0e-4:
            raise ValueError("vtdex_camera_eye_local and target must be different points")
        eyes_w = self.scene.env_origins + eye_local
        targets_w = self.scene.env_origins + target_local
        self._vtdex_camera.set_world_poses_from_view(eyes_w, targets_w)

    def _hide_dg5f_tip_visuals(self):
        """Hide DG5F's detached silicone visuals without changing physics."""

        if not self.cfg.hide_dg5f_tip_visuals:
            return
        for env_path in self.scene.env_prim_paths:
            for finger_index in range(1, 6):
                visual_path = f"{env_path}/Robot/rl_dg_{finger_index}_tip/visuals"
                visual_prim = self.scene.stage.GetPrimAtPath(visual_path)
                if not visual_prim.IsValid():
                    raise RuntimeError(
                        f"Missing DG5F fingertip visual prim required by hide_dg5f_tip_visuals: "
                        f"{visual_path}"
                    )
                UsdGeom.Imageable(visual_prim).MakeInvisible()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        rgb = self._vtdex_camera.data.output["rgb"]
        tactile = self.fingertip_force_binary_results.to(dtype=torch.float32)
        self.vtdex_embeddings = self.vtdex_encoder(rgb, tactile)

        # Match reorient_down's policy boundary: only hand proprioception and
        # the frozen joint RGB/touch representation are exposed to the actor.
        policy_obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.vtdex_embeddings,
            ),
            dim=-1,
        )

        # Simulator object pose/velocity is privileged and never enters policy_obs.
        critic_obs = self.compute_full_state()
        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"VTDex policy observation has {policy_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.observation_space}"
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"VTDex critic observation has {critic_obs.shape[-1]} values; "
                f"configuration declares {self.cfg.state_space}"
            )
        return {"policy": policy_obs, "critic": critic_obs}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Map normalized actions to absolute targets like VTDexManip."""

        self.raw_actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = self.raw_actions.clone()
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        scaled_actions = torch.clamp(
            float(self.cfg.action_scale) * self.raw_actions, -1.0, 1.0
        )
        self.target_pos[:, self.actuated_dof_indices] = (
            0.5 * (scaled_actions + 1.0) * (upper - lower) + lower
        )

    def _table_task_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return planar drift, rotation error, excessive tilt and table fall."""

        planar_drift = torch.linalg.vector_norm(
            self.object_pos[:, :2] - self.in_hand_pos[:, :2], dim=-1
        )
        rotation_error = rotation_distance(self.object_rot, self.goal_rot)
        object_up = quat_apply(self.object_rot, self.z_unit_tensor)
        tilt_cosine = object_up[:, 2]
        tilt_limit_cosine = torch.cos(
            torch.deg2rad(
                torch.tensor(float(self.cfg.table_tilt_limit_deg), device=self.device)
            )
        )
        excessive_tilt = tilt_cosine <= tilt_limit_cosine
        below_table = self.object_pos[:, 2] < float(self.cfg.table_top_z) - 0.025
        return planar_drift, rotation_error, excessive_tilt, below_table

    def _get_rewards(self) -> torch.Tensor:
        """Reproduce reorient_down's rotation, velocity and fingertip shaping."""

        planar_drift, rotation_error, excessive_tilt, below_table = self._table_task_metrics()
        fingertip_pos_w = self.hand.data.body_pos_w.index_select(1, self._reward_fingertip_body_ids)
        fingertip_z = fingertip_pos_w[:, :, 2] - self.scene.env_origins[:, 2:3]
        fingertip_height_error = torch.abs(
            fingertip_z - self.object_pos[:, 2:3]
        ).sum(dim=-1)

        distance_reward = planar_drift * float(self.cfg.dist_reward_scale)
        rotation_reward = float(self.cfg.rot_reward_scale) / (
            torch.abs(rotation_error) + float(self.cfg.rot_eps)
        )
        velocity_reward = (
            torch.clamp(self.object_angvel[:, 2], -10.0, 10.0)
            * float(self.cfg.vel_reward_scale)
        )
        fingertip_reward = (
            torch.exp(-10.0 * fingertip_height_error)
            * float(self.cfg.fingertip_distance_reward_scale)
        )
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        reward = (
            distance_reward
            + rotation_reward
            + velocity_reward
            + fingertip_reward
            + action_penalty * float(self.cfg.action_penalty_scale)
        )

        success = rotation_error <= float(self.cfg.success_tolerance)
        displaced = planar_drift >= float(self.cfg.fall_dist)
        failed = displaced | excessive_tilt | below_table
        reward = torch.where(success, reward + float(self.cfg.reach_goal_bonus), reward)
        reward = torch.where(failed, reward + float(self.cfg.fall_penalty), reward)

        self.reset_goal_buf[:] = success
        self.successes[:] = torch.where(success, torch.ones_like(self.successes), self.successes)
        self.extras.setdefault("log", {}).update(
            {
                "table_rotation_error_rad": rotation_error.mean(),
                "table_planar_drift_m": planar_drift.mean(),
                "table_success_rate": success.float().mean(),
                "table_tilt_failure_rate": excessive_tilt.float().mean(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        planar_drift, rotation_error, excessive_tilt, below_table = self._table_task_metrics()
        success = rotation_error <= float(self.cfg.success_tolerance)
        failed = (
            (planar_drift >= float(self.cfg.fall_dist))
            | excessive_tilt
            | below_table
        )
        terminated = success | failed
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _compute_tactile_observations(self):
        """Use VTDex-compatible net contact forces for the 20 tactile bits."""

        if not hasattr(self, "_vtdex_contact_body_ids"):
            raise RuntimeError("VTDex tactile body mapping has not been initialized")

        net_forces_w = self._vtdex_contact_sensor.data.net_forces_w
        if net_forces_w is None or net_forces_w.shape[:2] != (
            self.num_envs,
            self._vtdex_contact_sensor.num_bodies,
        ):
            raise RuntimeError(
                "Unexpected VTDex ContactSensor force tensor shape: "
                f"{None if net_forces_w is None else tuple(net_forces_w.shape)}"
            )
        tactile_forces_w = net_forces_w.index_select(1, self._vtdex_contact_body_ids)
        tactile_force_norms = torch.linalg.vector_norm(tactile_forces_w, dim=-1)
        self.fingertip_force_sensors = tactile_forces_w
        self.fingertip_force_binary_results = (
            tactile_force_norms > float(self.cfg.vtdex_contact_threshold)
        ).to(dtype=torch.int32)
        if hasattr(self, "extras"):
            self.extras.setdefault("log", {})["vtdex_tactile_active_ratio"] = (
                self.fingertip_force_binary_results.float().mean()
            )
            self.extras["log"]["vtdex_tactile_force_mean_n"] = tactile_force_norms.mean()
            self.extras["log"]["vtdex_tactile_force_max_n"] = tactile_force_norms.max()

    def _reset_target_pose(self, env_ids: Sequence[int]):
        object_rot = self.object.data.root_quat_w[env_ids]
        target_delta = quat_from_angle_axis(
            torch.full(
                (len(env_ids),),
                float(self.cfg.target_yaw_delta_rad),
                dtype=torch.float32,
                device=self.device,
            ),
            self.z_unit_tensor[env_ids],
        )
        self.goal_rot[env_ids] = quat_mul(object_rot, target_delta)
        self.in_hand_pos[env_ids] = self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.goal_pos[env_ids] = self.in_hand_pos[env_ids]
        self.goal_pos[env_ids, 2] -= 0.03
        self.reset_goal_buf[env_ids] = False
        self._write_goal_object_pose(env_ids)
        self._hide_goal_marker()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset on-table position/yaw after the shared DG5F state reset."""

        if env_ids is None:
            env_ids = self.hand._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        object_root_state = self.object.data.default_root_state[env_ids].clone()
        xy_noise = sample_uniform(
            -float(self.cfg.reset_position_noise),
            float(self.cfg.reset_position_noise),
            (len(env_ids), 2),
            device=self.device,
        )
        object_root_state[:, :2] += xy_noise
        yaw = sample_uniform(
            -torch.pi,
            torch.pi,
            (len(env_ids),),
            device=self.device,
        )
        yaw_rotation = quat_from_angle_axis(yaw, self.z_unit_tensor[env_ids])
        object_root_state[:, 3:7] = quat_mul(object_root_state[:, 3:7], yaw_rotation)
        object_root_state[:, :3] += self.scene.env_origins[env_ids]
        object_root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids)

        object_pos_env = object_root_state[:, :3] - self.scene.env_origins[env_ids]
        self.in_hand_pos[env_ids] = object_pos_env
        self.goal_pos[env_ids] = object_pos_env
        self.goal_pos[env_ids, 2] -= 0.03
        target_delta = quat_from_angle_axis(
            torch.full(
                (len(env_ids),),
                float(self.cfg.target_yaw_delta_rad),
                dtype=torch.float32,
                device=self.device,
            ),
            self.z_unit_tensor[env_ids],
        )
        self.goal_rot[env_ids] = quat_mul(object_root_state[:, 3:7], target_delta)
        self.reset_goal_buf[env_ids] = False
        self._write_goal_object_pose(env_ids)
        self._hide_goal_marker()
        self._compute_intermediate_values()

    def _write_goal_object_pose(self, env_ids: Sequence[int]):
        """Move the tiny non-colliding goal actor to the current target pose."""

        goal_pos_w = self.goal_pos[env_ids] + self.scene.env_origins[env_ids]
        goal_pose_w = torch.cat((goal_pos_w, self.goal_rot[env_ids]), dim=-1)
        self.goal_object.write_root_pose_to_sim(goal_pose_w, env_ids)

    def _hide_goal_marker(self):
        hidden_goal_pos_w = torch.full_like(self.goal_pos, -10.0)
        self.goal_markers.visualize(hidden_goal_pos_w, self.goal_rot)

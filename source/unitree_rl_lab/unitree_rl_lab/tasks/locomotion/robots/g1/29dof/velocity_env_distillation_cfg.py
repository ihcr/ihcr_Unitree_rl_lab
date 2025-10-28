# Copyright (c) 2022-2025,
# The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.sensors import LidarSensorCfg  # 可选
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg  # 可选

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# 预置地形
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# =========================
# Scene
# =========================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Terrain scene with a legged robot."""

    # 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 机器人
    robot: ArticulationCfg = MISSING

    # 高度扫描（teacher 用）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    # lidar_sensor = LidarSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=LidarSensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0, 1, 0., 0.)),
    #     ray_alignment = "yaw",
    #     pattern_cfg=LivoxPatternCfg(
    #         sensor_type="mid360",
    #         # samples=24000,  # Reduced for better performance with 1024 envs
    #         samples=20000,  # Reduced for better performance with 1024 envs
    #         # samples=20,  # Reduced for better performance with 1024 envs
    #     ),
    #     mesh_prim_paths=["/World/ground","/World/static"],
    #     max_distance=20.0,
    #     # max_distance=20.0,
    #     min_range=0.2,
    #     return_pointcloud=True,  # Disable pointcloud for performance
    #     pointcloud_in_world_frame=False,
    #     enable_sensor_noise=False,  # Disable noise for pure performance test
    #     random_distance_noise=0.0,
    #     update_frequency=25.0,  # 25 Hz for better performance
    #     debug_vis=False,  # Disable visualization for performance
    # )


    # 深度相机（student 用）
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_depth_cam",
        update_period=0.1,                  # 你已有逻辑会按物理步长覆盖
        height=60,                          # 建议 48x64 或 60x80
        width=80,
        data_types=["depth"],  # ★ 用这个 key
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=16.0,
            clipping_range=(0.1, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.12),
            # rot=(0.5, -0.5, 0.5, -0.5),     # 你后续可调整到“头部、俯视”的真实位姿
            rot = (0.436, -0.560, 0.560, -0.430),



            convention="ros",
        ),
    )

    # 接触力
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # 光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# =========================
# MDP
# =========================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # ---------- Teacher 观测（privileged） ----------
    @configclass
    class TeacherObsCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # ---------- Student 观测（深度特征） ----------
    @configclass
    class PolicyStudentCfg(ObsGroup):
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        # 深度 → 预训练网络 → 特征向量
        # depth_latent = ObsTerm(
        #     func=mdp.depth_cnn_latent,
        #     params=dict(
        #         sensor_cfg=SceneEntityCfg("camera"),
        #         data_type="distance_to_camera",
        #         convert_perspective_to_orthogonal=True,
        #         clip_min=0.0,
        #         clip_max=6.0,
        #         add_gaussian_noise=True,
        #         noise_std=0.02,
        #         out_dim=128,     # ★ 想要多少维 latent 就写多少
        #     ),
        #     noise=None,
        #     clip=None,
        # )

        forward_depth = ObsTerm(
            func=mdp.depth_image_flat,
            params=dict(
                sensor_cfg=SceneEntityCfg("camera"),
                data_type="depth",
                convert_perspective_to_orthogonal=True,
                crop_top_bottom=(0, 0),     # 与官方/你自己的相机几何对齐
                crop_left_right=(0, 0),
                out_hw=(48, 64),
                clip_min=0.12,
                clip_max=2.0,
                refresh_hz=10.0,
                delay_s=0.20,
                # control_dt 可不填，函数会从 env 上推断
            ),
            noise=AdditiveGaussianNoiseCfg(std=0.01),
            clip=None,
        )

        # lidar_distances = ObsTerm(
        #     func=mdp.height_scan,  # We can reuse height_scan function as it works with any ray caster
        #     params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  

    # 绑定观测组
    policy: PolicyStudentCfg = PolicyStudentCfg()
    teacher: TeacherObsCfg = TeacherObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset", params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)}
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    camera_offset_startup = EventTerm(
        func=mdp.randomize_camera_offset_simple,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "pos_jitter": {"x": (-0.01, 0.01), "y": (-0.005, 0.005), "z": (-0.005, 0.01)},
            "pitch_range_deg": (30.0, 60.0),
            "lock_pitch": False,     # 需要随机俯仰
            "pitch_sign": -1.0,      # 俯视为负；若相反改为 +1.0
        },
    )

    # reset：只抖位置，不改俯仰（lock_pitch=True）
    camera_offset_reset = EventTerm(
        func=mdp.randomize_camera_offset_simple,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "pos_jitter": {"x": (-0.005, 0.005), "y": (-0.003, 0.003), "z": (-0.003, 0.006)},
            "pitch_range_deg": (30.0, 60.0),  # 传也行，但 lock_pitch=True 会忽略
            "lock_pitch": True,
            "pitch_sign": -1.0,
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "command_name": "base_velocity", "threshold": 0.5},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )

    # optional
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =========================
# Env Config
# =========================
@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # simulation basics
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器更新周期对齐

        # if self.scene.lidar_sensor is not None:
        #     self.scene.lidar_sensor.update_period = self.decimation * self.sim.dt
        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if getattr(self.scene, "camera", None) is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt * 5   # 0.02 * 5 = 0.1 s
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # curriculum 启用/禁用
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

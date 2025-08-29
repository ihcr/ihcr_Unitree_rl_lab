import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip



from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


COBBLESTONE_AND_STAIRS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        # 50% 是鹅卵石 flat 区域
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5,
        flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=0.15,          # ✅ 稍微缩小半径，便于采样
                    max_height_diff=0.02,       # ✅ 更松一点，但仍平整
                    x_range=(-0.3, 0.3),        # ✅ 中心区域即可
                    y_range=(-0.3, 0.3),
                    z_range=(-0.02, 0.02),
                )
            },
        ),
        # 50% 是楼梯区，可以细化多个难度
        # 上楼梯（在 sub-terrain 中控制方向很难，需靠 agent 出生位置）
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.03, 0.24),
            step_width=0.3,
            platform_width=3.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=0.15,          # ✅ 稍微缩小半径，便于采样
                    max_height_diff=0.02,       # ✅ 更松一点，但仍平整
                    x_range=(-0.3, 0.3),        # ✅ 中心区域即可
                    y_range=(-0.3, 0.3),
                    z_range=(-0.02, 0.02),
                )
            },
            
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.03, 0.24),
            step_width=0.3,
            platform_width=3.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=1,
                    patch_radius=0.15,          # ✅ 稍微缩小半径，便于采样
                    max_height_diff=0.02,       # ✅ 更松一点，但仍平整
                    x_range=(-0.3, 0.3),        # ✅ 中心区域即可
                    y_range=(-0.3, 0.3),
                    z_range=(-0.02, 0.02),
                )
            },
        ),


                # 斜坡（上坡 + 下坡）
        # "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=(0.01, 0.5),  # 弧度，约等于 (5.7°, 22.9°)
        #     platform_width=3.0,
        #     inverted=False,  # 默认为“从高处向下斜”

        #     flat_patch_sampling={
        #         "init_pos": FlatPatchSamplingCfg(
        #             num_patches=1,
        #             patch_radius=0.15,          # ✅ 稍微缩小半径，便于采样
        #             max_height_diff=0.02,       # ✅ 更松一点，但仍平整
        #             x_range=(-0.3, 0.3),        # ✅ 中心区域即可
        #             y_range=(-0.3, 0.3),
        #             z_range=(-0.02, 0.02),
        #         )
        #     },
        # ),

        # "slope_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1,
        #     slope_range=(0.01, 0.5),  # 弧度，约等于 (5.7°, 22.9°)
        #     platform_width=3.0,
        #     inverted=True,  # 默认为“从高处向下斜”

        #     flat_patch_sampling={
        #         "init_pos": FlatPatchSamplingCfg(
        #             num_patches=1,
        #             patch_radius=0.15,          # ✅ 稍微缩小半径，便于采样
        #             max_height_diff=0.02,       # ✅ 更松一点，但仍平整
        #             x_range=(-0.3, 0.3),        # ✅ 中心区域即可
        #             y_range=(-0.3, 0.3),
        #             z_range=(-0.02, 0.02),
        #         )
        #     },
        # ),

        # "Star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.1,
        #     height_range=(0.05, 0.15),
        #     block_size_range=(0.1, 0.3),
        #     gap_size_range=(0.05, 0.15),
        # ),

        # "Rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=0.1,
        #     rail_thickness_range=(0.05, 0.15),
        #     rail_height_range=(0.05, 0.2),
        #     platform_width=1.0
        # ),
        # "stairs_hard": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.05,
        #     step_height_range=(0.10, 0.20),
        #     step_width=0.25,
        #     platform_width=0.5,
        # ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",  # "plane", "generator"
    #     terrain_generator=COBBLESTONE_AND_STAIRS_CFG,  # None, ROUGH_TERRAINS_CFG
    #     max_init_terrain_level=COBBLESTONE_AND_STAIRS_CFG.num_rows - 1,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )


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
    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # attach_yaw_only=True,
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_from_terrain,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.1, 0.15), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 2.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        # ),
        # limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-0.5, 2.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        # ),

        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(0.5, 0.5), ang_vel_z=(-1.0, 1.0)

        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True


    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""



    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)


    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )


    # stand_still = RewTerm(
    #     func=mdp.stand_still,
    #     weight=0.2,
    #     params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*",
    #                 ".*_hip_roll_joint", 
    #                 ".*_hip_yaw_joint",
    #                 ".*_hip_pitch_joint",
    #                 ".*_ankle_.*_joint",
    #             ],
    #         )},
    # )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    # base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})
    base_height = RewTerm(
        func=mdp.base_height_l2,            # 就是你贴的那个函数
        weight=-10.0,                        # 先别太大，-2 ~ -5 都行；和 termination 是互补关系
        params={
            "target_height": 0.78,          # 取你“平地自然站姿的 base_z - 地面高度”，G1 常见 0.74~0.78
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            # asset_cfg 可不写，默认 "robot"
        },
    )


    # -- feet
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )
    # gait = RewTerm(
    #     func=mdp.contact_event_reward,
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_terrain_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "ground_sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "margin": 0.10,        # 楼梯先用 0.10~0.12，学稳了再降
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )

    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_rel_terrain,
    #     weight=1.2,   
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #         "ray_sensor": SceneEntityCfg("height_scanner"),
    #         "contact_sensor": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         "safety_margin": 0.03,
    #         "target_clear": 0.06,
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "c_under": 50.0,
    #         "c_over": 1.0,
    #         "num_scan_pts": 5,
    #         # "ahead": 0.20,
    #         "ahead": 0.14,     # ✅ 缩短到一阶内

    #         # "half_width": 0.06,
    #         "half_width": 0.05 # ✅ 与上面一致
    #     },
    # )


    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # feet_edge_align = RewTerm(
    #     func=mdp.foot_edge_alignment,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #         "ray_sensor": SceneEntityCfg("height_scanner"),
    #         "contact_sensor": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #         "h_jump_thresh": 0.05,
    #         "safe_dist": 0.05,
    #         "w_pos": 0.6,
    #         "w_neg": 1.2,
    #         "search_radius": 0.30,
    #     },
    # )



    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )



    # -- other
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1,
    #     params={
    #         "threshold": 2,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
    #     },
    # )







@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    # base_height = DoneTerm(func=mdp.root_height_below_terrain_origin_map, params={"min_relative_height": 0.2})
    base_height = DoneTerm(
        func=mdp.root_height_below_mesh_terrain,
        params={
            "min_relative_height": 0.2,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    
    # base_height = DoneTerm(func=mdp.root_height_below_relative_minimum, params={"minimum_rel_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.num_cols = 10
        # self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


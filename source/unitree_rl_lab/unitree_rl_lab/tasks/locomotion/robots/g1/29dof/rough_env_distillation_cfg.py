# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_distillation_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, TerminationsCfg

from isaaclab.utils.noise import AdditiveGaussianNoiseCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG, UNITREE_G1_29DOF_CFG, UNITREE_G1_29DOF_MINIMAL_CFG  # isort: skip


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",

                    
                ],
            )
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_roll_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_pitch_joint",
    #                 ".*_elbow_roll_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_five_joint",
    #                 ".*_three_joint",
    #                 ".*_six_joint",
    #                 ".*_four_joint",
    #                 ".*_zero_joint",
    #                 ".*_one_joint",
    #                 ".*_two_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    # )
    joint_deviation_waists = RewTerm(
            func=mdp.joint_deviation_l1,
            # weight=-0.1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                )
            },
    )

    # base_height = RewTerm(
    #     func=mdp.base_height_l2,            # 就是你贴的那个函数
    #     weight=-2.0,                        # 先别太大，-2 ~ -5 都行；和 termination 是互补关系
    #     params={
    #         "target_height": 0.74,          # 取你“平地自然站姿的 base_z - 地面高度”，G1 常见 0.74~0.78
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         # asset_cfg 可不写，默认 "robot"
    #     },
    # )

    base_height = RewTerm(
        func=mdp.base_height_l2_robust,
        weight=-5.0,
        # weight=-2.0,
        params={
            "target_height": 0.76,
            # "target_height": 0.74,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            # "max_terrain_adjust": 0.3,  # 可调范围 ±0.3m
        }
    )

    # ----Phase 2 training----
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

@configclass
class G1Termination(TerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(
    #     func=mdp.root_height_below_mesh_terrain,
    #     params={
    #         "min_relative_height": 0.2,
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     }
    # )

    
    # base_height = DoneTerm(func=mdp.root_height_below_relative_minimum, params={"minimum_rel_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()
    terminations: G1Termination = G1Termination()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene

        # print("[DEBUG] G1_MINIMAL_CFG.spawn.usd_path =", G1_MINIMAL_CFG.spawn.usd_path)
        # self.scene.robot = UNITREE_G1_29DOF_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # self.scene.lidar_sensor.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/front_cam"

        # Randomization
        # self.events.push_robot = None
        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 5.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}

        # self.events.add_base_mass = None
        self.events.physics_material.mode = "startup"
        self.events.physics_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
        self.events.add_base_mass.mode = "startup"
        self.events.add_base_mass.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        }

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        camera_offset_startup = EventTerm( 
            func=mdp.randomize_camera_offset_simple, 
            mode="startup", 
            params={ 
                "sensor_cfg": SceneEntityCfg("camera"), 
                "pos_jitter": {"x": (-0.01, 0.01), "y": (-0.005, 0.005), "z": (-0.005, 0.01)}, 
                "pitch_range_deg": (35.0, 55.0), 
                "lock_pitch": False, # 需要随机俯仰 "pitch_sign": -1.0, # 俯视为负；若相反改为 +1.0 
                },
        )
        camera_offset_reset = EventTerm(
            func=mdp.randomize_camera_offset_simple,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("camera"), 
                "pos_jitter": {"x": (-0.005, 0.005), "y": (-0.003, 0.003), "z": (-0.003, 0.006)}, 
                "pitch_range_deg": (40.0, 50.0), # 传也行，但 lock_pitch=True 会忽略 
                "lock_pitch": True, 
                "pitch_sign": -1.0,
            },
        )
        # self.events.camera_offset_startup.mode = "startup"
        # self.events.camera_offset_startup.params = {
        #     "sensor_cfg": SceneEntityCfg("camera"),
        #     "pos_jitter": {"x": (-0.01, 0.01), "y": (-0.005, 0.005), "z": (-0.005, 0.01)},
        #     "pitch_range_deg": (30.0, 60.0),
        #     "lock_pitch": False,   # 需要随机俯仰
        #     "pitch_sign": -1.0,    # 俯视为负；若方向反了改 +1.0
        # }

        # —— camera_offset_reset：只抖位置，锁定俯仰，同样不要加行尾逗号 ——
        # self.events.camera_offset_reset.mode = "reset"
        # self.events.camera_offset_reset.params = {
        #     "sensor_cfg": SceneEntityCfg("camera"),
        #     "pos_jitter": {"x": (-0.005, 0.005), "y": (-0.003, 0.003), "z": (-0.003, 0.006)},
        #     "pitch_range_deg": (30.0, 60.0),  # 传不传都行；lock_pitch=True 会忽略
        #     "lock_pitch": True,
        #     "pitch_sign": -1.0,
        # }


        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/front_cam"
        # —— 不要用 self.scene.camera.prim_path（大多不存在），用 'camera' 实体 ——
        cam = None
        if hasattr(self.scene, "sensors") and hasattr(self.scene.sensors, "cameras") and "camera" in self.scene.sensors.cameras:
            cam = self.scene.sensors.cameras["camera"]
        elif hasattr(self.scene, "cameras") and "camera" in self.scene.cameras:
            cam = self.scene.cameras["camera"]

        if cam is not None:
            # 分辨率对齐 forward_depth=3072 -> 48x64
            cam.width  = 64
            cam.height = 48
            # ★ 合法的数据类型：深度 + RGB 可视化源
            cam.data_types = ["distance_to_camera", "rgba"]
            # 建议避免 inf：可选
            if hasattr(cam, "depth_clipping_behavior"):
                cam.depth_clipping_behavior = "max"
            # 每步更新（若有该字段）
            if hasattr(cam, "update_period"):
                cam.update_period = 0.0
            # 把相机挂到你想要的 prim 上
            cam.prim_path = "{ENV_REGEX_NS}/Robot/torso_link/front_cam"
            # 如需位姿（若 CameraCfg 支持 offset）：
            # cam.offset.pos = (0.10, 0.0, 0.12)
            # cam.offset.rot = (0.5, -0.5, 0.5, -0.5)
            # cam.offset.convention = "ros"

        # 将 policy 的 forward_depth 绑定到这台已存在的 'camera'
        self.observations.policy.forward_depth.params["sensor_cfg"] = SceneEntityCfg("camera")
        # 若该 term 支持 data_type 参数，也显式声明（有些模板不需要）
        # self.observations.policy.forward_depth.params["data_type"] = "distance_to_camera"

        # 打开调试可视化（Scene Debug 的开关；图像还是要在 Viewport 切换 AOV）
        if hasattr(self.scene, "debug_vis"):
            self.scene.debug_vis.camera = True
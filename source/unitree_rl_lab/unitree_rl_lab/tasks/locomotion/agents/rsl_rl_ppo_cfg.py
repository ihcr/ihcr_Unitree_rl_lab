# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlPpoActorCriticRecurrentCfg, RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=0.8,
    #     actor_hidden_dims=[32],
    #     critic_hidden_dims=[32],
    #     activation='elu',
    #     rnn_type='lstm',
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1,
    # )

    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=0.6,                 # RNN 往往用小一些更稳
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[256, 128],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,                 # 建议 128 起步
    #     rnn_num_layers=1,
    #     class_name="__import__('unitree_rl_lab.utils.actor_critic_scan', fromlist=['ActorCriticScanRecurrent']).ActorCriticScanRecurrent",
    # )



    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=0.8,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     # 关键：用内联 __import__，避免 NameError
    #     class_name="__import__('unitree_rl_lab.utils.actor_critic_scan', fromlist=['ActorCriticScan']).ActorCriticScan",
    # )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RoughDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "unitree_g1_29dof_velocity"          # 要与老师训练时的 experiment 目录一致
    empirical_normalization = False

    policy = RslRlDistillationStudentTeacherCfg(
        # class_name="StudentTeacher",
        class_name="StudentTeacherVisual",          # ★ 用新类

        init_noise_std=0.5,          # 建议初期减小一点，稳定分布
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",

        # 视觉参数
        # ---- 视觉定位 ----
        depth_slice=[96, 3168],         # 从 obs 中切出 depth_flat 的位置
        image_shape=[1, 48, 64],        # 单帧尺寸 (C,H,W)
        history_K=5,                    # ★ 这里控制拼多少帧 latent
        visual_latent_size=128,
        visual_kwargs=dict(
            channels=[16, 32, 32],
            kernel_sizes=[5, 3, 3],
            strides=[2, 2, 1],
            hidden_sizes=[128],
        ),

    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        gradient_length=32,
        learning_rate=5.0e-4,
        max_grad_norm=1.0,
        # 如果你当前 RslRlDistillationAlgorithmCfg 没有 loss_type 字段，就别加；
        # Distillation 里默认 loss_type="mse"
        # loss_type="mse",
    )
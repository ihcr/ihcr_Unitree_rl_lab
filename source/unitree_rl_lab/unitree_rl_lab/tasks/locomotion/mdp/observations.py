from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def curr_demo_dof_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current demonstration joint positions."""
    if not hasattr(env, '_curr_demo_dof_pos'):
        # Return zeros during initialization before motion lib is set up
        return torch.zeros(env.num_envs, 14, device=env.device)
    return env._curr_demo_dof_pos


def curr_demo_dof_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current demonstration joint velocities."""
    if not hasattr(env, '_curr_demo_dof_vel'):
        # Return zeros during initialization before motion lib is set up  
        return torch.zeros(env.num_envs, 14, device=env.device)
    return env._curr_demo_dof_vel


def next_demo_dof_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next step demonstration joint positions."""
    if not hasattr(env, '_next_demo_dof_pos'):
        # Return zeros during initialization before motion lib is set up
        return torch.zeros(env.num_envs, 14, device=env.device)
    return env._next_demo_dof_pos


def next_demo_dof_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next step demonstration joint velocities."""
    if not hasattr(env, '_next_demo_dof_vel'):
        # Return zeros during initialization before motion lib is set up  
        return torch.zeros(env.num_envs, 14, device=env.device)
    return env._next_demo_dof_vel



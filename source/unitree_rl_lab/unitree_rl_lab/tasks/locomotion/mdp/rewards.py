from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv





"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def foot_edge_alignment_soft(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ray_sensor: SceneEntityCfg,
    contact_sensor: SceneEntityCfg,
    h_jump_thresh: float = 0.05,   # 认为是“边”的高度跳变阈值
    safe_dist: float = 0.05,       # 离边的安全距离（越近惩罚）
    w_pos: float = 0.6,            # 远离边的上限奖励（适度，不要太大）
    w_neg: float = 1.2,            # 贴近边的二次惩罚系数
    search_radius: float = 0.30,   # 仅考虑脚附近的射线命中点
) -> torch.Tensor:
    """
    边缘对齐奖励（Soft 版）
    - 仅对 接触脚 计算；
    - 在脚一定半径内找“边”（通过近邻射线命中点的高度跳变判断）；
    - 与最近边的水平距离 dist_edge：
        * dist_edge >= safe_dist: 给予小额正奖励（偏向踩在平台面中央）
        * dist_edge <  safe_dist: 二次惩罚（避免踩在边缘）
    - 若邻域内未检测到边：返回 0（中性），避免过度保守，保障可上台阶。
    """
    # --- 取对象/传感器 ---
    robot: RigidObject = env.scene[asset_cfg.name]
    rc: RayCaster = env.scene.sensors[ray_sensor.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]
    feet_ids = asset_cfg.body_ids  # [F]

    # --- 脚状态（world）---
    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]      # [B,F,3]
    feet_xy = feet_pos_w[..., :2]                           # [B,F,2]
    feet_z  = feet_pos_w[..., 2]                            # [B,F]

    # 接触脚（stance）mask：只在接触期评估 edge
    # 更稳定的接触：按接触持续时间
    is_contact = (cs.data.current_contact_time[:, feet_ids] > 0).float()  # [B,F]

    # --- RayCaster 命中点（world）---
    hits = rc.data.ray_hits_w                                # [B,M,3]
    miss = getattr(rc.data, "miss_hits", None)               # [B,M] (bool)
    if miss is None:
        miss = torch.isnan(hits[..., 2])
    hit_xy = hits[..., :2]                                   # [B,M,2]
    hit_z  = torch.where(miss, torch.full_like(hits[..., 2], float('nan')), hits[..., 2])  # [B,M]

    B, F = feet_z.shape
    M = hit_z.shape[1]
    eps = 1e-6

    # --- 计算脚与射线点的距离，并筛选近邻 ---
    # dist: [B,F,M]
    dist = torch.norm(hit_xy.unsqueeze(1) - feet_xy.unsqueeze(2), dim=-1)
    near = (dist <= search_radius) & (~miss.unsqueeze(1))    # [B,F,M] 有效近邻

    # --- 以“近邻命中点的中位高度”为局部地形基准，找“边”候选 ---
    # 对每只脚 f，取其近邻命中点的中位数高度（忽略 NaN），作为局部平面参考
    # 先把 [B,F,M] 堆成 (B*F, M)，便于一次性求中位
    hit_z_bfm = hit_z.unsqueeze(1).expand(B, F, M)                  # [B,F,M]
    hit_z_bfm = torch.where(near, hit_z_bfm, torch.full_like(hit_z_bfm, float('nan')))
    hit_z_bfm_flat = hit_z_bfm.reshape(B * F, M)                    # [B*F, M]
    med_flat = torch.nanmedian(hit_z_bfm_flat, dim=1, keepdim=True).values  # [B*F,1]
    # 若某些脚附近全是 NaN，用脚本身高度兜底
    feet_z_flat = feet_z.reshape(B * F, 1)                           # [B*F,1]
    med_flat = torch.where(torch.isnan(med_flat), feet_z_flat, med_flat)
    med = med_flat.reshape(B, F, 1)                                  # [B,F,1]

    # 高度跳变：|hit_z - med| >= h_jump_thresh 即视作“疑似边”
    dz = torch.abs(hit_z.unsqueeze(1) - med)                         # [B,F,M]
    edge_cand = (dz >= h_jump_thresh) & near                         # [B,F,M]

    # --- 最近边的距离（没有边 → 设成很大，回到中性奖励 0）---
    big = torch.full_like(dist, 1e6)
    dist_edge = torch.where(edge_cand, dist, big).amin(dim=2)  # [B,F]
    # 若没有候选点，保持“很大”的距离
    dist_edge = torch.where(dist_edge >= 1e5,
                            torch.full_like(dist_edge, 10.0),
                            dist_edge)

    # --- 奖励/惩罚（Soft）---
    # 正项：离边越远，越接近平台中央，给最多 w_pos（再大也不再增加）
    pos_term = ( (dist_edge - safe_dist).clamp(min=0.0) / (safe_dist + eps) ).clamp(max=1.0) * w_pos
    # 负项：靠边（dist_edge < safe_dist）二次惩罚
    neg_term = ( (safe_dist - dist_edge).clamp(min=0.0) / (safe_dist + eps) ).pow(2) * w_neg
    rew_per_foot = pos_term - neg_term                           # [B,F]

    # 仅对接触脚有效；没有边（dist 很大）时 pos≈w_pos，但注意我们是以“检测到边”为前提：
    # 如果你更希望“没边→中性0”，可以乘以 has_edge 标志：
    has_edge = (edge_cand.any(dim=2)).float()                    # [B,F]
    rew_per_foot = rew_per_foot * has_edge

    # 平均到接触脚
    denom = is_contact.sum(dim=1) + eps
    rew = (rew_per_foot * is_contact).sum(dim=1) / denom         # [B]

    return rew

def approach_up_step_intent(
    env,
    ray_sensor: SceneEntityCfg,
    command_name: str = "base_velocity",
    v_pref: float = 0.3,
    w_yaw_pen: float = 0.1,
    h_jump_thresh: float = 0.04,
):
    import torch
    hits = env.scene.sensors[ray_sensor.name].data.ray_hits_w  # [B,M,3]
    hit_z = hits[..., 2]
    miss  = ~torch.isfinite(hit_z)
    hit_z = torch.where(miss, torch.full_like(hit_z, -10.0), hit_z)

    # 取传感器自身高度的近似（中位数更鲁棒）
    z_med = torch.median(hit_z, dim=1).values  # [B]

    # 是否“前方有较明显上台阶”（简单 proxy：最高点比中位数高出阈值）
    up_present = ((hit_z.max(dim=1).values - z_med) >= h_jump_thresh).float()  # [B]

    # 命令（或者直接用 base_lin_vel_x 也可以）
    cmd = env.command_manager.get_command(command_name)
    vx_cmd = cmd[:, 0]                # 期望前进
    yaw_cmd= cmd[:, 2]

    # 实际速度（根坐标 z 方向不关心）
    base_vel = env.scene["robot"].data.root_lin_vel_b   # [B,3] 机体坐标
    vx = base_vel[:, 0]
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2].abs()

    # 引导：上阶存在时，提高正向，抑制负向/大转向
    pos = torch.relu(vx) / (v_pref + 1e-6)
    neg = torch.relu(-vx) / (v_pref + 1e-6) + w_yaw_pen * yaw_rate
    return up_present * (pos - neg)

def foot_edge_alignment_dense_upaware(
    env,
    asset_cfg: SceneEntityCfg,
    ray_sensor: SceneEntityCfg,
    contact_sensor: SceneEntityCfg,
    # 感知窗口（比原来更短更窄，避免一次扫到多级台阶）
    search_radius: float = 0.22,
    # 边缘判定
    h_jump_thresh: float = 0.04,
    # 下沿安全距离（严一些，防踩空）
    safe_dist: float = 0.06,
    # 上沿“理想落点”的距离窗（踏面深度~0.20 时，取中间 0.08~0.14）
    tread_pref: tuple[float, float] = (0.08, 0.14),
    # 权重
    w_pos: float = 0.6,     # 上沿对齐奖励
    w_neg: float = 0.8,     # 下沿贴近惩罚（比你之前的 1.5 温和）
    # 形状（平滑）
    tau: float = 0.01,      # 惩罚软阈值
    sigma: float = 0.03,    # 上沿高斯窗宽度
):
    import torch
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import ContactSensor

    robot: RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]
    rc = env.scene.sensors[ray_sensor.name]

    feet_ids = asset_cfg.body_ids
    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]     # [B,F,3]
    feet_xy = feet_pos_w[..., :2]                          # [B,F,2]
    feet_z  = feet_pos_w[..., 2]                           # [B,F]
    stance  = (cs.data.current_contact_time[:, feet_ids] > 0).float()  # [B,F]

    hits = rc.data.ray_hits_w                               # [B,M,3]（你的传感器就是这个）
    hit_xy = hits[..., :2]                                  # [B,M,2]
    hit_z  = hits[...,  2]                                  # [B,M]
    miss   = ~torch.isfinite(hit_z)                         # [B,M]
    hit_z  = torch.where(miss, torch.full_like(hit_z, -10.0), hit_z)

    B, F = feet_z.shape
    M    = hit_z.shape[1]
    big  = torch.tensor(1e6, device=feet_z.device)

    rews = []
    for f in range(F):
        fx = feet_xy[:, f, :]                  # [B,2]
        fz = feet_z[:, f]                      # [B]
        # 与脚的相对距离
        dist = torch.norm(hit_xy - fx.unsqueeze(1), dim=-1)          # [B,M]
        in_rng = (dist <= search_radius) & (~miss)                    # [B,M]

        # 脚下参考高度：脚下小圆（<= 6cm）内点的中位数；没有则退回 fz
        near = (dist <= 0.06) & (~miss)
        z_under = torch.nanmedian(torch.where(near, hit_z, torch.nan), dim=1).values
        z_under = torch.where(torch.isfinite(z_under), z_under, fz)

        # 上沿/下沿候选
        up_cand   = in_rng & ((hit_z - z_under.unsqueeze(1)) >= h_jump_thresh)
        down_cand = in_rng & ((z_under.unsqueeze(1) - hit_z) >= h_jump_thresh)

        # 下沿最近距离 -> 近则惩罚
        d_down = torch.where(down_cand, dist, big).amin(dim=1)             # [B]
        down_pen = (torch.nn.functional.softplus((safe_dist - d_down)/(tau+1e-6))**2) * w_neg

        # 上沿最近距离 -> 奖励落在踏面中部（高斯窗）
        d_up   = torch.where(up_cand, dist, big).amin(dim=1)               # [B]
        d1, d2 = tread_pref
        d_star = 0.5 * (d1 + d2)
        up_bonus = torch.exp(-0.5 * ((d_up - d_star)/(sigma+1e-6))**2) * w_pos
        # 若没有上沿候选，则 up_bonus=0
        up_bonus = torch.where(d_up < big*0.5, up_bonus, torch.zeros_like(up_bonus))

        # 只在接触脚有效
        rew_f = (up_bonus - down_pen) * stance[:, f]
        rews.append(rew_f)

    return torch.stack(rews, dim=1).sum(dim=1)  # [B]



    
def foot_edge_alignment_dense(
    env,
    asset_cfg: SceneEntityCfg,
    ray_sensor: SceneEntityCfg,
    contact_sensor: SceneEntityCfg,
    h_jump_thresh: float = 0.045,   # ≥4.5cm 视为落差/边
    safe_dist: float = 0.05,        # 离踏鼻至少 5cm
    w_pos: float = 0.3,             # 正对齐奖励
    w_neg: float = 1.5,             # 误踩边惩罚
    search_radius: float = 0.30,    # 搜索半径
) -> torch.Tensor:
    """
    边缘对齐奖励（Dense版）：
      - 仅在支撑脚（接触）计算
      - 脚尖附近的台阶边缘对齐时奖励，踩在边缘外/过近惩罚
      - 使用高密度 height_scanner，减少稀疏
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]
    feet_ids = asset_cfg.body_ids

    # 脚部位置和接触状态
    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]      # [B,F,3]
    feet_xy = feet_pos_w[..., :2]                           # [B,F,2]
    contact_time = cs.data.current_contact_time[:, feet_ids]  # [B,F]
    stance_mask = (contact_time > 0).float()                # 只保留接触脚

    # 查询 height_scanner 数据
    scan_heights = env.scene.sensors[ray_sensor.name].data  # [B, num_rays]
    scan_xy = env.scene.sensors[ray_sensor.name].pattern.points_xy  # 射线在 robot local frame 的 XY

    # 遍历每只脚
    rew_list = []
    for f in range(feet_ids.shape[0]):
        foot_xy = feet_xy[:, f, :]  # [B,2]
        # 找到 search_radius 范围内的射线索引
        dist_xy = torch.norm(scan_xy.unsqueeze(0) - foot_xy.unsqueeze(1), dim=2)  # [B,num_rays]
        in_range = dist_xy < search_radius

        # 取这些射线的高度差
        heights_near = torch.where(in_range, scan_heights, torch.nan)  # [B,num_rays]
        h_min = torch.nanmin(heights_near, dim=1).values
        h_max = torch.nanmax(heights_near, dim=1).values
        h_jump = (h_max - h_min)  # 落差

        # 判断是否为边缘
        is_edge = (h_jump > h_jump_thresh).float()

        # 计算脚与最近边缘的水平距离
        min_dist = torch.where(in_range, dist_xy, torch.inf).min(dim=1).values
        dist_good = (min_dist >= safe_dist).float()

        # 奖励/惩罚
        rew_f = w_pos * (is_edge * dist_good) - w_neg * (is_edge * (1 - dist_good))
        rew_list.append(rew_f)

    rew_all = torch.stack(rew_list, dim=1)  # [B,F]
    rew_all = rew_all * stance_mask  # 只对接触脚有效

    return torch.sum(rew_all, dim=1)  # [B]

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def foot_clearance_terrain_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,            # 脚 link
    ground_sensor_cfg: SceneEntityCfg,    # RayCaster
    contact_sensor_cfg: SceneEntityCfg,   # ContactSensor
    margin: float = 0.10,                 # 相对地形的抬脚高度
    std: float = 0.05,
    tanh_mult: float = 3.0,
) -> torch.Tensor:
    """
    在楼梯/不平地形上奖励抬脚高度（相对当前地形），仅在摆动期生效。
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    ray: RayCaster = env.scene[ground_sensor_cfg.name]
    contact: ContactSensor = env.scene[contact_sensor_cfg.name]

    # === 地形高度（用中位数更稳） ===
    hits_z = ray.data.ray_hits_w[..., 2]                            # [B, Nrays]
    hits_z = torch.where(torch.isfinite(hits_z), hits_z, torch.nan)
    terrain_z = torch.nanmedian(hits_z, dim=1).values               # [B]

    # === 目标高度 ===
    target = terrain_z[:, None] + margin                            # [B, 1]

    # === 脚的世界高度 ===
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]        # [B, Nfeet]

    # === 是否在支撑期 ===
    in_contact = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    is_swing = (~in_contact).float()                                # [B, Nfeet]

    # === 误差（脚高度 vs 目标） ===
    err = foot_z - target                                           # [B, Nfeet]

    # === 水平速度 gating（避免慢慢抬脚蹭） ===
    foot_v_xy = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    speed_gate = torch.tanh(tanh_mult * foot_v_xy)

    # === 高斯核奖励 ===
    rew_per_foot = torch.exp(-(err * err) / (2.0 * (std * std))) * is_swing * speed_gate

    return rew_per_foot.mean(dim=1)   


def foot_clearance_rel_terrain(
    env,
    asset_cfg: SceneEntityCfg,
    ray_sensor: SceneEntityCfg,
    contact_sensor: SceneEntityCfg,
    safety_margin: float = 0.03,
    target_clear: float = 0.06,
    std: float = 0.05,
    tanh_mult: float = 2.0,
    c_under: float = 50.0,
    c_over: float = 1.0,
    num_scan_pts: int = 5,
    ahead: float = 0.20,
    half_width: float = 0.06,
) -> torch.Tensor:
    """地形感知抬脚净空奖励"""
    robot: RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]
    feet_ids = asset_cfg.body_ids

    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]      
    feet_xy = feet_pos_w[..., :2]                           
    foot_z = feet_pos_w[..., 2]                             
    foot_vel_xy = robot.data.body_lin_vel_w[:, feet_ids, :2]    

    # 精确接触判定
    contact_force = torch.norm(cs.data.net_forces_w[:, feet_ids, :2], dim=2)
    is_contact = (contact_force > 1.0).float()
    swing_mask = 1.0 - is_contact

    # 前方地形高度
    terrain_z_fwd = _terrain_z_forward_from_scan(env, ray_sensor, feet_xy,
                                                 num_scan_pts=num_scan_pts,
                                                 ahead=ahead,
                                                 half_width=half_width)
    terrain_z_fwd = torch.nan_to_num(terrain_z_fwd, nan=foot_z.min())  # 防 NaN

    # 净空 margin
    margin = foot_z - terrain_z_fwd - safety_margin

    # 低于地形 -> 二次惩罚
    cost_under = (-margin).clamp(min=0.0).pow(2) * c_under
    # 超过理想净空 -> 平方惩罚
    cost_over  = ((margin - target_clear).clamp(min=0.0) / (target_clear + 1e-6)).pow(2) * c_over
    cost = (cost_under + cost_over) * swing_mask

    # 速度调制
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_xy, dim=2))
    cost = cost * foot_velocity_tanh

    return torch.exp(-torch.sum(cost, dim=1) / (std + 1e-6))


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def contact_event_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    feet_body_ids: list[int] = None,
    min_normal_force: float = 0.0,   # 若用 current_contact_time>0，可保持为0
    stable_w: float = 0.4,
    touchdown_w: float = 0.6,
    state_key: str = "prev_contact_buffer",
) -> torch.Tensor:
    """
    事件驱动接触奖励：至少一脚在地 + 落地事件（上一帧空中，这一帧触地）。
    与 _reward_contact 逻辑一致，但使用 ContactSensor 的 current_contact_time。
    """
    device = env.device
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 脚的索引
    if feet_body_ids is None:
        feet_body_ids = sensor_cfg.body_ids  # 默认用配置里的 body_ids

    # 当前接触（bool）: 用 contact_time>0 更稳；若要按法向力阈值，可改为 forces[...,2] > min_normal_force
    is_contact = (cs.data.current_contact_time[:, feet_body_ids] > 0)  # [N, L] bool

    # 取上一帧缓存；若无则初始化为 False
    if not hasattr(env, state_key):
        setattr(env, state_key, torch.zeros_like(is_contact, dtype=torch.bool, device=device))
    prev_contact = getattr(env, state_key)  # [N, L] bool

    # touchdown: 上一帧 False，这一帧 True
    touchdown = is_contact & (~prev_contact)                 # [N, L] bool
    touchdown_reward = touchdown.float().sum(dim=1)          # [N]

    # 稳定：至少一脚接触
    stable_contact = is_contact.any(dim=1).float()           # [N]

    # 组合
    reward = stable_w * stable_contact + touchdown_w * touchdown_reward

    # 更新缓存
    setattr(env, state_key, is_contact.clone())

    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def foot_edge_alignment(
    env,
    asset_cfg: SceneEntityCfg,
    ray_sensor: SceneEntityCfg,
    contact_sensor: SceneEntityCfg,
    h_jump_thresh: float = 0.05,   # 稍微保守一点，先 5cm
    safe_dist: float = 0.05,       # 与边至少 5cm
    w_pos: float = 0.6,            # 远离边的奖励上限
    w_neg: float = 1.2,            # 过近的二次惩罚
    search_radius: float = 0.30,   # 搜索脚附近 30cm 的射线点
):
    """仅在接触脚上奖励“远离边缘”（高度突变区域）"""

    # --- 取状态（世界系）---
    robot: RigidObject = env.scene[asset_cfg.name]
    rc: RayCaster = env.scene.sensors[ray_sensor.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]

    feet_ids = asset_cfg.body_ids                              # [F]
    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]         # [B,F,3]
    feet_xy = feet_pos_w[..., :2]                              # [B,F,2]
    feet_z  = feet_pos_w[..., 2]                               # [B,F]
    is_contact = (cs.data.current_contact_time[:, feet_ids] > 0).float()  # [B,F]

    device = feet_z.device

    # --- RayCaster 命中点（世界系）---
    hits = rc.data.ray_hits_w.to(device)                       # [B,M,3]
    miss = torch.isnan(hits[..., 2])                           # [B,M]
    hit_xy = hits[..., :2]                                     # [B,M,2]
    # miss 的 z 用一个很低的地面值替换，避免 NaN 传播（且不会被选作“边”）
    hit_z  = torch.where(miss, torch.full_like(hits[..., 2], -10.0), hits[..., 2]).to(device)

    B, F = feet_z.shape
    M = hit_z.shape[1]

    # --- 只看脚附近的射线点 ---
    # dist: [B,F,M]
    dist = torch.norm(hit_xy.unsqueeze(1) - feet_xy.unsqueeze(2), dim=-1)
    near_mask = dist <= search_radius                          # [B,F,M]

    # --- “边”候选：与脚高差超过阈值的近邻点 ---
    dz = torch.abs(hit_z.unsqueeze(1) - feet_z.unsqueeze(2))   # [B,F,M]
    edge_cand = near_mask & (dz >= h_jump_thresh) & (~miss.unsqueeze(1))  # [B,F,M]

    # --- 取最近的边缘距离 ---
    big = torch.full_like(dist, 1e6)
    dist_edge = torch.where(edge_cand, dist, big).amin(dim=2)            # [B,F]
    # 没候选当作很远
    dist_edge = torch.where(dist_edge >= 1e5, torch.full_like(dist_edge, 10.0), dist_edge)

    # --- 奖励：远离边正奖励，过近二次惩罚（仅接触脚）---
    pos_term = (dist_edge / (safe_dist + 1e-6)).clamp(max=1.0) * w_pos
    neg_term = ((safe_dist - dist_edge).clamp(min=0.0) / (safe_dist + 1e-6)).pow(2) * w_neg
    rew_per_foot = pos_term - neg_term                         # [B,F]

    # 只在接触脚上平均；若该步没有任何接触脚，则奖励置 0
    contact_count = is_contact.sum(dim=1)                      # [B]
    rew = torch.where(
        contact_count > 0,
        (rew_per_foot * is_contact).sum(dim=1) / (contact_count + 1e-6),
        torch.zeros(B, device=device),
    )
    return rew


def base_height_rel_support_l2(
    env,
    target_offset: float,
    contact_sensor: SceneEntityCfg,
    ray_sensor: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    near_radius: float = 0.25,
):
    """
    让根部高度贴近“支撑面高度 + target_offset”的 L2 惩罚。
    支撑面高度优先用接触脚(stance)的脚底 z；若无接触脚，则回退用 raycast 在脚附近估计地形高度。
    """
    # 1) 取机器人和传感器
    robot: RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[contact_sensor.name]
    feet_ids = asset_cfg.body_ids  # 你在 RewTerm 里把 ankle_roll.* 传进来

    # 2) 脚的世界坐标和接触状态
    feet_pos_w = robot.data.body_pos_w[:, feet_ids, :]          # [B,F,3]
    foot_z = feet_pos_w[..., 2]                                 # [B,F]
    contact = (cs.data.current_contact_time[:, feet_ids] > 0)   # [B,F] bool
    contact_f = contact.float()
    has_contact = contact.any(dim=1)                            # [B]

    # 3) 首选：用接触脚的平均高度做支撑面
    support_z_contact = (foot_z * contact_f).sum(dim=1) / (contact_f.sum(dim=1) + 1e-6)  # [B]

    support_z = support_z_contact.clone()

    # 4) 回退：无接触时用 raycast 估计脚下/脚前地形高度
    if ray_sensor is not None and (~has_contact).any():
        rc: RayCaster = env.scene.sensors[ray_sensor.name]
        hits = rc.data.ray_hits_w[..., :3]                      # [B,M,3]
        miss = getattr(rc.data, "miss_hits", None)
        if miss is None:
            miss = torch.isnan(hits[..., 2])                    # [B,M]
        hit_xy = hits[..., :2]                                  # [B,M,2]
        hit_z  = torch.where(miss, torch.full_like(hits[..., 2], -10.0), hits[..., 2])  # miss 给很低

        # 取所有脚的 xy（平均一下当“机器人足下区域”中心）
        feet_xy_mean = feet_pos_w[..., :2].mean(dim=1, keepdim=True)  # [B,1,2]
        dist = torch.norm(hit_xy - feet_xy_mean, dim=-1)              # [B,M]
        near = dist <= near_radius
        # median 更稳健
        z_near = torch.where(near, hit_z, torch.nan)
        z_est  = torch.nanmedian(z_near, dim=1).values                 # [B]
        # 无 near 命中时兜底：用当前 contact 支撑（其实该分支也不会触发，因为 has_contact=False）
        z_est  = torch.where(torch.isfinite(z_est), z_est, support_z_contact)
        support_z = torch.where(has_contact, support_z_contact, z_est)

    # 5) 计算目标高度与 L2
    root_z = robot.data.root_pos_w[:, 2]                         # [B]
    target_z = support_z + target_offset
    err = root_z - target_z
    return err * err  # [B]



@torch.no_grad()
def _terrain_z_forward_from_scan(env, ray_sensor, feet_xy: torch.Tensor,
                                 num_scan_pts: int = 5, ahead: float = 0.20, half_width: float = 0.06) -> torch.Tensor:
    """
    用 RayCaster 网格命中点近似估计“脚前方落脚区域”的地形高度（取邻域最高值，保守）。
    feet_xy: [B, F, 2] 世界坐标
    返回: [B, F] 估计的前方地形 z
    """
    rc = env.scene.sensors[ray_sensor.name]
    hits = rc.data.ray_hits_w              # [B,M,3]
    miss = getattr(rc.data, "miss_hits", None)
    if miss is None:
        miss = torch.isnan(hits[..., 2])   # [B,M]
    hit_xy = hits[..., :2]                 # [B,M,2]
    hit_z  = torch.where(miss, torch.full_like(hits[..., 2], -10.0), hits[..., 2])  # miss 置低

    B, F, _ = feet_xy.shape
    M = hit_xy.shape[1]

    # 仅考虑脚前方一个长方形邻域（沿机器人前进方向近似，用“最近点”近似也够稳）
    # 简化：以欧氏距离最近的 K 条射线作为“前方候选”，取其最大 z
    centers = feet_xy.unsqueeze(2).expand(B, F, M, 2)      # [B,F,M,2]
    d2 = torch.sum((hit_xy.unsqueeze(1) - centers) ** 2, dim=-1)  # [B,F,M]
    topk = torch.topk(-d2, k=min(num_scan_pts, M), dim=-1).indices    # 最小距离

    idx_b = torch.arange(B, device=env.device)[:, None, None]
    sel_z = hit_z[idx_b, topk]     # [B,F,K]
    z_forward = sel_z.max(dim=-1).values  # [B,F]
    return z_forward
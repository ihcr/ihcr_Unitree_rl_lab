from isaaclab_rl.rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.unitree_rl_lab.utils.actor_critic_scan import ActorCriticScan

class OnPolicyRunnerScan(OnPolicyRunner):
    """与原 OnPolicyRunner 一样，但覆盖建模步骤，接入 ActorCriticScan。"""

    def _build_policy(self):
        # === 从已构建的 env/manager 中拿维度 ===
        obs_mgr = self.obs_mgr  # IsaacLab 的 observation manager
        act_mgr = self.action_mgr

        policy_obs_dim = obs_mgr.group_obs_dim["policy"]
        critic_obs_dim = obs_mgr.group_obs_dim.get("critic", policy_obs_dim)
        act_dim = act_mgr.num_actions

        # 你的 height_scanner 在 obs 的末尾，长度 = 187
        scan_dim = 187
        # 如果以后想从 manager 自动拿 term 维度，可以用：
        # scan_dim = obs_mgr.get_term_dim("policy", "height_scanner")

        # === 建你的自定义 ActorCritic ===
        ac = ActorCriticScan(
            obs_dim=policy_obs_dim,
            act_dim=act_dim,
            scan_dim=scan_dim,
            critic_obs_dim=critic_obs_dim,
        )

        # 把自定义网络塞回算法对象，保持其他 PPO 超参不变
        self.algo.actor_critic = ac
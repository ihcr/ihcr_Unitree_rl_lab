# actor_critic_scan.py
import torch
import torch.nn as nn
# actor_critic_scan.py 顶部
try:
    from rsl_rl.modules.actor_critic import ActorCritic   # rsl-rl 2.x
except ImportError:
    from rsl_rl.models.actor_critic import ActorCritic    # rsl-rl 1.x 旧版兼容

try:
    from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent   # rsl-rl 2.x
except ImportError:
    from rsl_rl.models.actor_critic_recurrent import ActorCriticRecurrent    # rsl-r

_ACTS = {

    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,   # 仅用于输出激活
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}

def _mlp(in_dim, hidden, act_name="elu", out_act=None):
    layers = []
    Act = _ACTS.get(act_name, nn.ELU)
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), Act()]
        last = h
    if out_act is not None:
        layers += [ _ACTS[out_act]() ]
    return nn.Sequential(*layers), last

try:
    from rsl_rl.modules.actor_critic import ActorCritic  # 2.x
except ImportError:
    from rsl_rl.models.actor_critic import ActorCritic   # 1.x

# class ActorCriticScan(ActorCritic):
class ActorCriticScanRecurrent(ActorCriticRecurrent):

    def __init__(self, num_obs, num_privileged_obs, num_actions, **kwargs):
        # ---- 扫描相关的固定参数 ----
        self.single_scan_dim = 187     # 单帧高度扫描长度
        self.scan_out_dim    = 32
        self.scan_hidden     = [128, 64]

        # 先设默认（只取最后一帧）
        self.history_length  = 5
        self._scan_start     = None    # 等看到真实 obs 再算
        self.scan_dim_total  = self.single_scan_dim  # 先用单帧

        # 父类网络的有效输入维度 = num_obs - 187 + 32  (只替换最后一帧)
        eff_num_obs = num_obs - self.single_scan_dim + self.scan_out_dim
        eff_num_privileged_obs = num_privileged_obs - self.single_scan_dim + self.scan_out_dim
        if eff_num_obs <= 0:
            raise ValueError(f"num_obs({num_obs}) must be > single_scan_dim({self.single_scan_dim}).")

        super().__init__(eff_num_obs, eff_num_privileged_obs, num_actions, **kwargs)

        # 扫描编码器
        layers = []
        in_dim = self.single_scan_dim
        for h in self.scan_hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, self.scan_out_dim), nn.Tanh()]
        self.scan_encoder = nn.Sequential(*layers)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # 假设在 __init__ 里：self.single_scan_dim = 187, self.scan_encoder 输出 32 维
        D = obs.shape[-1]
        scan_dim = self.single_scan_dim
        if D < scan_dim:
            raise RuntimeError(f"obs_dim({D}) < scan_dim({scan_dim})，检查 height-scan 是否拼在观察末尾。")

        scan_start = D - scan_dim
        state = obs[..., :scan_start]        # [B, D-187]
        scan  = obs[..., scan_start:]        # [B, 187]

        # 设备与 dtype 对齐（以 encoder 为准）
        dev = next(self.scan_encoder.parameters()).device
        dtype = next(self.scan_encoder.parameters()).dtype
        state = state.to(device=dev, dtype=dtype)
        scan  = scan.to(device=dev, dtype=dtype)

        feat = self.scan_encoder(scan)       # [B, 32]
        return torch.cat([state, feat], dim=-1)  # [B, D-187+32]

    def act(self, obs, **kwargs):
        return super().act(self._encode_obs(obs), **kwargs)

    def evaluate(self, obs, **kwargs):
        return super().evaluate(self._encode_obs(obs), **kwargs)

    def act_inference(self, obs):
        return super().act_inference(self._encode_obs(obs))
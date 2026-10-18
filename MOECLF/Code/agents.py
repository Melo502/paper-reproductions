"""强化学习层：FSM 状态编码、Q-learning、PPO(MobileNetV2 + ECA + Actor-Critic)。

统一动作空间 = 10 个 LLH；统一奖励 = 支配关系(式38)。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ehfsp_hmc import Instance, Schedule

N_ACTIONS = 10  # LLH1..LLH10


# --------------------------------------------------------------------------
# FSM 状态编码 (Fig 7)
# --------------------------------------------------------------------------


def build_fsm(inst: Instance, sch: Schedule, obj: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """构建三通道 FSM，pad 到固定 (3, rows, cols)。

    通道0: job code (job+1)/I  —— 机器/工人的操作路线；
    通道1: 加工时间 / Cmax（机器 t_hat+p_hat，工人 t_hat）；
    通道2: 能耗 / TEC。
    """
    fsm = np.zeros((3, rows, cols), dtype=np.float32)
    n_m = inst.n_machines()
    for r in range(n_m):
        for p, (job, time, energy) in enumerate(sch.mach_ops[r]):
            if p < cols:
                fsm[0, r, p] = (job + 1) / inst.I
                fsm[1, r, p] = time / (obj[0] + 1e-8)
                fsm[2, r, p] = energy / (obj[1] + 1e-8)
    for r in range(inst.n_workers()):
        row = n_m + r
        for p, (job, time, energy) in enumerate(sch.worker_ops[r]):
            if p < cols:
                fsm[0, row, p] = (job + 1) / inst.I
                fsm[1, row, p] = time / (obj[0] + 1e-8)
                fsm[2, row, p] = energy / (obj[1] + 1e-8)
    return fsm


# --------------------------------------------------------------------------
# ECA 通道注意力 (式31-33)
# --------------------------------------------------------------------------


class ECAModule(nn.Module):
    """高效通道注意力：GAP -> 1D 卷积(自适应核) -> sigmoid -> 逐通道加权。"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 == 1 else t + 1  # 最近奇数 (式31)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = x.mean(dim=(2, 3), keepdim=True)   # GAP -> (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)    # (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y


# --------------------------------------------------------------------------
# MobileNetV2 + ECA + Actor/Critic 头
# --------------------------------------------------------------------------


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块，末尾嵌入 ECA（官方结构）。"""

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        layers.append(ECAModule(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class FSMPolicy(nn.Module):
    """手写 MobileNetV2 + 每块嵌 ECA + Actor/Critic 头（官方结构）。"""

    def __init__(self, n_actions: int = N_ACTIONS, dropout: float = 0.2, width_mult: float = 1.0):
        super().__init__()
        last_channel = int(1280 * width_mult)
        # 官方：actor/critic 各为 Dropout + 单个 Linear（无中间 128 层）
        self.actor = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_channel, n_actions))
        self.critic = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_channel, 1))

        features = [ConvBNReLU(3, 32, stride=2)]  # 3 通道输入
        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = 32
        for t, c, n, s in setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # 权重初始化（官方）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fsm: torch.Tensor):
        x = self.features(fsm)
        x = x.mean(-1).mean(-1)  # 全局平均池化
        return self.actor(x), self.critic(x)


# --------------------------------------------------------------------------
# Q-learning (LLH state, 式29-30)
# --------------------------------------------------------------------------


class QLearningAgent:
    """状态 = 上一个 LLH(0..n-1)，动作 = 下一个 LLH(0..n-1)。Q 表 n x n。"""

    def __init__(self, alpha: float = 0.15, gamma: float = 0.98, eps: float = 0.4, n_actions: int = N_ACTIONS):
        self.n_actions = n_actions
        self.Q = np.zeros((n_actions, n_actions), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.state = None

    def reset_state(self, rng: np.random.Generator):
        self.state = int(rng.integers(0, self.n_actions))

    def act(self, rng: np.random.Generator) -> int:
        if rng.random() < self.eps:
            return int(rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[self.state]))

    def update(self, action: int, reward: float):
        nxt = action  # 下一个 LLH 状态 = 本次动作
        self.Q[self.state, action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[nxt]) - self.Q[self.state, action]
        )
        self.state = nxt


# --------------------------------------------------------------------------
# PPO (FSM state, 式34-37 + GAE)
# --------------------------------------------------------------------------


class PPOAgent:
    """on-policy PPO，Actor-Critic 骨干为 FSMPolicy。经验经 GAE 后 clip 更新。"""

    def __init__(
        self,
        fsm_rows: int,
        fsm_cols: int,
        alpha: float = 0.001,
        gamma: float = 0.98,
        lam: float = 0.98,
        clip: float = 0.2,
        c_theta: float = 2.0,
        c_phi: float = 1.0,
        c_entropy: float = 0.01,
        dropout: float = 0.2,
        device: str = "cpu",
        n_actions: int = N_ACTIONS,
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.c_theta = c_theta
        self.c_phi = c_phi
        self.c_entropy = c_entropy
        self.device = device
        self.policy = FSMPolicy(n_actions=n_actions, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.buffer: list[dict] = []

    @torch.no_grad()
    def act(self, fsm: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
        s = torch.from_numpy(fsm).float().unsqueeze(0).to(self.device)
        logits, value = self.policy(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def store(self, s, a, r, s_next, logprob, done):
        self.buffer.append(dict(s=s, a=a, r=r, s_next=s_next, logprob=logprob, done=done))

    def update(self, epochs: int = 10, batch_size: int = 128):
        buf = self.buffer
        if not buf:
            return
        n = len(buf)
        S = torch.stack([torch.from_numpy(x["s"]) for x in buf]).float().to(self.device)
        Sn = torch.stack([torch.from_numpy(x["s_next"]) for x in buf]).float().to(self.device)
        with torch.no_grad():
            _, V = self.policy(S)
            _, Vn = self.policy(Sn)
        V = V.squeeze(-1).cpu().numpy()
        Vn = Vn.squeeze(-1).cpu().numpy()

        adv = np.zeros(n)
        ret = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if buf[t]["done"] else Vn[t]
            delta = buf[t]["r"] + self.gamma * next_val - V[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
            ret[t] = gae + V[t]

        adv_t = torch.tensor(adv).float().to(self.device)
        ret_t = torch.tensor(ret).float().to(self.device)
        actions = torch.tensor([x["a"] for x in buf]).long().to(self.device)
        old_lp = torch.tensor([x["logprob"] for x in buf]).float().to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                logits, v_est = self.policy(S[b])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions[b])
                ratio = torch.exp(new_lp - old_lp[b])

                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[b]
                loss_theta = -torch.min(surr1, surr2).mean()          # 式(34)
                loss_entropy = -dist.entropy().mean()                  # 式(35)
                loss_phi = F.mse_loss(v_est.squeeze(-1), ret_t[b])     # 式(36)

                loss = (
                    self.c_theta * loss_theta
                    + self.c_entropy * loss_entropy
                    + self.c_phi * loss_phi                             # 式(37)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer = []

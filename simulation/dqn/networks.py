"""Q 网络架构: MLP / Dueling MLP / Transformer / Enhanced (NoisyNet)。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class IntraQNetwork(nn.Module):
    """基础 Q 网络 (3 层 MLP)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntraDuelingQNetwork(nn.Module):
    """Dueling Q 网络 (3层 MLP + LayerNorm): Q(s,a) = V(s) + A(s,a) - mean(A)"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class PerActionDuelingQNetwork(nn.Module):
    """Per-Action Dueling Q 网络.

    每个邻居独立编码后与全局特征拼接，生成各自的 Advantage，
    避免 flat MLP 中不同邻居特征的伪相关。推理速度接近原版 MLP。

    结构:
      base_feat (base_dim)  → base_encoder → base_embed (embed_dim)
      nb_feat_i (nb_feat_dim) → nb_encoder → nb_embed_i (embed_dim)  (共享权重)
      concat(base_embed, nb_embed_i) → action_head → A(s, a_i)
      base_embed → value_head → V(s)
      Q(s,a_i) = V(s) + A(s,a_i) - mean(A)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 base_dim: int = 5, nb_feat_dim: int = 7,
                 embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.base_dim = base_dim
        self.nb_feat_dim = nb_feat_dim

        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.nb_encoder = nn.Sequential(
            nn.Linear(nb_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        base_feat = x[:, :self.base_dim]
        nb_feat = x[:, self.base_dim:].view(B, self.action_dim, self.nb_feat_dim)

        base_embed = self.base_encoder(base_feat)                    # (B, embed)
        nb_embed = self.nb_encoder(nb_feat)                          # (B, K, embed)

        base_expand = base_embed.unsqueeze(1).expand_as(nb_embed)    # (B, K, embed)
        pair = torch.cat([base_expand, nb_embed], dim=-1)            # (B, K, embed*2)

        advantage = self.action_head(pair).squeeze(-1)               # (B, K)
        value = self.value_head(base_embed)                          # (B, 1)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ContextPerActionDuelingQNetwork(nn.Module):
    """Permutation-equivariant Per-Action network with set context.

    The original Per-Action head scores each neighbor independently.  This
    variant keeps the shared neighbor encoder but augments every action with
    a permutation-invariant mean summary of the complete valid-neighbor
    set, allowing Top-1 decisions to depend on relative candidate quality.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 base_dim: int = 5, nb_feat_dim: int = 7,
                 embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.base_dim = base_dim
        self.nb_feat_dim = nb_feat_dim

        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.nb_encoder = nn.Sequential(
            nn.Linear(nb_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        # base + current neighbor + mean-pool
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # State value also observes the invariant neighbor-set summary.
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        base_feat = x[:, :self.base_dim]
        nb_feat = x[:, self.base_dim:].view(
            batch_size, self.action_dim, self.nb_feat_dim
        )
        # Padding has normalized degree 0; every real neighbor has degree > 0.
        valid = nb_feat[:, :, 2] > 0
        valid_f = valid.unsqueeze(-1).to(nb_feat.dtype)
        valid_count = valid_f.sum(dim=1).clamp_min(1.0)

        base_embed = self.base_encoder(base_feat)
        nb_embed = self.nb_encoder(nb_feat)
        mean_context = (nb_embed * valid_f).sum(dim=1) / valid_count

        base_expand = base_embed.unsqueeze(1).expand_as(nb_embed)
        mean_expand = mean_context.unsqueeze(1).expand_as(nb_embed)
        action_input = torch.cat(
            [base_expand, nb_embed, mean_expand], dim=-1
        )
        advantage = self.action_head(action_input).squeeze(-1)

        advantage_mean = (
            (advantage * valid.to(advantage.dtype)).sum(dim=1, keepdim=True)
            / valid.sum(dim=1, keepdim=True).clamp_min(1)
        )
        value_input = torch.cat([base_embed, mean_context], dim=-1)
        value = self.value_head(value_input)
        return value + advantage - advantage_mean


class IntraTransformerDuelingQNetwork(nn.Module):
    """Transformer-based Dueling Q Network.

    将每个邻居视为一个 token，用 self-attention 学习邻居间的比较关系。
    全局特征（距离、跳数比例、方向向量）作为 [CLS] token 提供上下文。

    结构:
      [CLS_token, nb_token_0, ..., nb_token_{k-1}] → Transformer Encoder
      → CLS_out → V(s)
      → nb_out_i → A(s, a_i)
      → Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    BASE_DIM = 5
    NB_FEAT_DIM = 7

    def __init__(self, state_dim: int, action_dim: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.action_dim = action_dim

        self.base_proj = nn.Sequential(
            nn.Linear(self.BASE_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.nb_proj = nn.Sequential(
            nn.Linear(self.NB_FEAT_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + action_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.advantage_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        base_feat = x[:, :self.BASE_DIM]
        nb_feat = x[:, self.BASE_DIM:].view(B, self.action_dim, self.NB_FEAT_DIM)

        cls_token = self.base_proj(base_feat).unsqueeze(1)
        nb_tokens = self.nb_proj(nb_feat)
        tokens = torch.cat([cls_token, nb_tokens], dim=1)
        tokens = tokens + self.pos_embed

        out = self.transformer(tokens)

        value = self.value_stream(out[:, 0])
        advantage = self.advantage_fc(out[:, 1:]).squeeze(-1)

        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# NoisyNet + Enhanced Dueling (用于 EnhancedD3QNAgent)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet 线性层.

    替代 ε-greedy: 网络权重本身带有可学习噪声参数,
    训练过程中自适应调节探索强度.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class EnhancedDuelingQNetwork(nn.Module):
    """增强版 Dueling Q 网络 (NoisyLinear + LayerNorm)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, hidden // 2),
            nn.ReLU(),
            NoisyLinear(hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, hidden // 2),
            nn.ReLU(),
            NoisyLinear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

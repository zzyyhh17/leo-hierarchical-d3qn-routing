"""DQN / D3QN / Enhanced D3QN Agent 实现。"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import IntraReplayBuffer, PrioritizedReplayBuffer, NStepBuffer
from .networks import (
    IntraQNetwork, IntraDuelingQNetwork,
    IntraTransformerDuelingQNetwork, PerActionDuelingQNetwork,
    ContextPerActionDuelingQNetwork,
    EnhancedDuelingQNetwork,
)


class IntraDQNAgent:
    """基础 DQN Agent。"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 8000, target_update: int = 500,
                 batch_size: int = 128, buffer_size: int = 100000,
                 hidden: int = 128, device: str = "cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q_net = IntraQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = IntraQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = IntraReplayBuffer(buffer_size)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self.steps / self.epsilon_decay)

    def select_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            valid = mask.nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                return 0
            return valid[random.randint(0, len(valid) - 1)].item()
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def greedy_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        dev = self.device
        states = torch.stack([t[0] for t in batch]).to(dev)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=dev).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=dev)
        next_states = torch.stack([t[3] for t in batch]).to(dev)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=dev)
        next_masks = torch.stack([t[5] for t in batch]).to(dev)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q[~next_masks] = -1e9
            max_next_q = next_q.max(dim=1).values
            max_next_q[dones.bool()] = 0.0
            target = rewards + self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


class IntraD3QNAgent:
    """Dueling Double DQN Agent (D3QN)

    支持三种网络:
      - "mlp": 3层全连接 + LayerNorm
      - "per_action": Per-Action 结构化网络 (推荐)
      - "context_per_action": 带排列不变候选集合上下文的 Per-Action
      - "transformer": Transformer encoder + Dueling head

    训练优化:
      - Smooth L1 loss (Huber): 比 MSE 对异常 TD-error 更鲁棒
      - Soft target update (Polyak τ=0.005): 比硬更新更平滑稳定
      - Cosine Annealing LR: 后期精细调参
      - Gradient clipping = 1.0: 防止梯度爆炸
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 1e-3, lr_min: float = 1e-5,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 8000,
                 target_tau: float = 0.005,
                 batch_size: int = 128, buffer_size: int = 100000,
                 hidden: int = 128, device: str = "cpu",
                 network: str = "mlp",
                 lr_T_max: int = 50000,
                 base_dim: int = 5, nb_feat_dim: int = 7,
                 use_per: bool = False,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_tau = target_tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.network_type = network
        self.use_per = use_per

        if network == "transformer":
            self.q_net = IntraTransformerDuelingQNetwork(
                state_dim, action_dim, d_model=64, nhead=4, num_layers=3,
            ).to(self.device)
            self.target_net = IntraTransformerDuelingQNetwork(
                state_dim, action_dim, d_model=64, nhead=4, num_layers=3,
            ).to(self.device)
        elif network in {"per_action", "context_per_action"}:
            network_cls = (
                ContextPerActionDuelingQNetwork
                if network == "context_per_action"
                else PerActionDuelingQNetwork
            )
            self.q_net = network_cls(
                state_dim, action_dim, base_dim=base_dim,
                nb_feat_dim=nb_feat_dim, embed_dim=64, hidden=128,
            ).to(self.device)
            self.target_net = network_cls(
                state_dim, action_dim, base_dim=base_dim,
                nb_feat_dim=nb_feat_dim, embed_dim=64, hidden=128,
            ).to(self.device)
        else:
            self.q_net = IntraDuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
            self.target_net = IntraDuelingQNetwork(state_dim, action_dim, hidden).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=lr_T_max, eta_min=lr_min)
        if use_per:
            self.buffer = PrioritizedReplayBuffer(buffer_size, per_alpha,
                                                  per_beta_start)
        else:
            self.buffer = IntraReplayBuffer(buffer_size)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self.steps / self.epsilon_decay)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def select_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            valid = mask.nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                return 0
            return valid[random.randint(0, len(valid) - 1)].item()
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def greedy_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def update(
        self,
        expert_batch: list[tuple[torch.Tensor, int, torch.Tensor]] | None = None,
        expert_margin: float = 0.5,
        expert_weight: float = 0.0,
    ) -> float | None:
        """Run one D3QN update, optionally with an expert ranking loss.

        Each expert tuple is ``(state, expert_action, valid_action_mask)``.
        The large-margin term trains the expert action to outrank every other
        valid action and therefore directly targets greedy Top-1 accuracy.
        """
        if len(self.buffer) < self.batch_size:
            return None

        if self.use_per:
            batch, per_indices, is_weights = self.buffer.sample(self.batch_size)
            is_weights = is_weights.to(self.device)
        else:
            batch = self.buffer.sample(self.batch_size)

        dev = self.device
        states = torch.stack([t[0] for t in batch]).to(dev)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=dev).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=dev)
        next_states = torch.stack([t[3] for t in batch]).to(dev)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=dev)
        next_masks = torch.stack([t[5] for t in batch]).to(dev)

        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.q_net(next_states)
            next_q_policy[~next_masks] = -1e9
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            max_next_q[dones.bool()] = 0.0
            target = rewards + self.gamma * max_next_q

        if self.use_per:
            td_errors = (q_values - target).detach()
            td_loss = (is_weights * nn.functional.smooth_l1_loss(
                q_values, target, reduction='none')).mean()
        else:
            td_loss = nn.functional.smooth_l1_loss(q_values, target)

        loss = td_loss
        if expert_batch and expert_weight > 0.0:
            expert_states = torch.stack([t[0] for t in expert_batch]).to(dev)
            expert_actions = torch.tensor(
                [t[1] for t in expert_batch], dtype=torch.long, device=dev
            ).unsqueeze(1)
            expert_masks = torch.stack([t[2] for t in expert_batch]).to(dev)

            ranked_q = self.q_net(expert_states)
            ranked_q = ranked_q.masked_fill(~expert_masks, -1e9)
            expert_q = ranked_q.gather(1, expert_actions).squeeze(1)
            competitor_q = ranked_q.clone()
            competitor_q.scatter_(1, expert_actions, -1e9)
            best_competitor = competitor_q.max(dim=1).values
            ranking_loss = torch.relu(
                expert_margin + best_competitor - expert_q
            ).mean()
            loss = td_loss + expert_weight * ranking_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.use_per:
            self.buffer.update_priorities(per_indices, td_errors.abs())

        self.steps += 1
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.target_tau).add_(p.data * self.target_tau)

        return loss.item()


class EnhancedD3QNAgent:
    """增强版 D3QN Agent (NoisyNet + PER + N-step)。"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 5e-4, gamma: float = 0.99,
                 n_steps: int = 3,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4,
                 target_tau: float = 0.005,
                 batch_size: int = 128, buffer_size: int = 100000,
                 hidden: int = 256, device: str = "cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.gamma_n = gamma ** n_steps
        self.n_steps = n_steps
        self.target_tau = target_tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q_net = EnhancedDuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = EnhancedDuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size, per_alpha, per_beta_start)
        self.nstep_buffer = NStepBuffer(n_steps, gamma)
        self.steps = 0

    def select_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def greedy_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            action = q.argmax().item()
        self.q_net.train()
        return action

    def store(self, state, action, reward, next_state, done, next_mask):
        transition = self.nstep_buffer.push(
            state, action, reward, next_state, done, next_mask
        )
        if transition is not None:
            self.buffer.push(*transition)
        if done:
            for t in self.nstep_buffer.flush():
                self.buffer.push(*t)
            self.nstep_buffer.reset()

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch, indices, is_weights = self.buffer.sample(self.batch_size)
        dev = self.device

        states = torch.stack([t[0] for t in batch]).to(dev)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=dev).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=dev)
        next_states = torch.stack([t[3] for t in batch]).to(dev)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=dev)
        next_masks = torch.stack([t[5] for t in batch]).to(dev)
        is_weights = is_weights.to(dev)

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.q_net(next_states)
            next_q_policy[~next_masks] = -1e9
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            max_next_q[dones.bool()] = 0.0
            target = rewards + self.gamma_n * max_next_q

        td_errors = q_values - target
        loss = (is_weights * nn.functional.smooth_l1_loss(
            q_values, target, reduction='none'
        )).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.abs())

        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.mul_(1.0 - self.target_tau).add_(p.data * self.target_tau)

        self.steps += 1
        return loss.item()

    @property
    def epsilon(self) -> float:
        return 0.0

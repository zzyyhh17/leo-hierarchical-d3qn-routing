"""经验回放缓冲区: 均匀采样 / 优先经验回放 (PER) / N-step Return。"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch


Transition = tuple[torch.Tensor, int, float, torch.Tensor, bool, torch.Tensor]


class IntraReplayBuffer:
    """均匀采样回放缓冲区。"""

    def __init__(self, capacity: int = 100000):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buf.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)


class SumTree:
    """SumTree 数据结构, 用于 O(log n) 的优先级采样."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data: list = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """优先经验回放 (PER).

    TD-error 大的经验被采样的概率更高 → 学习效率↑.
    用 importance sampling weights 修正偏差.
    """

    def __init__(self, capacity: int = 100000,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self._max_priority = 1.0
        self._min_priority = 1e-6

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        self.tree.add(self._max_priority ** self.alpha, args)

    def sample(self, batch_size: int):
        self.frame += 1
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total)
                idx, p, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(p)
            batch.append(data)

        total = self.tree.total
        min_prob = min(priorities) / total if total > 0 else 1e-6
        min_prob = max(min_prob, 1e-6)
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)

        weights = []
        for p in priorities:
            prob = p / total if total > 0 else 1e-6
            prob = max(prob, 1e-6)
            weight = (prob * self.tree.n_entries) ** (-self.beta) / max_weight
            weights.append(weight)

        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: list[int], td_errors: torch.Tensor):
        for idx, td in zip(indices, td_errors.detach().cpu().numpy()):
            priority = (abs(td) + self._min_priority) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class NStepBuffer:
    """N-step Return 缓冲区.

    累积 n 步奖励: R = r_1 + γr_2 + γ²r_3 + ... + γ^{n-1}r_n
    然后存 (s_1, a_1, R, s_{n+1}, done) 到主 buffer.
    """

    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n_steps)

    def push(self, state, action, reward, next_state, done, next_mask) -> tuple | None:
        self.buffer.append((state, action, reward, next_state, done, next_mask))
        if len(self.buffer) < self.n_steps:
            return None
        return self._compute()

    def _compute(self) -> tuple:
        R = 0.0
        for i in reversed(range(len(self.buffer))):
            _, _, r, _, d, _ = self.buffer[i]
            R = r + self.gamma * R * (1.0 - float(d))
        state, action, _, _, _, _ = self.buffer[0]
        _, _, _, next_state, done, next_mask = self.buffer[-1]
        return (state, action, R, next_state, done, next_mask)

    def flush(self) -> list[tuple]:
        results = []
        while len(self.buffer) > 0:
            results.append(self._compute())
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()

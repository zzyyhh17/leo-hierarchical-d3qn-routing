"""
DQN 组间路由：将每个分组视为一个节点，ISL 跨组连接视为边，
用 Deep Q-Network 学习从源组到目标组的最优下一跳策略。

状态: (当前组, 目标组, 各邻居组的负载/距离特征)
动作: 选择一个邻居组作为下一跳
奖励: 到达目标 +10, 每跳 -1, 回环 -5, 超最大跳数 -10
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from starlink_model import Satellite
from group import distance_km


# ---------------------------------------------------------------------------
# 1. 组间拓扑图
# ---------------------------------------------------------------------------

@dataclass
class GroupNode:
    """分组节点：记录组质心位置和组内卫星。"""
    group_id: int
    sat_indices: list[int]
    center_lat: float = 0.0
    center_lon: float = 0.0
    center_alt: float = 0.0
    neighbors: set[int] = field(default_factory=set)


def build_group_graph(satellites: list[Satellite]) -> dict[int, GroupNode]:
    """从带 group_id 和 ISL 信息的卫星列表构建组间邻接图。

    两个组之间存在边 ⟺ 存在 ISL 连接跨越这两个组的卫星。
    """
    cat_to_idx = {s.catalog_number: i for i, s in enumerate(satellites)}

    groups: dict[int, GroupNode] = {}
    for i, s in enumerate(satellites):
        gid = s.group_id
        if gid < 0:
            continue
        if gid not in groups:
            groups[gid] = GroupNode(group_id=gid, sat_indices=[])
        groups[gid].sat_indices.append(i)

    for gid, node in groups.items():
        lats, lons, alts = [], [], []
        for idx in node.sat_indices:
            p = satellites[idx].position
            lats.append(p.latitude_deg)
            lons.append(p.longitude_deg)
            alts.append(p.height_km)
        node.center_lat = sum(lats) / len(lats)
        node.center_lon = sum(lons) / len(lons)
        node.center_alt = sum(alts) / len(alts)

    for i, s in enumerate(satellites):
        gid = s.group_id
        if gid < 0:
            continue
        for peer_cat in s.isl.connected_peers:
            j = cat_to_idx.get(peer_cat)
            if j is None:
                continue
            peer_gid = satellites[j].group_id
            if peer_gid >= 0 and peer_gid != gid:
                groups[gid].neighbors.add(peer_gid)

    return groups


def dijkstra_group_path(
    graph: dict[int, GroupNode],
    src_gid: int,
    dst_gid: int,
    satellites: list[Satellite] | None = None,
) -> list[int] | None:
    """在组图上用 Dijkstra 找最短路径 (按组间距离加权)。"""
    import heapq
    if src_gid == dst_gid:
        return [src_gid]
    dist = {src_gid: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, src_gid)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        if cur == dst_gid:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(src_gid)
            return path[::-1]
        for nb in graph[cur].neighbors:
            w = group_distance(graph[cur], graph[nb], satellites)
            nd = d + w
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = cur
                heapq.heappush(heap, (nd, nb))
    return None


def dijkstra_group_path_hops(
    graph: dict[int, GroupNode],
    src_gid: int,
    dst_gid: int,
) -> list[int] | None:
    """在组图上用 Dijkstra 找最短跳数路径 (等权)。"""
    import heapq
    if src_gid == dst_gid:
        return [src_gid]
    dist = {src_gid: 0}
    prev: dict[int, int] = {}
    heap = [(0, src_gid)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        if cur == dst_gid:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(src_gid)
            return path[::-1]
        for nb in graph[cur].neighbors:
            nd = d + 1
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = cur
                heapq.heappush(heap, (nd, nb))
    return None


def group_distance(
    g1: GroupNode,
    g2: GroupNode,
    satellites: list[Satellite] | None = None,
) -> float:
    """两组之间最近卫星对的距离 (km)。

    遍历 g1 × g2 中所有卫星对，返回最小 ECEF 距离。
    若未提供 satellites 则退化为质心距离。
    """
    if satellites is None:
        return distance_km(
            g1.center_lat, g1.center_lon, g1.center_alt,
            g2.center_lat, g2.center_lon, g2.center_alt,
        )
    min_d = float("inf")
    for i in g1.sat_indices:
        p1 = satellites[i].position
        for j in g2.sat_indices:
            p2 = satellites[j].position
            d = distance_km(
                p1.latitude_deg, p1.longitude_deg, p1.height_km,
                p2.latitude_deg, p2.longitude_deg, p2.height_km,
            )
            if d < min_d:
                min_d = d
    return min_d


# ---------------------------------------------------------------------------
# 2. 路由环境
# ---------------------------------------------------------------------------

class RoutingEnv:
    """组间路由 Gym 风格环境。

    每个 episode: 随机选源组和目标组, agent 逐跳选下一跳, 直到到达或超限。
    """

    def __init__(
        self,
        graph: dict[int, GroupNode],
        satellites: list[Satellite],
        max_hops: int = 30,
        max_neighbors: int = 12,
    ):
        self.graph = graph
        self.satellites = satellites
        self.group_ids = sorted(graph.keys())
        self.n_groups = len(self.group_ids)
        self.max_hops = max_hops
        self.max_neighbors = max_neighbors

        self._dist_cache: dict[tuple[int, int], float] = {}
        self._max_dist = 1.0
        for g1 in self.group_ids:
            for g2 in self.group_ids:
                if g1 < g2:
                    d = group_distance(graph[g1], graph[g2], satellites)
                    self._dist_cache[(g1, g2)] = d
                    self._dist_cache[(g2, g1)] = d
                    if d > self._max_dist:
                        self._max_dist = d

        self._border_counts: dict[tuple[int, int], int] = {}
        cat_to_idx = {satellites[i].catalog_number: i for i, s in enumerate(satellites)}
        for gid, node in graph.items():
            for nb_gid in node.neighbors:
                nb_sat_set = set(graph[nb_gid].sat_indices)
                count = 0
                for idx in node.sat_indices:
                    for peer_cat in satellites[idx].isl.connected_peers:
                        j = cat_to_idx.get(peer_cat)
                        if j is not None and j in nb_sat_set:
                            count += 1
                self._border_counts[(gid, nb_gid)] = count
        self._max_border = max(self._border_counts.values()) if self._border_counts else 1

        self.state_dim = 2 + self.max_neighbors * 3
        self.action_dim = self.max_neighbors

        self.current: int = -1
        self.dest: int = -1
        self.hops: int = 0
        self.visited: set[int] = set()
        self._neighbors: list[int] = []

    def _get_dist(self, g1: int, g2: int) -> float:
        if g1 == g2:
            return 0.0
        return self._dist_cache.get((g1, g2), self._max_dist)

    def _get_neighbors(self, gid: int) -> list[int]:
        """返回当前组的邻居列表 (固定长度, 不足补 -1)。"""
        nbs = sorted(self.graph[gid].neighbors)
        nbs = nbs[: self.max_neighbors]
        while len(nbs) < self.max_neighbors:
            nbs.append(-1)
        return nbs

    def _make_state(self) -> torch.Tensor:
        """状态向量:
        [0]           当前组到目标的归一化距离
        [1]           已用跳数 / max_hops
        [2..2+K-1]    每个邻居到目标的归一化距离 (无邻居=1)
        [2+K..2+2K-1] 每个邻居是否已访问过 (1=visited, 0=not)
        [2+2K..2+3K-1] 当前组到每个邻居的 ISL 边界连接数 (归一化, 无邻居=0)
        """
        d_cur = self._get_dist(self.current, self.dest) / self._max_dist
        hop_ratio = self.hops / self.max_hops
        feats = [d_cur, hop_ratio]
        for nb in self._neighbors:
            if nb < 0:
                feats.append(1.0)
            else:
                feats.append(self._get_dist(nb, self.dest) / self._max_dist)
        for nb in self._neighbors:
            feats.append(1.0 if nb in self.visited else 0.0)
        for nb in self._neighbors:
            if nb < 0:
                feats.append(0.0)
            else:
                bc = self._border_counts.get((self.current, nb), 0)
                feats.append(bc / self._max_border)
        return torch.tensor(feats, dtype=torch.float32)

    def reset(self, src: int | None = None, dst: int | None = None) -> torch.Tensor:
        if src is None or dst is None:
            src, dst = random.sample(self.group_ids, 2)
        self.current = src
        self.dest = dst
        self.hops = 0
        self.visited = {src}
        self._neighbors = self._get_neighbors(self.current)
        return self._make_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        """执行一步, 返回 (next_state, reward, done)。"""
        nb = self._neighbors[action] if action < len(self._neighbors) else -1

        if nb < 0 or nb not in self.graph:
            return self._make_state(), -5.0, True

        self.hops += 1
        loop = nb in self.visited
        prev = self.current
        self.current = nb
        self.visited.add(nb)
        self._neighbors = self._get_neighbors(self.current)

        if nb == self.dest:
            return self._make_state(), 10.0, True

        if self.hops >= self.max_hops:
            return self._make_state(), -10.0, True

        d_before = self._get_dist(prev, self.dest)
        d_after = self._get_dist(self.current, self.dest)
        progress = (d_before - d_after) / self._max_dist * 5.0

        reward = -0.5 + progress
        if loop:
            reward -= 5.0

        return self._make_state(), reward, False

    def valid_action_mask(self) -> torch.Tensor:
        """返回合法动作掩码 (1=可选, 0=无效/填充)。"""
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        for i, nb in enumerate(self._neighbors):
            if nb >= 0 and nb in self.graph:
                mask[i] = True
        return mask


# ---------------------------------------------------------------------------
# 3. DQN 网络
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
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


# ---------------------------------------------------------------------------
# 4. 经验回放
# ---------------------------------------------------------------------------

Transition = tuple[torch.Tensor, int, float, torch.Tensor, bool, torch.Tensor]


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buf.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# 5. DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        target_update: int = 200,
        batch_size: int = 64,
        buffer_size: int = 50000,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
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
            q = self.q_net(state.to(self.device).unsqueeze(0)).squeeze(0)
            q[~mask.to(self.device)] = -1e9
            return q.argmax().item()

    def greedy_action(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.q_net(state.to(self.device).unsqueeze(0)).squeeze(0)
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


# ---------------------------------------------------------------------------
# 6. 训练与评估
# ---------------------------------------------------------------------------

def train(
    env: RoutingEnv,
    agent: DQNAgent,
    n_episodes: int = 10000,
    log_interval: int = 500,
) -> list[dict]:
    """训练 DQN agent, 返回训练统计列表。"""
    history: list[dict] = []
    ep_rewards: list[float] = []
    ep_hops: list[float] = []
    ep_success: list[float] = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        mask = env.valid_action_mask()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.valid_action_mask()
            agent.buffer.push(state, action, reward, next_state, done, next_mask)
            agent.update()
            state = next_state
            mask = next_mask
            total_reward += reward

        ep_rewards.append(total_reward)
        ep_hops.append(env.hops)
        ep_success.append(1.0 if env.current == env.dest else 0.0)

        if ep % log_interval == 0:
            n = log_interval
            avg_r = sum(ep_rewards[-n:]) / n
            avg_h = sum(ep_hops[-n:]) / n
            avg_s = sum(ep_success[-n:]) / n
            rec = {
                "episode": ep,
                "avg_reward": round(avg_r, 2),
                "avg_hops": round(avg_h, 2),
                "success_rate": round(avg_s, 3),
                "epsilon": round(agent.epsilon, 4),
                "buffer_size": len(agent.buffer),
            }
            history.append(rec)
            print(
                f"Ep {ep:>6d} | reward={avg_r:>7.2f} | hops={avg_h:>5.2f} | "
                f"success={avg_s:.3f} | ε={agent.epsilon:.4f} | buf={len(agent.buffer)}"
            )

    return history


def evaluate(
    env: RoutingEnv,
    agent: DQNAgent,
    n_episodes: int = 500,
) -> dict:
    """贪心策略评估, 返回统计摘要。"""
    rewards, hops, successes = [], [], []
    for _ in range(n_episodes):
        state = env.reset()
        mask = env.valid_action_mask()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.greedy_action(state, mask)
            state, reward, done = env.step(action)
            mask = env.valid_action_mask()
            total_reward += reward
        rewards.append(total_reward)
        hops.append(env.hops)
        successes.append(1.0 if env.current == env.dest else 0.0)
    return {
        "avg_reward": round(sum(rewards) / len(rewards), 2),
        "avg_hops": round(sum(hops) / len(hops), 2),
        "success_rate": round(sum(successes) / len(successes), 3),
        "n_episodes": n_episodes,
    }


def trace_route(
    env: RoutingEnv,
    agent: DQNAgent,
    src: int,
    dst: int,
) -> tuple[list[int], float]:
    """用贪心策略跟踪一条路由路径, 返回 (组序列, 总奖励)。"""
    state = env.reset(src, dst)
    mask = env.valid_action_mask()
    path = [src]
    total_reward = 0.0
    done = False
    while not done:
        action = agent.greedy_action(state, mask)
        state, reward, done = env.step(action)
        mask = env.valid_action_mask()
        total_reward += reward
        path.append(env.current)
    return path, total_reward


# ---------------------------------------------------------------------------
# 7. 封装: 训练 pipeline & 评估 pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    satellites: list[Satellite],
    n_episodes: int = 10000,
    max_hops: int = 30,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_decay: int = 3000,
    target_update: int = 200,
    batch_size: int = 64,
    hidden: int = 128,
    log_interval: int = 500,
    save_path: str | None = None,
) -> tuple[DQNAgent, RoutingEnv, list[dict]]:
    """从已完成 ISL + 分组的卫星列表一键训练 DQN 路由。

    返回 (agent, env, history)。
    """
    import time as _time

    graph = build_group_graph(satellites)
    max_nb = max(len(n.neighbors) for n in graph.values())
    avg_nb = sum(len(n.neighbors) for n in graph.values()) / len(graph)
    print(f"组间图: {len(graph)} 节点, 平均邻居 {avg_nb:.1f}, 最大邻居 {max_nb}")

    env = RoutingEnv(graph, satellites, max_hops=max_hops, max_neighbors=max_nb)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=lr, gamma=gamma, epsilon_decay=epsilon_decay,
        target_update=target_update, batch_size=batch_size, hidden=hidden,
    )
    print(f"状态维度: {env.state_dim}, 动作维度: {env.action_dim}")

    t0 = _time.time()
    history = train(env, agent, n_episodes=n_episodes, log_interval=log_interval)
    print(f"训练完成, 耗时 {_time.time() - t0:.1f}s")

    if save_path is not None:
        torch.save({
            "q_net": agent.q_net.state_dict(),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "history": history,
        }, save_path)
        print(f"模型已保存: {save_path}")

    return agent, env, history


def eval_pipeline(
    agent: DQNAgent,
    env: RoutingEnv,
    n_episodes: int = 5000,
    n_samples: int = 10,
) -> dict:
    """评估已训练的 agent, 返回统计摘要并打印示例路由。"""
    result = evaluate(env, agent, n_episodes=n_episodes)
    print(f"评估: 平均奖励 {result['avg_reward']}, "
          f"平均跳数 {result['avg_hops']}, 成功率 {result['success_rate']}")

    graph = env.graph
    pairs = [(a, b) for a in graph for b in graph if a != b]
    sample_pairs = random.sample(pairs, min(n_samples, len(pairs)))
    routes: list[dict] = []
    for src, dst in sample_pairs:
        path, r = trace_route(env, agent, src, dst)
        ok = path[-1] == dst
        routes.append({"src": src, "dst": dst, "path": path, "reward": r, "ok": ok})
        tag = "OK" if ok else "FAIL"
        print(f"  {src:>3d} -> {dst:>3d}: {' -> '.join(str(g) for g in path)} "
              f"({len(path)-1} 跳, reward={r:.1f}) {tag}")

    result["sample_routes"] = routes
    return result


# ---------------------------------------------------------------------------
# 8. 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _GOOD = Path(__file__).resolve().parent
    if str(_GOOD) not in sys.path:
        sys.path.insert(0, str(_GOOD))

    from data_loader import load_gen11_at_time
    from isl import build_isl_from_satellites, update_satellite_isl_peers
    from group import run_grouping
    from datetime import datetime, timezone

    print("=== DQN 组间路由训练 ===")
    ref_time = datetime.now(timezone.utc)
    sats = load_gen11_at_time(ref_time, source="ideal")
    print(f"加载 {len(sats)} 颗卫星")

    build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    run_grouping(sats, max_size=9)

    model_path = str(_GOOD / "dqn_routing_model.pt")
    agent, env, history = train_pipeline(sats, save_path=model_path)

    print("\n=== 贪心策略评估 ===")
    eval_pipeline(agent, env)

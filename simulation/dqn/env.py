"""卫星级路由环境 (ISL 图上)。

状态 (base 8 + per-nb 11): 纯几何 + 局部拓扑特征
奖励: -1 + 欧氏进步(×5) + 方向余弦(×2)
"""
from __future__ import annotations

import math

import torch


class IntraSatEnv:
    """卫星级路由环境 (ISL 图上)。

    状态特征 (base 8 + per-nb 11):
      base: dist, hop_ratio, goal_dir(3), visit_count_self, clustering, 2hop_unvis_ratio
      per-nb: nb_dist, visit_count, degree, unvis_ratio, cos_goal, dist_delta,
              dead_end, nb_2hop_unvis, nb_goal_closer, nb_common, nb_clustering

    奖励: -1 + 欧氏进步(×5) + 方向余弦(×2) - 回路惩罚
    """

    def __init__(self, adj: dict[int, set[int]], sats: list,
                 max_hops: int = 50, max_neighbors: int = 6,
                 use_bfs: bool = False):
        self.adj = adj
        self.sats = sats
        self.max_hops = max_hops
        self.max_neighbors = max_neighbors
        self.n_sats = len(sats)

        self._ecef: dict[int, tuple[float, float, float]] = {}
        R = 6371.0
        for i, s in enumerate(sats):
            p = s.position
            lat, lon = math.radians(p.latitude_deg), math.radians(p.longitude_deg)
            r = R + p.height_km
            cl, sl = math.cos(lat), math.sin(lat)
            co, so = math.cos(lon), math.sin(lon)
            self._ecef[i] = (r * cl * co, r * cl * so, r * sl)

        self._max_dist = 1.0
        for i in range(min(self.n_sats, 200)):
            for j in range(i + 1, min(self.n_sats, 200)):
                d = self._dist(i, j)
                if d > self._max_dist:
                    self._max_dist = d

        self._degree: dict[int, int] = {i: len(adj.get(i, set())) for i in range(self.n_sats)}
        self._max_degree = max(self._degree.values()) if self._degree else 1

        self._clustering = self._precompute_clustering()
        self._max_2hop = self._precompute_max_2hop()

        self.base_dim = 8
        self.per_nb_dim = 11
        self.state_dim = self.base_dim + self.max_neighbors * self.per_nb_dim
        self.action_dim = self.max_neighbors

        self.current = -1
        self.dest = -1
        self.hops = 0
        self.visited: set[int] = set()
        self._visit_count: dict[int, int] = {}
        self._neighbors: list[int] = []

    def _precompute_clustering(self) -> dict[int, float]:
        """预计算每个节点的聚类系数。"""
        clustering: dict[int, float] = {}
        for node in range(self.n_sats):
            nbs = list(self.adj.get(node, set()))
            k = len(nbs)
            if k < 2:
                clustering[node] = 0.0
                continue
            edges = 0
            nb_set = set(nbs)
            for i in range(k):
                for j in range(i + 1, k):
                    if nbs[j] in self.adj.get(nbs[i], set()):
                        edges += 1
            clustering[node] = 2.0 * edges / (k * (k - 1))
        return clustering

    def _precompute_max_2hop(self) -> int:
        """预计算 2 跳可达节点数的最大值（用于归一化）。"""
        max_val = 1
        for node in range(min(self.n_sats, 200)):
            reach = set()
            for nb in self.adj.get(node, set()):
                reach.add(nb)
                for nn in self.adj.get(nb, set()):
                    if nn != node:
                        reach.add(nn)
            if len(reach) > max_val:
                max_val = len(reach)
        return max_val

    def _2hop_reachable(self, node: int) -> set[int]:
        """返回从 node 出发 1-2 跳可达的节点集合（不含 node 自身）。"""
        reach = set()
        for nb in self.adj.get(node, set()):
            reach.add(nb)
            for nn in self.adj.get(nb, set()):
                if nn != node:
                    reach.add(nn)
        return reach

    def _dist(self, i: int, j: int) -> float:
        x1, y1, z1 = self._ecef[i]
        x2, y2, z2 = self._ecef[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    def _direction(self, frm: int, to: int) -> tuple[float, float, float]:
        """frm→to 的单位方向向量，距离为 0 时返回 (0,0,0)。"""
        x1, y1, z1 = self._ecef[frm]
        x2, y2, z2 = self._ecef[to]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm < 1e-9:
            return (0.0, 0.0, 0.0)
        return (dx / norm, dy / norm, dz / norm)

    def _cos_sim(self, a: tuple[float, float, float],
                 b: tuple[float, float, float]) -> float:
        dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        return max(-1.0, min(1.0, dot))

    def _get_neighbors(self, idx: int) -> list[int]:
        nbs = sorted(self.adj.get(idx, set()))[:self.max_neighbors]
        while len(nbs) < self.max_neighbors:
            nbs.append(-1)
        return nbs

    def _make_state(self) -> torch.Tensor:
        d_cur = self._dist(self.current, self.dest) / self._max_dist
        hop_ratio = self.hops / self.max_hops

        goal_dir = self._direction(self.current, self.dest)

        vc_self = min(self._visit_count.get(self.current, 0), 5) / 5.0
        two_hop = self._2hop_reachable(self.current)
        two_hop_unvis = sum(1 for x in two_hop if x not in self.visited)

        feats = [d_cur, hop_ratio, goal_dir[0], goal_dir[1], goal_dir[2],
                 vc_self, self._clustering.get(self.current, 0.0),
                 two_hop_unvis / self._max_2hop]

        pad = [0.0] * self.per_nb_dim
        pad[0] = 1.0   # nb_dist = far
        pad[6] = 1.0   # dead_end = true

        cur_nbs_set = set(self.adj.get(self.current, set()))
        d_cur_raw = self._dist(self.current, self.dest)

        for nb in self._neighbors:
            if nb < 0:
                feats.extend(pad)
            else:
                nb_dist = self._dist(nb, self.dest) / self._max_dist
                visited = min(self._visit_count.get(nb, 0), 5) / 5.0
                degree = self._degree[nb] / self._max_degree
                nb_nbs = self.adj.get(nb, set())
                total = len(nb_nbs)
                unvis = sum(1 for x in nb_nbs if x not in self.visited) if total > 0 else 0
                unvis_ratio = unvis / total if total > 0 else 0.0

                nb_dir = self._direction(self.current, nb)
                cos_goal = self._cos_sim(nb_dir, goal_dir)

                dist_delta = d_cur - nb_dist

                dead_end = 1.0 if (total <= 1 and nb != self.dest) else 0.0

                feats.extend([nb_dist, visited, degree, unvis_ratio,
                              cos_goal, dist_delta, dead_end])

                nb_2hop = self._2hop_reachable(nb)
                nb_2hop_unvis = sum(1 for x in nb_2hop if x not in self.visited)
                feats.append(nb_2hop_unvis / self._max_2hop)

                closer = sum(1 for nn in nb_nbs
                             if self._dist(nn, self.dest) < d_cur_raw)
                feats.append(closer / total if total > 0 else 0.0)

                common = len(nb_nbs & cur_nbs_set)
                feats.append(common / total if total > 0 else 0.0)

                feats.append(self._clustering.get(nb, 0.0))

        return torch.tensor(feats, dtype=torch.float32)

    def reset(self, src: int, dst: int) -> torch.Tensor:
        self.current = src
        self.dest = dst
        self.hops = 0
        self.visited = {src}
        self._visit_count = {src: 1}
        self._neighbors = self._get_neighbors(self.current)
        return self._make_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        nb = self._neighbors[action] if action < len(self._neighbors) else -1
        if nb < 0:
            return self._make_state(), -10.0, True

        self.hops += 1
        loop = nb in self.visited
        prev = self.current
        self.current = nb
        self.visited.add(nb)
        self._visit_count[nb] = self._visit_count.get(nb, 0) + 1
        self._neighbors = self._get_neighbors(self.current)

        if nb == self.dest:
            return self._make_state(), 20.0, True
        if self.hops >= self.max_hops:
            return self._make_state(), -20.0, True

        d_before = self._dist(prev, self.dest)
        d_after = self._dist(self.current, self.dest)
        euclid_progress = (d_before - d_after) / self._max_dist * 5.0

        goal_dir = self._direction(self.current, self.dest)
        step_dir = self._direction(prev, self.current)
        cos_bonus = self._cos_sim(step_dir, goal_dir) * 2.0

        reward = -1.0 + euclid_progress + cos_bonus
        if loop:
            vc = min(self._visit_count.get(nb, 1), 5)
            reward -= (5.0 + vc * 3.0)
        return self._make_state(), reward, False

    def valid_action_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        for i, nb in enumerate(self._neighbors):
            if nb >= 0:
                mask[i] = True
        return mask

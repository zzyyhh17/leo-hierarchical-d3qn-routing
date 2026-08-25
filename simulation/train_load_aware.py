"""端到端训练负载感知 Per-Action D3QN（而非推理期叠加）。

相对原 52 维局部特征，每个邻居新增 1 维"链路当前利用率 ρ"特征（→ 56 维状态），
并在奖励中对选择高负载链路施加惩罚；训练时每个 episode 随机注入背景拥塞，
使智能体学会在到达目的的同时主动绕开拥塞链路。训练好的模型用于动态拥塞实验，
其负载规避来自学习（推理时无需叠加惩罚）。
"""
from __future__ import annotations
import sys, math, time, random, argparse
from pathlib import Path
from collections import deque

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from dqn.env import IntraSatEnv
from dqn.agents import IntraD3QNAgent
from dqn.train import _bfs_path, _build_hop_neighborhoods, _curriculum_sample


def ekey(u, v): return (u, v) if u < v else (v, u)


class LoadAwareSatEnv(IntraSatEnv):
    """在 IntraSatEnv 基础上加入逐邻居链路负载特征与负载奖励惩罚。"""

    def __init__(self, adj, sats, max_hops=60, max_neighbors=4,
                 cap=25, t_serv=0.6, beta_load=0.5, cong_frac=0.18):
        super().__init__(adj, sats, max_hops=max_hops, max_neighbors=max_neighbors, use_bfs=False)
        self.per_nb_dim = 12                       # 原 11 + 1 负载特征
        self.state_dim = self.base_dim + self.max_neighbors * self.per_nb_dim  # 8 + 4*12 = 56
        self.cap = cap; self.t_serv = t_serv; self.beta_load = beta_load; self.cong_frac = cong_frac
        self.link_load = {}
        self.randomize_on_reset = True
        self.external_load = False          # True: 由外部(多流累积)管理 link_load, reset 不清空/不随机
        self._all_edges = [ekey(u, v) for u in adj for v in adj[u] if u < v]

    def _rho(self, u, v):
        return self.link_load.get(ekey(u, v), 0.0) / self.cap

    def _qpen(self, rho):
        if rho >= 1.0:
            return 10.0
        return self.t_serv * rho / (1.0 - rho)

    def reset(self, src, dst):
        if not self.external_load:
            self.link_load = {}
            if self.randomize_on_reset and random.random() < 0.75:
                k = int(self.cong_frac * len(self._all_edges))
                for e in random.sample(self._all_edges, k):
                    self.link_load[e] = random.uniform(0.6, 1.3) * self.cap
        return super().reset(src, dst)

    def step(self, action):
        prev = self.current
        nb = self._neighbors[action] if action < len(self._neighbors) else -1
        ns, reward, done = super().step(action)
        if nb is not None and nb >= 0:                 # 合法移动才计链路负载惩罚
            reward = reward - self.beta_load * self._qpen(self._rho(prev, nb))
        return ns, reward, done

    def _make_state(self):
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
        pad[0] = 1.0; pad[6] = 1.0                     # nb_dist=far, dead_end=true
        cur_nbs_set = set(self.adj.get(self.current, set()))
        d_cur_raw = self._dist(self.current, self.dest)
        for nb in self._neighbors:
            if nb < 0:
                feats.extend(pad)
            else:
                nb_dist = self._dist(nb, self.dest) / self._max_dist
                visited = min(self._visit_count.get(nb, 0), 5) / 5.0
                degree = self._degree[nb] / self._max_degree
                nb_nbs = self.adj.get(nb, set()); total = len(nb_nbs)
                unvis = sum(1 for x in nb_nbs if x not in self.visited) if total else 0
                unvis_ratio = unvis / total if total else 0.0
                cos_goal = self._cos_sim(self._direction(self.current, nb), goal_dir)
                dist_delta = d_cur - nb_dist
                dead_end = 1.0 if (total <= 1 and nb != self.dest) else 0.0
                feats.extend([nb_dist, visited, degree, unvis_ratio, cos_goal, dist_delta, dead_end])
                nb_2hop = self._2hop_reachable(nb)
                feats.append(sum(1 for x in nb_2hop if x not in self.visited) / self._max_2hop)
                feats.append((sum(1 for nn in nb_nbs if self._dist(nn, self.dest) < d_cur_raw) / total) if total else 0.0)
                feats.append((len(nb_nbs & cur_nbs_set) / total) if total else 0.0)
                feats.append(self._clustering.get(nb, 0.0))
                feats.append(min(self._rho(self.current, nb), 1.5) / 1.5)   # 新增：链路负载特征
        return torch.tensor(feats, dtype=torch.float32)


def prefill_demos(adj, env, buffer, n=1000):
    """干净（无背景负载）BFS 最短路专家示范预填充。"""
    env.randomize_on_reset = False
    n_idx = env.n_sats; cnt = 0; trans = 0
    rng = random.Random(7)
    for _ in range(n):
        s, d = rng.sample(range(n_idx), 2)
        path = _bfs_path(adj, s, d, max_d=env.max_hops)
        if not path or len(path) < 2:
            continue
        st = env.reset(s, d); mk = env.valid_action_mask()
        ok = True
        for k in range(len(path) - 1):
            nxt = path[k + 1]
            a = None
            for i, nb in enumerate(env._neighbors):
                if nb == nxt: a = i; break
            if a is None: ok = False; break
            ns, r, done = env.step(a); nmk = env.valid_action_mask()
            buffer.push(st, a, r, ns, done, nmk); trans += 1
            st, mk = ns, nmk
        if ok: cnt += 1
    env.randomize_on_reset = True
    return cnt, trans


def quick_eval(env, agent, n=60, seed=99):
    """干净拓扑(无背景负载)下的 greedy 成功率与平均跳数。"""
    env.randomize_on_reset = False
    rng = random.Random(seed); succ = 0; hops = []
    for _ in range(n):
        s, d = rng.sample(range(env.n_sats), 2)
        st = env.reset(s, d); mk = env.valid_action_mask()
        for _ in range(env.max_hops):
            a = agent.greedy_action(st, mk)
            st, _, done = env.step(a); mk = env.valid_action_mask()
            if done: break
        if env.current == env.dest:
            succ += 1; hops.append(env.hops)
    env.randomize_on_reset = True
    sr = succ / n
    return sr, (sum(hops) / len(hops) if hops else env.max_hops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--beta_load", type=float, default=0.5)
    ap.add_argument("--cong_frac", type=float, default=0.18)
    ap.add_argument("--out", default="model/d3qn_load_aware.pt")
    args = ap.parse_args()

    from dynamic_experiment import load_topology
    print("加载拓扑 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology("ideal")
    print(f"  卫星={len(sats)} 链路={len(edges)}", flush=True)

    dev = "cpu"
    env = LoadAwareSatEnv(adj, sats, max_hops=60, max_neighbors=4,
                          beta_load=args.beta_load, cong_frac=args.cong_frac)
    print(f"  state_dim={env.state_dim} (含负载特征) per_nb={env.per_nb_dim}", flush=True)
    n_ep = args.episodes
    agent = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=256,
                           epsilon_decay=max(n_ep*3//4, 3000), device=dev, network="per_action",
                           lr_T_max=n_ep*60//4, lr=5e-4, target_tau=0.003,
                           base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim, use_per=False)
    print("  专家示范预填充 ...", flush=True)
    c, t = prefill_demos(adj, env, agent.buffer, n=1000)
    print(f"    {c} 条路径 → {t} 条转移", flush=True)

    all_idx = list(range(len(sats)))
    near = _build_hop_neighborhoods(adj, len(sats), max_hop=5)
    mid = _build_hop_neighborhoods(adj, len(sats), max_hop=15)
    near_src = [i for i in all_idx if near[i]]; mid_src = [i for i in all_idx if mid[i]]

    succ = deque(maxlen=500); gstep = 0
    best_sr = 0.0; best_sd = None; t0 = time.time()
    print(f"  训练 {n_ep} episodes (beta_load={args.beta_load}, cong_frac={args.cong_frac}) ...", flush=True)
    for ep in range(1, n_ep + 1):
        s, d = _curriculum_sample(ep, n_ep, all_idx, near, near_src, mid, mid_src)
        st = env.reset(s, d); mk = env.valid_action_mask(); done = False
        while not done:
            a = agent.select_action(st, mk)
            ns, r, done = env.step(a); nmk = env.valid_action_mask()
            agent.buffer.push(st, a, r, ns, done, nmk)
            gstep += 1
            if gstep % 4 == 0: agent.update()
            st, mk = ns, nmk
        succ.append(1.0 if env.current == env.dest else 0.0)
        if ep % max(n_ep // 16, 500) == 0:
            sr, ah = quick_eval(env, agent)
            tr = sum(succ) / len(succ)
            print(f"    ep {ep:5d} | train_sr={tr:.0%} eval_sr={sr:.0%} eval_hops={ah:.1f} "
                  f"eps={agent.epsilon:.2f} [{time.time()-t0:.0f}s]", flush=True)
            if sr > best_sr:
                best_sr = sr; best_sd = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}

    if best_sd is None:
        best_sd = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}
    torch.save({"state_dict": best_sd, "state_dim": env.state_dim, "action_dim": env.action_dim,
                "max_hops": env.max_hops, "max_neighbors": env.max_neighbors,
                "network": "per_action", "nb_feat_dim": env.per_nb_dim, "base_dim": env.base_dim,
                "best_eval_sr": best_sr}, args.out)
    print(f"\n训练完成，最佳 eval_sr={best_sr:.0%}，模型已存 {args.out}（耗时 {time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()

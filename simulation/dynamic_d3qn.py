"""动态拥塞下的训练好 D3QN 实测（负载感知推理）。

复用 dynamic_experiment 的拓扑/拥塞/Dijkstra 机制，加入：
  D3QN-blind  训练好的 Per-Action D3QN，负载盲目（自然行为，λ=0）
  D3QN-LA     同一模型 + 负载感知推理：每跳 Q 值减去 λ·局部链路排队惩罚，
              贪心失败则回退负载感知 Dijkstra（三级容错）。

对比 SP-hop（负载盲目最短路）与 LA-Dijkstra（集中式负载感知，近似最优）。
"""
from __future__ import annotations
import sys, json, argparse, time, random
from pathlib import Path
from datetime import datetime

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from dynamic_experiment import (
    load_topology, ekey, queue_delay_ms, dijkstra,
    route_static, route_la_dijkstra, evaluate,
)
from dqn.env import IntraSatEnv
from dqn.agents import IntraD3QNAgent
from route import _remove_path_loops


def build_d3qn(adj, sats, ckpt_path="model/d3qn_per_action_best.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    env = IntraSatEnv(adj, sats, max_hops=ckpt["max_hops"],
                      max_neighbors=ckpt["max_neighbors"], use_bfs=False)
    agent = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                           hidden=256, network=ckpt["network"],
                           base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim, device="cpu")
    agent.q_net.load_state_dict(ckpt["state_dict"])
    agent.q_net.eval()
    return env, agent


def route_d3qn(env, agent, adj, dprop, flows, cap, t_serv, lam, beta_fb=0.3, hops_cap=60):
    """D3QN 路由：每跳 Q 值减 λ·局部排队惩罚（λ=0 即负载盲目）；失败回退负载感知 Dijkstra。

    返回 (paths, autonomous)。负载随路由实时累加，体现流间相互影响。
    """
    load = {}
    def la_w(u, v):
        rho = load.get(ekey(u, v), 0) / cap
        pen = 1e7 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
        return dprop[(u, v)] + beta_fb * pen
    paths = []
    autonomous = 0
    for s, d in flows:
        st = env.reset(s, d)
        path = [s]
        ok = False
        for _ in range(hops_cap):
            if env.current == d:
                ok = True
                break
            with torch.no_grad():
                q = agent.q_net(st.unsqueeze(0)).squeeze(0).clone()
            mask = env.valid_action_mask()
            nbs = env._neighbors
            if lam > 0:
                cur = env.current
                for i, nb in enumerate(nbs):
                    if nb is not None and nb >= 0:
                        rho = load.get(ekey(cur, nb), 0) / cap
                        pen = 1e3 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
                        q[i] = q[i] - lam * pen
            q[~mask] = -1e18
            a = int(q.argmax().item())
            st, _, done = env.step(a)
            path.append(env.current)
            if env.current == d:
                ok = True
                break
            if done:
                break
        if env.current == d:
            ok = True
        if ok:
            autonomous += 1
            path = _remove_path_loops(path)
        else:
            path = dijkstra(adj, s, d, la_w)   # 三级容错兜底
        if path:
            for a_, b_ in zip(path, path[1:]):
                load[ekey(a_, b_)] = load.get(ekey(a_, b_), 0) + 1
        paths.append(path)
    return paths, autonomous


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ideal")
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--tserv", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3, help="D3QN-LA 负载惩罚权重")
    ap.add_argument("--K", type=str, default="500,1000,1500,2000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("加载拓扑 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology(args.source)
    print(f"  卫星={len(sats)} 链路={len(edges)}", flush=True)
    print("加载训练好的 D3QN ...", flush=True)
    env, agent = build_d3qn(adj, sats)
    print(f"  state_dim={env.state_dim} network=per_action 已就绪", flush=True)

    K_list = [int(x) for x in args.K.split(",")]
    n = len(sats)
    rows = []
    print(f"\nD3QN 动态拥塞实测 (cap={args.cap}, tserv={args.tserv}, beta={args.beta}, lam={args.lam}):", flush=True)
    for K in K_list:
        rng = random.Random(args.seed)
        flows = []
        while len(flows) < K:
            s, d = rng.randrange(n), rng.randrange(n)
            if s != d:
                flows.append((s, d))
        row = {"K": K}
        row["SP-hop"] = evaluate(route_static(adj, flows, lambda u, v: 1.0), dprop, args.cap, args.tserv)
        row["LA-Dijkstra"] = evaluate(route_la_dijkstra(adj, dprop, flows, args.cap, args.tserv, args.beta), dprop, args.cap, args.tserv)
        t = time.time()
        pb, ab = route_d3qn(env, agent, adj, dprop, flows, args.cap, args.tserv, lam=0.0)
        row["D3QN-blind"] = evaluate(pb, dprop, args.cap, args.tserv); row["D3QN-blind"]["autonomous_rate"] = round(ab/len(flows), 3)
        pl, al = route_d3qn(env, agent, adj, dprop, flows, args.cap, args.tserv, lam=args.lam)
        row["D3QN-LA"] = evaluate(pl, dprop, args.cap, args.tserv); row["D3QN-LA"]["autonomous_rate"] = round(al/len(flows), 3)
        rows.append(row)
        sp, la, db, dl = row["SP-hop"], row["LA-Dijkstra"], row["D3QN-blind"], row["D3QN-LA"]
        print(f"  K={K:5d} | SP-hop drop={sp['drop_rate']:.1%} util={sp['max_util']:.2f} "
              f"| D3QN-blind drop={db['drop_rate']:.1%} util={db['max_util']:.2f} auton={db['autonomous_rate']:.0%} "
              f"| D3QN-LA drop={dl['drop_rate']:.1%} util={dl['max_util']:.2f} auton={dl['autonomous_rate']:.0%} "
              f"| LA-Dij drop={la['drop_rate']:.1%} util={la['max_util']:.2f}  [{time.time()-t:.0f}s]", flush=True)

    out = args.out or f"results_d3qn_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump({"params": vars(args), "rows": rows}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}", flush=True)

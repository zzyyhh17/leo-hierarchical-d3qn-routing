"""评估端到端训练的负载感知 D3QN（56维，纯贪心，负载感知来自模型）在动态拥塞下的表现，
并与 SP-hop / LA-Dijkstra / 推理叠加版 D3QN-LA 对比。"""
from __future__ import annotations
import sys, json, time, random, argparse
from pathlib import Path
from datetime import datetime

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from dynamic_experiment import (load_topology, ekey, queue_delay_ms, dijkstra,
                                 route_static, route_la_dijkstra, evaluate)
from dynamic_d3qn import build_d3qn, route_d3qn      # 旧 52维模型 + 推理叠加
from train_load_aware import LoadAwareSatEnv
from dqn.agents import IntraD3QNAgent
from route import _remove_path_loops


def build_trained(adj, sats, ckpt_path="model/d3qn_load_aware.pt"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    env = LoadAwareSatEnv(adj, sats, max_hops=ck["max_hops"], max_neighbors=ck["max_neighbors"])
    env.randomize_on_reset = False
    agent = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=256,
                           network="per_action", base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim, device="cpu")
    agent.q_net.load_state_dict(ck["state_dict"]); agent.q_net.eval()
    return env, agent, ck.get("best_eval_sr")


def route_trained(env, agent, adj, dprop, flows, cap, t_serv, hops_cap=60):
    """纯贪心路由：负载感知来自模型的负载特征（推理时不叠加惩罚）；失败回退负载感知 Dijkstra。"""
    load = {}
    def la_w(u, v):
        rho = load.get(ekey(u, v), 0) / cap
        pen = 1e7 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
        return dprop[(u, v)] + 0.3 * pen
    paths = []; auton = 0
    for s, d in flows:
        env.reset(s, d)              # 清空 link_load
        env.link_load = load         # 接入实时累加负载
        st = env._make_state()       # 用实时负载重算初始状态
        ok = False
        path = [s]
        for _ in range(hops_cap):
            if env.current == d:
                ok = True; break
            mk = env.valid_action_mask()
            a = agent.greedy_action(st, mk)
            st, _, done = env.step(a)
            path.append(env.current)
            if env.current == d:
                ok = True; break
            if done:
                break
        if env.current == d:
            ok = True
        if ok:
            auton += 1; path = _remove_path_loops(path)
        else:
            path = dijkstra(adj, s, d, la_w)
        if path:
            for a_, b_ in zip(path, path[1:]):
                load[ekey(a_, b_)] = load.get(ekey(a_, b_), 0) + 1
        paths.append(path)
    return paths, auton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--tserv", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--K", type=str, default="500,1000,1500,2000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_d3qn_trained.json")
    args = ap.parse_args()

    print("加载拓扑 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology("ideal")
    print("加载旧 52维模型(推理叠加) ...", flush=True)
    env_o, agent_o = build_d3qn(adj, sats)
    print("加载端到端负载感知 56维模型 ...", flush=True)
    env_t, agent_t, tr_sr = build_trained(adj, sats)
    print(f"  端到端模型干净拓扑 eval_sr≈{tr_sr}", flush=True)

    n = len(sats); rows = []
    print(f"\n对比 (cap={args.cap}):", flush=True)
    for K in [int(x) for x in args.K.split(",")]:
        rng = random.Random(args.seed)
        flows = []
        while len(flows) < K:
            s, d = rng.randrange(n), rng.randrange(n)
            if s != d: flows.append((s, d))
        row = {"K": K}
        row["SP-hop"] = evaluate(route_static(adj, flows, lambda u, v: 1.0), dprop, args.cap, args.tserv)
        row["LA-Dijkstra"] = evaluate(route_la_dijkstra(adj, dprop, flows, args.cap, args.tserv, args.beta), dprop, args.cap, args.tserv)
        po, ao = route_d3qn(env_o, agent_o, adj, dprop, flows, args.cap, args.tserv, lam=args.lam)
        row["D3QN-LA-overlay"] = evaluate(po, dprop, args.cap, args.tserv); row["D3QN-LA-overlay"]["auton"] = round(ao/len(flows), 3)
        pt, at = route_trained(env_t, agent_t, adj, dprop, flows, args.cap, args.tserv)
        row["D3QN-trained"] = evaluate(pt, dprop, args.cap, args.tserv); row["D3QN-trained"]["auton"] = round(at/len(flows), 3)
        rows.append(row)
        sp, la, ov, tr = row["SP-hop"], row["LA-Dijkstra"], row["D3QN-LA-overlay"], row["D3QN-trained"]
        print(f"  K={K:5d} | SP util={sp['max_util']:.2f} good={sp['goodput']:4d} "
              f"| overlay util={ov['max_util']:.2f} good={ov['goodput']:4d} auton={ov['auton']:.0%} "
              f"| trained util={tr['max_util']:.2f} good={tr['goodput']:4d} auton={tr['auton']:.0%} "
              f"| LA-Dij util={la['max_util']:.2f} good={la['goodput']:4d}", flush=True)

    json.dump({"params": vars(args), "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()

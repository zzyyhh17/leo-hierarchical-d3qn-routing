"""最终对比：SP-hop / LA-Dijkstra / D3QN-LA(推理叠加) /
D3QN(随机背景训练) / D3QN(多流累积训练) 在动态拥塞下的峰值利用率与吞吐。"""
from __future__ import annotations
import sys, json, random, argparse
from pathlib import Path

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from dynamic_experiment import (load_topology, ekey, queue_delay_ms, dijkstra,
                                 route_static, route_la_dijkstra, evaluate,
                                 topology_fingerprint, flows_fingerprint,
                                 CANONICAL_REF_TIME)
from dynamic_d3qn import build_d3qn, route_d3qn
from train_load_aware import LoadAwareSatEnv
from dqn.agents import IntraD3QNAgent
from route import _remove_path_loops


def build_56(adj, sats, ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    env = LoadAwareSatEnv(adj, sats, max_hops=ck["max_hops"], max_neighbors=ck["max_neighbors"])
    env.external_load = True
    ag = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=256,
                        network="per_action", base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim, device="cpu")
    ag.q_net.load_state_dict(ck["state_dict"]); ag.q_net.eval()
    return env, ag


def route_56(env, agent, adj, dprop, flows, cap, t_serv):
    """纯贪心(负载感知来自模型) + 负载感知 Dijkstra 兜底。"""
    load = {}
    def la_w(u, v):
        rho = load.get(ekey(u, v), 0) / cap
        return dprop[(u, v)] + 0.3 * (1e7 if rho >= 1 else queue_delay_ms(rho, t_serv))
    paths = []; auton = 0
    for s, d in flows:
        env.link_load = load
        st = env.reset(s, d); mask = env.valid_action_mask()
        path = [s]; ok = False
        for _ in range(env.max_hops):
            if env.current == d: ok = True; break
            a = agent.greedy_action(st, mask)
            st, _, done = env.step(a); mask = env.valid_action_mask()
            path.append(env.current)
            if env.current == d: ok = True; break
            if done: break
        if env.current == d: ok = True
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
    ap.add_argument("--source", default="ideal")
    ap.add_argument("--ref-time", default=CANONICAL_REF_TIME,
                    help="固定拓扑快照UTC时间")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=25); ap.add_argument("--tserv", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--K", type=str, default="500,1000,1500,2000")
    ap.add_argument("--out", default="results_final.json")
    args = ap.parse_args()

    print("加载拓扑 + 模型 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology(args.source, args.ref_time)
    env_o, agent_o = build_d3qn(
        adj, sats, _GOOD / "model/d3qn_per_action_best.pt"
    )  # 旧52维 + 叠加
    env_r, agent_r = build_56(
        adj, sats, _GOOD / "model/d3qn_load_aware.pt"
    )  # 随机背景训练
    env_m, agent_m = build_56(
        adj, sats, _GOOD / "model/d3qn_load_aware_mf.pt"
    )  # 多流累积训练
    n = len(sats); rows = []
    print(f"\n最终对比 (cap={args.cap})  [峰值利用率/吞吐/自主率]:", flush=True)
    for K in [int(x) for x in args.K.split(",")]:
        rng = random.Random(args.seed); flows = []
        while len(flows) < K:
            s, d = rng.randrange(n), rng.randrange(n)
            if s != d: flows.append((s, d))
        row = {"K": K, "flows_sha256": flows_fingerprint(flows)}
        row["SP-hop"] = evaluate(route_static(adj, flows, lambda u, v: 1.0), dprop, args.cap, args.tserv)
        row["LA-Dijkstra"] = evaluate(route_la_dijkstra(adj, dprop, flows, args.cap, args.tserv, args.beta), dprop, args.cap, args.tserv)
        po, ao = route_d3qn(env_o, agent_o, adj, dprop, flows, args.cap, args.tserv, lam=args.lam)
        row["overlay"] = evaluate(po, dprop, args.cap, args.tserv); row["overlay"]["auton"] = round(ao/len(flows), 3)
        pr, ar = route_56(env_r, agent_r, adj, dprop, flows, args.cap, args.tserv)
        row["trained-rand"] = evaluate(pr, dprop, args.cap, args.tserv); row["trained-rand"]["auton"] = round(ar/len(flows), 3)
        pm, am = route_56(env_m, agent_m, adj, dprop, flows, args.cap, args.tserv)
        row["trained-mf"] = evaluate(pm, dprop, args.cap, args.tserv); row["trained-mf"]["auton"] = round(am/len(flows), 3)
        rows.append(row)
        def f(s): x = row[s]; return f"{x['max_util']:.2f}/{x['goodput']:4d}" + (f"/{int(x['auton']*100)}%" if 'auton' in x else "")
        print(f"  K={K:5d} | SP {f('SP-hop')} | overlay {f('overlay')} | rand {f('trained-rand')} | "
              f"mf {f('trained-mf')} | LA-Dij {f('LA-Dijkstra')}", flush=True)
    output = {
        "params": vars(args),
        "topology": {
            "n_sats": len(sats),
            "n_links": len(edges),
            "avg_link_prop_ms": round(sum(dprop.values()) / len(dprop), 3),
            "sha256": topology_fingerprint(adj, dprop),
        },
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""在真实多流累积负载（自致拥塞）上微调负载感知 D3QN。

相对 train_load_aware.py 的随机背景拥塞，本脚本按 eval 同样的方式生成训练负载：
每"轮"路由一批流，链路负载随智能体自身的路由决策实时累加，后续流必须绕开
前面流造成的拥塞——训练分布与评估分布一致。从 d3qn_load_aware.pt 热启动微调。
"""
from __future__ import annotations
import sys, time, random, argparse
from pathlib import Path
from collections import deque

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from dynamic_experiment import load_topology, ekey, queue_delay_ms, evaluate, route_static
from train_load_aware import LoadAwareSatEnv
from dqn.agents import IntraD3QNAgent
from route import _remove_path_loops


def route_batch(env, agent, load, flows, training, buffer=None, upd=None, max_hops=60):
    """按累积负载顺序路由一批流。training=True 用 ε-greedy 并收集转移。返回 (peak_util, goodput, auton)。"""
    auton = 0
    for s, d in flows:
        env.link_load = load
        st = env.reset(s, d)                  # external_load=True → 保留 load
        mask = env.valid_action_mask()
        path = [s]; ok = False; done = False
        for _ in range(max_hops):
            if env.current == d:
                ok = True; break
            a = agent.select_action(st, mask) if training else agent.greedy_action(st, mask)
            ns, r, done = env.step(a)
            nmask = env.valid_action_mask()
            if training and buffer is not None:
                buffer.push(st, a, r, ns, done, nmask)
                if upd: upd()
            path.append(env.current); st = ns; mask = nmask
            if env.current == d:
                ok = True; break
            if done:
                break
        if env.current == d:
            ok = True
        if ok:
            auton += 1
            cp = _remove_path_loops(path)
            for a_, b_ in zip(cp, cp[1:]):
                load[ekey(a_, b_)] = load.get(ekey(a_, b_), 0) + 1
    # 指标
    rhos = [c / env.cap for c in load.values()]
    return (max(rhos) if rhos else 0.0), auton


def cong_eval(env, agent, adj, flows, cap, t_serv):
    """固定测试批：纯贪心(负载感知来自模型)路由，返回 (peak_util, goodput)。"""
    env.external_load = True
    load = {}
    paths = []
    for s, d in flows:
        env.link_load = load
        st = env.reset(s, d); mask = env.valid_action_mask()
        path = [s]
        for _ in range(env.max_hops):
            if env.current == d: break
            a = agent.greedy_action(st, mask)
            st, _, done = env.step(a); mask = env.valid_action_mask()
            path.append(env.current)
            if env.current == d or done: break
        if env.current == d:
            cp = _remove_path_loops(path); paths.append(cp)
            for a_, b_ in zip(cp, cp[1:]):
                load[ekey(a_, b_)] = load.get(ekey(a_, b_), 0) + 1
        else:
            paths.append(None)
    # 用 dprop 估指标需要 dprop；这里只看 util/goodput（goodput=未过载交付流）
    from dynamic_experiment import _build_isl_adj  # noqa
    m = evaluate(paths, DPROP, cap, t_serv)
    return m["max_util"], m["goodput"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warm", default="model/d3qn_load_aware.pt")
    ap.add_argument("--beta_load", type=float, default=0.8)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--tserv", type=float, default=0.6)
    ap.add_argument("--out", default="model/d3qn_load_aware_mf.pt")
    args = ap.parse_args()

    global DPROP
    print("加载拓扑 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology("ideal")
    DPROP = dprop
    n = len(sats)

    env = LoadAwareSatEnv(adj, sats, max_hops=60, max_neighbors=4,
                          cap=args.cap, t_serv=args.tserv, beta_load=args.beta_load)
    env.external_load = True
    agent = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=256,
                           epsilon_start=0.30, epsilon_end=0.03, epsilon_decay=6000,
                           device="cpu", network="per_action",
                           base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim,
                           lr=3e-4, lr_T_max=40000, target_tau=0.003)
    ck = torch.load(args.warm, map_location="cpu", weights_only=False)
    agent.q_net.load_state_dict(ck["state_dict"]); agent.target_net.load_state_dict(ck["state_dict"])
    print(f"  从 {args.warm} 热启动 (state_dim={env.state_dim}, beta_load={args.beta_load})", flush=True)

    rng = random.Random(0)
    test_flows = []
    while len(test_flows) < 1000:
        s, d = rng.randrange(n), rng.randrange(n)
        if s != d: test_flows.append((s, d))

    gstep = [0]
    def upd():
        gstep[0] += 1
        if gstep[0] % 4 == 0:
            agent.update()

    best_score = -1e9; best_sd = None; t0 = time.time()
    print(f"  多流累积负载微调 {args.rounds} 轮 ...", flush=True)
    for rd in range(1, args.rounds + 1):
        K = 200 + (rd % 10) * 120          # 课程：200→1280 循环，覆盖不同拥塞
        load = {}
        rng2 = random.Random(1000 + rd)
        flows = []
        while len(flows) < K:
            s, d = rng2.randrange(n), rng2.randrange(n)
            if s != d: flows.append((s, d))
        env.external_load = True
        pu, au = route_batch(env, agent, load, flows, training=True, buffer=agent.buffer, upd=upd)
        if rd % 3 == 0 or rd == args.rounds:
            tu, tg = cong_eval(env, agent, adj, test_flows, args.cap, args.tserv)
            score = tg - 150 * max(0.0, tu - 1.0)     # 高吞吐 + 罚峰值过载
            tag = ""
            if score > best_score:
                best_score = score
                best_sd = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}
                tag = " ★"
            print(f"    round {rd:3d} (K={K:4d}) train_peak={pu:.2f} auton={au/len(flows):.0%} | "
                  f"test@1000 peak_util={tu:.2f} goodput={tg:4d} score={score:.0f}{tag} "
                  f"eps={agent.epsilon:.2f} [{time.time()-t0:.0f}s]", flush=True)

    if best_sd is None:
        best_sd = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}
    torch.save({"state_dict": best_sd, "state_dim": env.state_dim, "action_dim": env.action_dim,
                "max_hops": env.max_hops, "max_neighbors": env.max_neighbors,
                "network": "per_action", "nb_feat_dim": env.per_nb_dim, "base_dim": env.base_dim,
                "best_score": best_score}, args.out)
    print(f"\n微调完成，最佳 score={best_score:.0f}，模型已存 {args.out}（耗时 {time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()

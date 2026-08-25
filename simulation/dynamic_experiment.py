"""动态场景实验：负载/拥塞 + 链路故障（真实 Starlink Gen1-1 1584 拓扑，纯 Python，无需 torch）。

回答审稿核心质疑——静态等权跳数下 Dijkstra 即最优、学习路由无意义；
本实验在多流并发负载与链路故障下，验证：
 (1) 负载盲目最短路 Dijkstra 在高负载下严重拥塞，而负载感知双权重路由
     （传播时延 + 排队拥塞代价）显著降低拥塞/丢包/时延；
 (2) 纯局部信息逐跳负载感知策略 + 三级容错（贪心→兜底）可分布式实现该收益；
 (3) 分层架构把单链路故障的影响范围隔离在域内（≈M_max 颗），远小于扁平网络的 O(N)。

路由策略：
  SP-hop      最小跳数 Dijkstra（负载盲目，传统基线）
  SP-dist     最小传播距离 Dijkstra（负载盲目）
  LA-Dijkstra 负载感知双权重 Dijkstra（集中式，近似最优负载均衡）
  Local-LA    纯局部信息逐跳负载感知 + LA-Dijkstra 兜底（分布式三级容错）
"""
from __future__ import annotations

import sys, os, json, math, heapq, random, argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

from data_loader import load_gen11_at_time
from isl import build_isl_from_satellites, update_satellite_isl_peers
from route import _build_isl_adj, _sat_distance
from group import run_grouping

C_LIGHT_KM_S = 299792.458   # 真空光速，激光 ISL
CANONICAL_REF_TIME = "2026-03-13T09:50:07Z"


# ───────────────────────── 拓扑 ─────────────────────────

def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def flows_fingerprint(flows) -> str:
    return _sha256_json([[int(s), int(d)] for s, d in flows])


def topology_fingerprint(adj, dprop) -> str:
    records = []
    for u, nbs in adj.items():
        for v in nbs:
            if u < v:
                records.append([u, v, round(float(dprop[(u, v)]), 9)])
    records.sort()
    return _sha256_json(records)


def load_topology(source="ideal", ref_time=CANONICAL_REF_TIME):
    if ref_time is None:
        ref_time = CANONICAL_REF_TIME
    sats = load_gen11_at_time(ref_time, source=source)
    build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    adj = _build_isl_adj(sats)
    edges = set()
    dprop = {}
    for u in adj:
        for v in adj[u]:
            if u < v:
                ms = _sat_distance(sats, u, v) / C_LIGHT_KM_S * 1000.0
                dprop[(u, v)] = dprop[(v, u)] = ms
                edges.add((u, v))
    ecef = {}
    R = 6371.0
    for i, s in enumerate(sats):
        p = s.position
        la, lo = math.radians(p.latitude_deg), math.radians(p.longitude_deg)
        r = R + p.height_km
        ecef[i] = (r*math.cos(la)*math.cos(lo), r*math.cos(la)*math.sin(lo), r*math.sin(la))
    return sats, adj, edges, dprop, ecef


def ekey(u, v): return (u, v) if u < v else (v, u)


# ───────────────────────── 拥塞模型 ─────────────────────────

def queue_delay_ms(rho, t_serv):
    r = min(rho, 0.99)
    return t_serv * r / (1.0 - r)


# ───────────────────────── Dijkstra ─────────────────────────

def dijkstra(adj, src, dst, w, banned=None):
    pq = [(0.0, src)]; dist = {src: 0.0}; prev = {}; seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u == dst: break
        for v in adj[u]:
            if v in seen: continue
            if banned and ekey(u, v) in banned: continue
            nd = d + w(u, v)
            if v not in dist or nd < dist[v]:
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if dst not in dist: return None
    path = [dst]
    while path[-1] != src: path.append(prev[path[-1]])
    return path[::-1]


def bfs_hops(adj, src, banned=None):
    """单源最短跳数距离 dict。"""
    dist = {src: 0}; q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if banned and ekey(u, v) in banned: continue
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


# ───────────────────────── 路由策略 ─────────────────────────

def route_static(adj, flows, w):
    return [dijkstra(adj, s, d, w) for s, d in flows]


def route_la_dijkstra(adj, dprop, flows, cap, t_serv, beta, hops_cap=80):
    load = {}
    def w(u, v):
        rho = load.get(ekey(u, v), 0) / cap
        pen = 1e7 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
        return dprop[(u, v)] + beta * pen
    paths = []
    for s, d in flows:
        p = dijkstra(adj, s, d, w)
        if p and len(p) - 1 <= hops_cap:
            for a, b in zip(p, p[1:]):
                load[ekey(a, b)] = load.get(ekey(a, b), 0) + 1
        paths.append(p)
    return paths


def route_local_la(adj, dprop, ecef, flows, cap, t_serv, beta_local, hops_cap=80):
    """纯局部信息逐跳负载感知 + 三级容错（贪心失败→ LA-Dijkstra 兜底）。

    返回 (paths, autonomous)：autonomous = 局部贪心自主到达(未用兜底)的流数。
    """
    load = {}
    def geo_ms(a, b):
        ax, ay, az = ecef[a]; bx, by, bz = ecef[b]
        return math.sqrt((ax-bx)**2+(ay-by)**2+(az-bz)**2) / C_LIGHT_KM_S * 1000.0
    def la_w(u, v):
        rho = load.get(ekey(u, v), 0) / cap
        pen = 1e7 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
        return dprop[(u, v)] + beta_local * pen
    paths = []; autonomous = 0
    for s, d in flows:
        cur = s; path = [s]; visited = {s}; ok = False
        for _ in range(hops_cap):
            if cur == d:
                ok = True; break
            best, best_c = None, float("inf")
            for v in adj[cur]:
                if v in visited: continue
                rho = load.get(ekey(cur, v), 0) / cap
                pen = 1e7 if rho >= 1.0 else queue_delay_ms(rho, t_serv)
                # 局部代价：本跳(传播+排队) + 到目的的几何下界（仅 1 跳邻居信息）
                c = dprop[(cur, v)] + beta_local * pen + geo_ms(v, d)
                if c < best_c:
                    best_c, best = c, v
            if best is None: break
            path.append(best); visited.add(best); cur = best
        if cur == d:
            ok = True
        if ok:
            autonomous += 1
        else:
            # 三级容错：局部失败 → 负载感知 Dijkstra 兜底
            path = dijkstra(adj, s, d, la_w)
        if path:
            for a, b in zip(path, path[1:]):
                load[ekey(a, b)] = load.get(ekey(a, b), 0) + 1
        paths.append(path)
    return paths, autonomous


# ───────────────────────── 指标 ─────────────────────────

def evaluate(paths, dprop, cap, t_serv, cong_thresh=0.85):
    load = {}; delivered = 0
    for p in paths:
        if not p: continue
        delivered += 1
        for a, b in zip(p, p[1:]):
            load[ekey(a, b)] = load.get(ekey(a, b), 0) + 1
    rhos = [c / cap for c in load.values()]
    max_util = max(rhos) if rhos else 0.0
    n_used = len(load) or 1
    n_cong = sum(1 for r in rhos if r > cong_thresh)
    lat = []; dropped = 0
    for p in paths:
        if not p:
            dropped += 1; continue
        d = 0.0; bad = False
        for a, b in zip(p, p[1:]):
            rho = load[ekey(a, b)] / cap
            if rho >= 1.0: bad = True
            d += dprop[(a, b)] + queue_delay_ms(rho, t_serv)
        if bad: dropped += 1
        else: lat.append(d)
    lat.sort()
    def pct(q): return lat[min(len(lat)-1, int(q*len(lat)))] if lat else 0.0
    total = len(paths)
    return {
        "avg_latency_ms": round(sum(lat)/len(lat), 2) if lat else 0.0,
        "p95_latency_ms": round(pct(0.95), 2),
        "max_util": round(max_util, 3),
        "congested_link_frac": round(n_cong/n_used, 4),
        "drop_rate": round(dropped/total, 4) if total else 0.0,
        "goodput": len(lat),
    }


# ───────────────────────── 实验 A：负载/拥塞 ─────────────────────────

def experiment_congestion(sats, adj, dprop, ecef, K_list, cap, t_serv, beta,
                          seed=42, hotspot=False):
    n = len(sats)
    rng0 = random.Random(seed)
    hot = None
    if hotspot:
        c = rng0.randrange(n)
        hot = sorted(range(n), key=lambda j: sum((ecef[j][k]-ecef[c][k])**2 for k in range(3)))[:max(20, n//40)]
    results = []
    for K in K_list:
        rng = random.Random(seed)
        flows = []
        while len(flows) < K:
            s = rng.randrange(n)
            d = hot[rng.randrange(len(hot))] if hotspot else rng.randrange(n)
            if s != d: flows.append((s, d))
        row = {"K": K, "flows_sha256": flows_fingerprint(flows)}
        row["SP-hop"]  = evaluate(route_static(adj, flows, lambda u, v: 1.0), dprop, cap, t_serv)
        row["SP-dist"] = evaluate(route_static(adj, flows, lambda u, v: dprop[(u, v)]), dprop, cap, t_serv)
        row["LA-Dijkstra"] = evaluate(route_la_dijkstra(adj, dprop, flows, cap, t_serv, beta), dprop, cap, t_serv)
        ll_paths, auton = route_local_la(adj, dprop, ecef, flows, cap, t_serv, beta_local=0.3)
        row["Local-LA"] = evaluate(ll_paths, dprop, cap, t_serv)
        row["Local-LA"]["autonomous_rate"] = round(auton/len(flows), 3)
        results.append(row)
        sp, la, ll = row["SP-hop"], row["LA-Dijkstra"], row["Local-LA"]
        print(f"  K={K:5d} | SP-hop drop={sp['drop_rate']:.1%} util={sp['max_util']:.2f} lat={sp['avg_latency_ms']:.0f} "
              f"|| LA-Dij drop={la['drop_rate']:.1%} util={la['max_util']:.2f} lat={la['avg_latency_ms']:.0f} "
              f"|| Local-LA drop={ll['drop_rate']:.1%} auton={ll['autonomous_rate']:.0%}", flush=True)
    return results


# ───────────────────────── 实验 C：链路故障 ─────────────────────────

def experiment_failure(sats, adj, dprop, fail_fracs, n_pairs=500, gs=20, seed=42):
    """链路故障下的可达率、路径拉伸，以及单链路故障的"影响范围(blast radius)"。"""
    n = len(sats)
    all_edges = [ekey(u, v) for u in adj for v in adj[u] if u < v]
    rng = random.Random(seed)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(n_pairs)]
    pairs = [(s, d) for s, d in pairs if s != d]

    # 基线最短跳
    base = {}
    for s, d in pairs:
        dist = bfs_hops(adj, s)
        base[(s, d)] = dist.get(d)

    # 分域（用于"影响范围"上界 M_max）
    for s in sats: s.group_id = -1
    run_grouping(sats, max_size=gs)
    gsize = {}
    for s in sats:
        gsize[s.group_id] = gsize.get(s.group_id, 0) + 1
    avg_dom = sum(gsize.values()) / len(gsize)

    results = []
    for f in fail_fracs:
        rng = random.Random(seed + int(f*1000))
        nban = int(f * len(all_edges))
        banned = set(rng.sample(all_edges, nban))
        deliv = 0; stretch = []
        for s, d in pairs:
            dist = bfs_hops(adj, s, banned=banned)
            if d in dist:
                deliv += 1
                if base[(s, d)]:
                    stretch.append(dist[d] / base[(s, d)])
        results.append({
            "fail_frac": f,
            "delivery_rate": round(deliv/len(pairs), 4),
            "avg_path_stretch": round(sum(stretch)/len(stretch), 4) if stretch else 0.0,
        })
        print(f"  fail={f:.0%} | 可达率={deliv/len(pairs):.1%} 路径拉伸={sum(stretch)/len(stretch) if stretch else 0:.3f}", flush=True)

    # blast radius：随机抽样单链路故障，统计有多少节点到某固定目的的最短跳距离改变
    rng = random.Random(seed)
    blast = []
    dests = [rng.randrange(n) for _ in range(20)]
    for d in dests:
        base_dist = bfs_hops(adj, d)
        for _ in range(5):
            e = all_edges[rng.randrange(len(all_edges))]
            nd = bfs_hops(adj, d, banned={e})
            changed = sum(1 for u in range(n)
                          if base_dist.get(u, -1) != nd.get(u, -1))
            blast.append(changed)
    avg_blast = sum(blast) / len(blast)
    print(f"  单链路故障平均影响节点数(扁平最短路)={avg_blast:.0f} / N={n}；分层域内上界≈{avg_dom:.0f}", flush=True)
    return {
        "per_fraction": results,
        "blast_radius_flat": round(avg_blast, 1),
        "N": n,
        "avg_domain_size": round(avg_dom, 1),
        "scope_reduction": round(1 - avg_dom / avg_blast, 4) if avg_blast else 0.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ideal")
    ap.add_argument("--ref-time", default=CANONICAL_REF_TIME,
                    help="固定拓扑快照UTC时间")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--tserv", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--K", type=str, default="100,250,500,1000,1500,2000,3000")
    ap.add_argument("--hotspot", action="store_true")
    ap.add_argument("--failfracs", type=str, default="0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--skip-failure", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("加载拓扑 ...", flush=True)
    sats, adj, edges, dprop, ecef = load_topology(args.source, args.ref_time)
    print(f"  卫星={len(sats)} 链路={len(edges)} 平均链路传播时延={sum(dprop.values())/len(dprop):.2f}ms", flush=True)

    out = {"params": vars(args), "topology": {"n_sats": len(sats), "n_links": len(edges),
            "avg_link_prop_ms": round(sum(dprop.values())/len(dprop), 3),
            "sha256": topology_fingerprint(adj, dprop)}}

    K_list = [int(x) for x in args.K.split(",")]
    print(f"\n实验A 负载/拥塞 (cap={args.cap}, t_serv={args.tserv}ms, beta={args.beta}, hotspot={args.hotspot}):", flush=True)
    out["congestion"] = experiment_congestion(
        sats, adj, dprop, ecef, K_list, args.cap, args.tserv, args.beta,
        seed=args.seed, hotspot=args.hotspot
    )

    if not args.skip_failure:
        print(f"\n实验C 链路故障:", flush=True)
        ffr = [float(x) for x in args.failfracs.split(",")]
        out["failure"] = experiment_failure(sats, adj, dprop, ffr, seed=args.seed)

    fn = args.out or f"dynamic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {fn}", flush=True)

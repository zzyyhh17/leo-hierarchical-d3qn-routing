"""
路由算法对比实验:
  1. dijkstra      — 全局最短路径 (理论最优, baseline)
  2. dqn_dijkstra  — DQN 组间 + Dijkstra 组内
  3. dqn_greedy    — DQN 组间 + 贪心角度组内
  4. greedy        — 全局贪心角度

对同一组随机 src/dst 测试所有算法, 统计成功率、平均跳数、与最优解的差距比.
"""
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timezone

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from data_loader import load_gen11_at_time
from isl import build_isl_from_satellites, update_satellite_isl_peers
from group import run_grouping
from dqn_routing import (
    build_group_graph, RoutingEnv, DQNAgent, trace_route,
    dijkstra_group_path, dijkstra_group_path_hops,
)
from route import (
    _build_isl_adj, _build_ecef_cache, _dijkstra_shortest_path,
    _greedy_angular_route, _find_border_pairs, _sat_distance, _remove_path_loops,
    _single_source_dijkstra,
)


def _stitch(sats, adj, group_members, group_path, src_idx, dst_idx, seg_fn):
    """组间路由的卫星级逐段拼接 (空间距离评分)。"""
    full_path = []
    cursor = src_idx
    for seg in range(len(group_path) - 1):
        g_cur, g_next = group_path[seg], group_path[seg + 1]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if not border_pairs:
            return None
        best_exit, best_entry, best_score = -1, -1, float("inf")
        for ex, en in border_pairs:
            sc = _sat_distance(sats, cursor, ex) + _sat_distance(sats, en, dst_idx)
            if sc < best_score:
                best_score, best_exit, best_entry = sc, ex, en
        intra = seg_fn(adj, cursor, best_exit)
        if intra is None:
            return None
        if not full_path:
            full_path.extend(intra)
        else:
            full_path.extend(intra[1:])
        full_path.append(best_entry)
        cursor = best_entry
    last = seg_fn(adj, cursor, dst_idx)
    if last is None:
        return None
    full_path.extend(last[1:])
    return _remove_path_loops(full_path)


def _stitch_topo(sats, adj, group_members, group_path, src_idx, dst_idx, seg_fn,
                 avg_isl_km=1200.0):
    """拓扑距离 + 空间距离联合评分。"""
    full_path = []
    cursor = src_idx
    for seg in range(len(group_path) - 1):
        g_cur, g_next = group_path[seg], group_path[seg + 1]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if not border_pairs:
            return None

        dists_from_cursor, _ = _single_source_dijkstra(adj, cursor)

        best_exit, best_entry, best_score = -1, -1, float("inf")
        for ex, en in border_pairs:
            topo_hops = dists_from_cursor.get(ex, 999)
            topo_km = topo_hops * avg_isl_km
            spatial_km = _sat_distance(sats, en, dst_idx)
            sc = topo_km + spatial_km
            if sc < best_score:
                best_score, best_exit, best_entry = sc, ex, en

        intra = seg_fn(adj, cursor, best_exit)
        if intra is None:
            return None
        if not full_path:
            full_path.extend(intra)
        else:
            full_path.extend(intra[1:])
        full_path.append(best_entry)
        cursor = best_entry
    last = seg_fn(adj, cursor, dst_idx)
    if last is None:
        return None
    full_path.extend(last[1:])
    return _remove_path_loops(full_path)


def _stitch_exact(sats, adj, group_members, group_path, src_idx, dst_idx, seg_fn):
    """最优出组选择: 用精确跳数双向评分。

    pre-compute dst 的全图 Dijkstra (一次),
    每段 compute cursor 的 Dijkstra,
    score = exact_hops(cursor→exit) + exact_hops(entry→dst)
    """
    dists_from_dst, _ = _single_source_dijkstra(adj, dst_idx)

    full_path = []
    cursor = src_idx
    for seg in range(len(group_path) - 1):
        g_cur, g_next = group_path[seg], group_path[seg + 1]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if not border_pairs:
            return None

        dists_from_cursor, _ = _single_source_dijkstra(adj, cursor)

        best_exit, best_entry, best_score = -1, -1, float("inf")
        for ex, en in border_pairs:
            h_to_exit = dists_from_cursor.get(ex, 9999)
            h_to_dst = dists_from_dst.get(en, 9999)
            sc = h_to_exit + h_to_dst
            if sc < best_score:
                best_score, best_exit, best_entry = sc, ex, en

        intra = seg_fn(adj, cursor, best_exit)
        if intra is None:
            return None
        if not full_path:
            full_path.extend(intra)
        else:
            full_path.extend(intra[1:])
        full_path.append(best_entry)
        cursor = best_entry
    last = seg_fn(adj, cursor, dst_idx)
    if last is None:
        return None
    full_path.extend(last[1:])
    return _remove_path_loops(full_path)


def _build_subgraph(adj, group_members, group_path, src_idx, dst_idx,
                    expand="none", graph_ref=None):
    """构建子图的允许节点集合。

    expand:
        "none"  — 仅 path 中各组卫星
        "1hop"  — + 1-hop ISL 邻居
        "2hop"  — + 2-hop ISL 邻居
        "group" — + 邻居组全部卫星
    """
    core = set()
    for gid in group_path:
        core.update(group_members.get(gid, []))
    core.add(src_idx)
    core.add(dst_idx)

    if expand == "1hop":
        allowed = set(core)
        for node in core:
            for nb in adj.get(node, []):
                allowed.add(nb)
    elif expand == "2hop":
        ring1 = set()
        for node in core:
            for nb in adj.get(node, []):
                if nb not in core:
                    ring1.add(nb)
        allowed = core | ring1
        for node in ring1:
            for nb in adj.get(node, []):
                allowed.add(nb)
    elif expand == "group" and graph_ref:
        gid_set = set(group_path)
        for gid in group_path:
            gn = graph_ref.get(gid)
            if gn:
                gid_set.update(gn.neighbors)
        allowed = set()
        for gid in gid_set:
            allowed.update(group_members.get(gid, []))
        allowed.add(src_idx)
        allowed.add(dst_idx)
    else:
        allowed = core

    return allowed


def _shortcut_group_path(group_path, graph_ref):
    """贪心简化组路径: 跳过可直达的中间组。

    例 [G3,G5,G7,G12] 若 G3→G7 直连 → [G3,G7,G12]
    """
    if len(group_path) <= 2 or graph_ref is None:
        return group_path
    short = [group_path[0]]
    i = 0
    while i < len(group_path) - 1:
        best_j = i + 1
        for j in range(len(group_path) - 1, i, -1):
            gn = graph_ref.get(group_path[i])
            if gn and group_path[j] in gn.neighbors:
                best_j = j
                break
        short.append(group_path[best_j])
        i = best_j
    return short


def _build_subgraph_border2hop(adj, group_members, sats, group_path,
                               src_idx, dst_idx):
    """边界定向 2hop: 只从组边界卫星做 2hop 扩展, 大幅减少搜索规模。

    border 卫星 = 在当前组但有 ISL 连接到相邻组的卫星。
    """
    path_gids = set(group_path)
    core = set()
    for gid in group_path:
        core.update(group_members.get(gid, []))
    core.add(src_idx)
    core.add(dst_idx)

    gid_of = {}
    for gid in path_gids:
        for sat_idx in group_members.get(gid, []):
            gid_of[sat_idx] = gid

    border_sats = set()
    for node in core:
        ng = gid_of.get(node, -1)
        for nb in adj.get(node, []):
            nb_g = gid_of.get(nb, -2)
            if nb_g != ng:
                border_sats.add(node)
                break

    ring1 = set()
    for node in border_sats:
        for nb in adj.get(node, []):
            if nb not in core:
                ring1.add(nb)
    allowed = core | ring1
    for node in ring1:
        for nb in adj.get(node, []):
            allowed.add(nb)

    return allowed


def _subgraph_dijkstra(adj, allowed, src_idx, dst_idx):
    """在 allowed 节点集上跑 Dijkstra, 返回 (path, size)。"""
    sub_adj: dict[int, list[int]] = {}
    for node in allowed:
        sub_adj[node] = [nb for nb in adj.get(node, []) if nb in allowed]
    return _dijkstra_shortest_path(sub_adj, src_idx, dst_idx), len(allowed)


def run_benchmark(n_tests: int = 100, source: str = "ideal",
                  max_group_size: int = 9, isl_mode: str = "orbit"):
    print("=" * 70)
    print(f"  路由算法对比实验 (max_group_size={max_group_size}, ISL={isl_mode})")
    print("=" * 70)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载卫星数据...")
    ref_time = datetime.now(timezone.utc)
    sats = load_gen11_at_time(ref_time, source=source)
    if isl_mode == "nearest":
        from isl import build_isl_nearest
        build_isl_nearest(sats)
        update_satellite_isl_peers(sats, mode="nearest")
        print(f"  ISL 模式: 最近原则 (k={4})")
    else:
        build_isl_from_satellites(sats)
        update_satellite_isl_peers(sats)
        print(f"  ISL 模式: 同轨+邻轨")
    run_grouping(sats, max_size=max_group_size)
    print(f"  卫星数: {len(sats)}")

    adj = _build_isl_adj(sats)
    ecef = _build_ecef_cache(sats)

    group_members: dict[int, list[int]] = {}
    for i, s in enumerate(sats):
        if s.group_id >= 0:
            group_members.setdefault(s.group_id, []).append(i)
    print(f"  分组数: {len(group_members)}")

    # ── 2. 加载 / 训练 DQN ──
    print("\n[2/4] 加载 DQN 模型...")
    graph = build_group_graph(sats)
    max_nb = max(len(n.neighbors) for n in graph.values())
    env = RoutingEnv(graph, sats, max_hops=30, max_neighbors=max_nb)

    isl_suffix = f"_{isl_mode}" if isl_mode != "orbit" else ""
    model_suffix = f"_g{max_group_size}" if max_group_size != 9 else ""
    model_path = _GOOD / f"dqn_routing_model{model_suffix}{isl_suffix}.pt"
    if model_path.is_file():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        saved_sd = ckpt.get("state_dim", env.state_dim)
        saved_ad = ckpt.get("action_dim", env.action_dim)
        if saved_sd != env.state_dim or saved_ad != env.action_dim:
            env = RoutingEnv(graph, sats, max_hops=30, max_neighbors=saved_ad)
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=128)
        if saved_sd == env.state_dim:
            agent.q_net.load_state_dict(ckpt["q_net"])
            agent.target_net.load_state_dict(ckpt["q_net"])
            print("  已加载训练好的 DQN 模型")
        else:
            print(f"  [WARN] state_dim 不匹配 ({saved_sd} vs {env.state_dim}), 使用未训练模型")
    else:
        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden=128)
        print("  [WARN] 未找到模型文件, 使用未训练模型")

    # ── 3. 生成测试对 ──
    print(f"\n[3/4] 生成 {n_tests} 组随机 src/dst 测试对...")
    grouped = [i for i, s in enumerate(sats) if s.group_id >= 0]
    test_pairs = []
    for _ in range(n_tests):
        si, di = random.sample(grouped, 2)
        test_pairs.append((si, di))

    # ── 4. 算法实现 ──
    def algo_dijkstra(src, dst):
        return _dijkstra_shortest_path(adj, src, dst)

    def algo_greedy(src, dst):
        path = _greedy_angular_route(adj, src, dst, ecef)
        if path is None:
            path = _dijkstra_shortest_path(adj, src, dst)
        return path

    def algo_dqn_dijkstra(src, dst):
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp, _ = trace_route(env, agent, src_gid, dst_gid)
        if gp[-1] != dst_gid:
            return None
        return _stitch(sats, adj, group_members, gp, src, dst,
                       lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    def algo_dqn_greedy(src, dst):
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            p = _greedy_angular_route(adj, src, dst, ecef)
            return p if p else _dijkstra_shortest_path(adj, src, dst)
        gp, _ = trace_route(env, agent, src_gid, dst_gid)
        if gp[-1] != dst_gid:
            return None
        def _seg(a, s, d):
            p = _greedy_angular_route(a, s, d, ecef)
            return p if p else _dijkstra_shortest_path(a, s, d)
        return _stitch(sats, adj, group_members, gp, src, dst, _seg)

    def algo_grp_dij_dist(src, dst):
        """组图 Dijkstra(距离权) + 卫星级 Dijkstra (空间评分)"""
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp = dijkstra_group_path(graph, src_gid, dst_gid, sats)
        if gp is None:
            return None
        return _stitch(sats, adj, group_members, gp, src, dst,
                       lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    def algo_grp_dij_hops(src, dst):
        """组图 Dijkstra(等权跳数) + 卫星级 Dijkstra (空间评分)"""
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp = dijkstra_group_path_hops(graph, src_gid, dst_gid)
        if gp is None:
            return None
        return _stitch(sats, adj, group_members, gp, src, dst,
                       lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    def algo_grp_dij_exact(src, dst):
        """组图 Dijkstra(跳数) + 精确跳数评分"""
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp = dijkstra_group_path_hops(graph, src_gid, dst_gid)
        if gp is None:
            return None
        return _stitch_exact(sats, adj, group_members, gp, src, dst,
                             lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    def algo_dqn_dij_exact(src, dst):
        """DQN 组间 + 精确跳数评分"""
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp, _ = trace_route(env, agent, src_gid, dst_gid)
        if gp[-1] != dst_gid:
            return None
        return _stitch_exact(sats, adj, group_members, gp, src, dst,
                             lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    def algo_grp_dist_exact(src, dst):
        """组图 Dijkstra(距离权) + 精确跳数评分"""
        src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
        if src_gid == dst_gid:
            return _dijkstra_shortest_path(adj, src, dst)
        gp = dijkstra_group_path(graph, src_gid, dst_gid, sats)
        if gp is None:
            return None
        return _stitch_exact(sats, adj, group_members, gp, src, dst,
                             lambda a, s, d: _dijkstra_shortest_path(a, s, d))

    n_sats = len(sats)

    def _make_subgr(group_fn, expand_mode, fallback=None, shortcut=False,
                    border_only=False):
        """生成子图算法。

        shortcut: 是否对组路径做贪心简化
        border_only: 是否只从边界卫星做扩展
        """
        def algo(src, dst):
            src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
            if src_gid == dst_gid:
                return _dijkstra_shortest_path(adj, src, dst), n_sats
            gp = group_fn(src_gid, dst_gid)
            if gp is None:
                return None, 0
            if shortcut:
                gp = _shortcut_group_path(gp, graph)
            if border_only:
                allowed = _build_subgraph_border2hop(
                    adj, group_members, sats, gp, src, dst)
            else:
                allowed = _build_subgraph(adj, group_members, gp, src, dst,
                                          expand=expand_mode, graph_ref=graph)
            path, size = _subgraph_dijkstra(adj, allowed, src, dst)
            if path is not None or fallback is None:
                return path, size
            allowed2 = _build_subgraph(adj, group_members, gp, src, dst,
                                       expand=fallback, graph_ref=graph)
            return _subgraph_dijkstra(adj, allowed2, src, dst)
        return algo

    def _grp_path(src_gid, dst_gid):
        return dijkstra_group_path_hops(graph, src_gid, dst_gid)

    def _dqn_path(src_gid, dst_gid):
        gp, _ = trace_route(env, agent, src_gid, dst_gid)
        return gp if gp[-1] == dst_gid else None

    def _wrap(fn):
        def wrapped(src, dst):
            return fn(src, dst), n_sats
        return wrapped

    algorithms = [
        ("dijkstra",        "全局 Dijkstra (baseline)",           _wrap(algo_dijkstra)),
        ("dqn+2hop",        "DQN + 2hop扩展",                    _make_subgr(_dqn_path, "2hop")),
        ("dqn+2hop→grp",    "DQN + 2hop→邻居组",                 _make_subgr(_dqn_path, "2hop", fallback="group")),
        ("dqn+sc+2hop",     "DQN + 路径简化 + 2hop",             _make_subgr(_dqn_path, "2hop", shortcut=True)),
        ("dqn+sc+2h→g",     "DQN + 简化 + 2hop→邻居组",         _make_subgr(_dqn_path, "2hop", shortcut=True, fallback="group")),
        ("dqn+border2h",    "DQN + 边界2hop",                    _make_subgr(_dqn_path, None, border_only=True)),
        ("dqn+bdr2h→g",     "DQN + 边界2hop→邻居组",            _make_subgr(_dqn_path, None, border_only=True, fallback="group")),
        ("dqn+sc+bdr→g",    "DQN + 简化 + 边界2hop→邻居组",     _make_subgr(_dqn_path, None, shortcut=True, border_only=True, fallback="group")),
        ("dqn+group",       "DQN + 邻居组扩展",                  _make_subgr(_dqn_path, "group")),
        ("grp+2hop→grp",    "组图Dij + 2hop→邻居组 (参考)",      _make_subgr(_grp_path, "2hop", fallback="group")),
    ]

    # ── 5. 运行对比 ──
    print(f"\n[4/4] 运行对比实验 ({n_tests} 组测试)...\n")
    results: dict[str, dict] = {}

    for name, desc, fn in algorithms:
        hops_list = []
        subgraph_sizes = []
        successes = 0
        t0 = time.time()

        for src, dst in test_pairs:
            path, sg_size = fn(src, dst)
            if path is not None and len(path) >= 2:
                successes += 1
                hops_list.append(len(path) - 1)
                subgraph_sizes.append(sg_size)

        elapsed = time.time() - t0
        avg_sg = sum(subgraph_sizes) / len(subgraph_sizes) if subgraph_sizes else 0
        results[name] = {
            "desc": desc,
            "success": successes,
            "hops": hops_list,
            "time": elapsed,
            "avg_subgraph": avg_sg,
        }

    # ── 6. 输出结果 ──
    dijkstra_hops = results["dijkstra"]["hops"]

    W = 90
    print("─" * W)
    print(f"{'算法':<18} {'成功率':>8} {'平均跳':>7} {'中位跳':>7} "
          f"{'最大':>5} {'vs最优':>7} {'搜索规模':>8} {'耗时':>8} {'单次ms':>7}")
    print("─" * W)

    for name, _, _ in algorithms:
        r = results[name]
        n = len(r["hops"])
        rate = f"{r['success']}/{n_tests}"
        if n == 0:
            print(f"{name:<18} {rate:>8} {'N/A':>7} {'N/A':>7} "
                  f"{'N/A':>5} {'N/A':>7} {'N/A':>8} {r['time']:>7.2f}s {'N/A':>7}")
            continue

        avg = sum(r["hops"]) / n
        sorted_h = sorted(r["hops"])
        med = sorted_h[n // 2]
        mx = max(r["hops"])
        avg_sg = r["avg_subgraph"]
        per_ms = r["time"] / n_tests * 1000

        if name == "dijkstra":
            ratio_str = "1.00x"
        elif len(dijkstra_hops) > 0:
            opt_avg = sum(dijkstra_hops) / len(dijkstra_hops)
            ratio_str = f"{avg / opt_avg:.2f}x"
        else:
            ratio_str = "N/A"

        sg_str = f"{avg_sg:.0f}" if avg_sg < n_sats else "全网"

        print(f"{name:<18} {rate:>8} {avg:>7.1f} {med:>7} "
              f"{mx:>5} {ratio_str:>7} {sg_str:>8} {r['time']:>7.2f}s {per_ms:>6.2f}")

    print("─" * W)

    dij_avg = sum(dijkstra_hops) / len(dijkstra_hops) if dijkstra_hops else 0

    # 逐对比较: 成功率最高且搜索规模最小的 DQN 方案
    dqn_candidates = ["dqn+1hop", "dqn+group", "dqn+pure", "dqn+exact"]
    dqn_key = next((k for k in dqn_candidates
                    if k in results and results[k]["success"] >= n_tests * 0.95), "dqn+exact")

    if dijkstra_hops and results.get(dqn_key, {}).get("hops"):
        dqn_dij = results[dqn_key]["hops"]
        n_compare = min(len(dijkstra_hops), len(dqn_dij))
        better, equal, worse, total_overhead = 0, 0, 0, 0
        for i in range(n_compare):
            diff = dqn_dij[i] - dijkstra_hops[i]
            total_overhead += diff
            if diff < 0: better += 1
            elif diff == 0: equal += 1
            else: worse += 1

        print(f"\n逐对比较 ({dqn_key} vs dijkstra, {n_compare} 对):")
        print(f"  DQN 更优: {better} ({better/n_compare*100:.1f}%)  "
              f"相同: {equal} ({equal/n_compare*100:.1f}%)  "
              f"Dijkstra 更优: {worse} ({worse/n_compare*100:.1f}%)")
        print(f"  平均额外跳数: {total_overhead/n_compare:+.1f}")

    print(f"\n{'=' * W}")
    print("结论:")
    print(f"  全局 Dijkstra: {dij_avg:.1f} 跳 (理论最优, 搜索 {n_sats} 节点)")
    show_keys = [n for n, _, _ in algorithms if n != "dijkstra" and n != "greedy"]
    for key in show_keys:
        r = results.get(key)
        if not r or not r["hops"]:
            continue
        avg = sum(r["hops"]) / len(r["hops"])
        pct = (avg / dij_avg - 1) * 100 if dij_avg > 0 else 0
        sg_pct = r["avg_subgraph"] / n_sats * 100
        print(f"  {key:<16s} {avg:5.1f} 跳 ({pct:+5.1f}%)  "
              f"成功 {r['success']:>3}/{n_tests}  "
              f"搜索 {r['avg_subgraph']:>5.0f} ({sg_pct:4.0f}%)  "
              f"{r['time']/n_tests*1000:.2f}ms")
    print("=" * W)


def sweep_group_sizes(n_tests: int = 200, source: str = "ideal",
                      sizes: list[int] | None = None):
    """批量测试不同 max_group_size, 仅用 grp_hop+subgr (无需 DQN), 找最优组大小。"""
    if sizes is None:
        sizes = [9, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500]

    print("=" * 78)
    print("  子图 Dijkstra 组大小扫描实验")
    print("=" * 78)

    ref_time = datetime.now(timezone.utc)
    sats = load_gen11_at_time(ref_time, source=source)
    build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    adj = _build_isl_adj(sats)
    n_sats = len(sats)
    print(f"  卫星数: {n_sats}")

    random.seed(42)
    all_indices = list(range(n_sats))
    test_pairs = []
    for _ in range(n_tests):
        si, di = random.sample(all_indices, 2)
        test_pairs.append((si, di))

    # 全局 Dijkstra baseline (不受 group_size 影响)
    print(f"\n  计算全局 Dijkstra baseline ({n_tests} 对)...")
    dij_hops = []
    for src, dst in test_pairs:
        path = _dijkstra_shortest_path(adj, src, dst)
        if path and len(path) >= 2:
            dij_hops.append(len(path) - 1)
    dij_avg = sum(dij_hops) / len(dij_hops) if dij_hops else 0
    print(f"  Dijkstra baseline: {dij_avg:.1f} 平均跳数 ({len(dij_hops)}/{n_tests} 成功)")

    print(f"\n  扫描 {len(sizes)} 种 group_size...\n")

    print("─" * 78)
    print(f"{'group_size':>10} {'分组数':>6} {'成功率':>10} {'平均跳':>8} {'中位跳':>8} "
          f"{'最大':>6} {'vs最优':>8} {'耗时':>8}")
    print("─" * 78)

    summary = []
    for gs in sizes:
        for s in sats:
            s.group_id = -1
        run_grouping(sats, max_size=gs)

        group_members: dict[int, list[int]] = {}
        for i, s in enumerate(sats):
            if s.group_id >= 0:
                group_members.setdefault(s.group_id, []).append(i)
        n_groups = len(group_members)

        graph = build_group_graph(sats)

        hops_list = []
        successes = 0
        t0 = time.time()

        for src, dst in test_pairs:
            src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
            if src_gid < 0 or dst_gid < 0:
                continue
            if src_gid == dst_gid:
                path = _dijkstra_shortest_path(adj, src, dst)
            else:
                gp = dijkstra_group_path_hops(graph, src_gid, dst_gid)
                path = None
                if gp is not None:
                    allowed = _build_subgraph(adj, group_members, gp, src, dst,
                                             expand="2hop", graph_ref=graph)
                    path, _ = _subgraph_dijkstra(adj, allowed, src, dst)
            if path is not None and len(path) >= 2:
                successes += 1
                hops_list.append(len(path) - 1)

        elapsed = time.time() - t0

        if hops_list:
            avg = sum(hops_list) / len(hops_list)
            sorted_h = sorted(hops_list)
            med = sorted_h[len(sorted_h) // 2]
            mx = max(hops_list)
            ratio = avg / dij_avg if dij_avg > 0 else 0
        else:
            avg = med = mx = 0
            ratio = 0

        rate_str = f"{successes}/{n_tests}"
        ratio_str = f"{ratio:.2f}x"
        print(f"{gs:>10} {n_groups:>6} {rate_str:>10} {avg:>8.1f} {med:>8} "
              f"{mx:>6} {ratio_str:>8} {elapsed:>7.2f}s")

        summary.append({
            "group_size": gs, "n_groups": n_groups,
            "success": successes, "avg": avg, "med": med, "max": mx,
            "ratio": ratio, "time": elapsed,
        })

    print("─" * 78)

    # 找最优: 在成功率 >= 95% 的方案中, ratio 最小的
    viable = [s for s in summary if s["success"] >= n_tests * 0.95]
    if viable:
        best = min(viable, key=lambda s: s["ratio"])
        print(f"\n最优组大小: {best['group_size']}")
        print(f"  分组数: {best['n_groups']}, 成功率: {best['success']}/{n_tests}")
        print(f"  平均跳数: {best['avg']:.1f} (vs Dijkstra {dij_avg:.1f}, "
              f"额外 {(best['ratio']-1)*100:+.1f}%)")
    else:
        print("\n未找到成功率 >= 95% 的方案")

    # 也找成功率 100% 中的最优
    perfect = [s for s in summary if s["success"] == n_tests]
    if perfect:
        best100 = min(perfect, key=lambda s: s["ratio"])
        if not viable or best100["group_size"] != best["group_size"]:
            print(f"\n100%成功率最优: group_size={best100['group_size']}")
            print(f"  分组数: {best100['n_groups']}")
            print(f"  平均跳数: {best100['avg']:.1f} (额外 {(best100['ratio']-1)*100:+.1f}%)")

    print("=" * 78)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="路由算法对比实验")
    parser.add_argument("--tests", type=int, default=100, help="测试用例数 (default: 100)")
    parser.add_argument("--source", default="ideal", help="数据来源 (default: ideal)")
    parser.add_argument("--group-size", type=int, default=9, help="每组最大卫星数 (default: 9)")
    parser.add_argument("--sweep", action="store_true", help="扫描多种 group_size")
    parser.add_argument("--sizes", type=str, default=None,
                        help="逗号分隔的 group_size 列表, 如 '20,30,50,80,120'")
    parser.add_argument("--isl", default="orbit", choices=["orbit", "nearest"],
                        help="ISL 模式: orbit=同轨邻轨, nearest=最近原则")
    args = parser.parse_args()

    if args.sweep:
        sizes = None
        if args.sizes:
            sizes = [int(x.strip()) for x in args.sizes.split(",")]
        sweep_group_sizes(n_tests=args.tests, source=args.source, sizes=sizes)
    else:
        run_benchmark(n_tests=args.tests, source=args.source,
                      max_group_size=args.group_size, isl_mode=args.isl)

"""
对比实验：全局 Dijkstra vs 组内D3QN+组间Dijkstra

串行扫描 group_size，对比两种算法：
  A) dijkstra        — 全局 Dijkstra (baseline)
  B) intra_d3qn      — 组内 D3QN + 组间 Dijkstra (优化拼接)
"""
import sys
import os
import time
import random
import pickle
from pathlib import Path
from datetime import datetime, timezone
import multiprocessing as mp

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from data_loader import load_gen11_at_time
from isl import build_isl_from_satellites, update_satellite_isl_peers
from group import run_grouping
from dqn_routing import (
    build_group_graph, dijkstra_group_path_hops,
)
from dqn import (
    train_intra_d3qn, intra_dqn_route, intra_dqn_beam_route,
    IntraSatEnv, IntraD3QNAgent,
)
from route import (
    _build_isl_adj, _dijkstra_shortest_path, _find_border_pairs,
    _sat_distance, _remove_path_loops,
)


def _quick_eval(env, agent, n_pairs=50, seed=99):
    """快速评估 greedy 成功率和平均跳数，返回 eval_score (越小越好)。"""
    rng = random.Random(seed)
    all_idx = list(range(env.n_sats))
    succ, hops_list = 0, []
    for _ in range(n_pairs):
        src, dst = rng.sample(all_idx, 2)
        st = env.reset(src, dst)
        mk = env.valid_action_mask()
        for _ in range(env.max_hops):
            a = agent.greedy_action(st, mk)
            st, _, d = env.step(a)
            mk = env.valid_action_mask()
            if d:
                break
        if env.current == env.dest:
            succ += 1
            hops_list.append(env.hops)
    sr = succ / n_pairs if n_pairs > 0 else 0
    avg_h = sum(hops_list) / len(hops_list) if hops_list else env.max_hops
    return avg_h / sr if sr >= 0.3 else float("inf"), sr


def _train_d3qn_worker(args):
    """训练 D3QN, 返回 (algo_name, state_dict, env_params, network)。

    支持多种子训练: args[9] = n_seeds (默认 1)。
    """
    algo, adj, sats_data, n_episodes, max_hops, log_interval, network = args[:7]
    use_bfs = args[7] if len(args) > 7 else True
    use_per = args[8] if len(args) > 8 else False
    n_seeds = args[9] if len(args) > 9 else 1
    sys.stdout.reconfigure(line_buffering=True)
    pid = os.getpid()
    tag_str = ", no-bfs" if not use_bfs else ""
    if use_per:
        tag_str += ", PER"
    if n_seeds > 1:
        tag_str += f", {n_seeds}seeds"
    print(f"  [PID {pid}] D3QN 训练开始 ({n_episodes} episodes, network={network}{tag_str})", flush=True)
    t0 = time.time()
    sats = pickle.loads(sats_data)

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    best_sd = None
    best_score = float("inf")
    best_env = None

    for seed_i in range(n_seeds):
        if n_seeds > 1:
            random.seed(42 + seed_i * 1000)
            torch.manual_seed(42 + seed_i * 1000)
            print(f"  [PID {pid}] --- seed {seed_i + 1}/{n_seeds} ---", flush=True)

        agent, env = train_intra_d3qn(
            adj, sats, n_episodes=n_episodes, max_hops=max_hops,
            log_interval=log_interval, device=train_device, update_every=4,
            network=network, use_bfs=use_bfs, use_per=use_per)

        score, sr = _quick_eval(env, agent)
        print(f"  [PID {pid}] seed {seed_i + 1} eval: score={score:.1f} sr={sr:.0%}",
              flush=True)
        if score < best_score:
            best_score = score
            best_sd = {k: v.cpu() for k, v in agent.q_net.state_dict().items()}
            best_env = env

    if n_seeds > 1:
        print(f"  [PID {pid}] ★ 最佳种子 eval_score={best_score:.1f}", flush=True)

    result = (algo, best_sd,
              {"max_hops": best_env.max_hops, "max_neighbors": best_env.max_neighbors,
               "use_bfs": use_bfs},
              network)
    print(f"  [PID {pid}] D3QN 训练完成, 耗时 {time.time() - t0:.1f}s", flush=True)
    return result


_d3qn_stats = {"calls": 0, "d3qn_ok": 0, "beam_ok": 0, "dij_fallback": 0}


def _d3qn_or_dij(adj, intra_env, intra_agent, src, dst):
    """D3QN 路由：greedy → beam search → Dijkstra fallback。"""
    _d3qn_stats["calls"] += 1
    path = intra_dqn_route(intra_env, intra_agent, src, dst)
    if path is not None:
        _d3qn_stats["d3qn_ok"] += 1
        return path
    path = intra_dqn_beam_route(intra_env, intra_agent, src, dst, beam_width=10)
    if path is not None:
        _d3qn_stats["beam_ok"] += 1
        return path
    _d3qn_stats["dij_fallback"] += 1
    return _dijkstra_shortest_path(adj, src, dst)


def _stitch_with_intra_dqn(sats, adj, group_members, group_path,
                            src_idx, dst_idx, intra_env, intra_agent):
    """组间 Dijkstra + 组内 D3QN 拼接。

    gateway 选择：按欧氏距离预筛 top-3，找到首条路径后提前终止。
    """
    full_path = []
    cursor = src_idx
    for seg in range(len(group_path) - 1):
        g_cur, g_next = group_path[seg], group_path[seg + 1]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if not border_pairs:
            return None

        scored = []
        for ex, en in border_pairs:
            sc = _sat_distance(sats, cursor, ex) + _sat_distance(sats, en, dst_idx)
            scored.append((sc, ex, en))
        scored.sort()

        best_seg_path = None
        best_entry = -1
        for _, ex, en in scored[:3]:
            seg_path = _d3qn_or_dij(adj, intra_env, intra_agent, cursor, ex)
            if seg_path is not None:
                if best_seg_path is None or len(seg_path) < len(best_seg_path):
                    best_seg_path = seg_path
                    best_entry = en
                    break

        if best_seg_path is None:
            return None
        if not full_path:
            full_path.extend(best_seg_path)
        else:
            full_path.extend(best_seg_path[1:])
        full_path.append(best_entry)
        cursor = best_entry

    last = _d3qn_or_dij(adj, intra_env, intra_agent, cursor, dst_idx)
    if last is None:
        return None
    full_path.extend(last[1:])
    return _remove_path_loops(full_path)


def _stitch_cross_group(sats, adj, group_members, group_path,
                        src_idx, dst_idx, intra_env, intra_agent):
    """跨组拼接：D3QN 直接路由到下一组的 entry 卫星（跨越组边界）。

    与 _stitch_with_intra_dqn 的区别：不拆分 cursor→exit + jump→entry，
    而是让 D3QN 直接从 cursor 路由到 entry，自然经过 ISL 跨组链路。
    """
    full_path = []
    cursor = src_idx

    for seg in range(len(group_path) - 1):
        g_cur, g_next = group_path[seg], group_path[seg + 1]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if not border_pairs:
            return None

        # 选下一组中距目标最近的 entry 卫星作为 D3QN 目标
        scored = []
        for ex, en in border_pairs:
            sc = _sat_distance(sats, cursor, en) + _sat_distance(sats, en, dst_idx)
            scored.append((sc, en))
        scored.sort()

        best_seg_path = None
        for _, target_en in scored[:3]:
            seg_path = _d3qn_or_dij(adj, intra_env, intra_agent, cursor, target_en)
            if seg_path is not None:
                if best_seg_path is None or len(seg_path) < len(best_seg_path):
                    best_seg_path = seg_path
                    break

        if best_seg_path is None:
            return None
        if not full_path:
            full_path.extend(best_seg_path)
        else:
            full_path.extend(best_seg_path[1:])
        cursor = full_path[-1]

    # 最后一段: cursor → dst
    last = _d3qn_or_dij(adj, intra_env, intra_agent, cursor, dst_idx)
    if last is None:
        return None
    full_path.extend(last[1:])
    return _remove_path_loops(full_path)


def _stitch_crossN_group(sats, adj, group_members, group_path,
                         src_idx, dst_idx, intra_env, intra_agent,
                         stride: int = 2):
    """跨 N 组拼接：D3QN 一次跨越 stride 个组边界。

    对 group_path 每隔 stride 个组取一个中继点。
    stride=1 等价于 cross1, stride=2 等价于 cross2, 以此类推。
    """
    full_path = []
    cursor = src_idx

    waypoints = []
    seg = 0
    while seg < len(group_path) - 1:
        next_seg = min(seg + stride, len(group_path) - 1)
        if next_seg == len(group_path) - 1:
            waypoints.append(dst_idx)
            break
        # 在 next_seg-1 → next_seg 的边界找 entry
        g_cur = group_path[next_seg - 1]
        g_next = group_path[next_seg]
        border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
        if border_pairs:
            scored = sorted(
                [(_sat_distance(sats, cursor, en) + _sat_distance(sats, en, dst_idx), en)
                 for _, en in border_pairs])
            waypoints.append(scored[0][1])
        else:
            # 退回到步长 1
            g_cur1 = group_path[seg]
            g_next1 = group_path[seg + 1]
            bp1 = _find_border_pairs(sats, adj, group_members, g_cur1, g_next1)
            if bp1:
                scored = sorted(
                    [(_sat_distance(sats, cursor, en) + _sat_distance(sats, en, dst_idx), en)
                     for _, en in bp1])
                waypoints.append(scored[0][1])
                next_seg = seg + 1
            else:
                return None
        seg = next_seg

    for target in waypoints:
        seg_path = _d3qn_or_dij(adj, intra_env, intra_agent, cursor, target)
        if seg_path is None:
            return None
        if not full_path:
            full_path.extend(seg_path)
        else:
            full_path.extend(seg_path[1:])
        cursor = full_path[-1]

    return _remove_path_loops(full_path)


def _eval_one_group_size(args):
    """单个 group_size 的评估任务。评估 Dijkstra vs 多种 D3QN 拼接策略。"""
    gs, sats_data, adj, n_tests, d3qn_state, network = args

    sats = pickle.loads(sats_data)
    d3qn_agent_sd, d3qn_env_params = d3qn_state

    t_start = time.time()
    for s in sats:
        s.group_id = -1
    run_grouping(sats, max_size=gs)
    print(f"    [gs={gs}] 分组完成, 耗时 {time.time() - t_start:.1f}s", flush=True)

    group_members: dict[int, list[int]] = {}
    for i, s in enumerate(sats):
        if s.group_id >= 0:
            group_members.setdefault(s.group_id, []).append(i)
    n_groups = len(group_members)

    grouped_idx = [i for i, s in enumerate(sats) if s.group_id >= 0]
    random.seed(42)
    test_pairs = [tuple(random.sample(grouped_idx, 2)) for _ in range(n_tests)]

    graph = build_group_graph(sats)

    d3qn_env = IntraSatEnv(adj, sats, max_hops=d3qn_env_params["max_hops"],
                           max_neighbors=d3qn_env_params["max_neighbors"],
                           use_bfs=d3qn_env_params.get("use_bfs", True))
    d3qn_agent = IntraD3QNAgent(
        state_dim=d3qn_env.state_dim, action_dim=d3qn_env.action_dim,
        hidden=256, network=network,
        base_dim=d3qn_env.base_dim, nb_feat_dim=d3qn_env.per_nb_dim)
    d3qn_agent.q_net.load_state_dict(d3qn_agent_sd)
    d3qn_agent.target_net.load_state_dict(d3qn_agent_sd)

    # ── 算法 A: 全局 Dijkstra ──
    dij_res = {"success": 0, "hops": []}
    for src, dst in test_pairs:
        path = _dijkstra_shortest_path(adj, src, dst)
        if path and len(path) >= 2:
            dij_res["success"] += 1
            dij_res["hops"].append(len(path) - 1)

    # ── 评估多种跨组步长 ──
    strides_to_test = [1, 2, 3, 4, 5, 6]
    results = {}
    for stride in strides_to_test:
        algo_name = f"cross{stride}"
        _d3qn_stats["calls"] = _d3qn_stats["d3qn_ok"] = _d3qn_stats["beam_ok"] = _d3qn_stats["dij_fallback"] = 0
        t_algo = time.time()
        res = {"success": 0, "hops": []}
        for src, dst in test_pairs:
            src_gid, dst_gid = sats[src].group_id, sats[dst].group_id
            if src_gid == dst_gid:
                path = _d3qn_or_dij(adj, d3qn_env, d3qn_agent, src, dst)
            else:
                gp = dijkstra_group_path_hops(graph, src_gid, dst_gid)
                if gp is None:
                    path = None
                else:
                    path = _stitch_crossN_group(
                        sats, adj, group_members, gp, src, dst,
                        d3qn_env, d3qn_agent, stride=stride)
            if path and len(path) >= 2:
                res["success"] += 1
                res["hops"].append(len(path) - 1)

        s = _d3qn_stats
        total = s["calls"]
        ok_pct = (s["d3qn_ok"] + s["beam_ok"]) / total * 100 if total > 0 else 0
        print(f"    [gs={gs}] {algo_name} 完成, 耗时 {time.time() - t_algo:.1f}s  "
              f"(greedy={s['d3qn_ok']} beam={s['beam_ok']} ok={ok_pct:.0f}% fallback={s['dij_fallback']})",
              flush=True)
        res["d3qn_greedy"] = s["d3qn_ok"]
        res["d3qn_beam"] = s["beam_ok"]
        res["d3qn_fallback"] = s["dij_fallback"]
        res["d3qn_calls"] = total
        results[algo_name] = res

    return {
        "group_size": gs, "n_groups": n_groups,
        "dijkstra": dij_res,
        **results,
    }


def run_sweep(group_sizes: list[int], n_tests: int = 100,
              intra_dqn_episodes: int = 3000,
              source: str = "ideal", network: str = "mlp",
              use_bfs: bool = True, use_per: bool = False,
              n_seeds: int = 1):
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    tag_str = "BFS" if use_bfs else "no-BFS+local"
    if use_per:
        tag_str += "+PER"
    print("=" * 80)
    print(f"  Dijkstra vs D3QN ({tag_str}) "
          f"(network={network}, 训练: {train_device}, update_every=4)")
    print("=" * 80)

    ref_time = datetime.now(timezone.utc)
    sats = load_gen11_at_time(ref_time, source=source)
    build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    adj = _build_isl_adj(sats)
    n_sats = len(sats)
    print(f"  卫星数: {n_sats}")

    # 全局 Dijkstra baseline
    random.seed(42)
    test_pairs_global = [tuple(random.sample(range(n_sats), 2)) for _ in range(n_tests)]
    print(f"\n  计算全局 Dijkstra baseline ({n_tests} 对)...")
    dij_hops = []
    for src, dst in test_pairs_global:
        path = _dijkstra_shortest_path(adj, src, dst)
        if path and len(path) >= 2:
            dij_hops.append(len(path) - 1)
    dij_avg = sum(dij_hops) / len(dij_hops) if dij_hops else 0
    print(f"  Dijkstra baseline: {dij_avg:.1f} 平均跳数 ({len(dij_hops)}/{n_tests} 成功)")

    log_iv = max(intra_dqn_episodes // 5, 500)
    sats_data_train = pickle.dumps(sats)

    train_args = ("d3qn", adj, sats_data_train, intra_dqn_episodes, 60, log_iv,
                  network, use_bfs, use_per, n_seeds)
    seeds_str = f", {n_seeds}seeds" if n_seeds > 1 else ""
    print(f"\n  训练 D3QN ({intra_dqn_episodes} episodes, network={network}, "
          f"bfs={use_bfs}, per={use_per}{seeds_str})...")
    t0 = time.time()
    _, sd, env_params, _ = _train_d3qn_worker(train_args)
    d3qn_state = (sd, env_params)
    print(f"  D3QN 训练完成, 耗时 {time.time() - t0:.1f}s")

    # ── 纯 D3QN (不分组) 直接端到端路由 ──
    print(f"\n  评估纯 D3QN (不分组) 端到端路由 ...")
    d3qn_env_global = IntraSatEnv(adj, sats,
                                  max_hops=env_params["max_hops"],
                                  max_neighbors=env_params["max_neighbors"],
                                  use_bfs=env_params.get("use_bfs", True))
    d3qn_agent_global = IntraD3QNAgent(
        state_dim=d3qn_env_global.state_dim,
        action_dim=d3qn_env_global.action_dim,
        hidden=256, network=network,
        base_dim=d3qn_env_global.base_dim, nb_feat_dim=d3qn_env_global.per_nb_dim)
    d3qn_agent_global.q_net.load_state_dict(sd)
    d3qn_agent_global.target_net.load_state_dict(sd)

    pure_d3qn_greedy = {"success": 0, "hops": [], "fail": 0}
    pure_d3qn_beam = {"success": 0, "hops": [], "fail": 0}
    t_pure = time.time()
    for src, dst in test_pairs_global:
        # greedy
        path_g = intra_dqn_route(d3qn_env_global, d3qn_agent_global, src, dst)
        if path_g and len(path_g) >= 2:
            pure_d3qn_greedy["success"] += 1
            pure_d3qn_greedy["hops"].append(len(path_g) - 1)
        else:
            pure_d3qn_greedy["fail"] += 1
        # beam search
        path_b = intra_dqn_beam_route(d3qn_env_global, d3qn_agent_global,
                                       src, dst, beam_width=10)
        if path_b and len(path_b) >= 2:
            pure_d3qn_beam["success"] += 1
            pure_d3qn_beam["hops"].append(len(path_b) - 1)
        else:
            pure_d3qn_beam["fail"] += 1

    def _fmt(res, label):
        n = len(res["hops"])
        avg = sum(res["hops"]) / n if n else 0
        ratio = avg / dij_avg if dij_avg > 0 else 0
        return (f"  {label}: 成功={res['success']}/{n_tests}, "
                f"平均跳数={avg:.1f}, vs Dijkstra={ratio:.2f}x, "
                f"失败={res['fail']}")

    print(f"  纯 D3QN 评估完成, 耗时 {time.time() - t_pure:.1f}s")
    print(_fmt(pure_d3qn_greedy, "纯D3QN(greedy)"))
    print(_fmt(pure_d3qn_beam, "纯D3QN(beam=10)"))

    sats_data = pickle.dumps(sats)

    tasks = [(gs, sats_data, adj, n_tests, d3qn_state, network)
             for gs in group_sizes]

    print(f"\n  开始串行评估 {len(group_sizes)} 种组大小 ...")
    t1 = time.time()
    all_results = []
    for idx, task in enumerate(tasks, 1):
        gs_val = task[0]
        print(f"\n  [{idx}/{len(tasks)}] 评估 group_size={gs_val} ...", flush=True)
        t_gs = time.time()
        result = _eval_one_group_size(task)
        print(f"  [{idx}/{len(tasks)}] group_size={gs_val} 完成, 耗时 {time.time() - t_gs:.1f}s", flush=True)
        all_results.append(result)
    elapsed = time.time() - t1
    print(f"\n  全部评估完成, 耗时 {elapsed:.1f}s")

    all_results.sort(key=lambda r: r["group_size"])

    strides = [1, 2, 3, 4, 5, 6]
    algo_keys = [(f"cross{s}", f"跨{s}组") for s in strides]

    W = 190
    print(f"\n{'─' * W}")
    hdr = f"{'grp':>4} {'组':>3} │ {'Dijkstra':^10}"
    for _, label in algo_keys:
        hdr += f" │ {label:^28}"
    print(hdr)

    sub = f"{'':>4} {'':>3} │ {'跳数':>6} {'':>3}"
    for _ in algo_keys:
        sub += f" │ {'成功':>4} {'跳数':>6} {'vs最优':>6} {'greedy':>6} {'beam':>5} {'dij':>4}"
    print(sub)
    print(f"{'─' * W}")

    for r in all_results:
        dij_n = len(r["dijkstra"]["hops"])
        dij_a = sum(r["dijkstra"]["hops"]) / dij_n if dij_n else 0
        line = f"{r['group_size']:>4} {r['n_groups']:>3} │ {dij_a:>6.1f} {'':>3}"
        for key, _ in algo_keys:
            if key in r and r[key]["hops"]:
                n = len(r[key]["hops"])
                a = sum(r[key]["hops"]) / n
                ratio = f"{a / dij_a:.2f}x" if dij_a > 0 else "N/A"
                calls = r[key].get("d3qn_calls", 0)
                greedy_pct = f"{r[key].get('d3qn_greedy', 0) / calls * 100:.0f}%" if calls > 0 else "N/A"
                beam_pct = f"{r[key].get('d3qn_beam', 0) / calls * 100:.0f}%" if calls > 0 else "N/A"
                dij_pct = f"{r[key].get('d3qn_fallback', 0) / calls * 100:.0f}%" if calls > 0 else "N/A"
                line += f" │ {r[key]['success']:>4} {a:>6.1f} {ratio:>6} {greedy_pct:>6} {beam_pct:>5} {dij_pct:>4}"
            else:
                line += f" │ {'':>4} {'':>6} {'N/A':>6} {'':>6} {'':>5} {'':>4}"
        print(line)

    print(f"{'─' * W}")

    # 找全局最优
    print(f"\n{'=' * W}")
    print("结论:")
    print(f"  全局 Dijkstra baseline: {dij_avg:.1f} 跳 (理论最优)")
    print(_fmt(pure_d3qn_greedy, "纯D3QN(greedy)"))
    print(_fmt(pure_d3qn_beam, "纯D3QN(beam=10)"))
    print()

    global_best = None
    for key, label in algo_keys:
        best = None
        for r in all_results:
            threshold = n_tests * 0.9
            if key in r and r[key]["success"] >= threshold and r[key]["hops"]:
                avg = sum(r[key]["hops"]) / len(r[key]["hops"])
                if best is None or avg < best[1]:
                    best = (r["group_size"], avg, r[key]["success"])
        if best:
            ratio = best[1] / dij_avg if dij_avg > 0 else 0
            print(f"  {label} 最优: gs={best[0]}, {best[1]:.1f}跳, "
                  f"成功={best[2]}/{n_tests}, {ratio:.2f}x")
            if global_best is None or best[1] < global_best[1]:
                global_best = (label, best[0], best[1], best[2])
        else:
            print(f"  {label}: 未找到成功率 >= 90% 的组大小")

    if global_best:
        ratio = global_best[2] / dij_avg if dij_avg > 0 else 0
        print(f"\n  ★ 全局最优: {global_best[0]}, gs={global_best[1]}, "
              f"{global_best[2]:.1f}跳, 成功={global_best[3]}/{n_tests}, "
              f"{ratio:.2f}x Dijkstra")

    print(f"{'=' * W}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dijkstra vs 组内D3QN+组间Dijkstra 对比实验")
    parser.add_argument("--tests", type=int, default=100, help="每种组大小的测试对数")
    parser.add_argument("--intra-episodes", type=int, default=3000,
                        help="组内D3QN训练轮次 (default: 3000)")
    parser.add_argument("--source", default="ideal")
    parser.add_argument("--sizes", type=str, default=None,
                        help="逗号分隔的 group_size, 如 '4,8,12'")
    parser.add_argument("--network", default="mlp",
                        choices=["mlp", "per_action", "transformer"],
                        help="D3QN 网络类型 (default: mlp)")
    parser.add_argument("--no-bfs", action="store_true",
                        help="禁用 BFS 距离表特征，仅用几何+局部拓扑特征")
    parser.add_argument("--per", action="store_true",
                        help="启用优先经验回放 (PER)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="多种子训练次数 (选最优模型)")
    args = parser.parse_args()

    if args.sizes:
        sizes = [int(x.strip()) for x in args.sizes.split(",")]
    else:
        sizes = list(range(4, 51))

    run_sweep(
        group_sizes=sizes,
        n_tests=args.tests,
        intra_dqn_episodes=args.intra_episodes,
        source=args.source,
        network=args.network,
        use_bfs=not args.no_bfs,
        use_per=args.per,
        n_seeds=args.seeds,
    )

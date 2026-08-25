"""
从 Starlink data 加载 Gen1-1 并做路由等后续处理；数据加载委托给 data_loader。
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

from data_loader import load_gen11_at_time, DEFAULT_DATA_DIR
from group import run_grouping
from isl import build_isl_from_satellites, update_satellite_isl_peers

# 演化目标时刻：当前时间（UTC）
REF_TIME = datetime.now(timezone.utc)


def write_route_grouping_log(
    sats: list,
    groups: list[list[int]],
    unassigned: list[int],
    ref_time: datetime,
    n_rounds: int = 0,
    round_records: list[tuple[int, list[list[int]], list[int]]] | None = None,
    log_dir: Path | None = None,
) -> Path:
    """将加载与分组演化结果写入日志文件，返回日志文件路径。若提供 round_records，会写入每轮演绎结果。"""
    if log_dir is None:
        log_dir = _GOOD / "logs"
    log_dir.mkdir(exist_ok=True)
    lines = [
        f"Loaded {len(sats)} Gen1-1 satellites (evolved to {ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')})",
        "",
        "Sample (first 30):",
    ]
    for s in sats[:30]:
        p = s.position
        lines.append(f"  {s.name} cat={s.catalog_number} lat={p.latitude_deg:.2f}° lon={p.longitude_deg:.2f}° alt={p.height_km:.1f} km")
    if round_records:
        lines.extend(["", "--- Per-round evolution ---"])
        for r, gs, u in round_records:
            assigned = sum(len(g) for g in gs)
            sizes = [len(g) for g in gs]
            lines.append(f"  Round {r}: {len(gs)} groups, assigned {assigned}, unassigned {len(u)}; group sizes {sizes[:20]}{'...' if len(sizes) > 20 else ''}")
    lines.extend([
        "",
        "--- Grouping result (final) ---",
        f"Satellites: {len(sats)}, Groups: {len(groups)}, Unassigned: {len(unassigned)}, Evolution rounds: {n_rounds}",
        "(catalog_number = NORAD 唯一编号)",
    ])
    for i, g in enumerate(groups):
        cats = [sats[idx].catalog_number for idx in g]
        cats_str = str(cats) if len(cats) == 9 else f"{cats[:5]}{'...' if len(cats) > 5 else ''}"
        lines.append(f"  Group {i+1}: {len(g)} sats, catalog_number {cats_str}")
    if unassigned:
        unassigned_cats = [sats[idx].catalog_number for idx in unassigned[:20]]
        lines.append("Unassigned (independent) satellite catalog_number: " + str(unassigned_cats) + (" ..." if len(unassigned) > 20 else ""))
    log_content = "\n".join(lines)
    log_file = log_dir / f"route_grouping_{ref_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file.write_text(log_content, encoding="utf-8")
    return log_file


def write_route_grouping_json(
    sats: list,
    groups: list[list[int]],
    unassigned: list[int],
    ref_time: datetime,
    n_rounds: int = 0,
    round_records: list[tuple[int, list[list[int]], list[int]]] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """将分组演化的最终结果写入 JSON 文件。groups/unassigned 以 catalog_number 存储便于复用。"""
    if out_dir is None:
        out_dir = _GOOD / "group-result"
    out_dir.mkdir(exist_ok=True)

    def idx_to_cats(indices: list[int]) -> list[int]:
        return [sats[i].catalog_number for i in indices]

    payload = {
        "ref_time": ref_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_rounds": n_rounds,
        "num_satellites": len(sats),
        "num_groups": len(groups),
        "num_unassigned": len(unassigned),
        "groups": [idx_to_cats(g) for g in groups],
        "unassigned": idx_to_cats(unassigned),
    }
    if round_records:
        payload["round_records"] = [
            {
                "round": r,
                "groups": [idx_to_cats(grp) for grp in gs],
                "unassigned": idx_to_cats(u),
            }
            for r, gs, u in round_records
        ]

    json_file = out_dir / f"route_grouping_{ref_time.strftime('%Y%m%d_%H%M%S')}.json"
    json_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_file


def write_isl_log(
    sats: list,
    ref_time: datetime,
    isl_edges_count: int,
    log_dir: Path | None = None,
    plane_sizes: list[int] | None = None,
) -> Path:
    """将每颗卫星的 ISL 情况（连接数、邻居 catalog_number）写入日志文件。若提供 plane_sizes 则写入各轨道面卫星数。"""
    if log_dir is None:
        log_dir = _GOOD / "logs"
    log_dir.mkdir(exist_ok=True)
    lines = [
        f"ISL log (ref_time={ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')}, total_edges={isl_edges_count})",
        "",
    ]
    if plane_sizes:
        lines.append(f"Orbital planes ({len(plane_sizes)}): satellites per plane")
        for i, count in enumerate(plane_sizes):
            lines.append(f"  Plane {i+1:2d}: {count} satellites")
        lines.extend(["", "Per-satellite: catalog_number, name, plane_id, lat, lon, alt_km, isl_links, connected_peers", "---"])
    else:
        lines.extend(["Per-satellite: catalog_number, name, plane_id, lat, lon, alt_km, isl_links, connected_peers", "---"])
    for s in sats:
        p = s.position
        peers = getattr(s.isl, "connected_peers", []) or []
        links = getattr(s.isl, "active_links", len(peers))
        plane_id = getattr(s, "plane_id", -1)
        peers_str = ",".join(str(c) for c in peers[:8])
        if len(peers) > 8:
            peers_str += f",...(+{len(peers)-8})"
        lines.append(
            f"  {s.catalog_number}  {s.name}  plane={plane_id}  "
            f"{p.latitude_deg:.4f}  {p.longitude_deg:.4f}  {p.height_km:.2f}  "
            f"links={links}  peers=[{peers_str}]"
        )
    log_file = log_dir / f"isl_{ref_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file.write_text("\n".join(lines), encoding="utf-8")
    return log_file


def _build_isl_adj(sats):
    """构建卫星级 ISL 邻接表: sat_index → set[sat_index]。"""
    cat_to_idx = {s.catalog_number: i for i, s in enumerate(sats)}
    adj: dict[int, set[int]] = {i: set() for i in range(len(sats))}
    for i, s in enumerate(sats):
        for peer_cat in s.isl.connected_peers:
            j = cat_to_idx.get(peer_cat)
            if j is not None:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _bfs_shortest_path(adj, src_idx, dst_idx, allowed=None):
    """BFS 最短路径, 返回 index 序列; 找不到返回 None。

    allowed: 可选的允许经过的节点集合 (用于限制在组内搜索)。
    """
    from collections import deque
    if src_idx == dst_idx:
        return [src_idx]
    visited = {src_idx}
    queue = deque([(src_idx, [src_idx])])
    while queue:
        cur, path = queue.popleft()
        for nb in adj[cur]:
            if nb in visited:
                continue
            if allowed is not None and nb not in allowed:
                continue
            new_path = path + [nb]
            if nb == dst_idx:
                return new_path
            visited.add(nb)
            queue.append((nb, new_path))
    return None


def _find_border_pairs(sats, adj, group_members, gid_a, gid_b):
    """找出两组之间所有跨组 ISL 边, 返回 [(idx_in_a, idx_in_b), ...]。"""
    members_b = set(group_members[gid_b])
    pairs = []
    for idx_a in group_members[gid_a]:
        for nb in adj[idx_a]:
            if nb in members_b:
                pairs.append((idx_a, nb))
    return pairs


def _dijkstra_shortest_path(adj, src_idx, dst_idx, allowed=None):
    """Dijkstra 最短路径 (等权), 返回 index 序列; 找不到返回 None。

    allowed: 可选允许经过的节点集合。
    """
    import heapq
    if src_idx == dst_idx:
        return [src_idx]
    dist = {src_idx: 0}
    prev: dict[int, int] = {}
    heap = [(0, src_idx)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        if cur == dst_idx:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(src_idx)
            return path[::-1]
        for nb in adj[cur]:
            if allowed is not None and nb not in allowed:
                continue
            nd = d + 1
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = cur
                heapq.heappush(heap, (nd, nb))
    return None


def _sat_distance(sats, i, j):
    """两颗卫星之间的 ECEF 距离 (km)。"""
    from group import distance_km
    p1, p2 = sats[i].position, sats[j].position
    return distance_km(
        p1.latitude_deg, p1.longitude_deg, p1.height_km,
        p2.latitude_deg, p2.longitude_deg, p2.height_km,
    )


def _single_source_dijkstra(adj, src, allowed=None):
    """单源 Dijkstra (等权), 返回 (dist_dict, prev_dict)。

    dist_dict: {node: 最短跳数}, prev_dict: {node: 前驱节点}
    """
    import heapq
    dist = {src: 0}
    prev: dict[int, int] = {}
    heap = [(0, src)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        for nb in adj[cur]:
            if allowed is not None and nb not in allowed:
                continue
            nd = d + 1
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = cur
                heapq.heappush(heap, (nd, nb))
    return dist, prev


def _extract_path_from_prev(prev, src, dst):
    """从 prev dict 中还原路径。"""
    if dst != src and dst not in prev:
        return None
    path = []
    cur = dst
    while cur != src:
        path.append(cur)
        cur = prev[cur]
    path.append(src)
    return path[::-1]


def _remove_path_loops(path: list[int]) -> list[int]:
    """移除路径中的环路: 若某节点出现多次, 从首次跳到末次的下一个。

    保证结果路径仍然合法 (每相邻对在原路径中存在)。
    """
    if len(path) <= 1:
        return path
    last_pos: dict[int, int] = {}
    for i, node in enumerate(path):
        last_pos[node] = i
    result: list[int] = []
    i = 0
    while i < len(path):
        node = path[i]
        result.append(node)
        jump = last_pos[node]
        i = (jump + 1) if jump > i else (i + 1)
    return result


def _build_ecef_cache(sats):
    """预计算所有卫星的 ECEF 坐标缓存, 避免重复三角函数计算。"""
    import math
    R = 6371.0
    cache: dict[int, tuple[float, float, float]] = {}
    for i, s in enumerate(sats):
        p = s.position
        lat = math.radians(p.latitude_deg)
        lon = math.radians(p.longitude_deg)
        r = R + p.height_km
        cl, sl = math.cos(lat), math.sin(lat)
        co, so = math.cos(lon), math.sin(lon)
        cache[i] = (r * cl * co, r * cl * so, r * sl)
    return cache


def _greedy_angular_route(adj, src_idx, dst_idx, ecef, max_hops=300):
    """贪心角度路由: 每一跳选择与到目标方向夹角最小的 ISL 邻居。

    ecef: {sat_idx: (x, y, z)} ECEF 坐标缓存。
    返回路径 index 序列; 无法到达返回 None。
    """
    import math

    if src_idx == dst_idx:
        return [src_idx]

    dx_t, dy_t, dz_t = ecef[dst_idx]

    path = [src_idx]
    visited = {src_idx}
    current = src_idx

    for _ in range(max_hops):
        if current == dst_idx:
            return path

        cx, cy, cz = ecef[current]
        vx, vy, vz = dx_t - cx, dy_t - cy, dz_t - cz
        d2 = vx * vx + vy * vy + vz * vz
        if d2 < 1.0:
            return path

        best_nb = -1
        best_cos = -2.0

        for nb in adj[current]:
            if nb in visited:
                continue
            nx, ny, nz = ecef[nb]
            ex, ey, ez = nx - cx, ny - cy, nz - cz
            e2 = ex * ex + ey * ey + ez * ez
            if e2 < 1.0:
                continue
            cos_a = (vx * ex + vy * ey + vz * ez) / math.sqrt(d2 * e2)
            if cos_a > best_cos:
                best_cos = cos_a
                best_nb = nb

        if best_nb < 0:
            return None

        path.append(best_nb)
        visited.add(best_nb)
        current = best_nb

    return None


def _compute_avg_isl_distance(sats, adj):
    """计算所有 ISL 边的平均距离 (km), 用于将空间距离转换为估计跳数。"""
    total = 0.0
    count = 0
    for i, neighbors in adj.items():
        for j in neighbors:
            if j > i:
                total += _sat_distance(sats, i, j)
                count += 1
    return total / count if count > 0 else 1500.0


def test_full_route(sats, agent, env, n_tests: int = 10):
    """完整路由测试: 组间 DQN + 组内贪心角度路由。

    组内路由逻辑:
      每一跳选择与「当前位置→段目标」方向夹角最小的 ISL 邻居,
      保证每步都朝目标方向前进。
    """
    import random
    from dqn_routing import trace_route

    adj = _build_isl_adj(sats)
    ecef = _build_ecef_cache(sats)

    group_members: dict[int, list[int]] = {}
    for i, s in enumerate(sats):
        if s.group_id >= 0:
            group_members.setdefault(s.group_id, []).append(i)

    grouped = [s for s in sats if s.group_id >= 0]

    print(f"\n=== 完整路由测试 ({n_tests} 组, 组间DQN + 贪心角度路由) ===")
    success_count = 0
    total_sat_hops = []

    for t in range(1, n_tests + 1):
        src_sat, dst_sat = random.sample(grouped, 2)
        src_idx = next(i for i, s in enumerate(sats) if s is src_sat)
        dst_idx = next(i for i, s in enumerate(sats) if s is dst_sat)
        src_gid, dst_gid = src_sat.group_id, dst_sat.group_id

        # 同组: 贪心角度路由
        if src_gid == dst_gid:
            path = _greedy_angular_route(adj, src_idx, dst_idx, ecef)
            if path is None:
                path = _dijkstra_shortest_path(adj, src_idx, dst_idx)
            if path:
                names = [sats[i].name for i in path]
                print(f"  [{t}] 同组(G{src_gid}) {src_sat.name} -> {dst_sat.name}")
                print(f"      卫星链路({len(path)-1}跳): {' -> '.join(names)}")
                success_count += 1
                total_sat_hops.append(len(path) - 1)
            else:
                print(f"  [{t}] 同组(G{src_gid}) {src_sat.name} -> {dst_sat.name}: 无通路")
            continue

        # 组间: DQN 路径
        group_path, reward = trace_route(env, agent, src_gid, dst_gid)
        if group_path[-1] != dst_gid:
            print(f"  [{t}] {src_sat.name}(G{src_gid}) -> {dst_sat.name}(G{dst_gid}): DQN 路由失败")
            continue

        # 逐段拼接: 每段用贪心角度路由
        full_path: list[int] = []
        cursor = src_idx
        ok = True

        for seg in range(len(group_path) - 1):
            g_cur, g_next = group_path[seg], group_path[seg + 1]

            border_pairs = _find_border_pairs(sats, adj, group_members, g_cur, g_next)
            if not border_pairs:
                print(f"  [{t}] G{g_cur}->G{g_next} 无跨组 ISL 边")
                ok = False
                break

            # 空间距离评分: 选综合距离最小的出组/入组卫星对
            best_exit, best_entry = -1, -1
            best_score = float("inf")
            for exit_idx, entry_idx in border_pairs:
                d_to_exit = _sat_distance(sats, cursor, exit_idx)
                d_remaining = _sat_distance(sats, entry_idx, dst_idx)
                score = d_to_exit + d_remaining
                if score < best_score:
                    best_score = score
                    best_exit = exit_idx
                    best_entry = entry_idx

            # 贪心角度路由: cursor → best_exit
            intra = _greedy_angular_route(adj, cursor, best_exit, ecef)
            if intra is None:
                intra = _dijkstra_shortest_path(adj, cursor, best_exit)
            if intra is None:
                print(f"  [{t}] G{g_cur} 内 {sats[cursor].name} 无法到达出组卫星 {sats[best_exit].name}")
                ok = False
                break

            if not full_path:
                full_path.extend(intra)
            else:
                full_path.extend(intra[1:])

            full_path.append(best_entry)
            cursor = best_entry

        if not ok:
            continue

        # 最后一段: cursor → dst_sat (贪心角度路由)
        last_seg = _greedy_angular_route(adj, cursor, dst_idx, ecef)
        if last_seg is None:
            last_seg = _dijkstra_shortest_path(adj, cursor, dst_idx)
        if last_seg is None:
            print(f"  [{t}] G{dst_gid} 内 {sats[cursor].name} 无法到达 {dst_sat.name}")
            continue

        full_path.extend(last_seg[1:])

        full_path = _remove_path_loops(full_path)
        sat_hops = len(full_path) - 1
        total_sat_hops.append(sat_hops)
        success_count += 1

        names = [sats[i].name for i in full_path]
        group_tags = [f"G{sats[i].group_id}" for i in full_path]

        print(f"  [{t}] {src_sat.name}(G{src_gid}) -> {dst_sat.name}(G{dst_gid})")
        print(f"      组间路由({len(group_path)-1}跳): {' -> '.join(str(g) for g in group_path)}")
        print(f"      卫星链路({sat_hops}跳): {' -> '.join(names)}")
        print(f"      分组标记: {' -> '.join(group_tags)}")

    print(f"\n  总计: {success_count}/{n_tests} 成功", end="")
    if total_sat_hops:
        avg = sum(total_sat_hops) / len(total_sat_hops)
        print(f", 平均卫星跳数 {avg:.1f}, 最大 {max(total_sat_hops)}, 最小 {min(total_sat_hops)}")
    else:
        print()


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Gen1-1 路由分组")
    parser.add_argument("--source", choices=["real", "ideal", "any"], default="ideal",
                        help="数据来源: real=真实TLE, ideal=理想星座, any=最新文件 (default: ideal)")
    args = parser.parse_args()

    sats = load_gen11_at_time(REF_TIME, source=args.source)

    # 星间链路：同轨前后、邻轨左右最近，写回每颗星的 isl.connected_peers
    isl_edges, plane_sizes = build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    logging.info("ISL: %d edges. Orbital planes (%d):", len(isl_edges), len(plane_sizes))
    for i, count in enumerate(plane_sizes):
        logging.info("  Plane %2d: %d satellites", i + 1, count)
    isl_log_file = write_isl_log(sats, REF_TIME, len(isl_edges), plane_sizes=plane_sizes)
    logging.info("Per-satellite ISL log: %s", isl_log_file)


    round_records: list[tuple[int, list[list[int]], list[int]]] = []
    groups, unassigned, n_rounds = run_grouping(
        sats,
        max_size=9,
        adjacent_radius_km=1500.0,
        on_round=lambda r, gs, u: round_records.append((r, gs, u)),
    )
    log_file = write_route_grouping_log(
        sats, groups, unassigned, REF_TIME, n_rounds=n_rounds, round_records=round_records
    )
    json_file = write_route_grouping_json(
        sats, groups, unassigned, REF_TIME, n_rounds=n_rounds, round_records=round_records
    )

    # ---- DQN 组间路由训练与评估 ----
    from dqn_routing import train_pipeline, eval_pipeline

    model_path = str(_GOOD / "dqn_routing_model.pt")
    agent, env, history = train_pipeline(
        sats, save_path=model_path,
        n_episodes=50000, epsilon_decay=8000, log_interval=2000,
    )
    eval_pipeline(agent, env)

    test_full_route(sats, agent, env, n_tests=20)

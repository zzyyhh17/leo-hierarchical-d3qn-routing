"""
演绎分组策略：从组中心出发，每轮从「与当前组相邻的独立卫星」中选最优的一颗加入组，
以「组内最大距离/组内卫星数」最小为优，直到组达到上限；最后标记未分组的独立卫星。
"""
import logging
import math
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_Sat = TypeVar("_Sat")


def _lat_lon_alt_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> tuple[float, float, float]:
    """大地坐标 → ECEF (km). WGS84 近似，R=6378.137 km."""
    a = 6378.137
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    x = (N + alt_km) * math.cos(lat) * math.cos(lon)
    y = (N + alt_km) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + alt_km) * math.sin(lat)
    return x, y, z


def distance_km(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float,
) -> float:
    """两点的欧氏距离 (km)，基于 ECEF."""
    x1, y1, z1 = _lat_lon_alt_to_ecef(lat1, lon1, alt1)
    x2, y2, z2 = _lat_lon_alt_to_ecef(lat2, lon2, alt2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def _distance_ecef(xyz1: tuple[float, float, float], xyz2: tuple[float, float, float]) -> float:
    """两点 ECEF 坐标的欧氏距离 (km)."""
    return math.sqrt((xyz1[0] - xyz2[0]) ** 2 + (xyz1[1] - xyz2[1]) ** 2 + (xyz1[2] - xyz2[2]) ** 2)


def _get_pos(sat: _Sat) -> tuple[float, float, float]:
    """从 Satellite 或带 .position 的对象取 (lat_deg, lon_deg, height_km)."""
    p = sat.position
    return p.latitude_deg, p.longitude_deg, p.height_km


def _max_intra_distance(satellites: list, indices: list[int]) -> float:
    """组内最大两两距离 (km)."""
    if len(indices) <= 1:
        return 0.0
    pos = [_get_pos(satellites[i]) for i in indices]
    out = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = distance_km(pos[i][0], pos[i][1], pos[i][2], pos[j][0], pos[j][1], pos[j][2])
            out = max(out, d)
    return out


def _max_dist_to_group(satellites: list, sat_idx: int, group_indices: list[int]) -> float:
    """卫星 sat_idx 到组内所有成员的最大距离."""
    if not group_indices:
        return 0.0
    p0 = _get_pos(satellites[sat_idx])
    out = 0.0
    for i in group_indices:
        p = _get_pos(satellites[i])
        d = distance_km(p0[0], p0[1], p0[2], p[0], p[1], p[2])
        out = max(out, d)
    return out


def _is_adjacent_to_group(
    satellites: list,
    sat_idx: int,
    group_indices: list[int],
    radius_km: float,
) -> bool:
    """独立卫星 sat_idx 是否在某个组成员的 radius_km 内（与组相邻）."""
    return _max_dist_to_group(satellites, sat_idx, group_indices) <= radius_km


def _build_adjacent_groups_from_grid(
    ecef: list[tuple[float, float, float]],
    assigned: set[int],
    sat_to_group: dict[int, int],
    radius_km: float,
    cell_size_km: float | None = None,
) -> tuple[dict[tuple[int, int, int], list[tuple[int, int]]], float]:
    """
    用 ECEF 网格建立「已分配卫星 → 组」的空间索引。
    返回 (grid, cell_size_km)。cell_size_km 若为 None 则取 radius_km，保证邻域只需 3^3 格。
    """
    if cell_size_km is None or cell_size_km < radius_km:
        cell_size_km = radius_km
    grid: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for idx in assigned:
        x, y, z = ecef[idx]
        cx = int(x / cell_size_km)
        cy = int(y / cell_size_km)
        cz = int(z / cell_size_km)
        cell = (cx, cy, cz)
        gi = sat_to_group[idx]
        if cell not in grid:
            grid[cell] = []
        grid[cell].append((idx, gi))
    return grid, cell_size_km


def _query_adjacent_group_ids(
    candidate_idx: int,
    ecef: list[tuple[float, float, float]],
    grid: dict[tuple[int, int, int], list[tuple[int, int]]],
    cell_size_km: float,
    radius_km: float,
) -> set[int]:
    """查询与候选卫星在 radius_km 内有成员的组 id 集合（仅检查网格邻格，再精确距离）。"""
    x, y, z = ecef[candidate_idx]
    cx, cy, cz = int(x / cell_size_km), int(y / cell_size_km), int(z / cell_size_km)
    out: set[int] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (cx + dx, cy + dy, cz + dz)
                for sat_idx, gi in grid.get(cell, []):
                    if _distance_ecef(ecef[candidate_idx], ecef[sat_idx]) <= radius_km:
                        out.add(gi)
    return out


class GroupingStrategy:
    """
    演绎分组：初始选 n_centers 个组中心，每轮从「与某组成员有 ISL 连接的未分组卫星」中
    选一个加入某组，使 组内最大距离/组内卫星数 最小；组大小不超过 max_size。
    结束后可查未分组卫星。
    """

    def __init__(
        self,
        satellites: list,
        max_size: int = 9,
        isl_neighbors: dict[int, set[int]] | None = None,
        adjacent_radius_km: float = 1500.0,
        center_indices: list[int] | None = None,
    ):
        """
        satellites: 卫星列表（需有 .position 含 latitude_deg, longitude_deg, height_km）。
        max_size: 每组最多卫星数。
        isl_neighbors: 基于卫星下标的 ISL 邻接表 {idx: {peer_idx, ...}}。
                       提供时按 ISL 拓扑判断「与组相邻」；为 None 时回退到空间距离判断。
        adjacent_radius_km: ISL 不可用时的空间相邻判定半径 (km)。
        center_indices: 若提供，则用这些下标作为初始组中心；否则取前 n_centers 个。
        """
        self.satellites = satellites
        self.n = len(satellites)
        self.max_size = max_size
        self.radius_km = adjacent_radius_km
        self._isl_neighbors = isl_neighbors
        # 组数 = ceil(卫星总数 / max_size)，保证 组数×每组上限 >= 卫星总数
        n_centers = math.ceil(self.n / max_size)
        self.n_centers = n_centers
        # groups: list[list[int]]，每组为卫星下标列表
        if center_indices is not None:
            idxs = [i for i in center_indices if 0 <= i < self.n][:n_centers]
        else:
            stride = max(1, self.n // n_centers)
            idxs = list(range(0, self.n, stride))[:n_centers]
        self.groups = [[i] for i in idxs]
        self.assigned = set(idxs)
        # 卫星下标 → 所属组编号，增量维护
        self._sat_to_group: dict[int, int] = {}
        for gi, idx in enumerate(idxs):
            self._sat_to_group[idx] = gi
        # ECEF 缓存，避免重复计算
        self._ecef: list[tuple[float, float, float]] = [
            _lat_lon_alt_to_ecef(*_get_pos(s)) for s in self.satellites
        ]
        # 每组当前组内最大两两距离，增量更新
        self.group_max_intra: list[float] = [0.0] * len(self.groups)

    def _score_if_add(self, group_idx: int, candidate_idx: int, current_max: float) -> float:
        """若将 candidate 加入 group，新得分 = 新组内最大距离 / 新组大小。"""
        g = self.groups[group_idx]
        new_max = max(current_max, _max_dist_to_group(self.satellites, candidate_idx, g))
        new_size = len(g) + 1
        return new_max / new_size

    def _adjacent_group_ids_isl(self, candidate: int) -> set[int]:
        """通过 ISL 邻接表查询与候选卫星有链路连接的组 id 集合。"""
        out: set[int] = set()
        for peer in self._isl_neighbors.get(candidate, set()):
            if peer in self._sat_to_group:
                out.add(self._sat_to_group[peer])
        return out

    def _adjacent_group_ids_radius(
        self, candidate: int, grid, cell_size: float
    ) -> set[int]:
        """通过 ECEF 空间网格查询与候选卫星在 radius_km 内的组 id 集合（ISL 不可用时回退）。"""
        return _query_adjacent_group_ids(
            candidate, self._ecef, grid, cell_size, self.radius_km
        )

    def evolve_one(self) -> bool:
        """
        执行一轮演绎：从「与某组成员有 ISL 连接（或空间相邻）且该组未满」的未分组卫星中，
        选使「组内最大距离/组内卫星数」最小的 (卫星, 组)，加入该组。
        返回是否成功添加了一颗。组内最大距离用 group_max_intra 增量更新。
        """
        t_start = time.perf_counter()
        n_unassigned = self.n - len(self.assigned)
        n_groups = len(self.groups)
        n_not_full = sum(1 for g in self.groups if len(g) < self.max_size)
        use_isl = self._isl_neighbors is not None

        t0 = time.perf_counter()
        grid = None
        cell_size = 0.0
        if not use_isl:
            sat_to_group: dict[int, int] = {}
            for gi, g in enumerate(self.groups):
                for idx in g:
                    sat_to_group[idx] = gi
            grid, cell_size = _build_adjacent_groups_from_grid(
                self._ecef, self.assigned, sat_to_group, self.radius_km
            )
        t_index = time.perf_counter() - t0

        best_score = float("inf")
        best_candidate: int | None = None
        best_group: int | None = None
        n_candidates_checked = 0
        n_with_adjacent = 0
        n_score_calls = 0

        t_loop_start = time.perf_counter()
        for candidate in range(self.n):
            if candidate in self.assigned:
                continue
            n_candidates_checked += 1
            if use_isl:
                adjacent_gi = self._adjacent_group_ids_isl(candidate)
            else:
                adjacent_gi = self._adjacent_group_ids_radius(
                    candidate, grid, cell_size
                )
            if not adjacent_gi:
                continue
            n_with_adjacent += 1
            for gi in adjacent_gi:
                g = self.groups[gi]
                if len(g) >= self.max_size:
                    continue
                n_score_calls += 1
                score = self._score_if_add(gi, candidate, self.group_max_intra[gi])
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
                    best_group = gi
        t_loop = time.perf_counter() - t_loop_start

        if best_candidate is None or best_group is None:
            t_total = time.perf_counter() - t_start
            logger.info(
                "evolve_one: no add | total=%.3fs | index=%.3fs loop=%.3fs | "
                "unassigned=%d groups=%d not_full=%d | candidates=%d with_adj=%d score_calls=%d",
                t_total, t_index, t_loop, n_unassigned, n_groups, n_not_full,
                n_candidates_checked, n_with_adjacent, n_score_calls,
            )
            return False
        new_diam = _max_dist_to_group(
            self.satellites, best_candidate, self.groups[best_group]
        )
        self.groups[best_group].append(best_candidate)
        self.assigned.add(best_candidate)
        self._sat_to_group[best_candidate] = best_group
        self.group_max_intra[best_group] = max(
            self.group_max_intra[best_group], new_diam
        )
        t_total = time.perf_counter() - t_start
        logger.info(
            "evolve_one: added sat %d -> group %d | total=%.3fs | index=%.3fs loop=%.3fs | "
            "unassigned=%d groups=%d not_full=%d | candidates=%d with_adj=%d score_calls=%d",
            best_candidate, best_group, t_total, t_index, t_loop, n_unassigned, n_groups, n_not_full,
            n_candidates_checked, n_with_adjacent, n_score_calls,
        )
        return True

    def evolve_until_done(self):
        """
        反复演绎直到无法再添加（或所有组都达到 max_size）。
        每执行一轮演绎，yield (当前轮数, 当前分组快照, 当前未分组下标列表)，方便外面输出。
        迭代结束后，最后一轮 yield 的当前轮数即为总演化轮数。
        """
        count = 0
        while self.evolve_one():
            count += 1
            yield count, [list(g) for g in self.groups], self.get_unassigned()

    def get_unassigned(self) -> list[int]:
        """返回仍未分到任何组的卫星下标列表。"""
        return [i for i in range(self.n) if i not in self.assigned]

    def get_groups(self) -> list[list[int]]:
        """返回当前分组（每组为卫星下标列表）。"""
        return self.groups


def _force_assign_remaining(gs: GroupingStrategy) -> None:
    """将剩余未分组卫星强制分配到距离最近的非满组，无视 ISL/半径约束。"""
    remaining = gs.get_unassigned()
    for sat_idx in remaining:
        best_gi = None
        best_dist = float("inf")
        for gi, g in enumerate(gs.groups):
            if len(g) >= gs.max_size:
                continue
            d = _max_dist_to_group(gs.satellites, sat_idx, g)
            if d < best_dist:
                best_dist = d
                best_gi = gi
        if best_gi is not None:
            gs.groups[best_gi].append(sat_idx)
            gs.assigned.add(sat_idx)
            gs._sat_to_group[sat_idx] = best_gi
            gs.group_max_intra[best_gi] = max(gs.group_max_intra[best_gi], best_dist)


def _build_isl_index_adj(satellites: list) -> dict[int, set[int]] | None:
    """从卫星的 isl.connected_peers (catalog_number) 构建基于下标的 ISL 邻接表。

    仅当卫星已填充了 ISL 信息时有效；若所有星都无 peers 则返回 None（回退到空间距离）。
    """
    cat_to_idx: dict[int, int] = {s.catalog_number: i for i, s in enumerate(satellites)}
    adj: dict[int, set[int]] = {}
    has_any = False
    for i, s in enumerate(satellites):
        peers = getattr(s.isl, "connected_peers", None) or []
        neighbors: set[int] = set()
        for cat in peers:
            j = cat_to_idx.get(cat)
            if j is not None:
                neighbors.add(j)
                has_any = True
        adj[i] = neighbors
    return adj if has_any else None


def run_grouping(
    satellites: list,
    max_size: int = 9,
    adjacent_radius_km: float = 1500.0,
    isl_neighbors: dict[int, set[int]] | None = None,
    on_round: Callable[[int, list[list[int]], list[int]], None] | None = None,
) -> tuple[list[list[int]], list[int], int]:
    """
    对卫星列表执行演绎分组，返回 (分组列表, 未分组卫星下标列表, 演化轮数)。
    分组列表的每一项为该组内的卫星下标列表。组数由卫星总数与 max_size 计算得出。
    演化轮数 = 成功执行 evolve_one 的次数（每轮最多加入 1 颗卫星）。

    isl_neighbors: 基于卫星下标的 ISL 邻接表。为 None 时自动从 satellites 的
                   isl.connected_peers 构建；若卫星尚未填充 ISL 则回退到空间距离。
    若提供 on_round(round_index, groups, unassigned)，则每轮演绎结束后会调用一次，便于外部输出。
    """
    if isl_neighbors is None:
        isl_neighbors = _build_isl_index_adj(satellites)
    gs = GroupingStrategy(
        satellites,
        max_size=max_size,
        isl_neighbors=isl_neighbors,
        adjacent_radius_km=adjacent_radius_km,
    )
    n_rounds = 0
    for n_rounds, groups_snapshot, unassigned_snapshot in gs.evolve_until_done():
        if on_round is not None:
            on_round(n_rounds, groups_snapshot, unassigned_snapshot)

    remaining = gs.get_unassigned()
    if remaining:
        gs._isl_neighbors = None
        gs.radius_km = max(adjacent_radius_km, 3000.0)
        for n_r, groups_snapshot, unassigned_snapshot in gs.evolve_until_done():
            n_rounds += 1
            if on_round is not None:
                on_round(n_rounds, groups_snapshot, unassigned_snapshot)

    remaining = gs.get_unassigned()
    if remaining:
        _force_assign_remaining(gs)
        n_rounds += len(remaining)

    groups = gs.get_groups()
    for gi, g in enumerate(groups):
        for idx in g:
            if hasattr(satellites[idx], "group_id"):
                satellites[idx].group_id = gi

    return groups, gs.get_unassigned(), n_rounds


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _GOOD = Path(__file__).resolve().parent
    if str(_GOOD) not in sys.path:
        sys.path.insert(0, str(_GOOD))

    from data_loader import load_gen11_at_time
    from isl import update_satellite_isl_peers, build_isl_from_satellites
    from datetime import datetime, timezone

    ref_time = datetime.now(timezone.utc)
    sats = load_gen11_at_time(ref_time)
    build_isl_from_satellites(sats)
    update_satellite_isl_peers(sats)
    groups, unassigned, n_rounds = run_grouping(
        sats,
        max_size=9,
    )
    lines = [
        f"Satellites: {len(sats)}, Groups: {len(groups)}, Unassigned: {len(unassigned)}, Evolution rounds: {n_rounds}",
        "(catalog_number = NORAD 唯一编号)",
    ]
    for i, g in enumerate(groups):
        cats = [sats[idx].catalog_number for idx in g]
        cats_str = str(cats) if len(cats) == 9 else f"{cats[:5]}{'...' if len(cats) > 5 else ''}"
        lines.append(f"  Group {i+1}: {len(g)} sats, catalog_number {cats_str}")
    if unassigned:
        unassigned_cats = [sats[idx].catalog_number for idx in unassigned[:20]]
        lines.append("Unassigned (independent) satellite catalog_number: " + str(unassigned_cats) + (" ..." if len(unassigned) > 20 else ""))
    log_content = "\n".join(lines)
    print(log_content)
    log_dir = _GOOD / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"grouping_{ref_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file.write_text(log_content, encoding="utf-8")
    print(f"Log written to {log_file}")

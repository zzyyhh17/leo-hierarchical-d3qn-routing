"""
星间链路 (ISL) 确定：根据卫星位置与轨道根数，按同轨前后、邻轨左右最近原则建立连接。

规则：
- 同轨道：一前一后 —— 同一轨道面内按 mean_anomaly 排序，相邻两颗相连（首尾成环）。
- 相邻轨道：左右取最近 —— 对每个轨道面，左邻轨、右邻轨各取与当前星 3D 距离最近的一颗相连。
- 每颗卫星的星间链路最多 4 条（同轨 2 条 + 邻轨左右各 1 条），添加边时若任一端已达上限则不再添加。
"""
from __future__ import annotations

import math
from typing import Any

MAX_ISL_PER_SAT = 4

from group import distance_km
from starlink_model import Satellite


def _get_pos(sat: Satellite) -> tuple[float, float, float]:
    """(lat_deg, lon_deg, height_km)."""
    p = sat.position
    return p.latitude_deg, p.longitude_deg, p.height_km


def _plane_key(sat: Satellite, incl_tol: float = 0.1, raan_tol: float = 2.0) -> tuple[float, float]:
    """将轨道面量化为 (incl, raan)，便于分组。raan_tol 越大同一轨道面内星越多，减少单星面。"""
    inc = round(sat.tle.inclination_deg / incl_tol) * incl_tol
    raan = round(sat.tle.raan_deg / raan_tol) * raan_tol
    return (inc, raan)


def _dist_km(sat_a: Satellite, sat_b: Satellite) -> float:
    """两星 ECEF 距离 (km)."""
    la, lo, ha = _get_pos(sat_a)
    lb, lo2, hb = _get_pos(sat_b)
    return distance_km(la, lo, ha, lb, lo2, hb)


# Shell 1 (Gen1-1) 设计轨道面数量与 RAAN 间隔（360/72 = 5°）
GEN1_1_EXPECTED_NUM_PLANES = 72
GEN1_1_RAAN_STEP_DEG = 360.0 / GEN1_1_EXPECTED_NUM_PLANES  # 5.0


def _teme_orbital_vectors(sat: Satellite):
    """从 TEME 位置/速度向量计算轨道要素中间量，返回 (hx,hy,hz, nx,ny, n_mag) 或 None。"""
    r = sat.position.position_teme_km
    v = sat.position.velocity_teme_km_s
    if not r or not v or all(x == 0.0 for x in r) or all(x == 0.0 for x in v):
        return None
    rx, ry, rz = r
    vx, vy, vz = v
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    nx, ny = -hy, hx
    n_mag = math.sqrt(nx * nx + ny * ny)
    return hx, hy, hz, nx, ny, n_mag


def _raan_from_teme(sat: Satellite) -> float | None:
    """从 SGP4 传播后的 TEME 位置/速度向量计算当前时刻的 RAAN (deg)。

    TLE 中的 raan_deg 是 TLE 历元时刻的值，由于 J2 摄动 RAAN 每天漂移约 -4~5°，
    历元与参考时刻差 1 天即可能跨越一个 5° 轨道面。此函数利用 SGP4 已传播到参考时刻
    的状态向量重新计算 RAAN，避免历元差异导致轨道面错分。
    """
    ov = _teme_orbital_vectors(sat)
    if ov is None:
        return None
    hx, hy, _hz, _nx, _ny, _n_mag = ov
    return math.degrees(math.atan2(hx, -hy)) % 360.0


def _arg_of_latitude_from_teme(sat: Satellite) -> float:
    """从 TEME 位置/速度计算纬度幅角 u (argument of latitude, deg)，即沿轨道的角度位置。

    u = 0° 在升交点，沿轨道运动方向递增到 360°。用于同一轨道面内卫星的正确排序。
    """
    ov = _teme_orbital_vectors(sat)
    if ov is None:
        return 0.0
    hx, hy, hz, nx, ny, n_mag = ov
    if n_mag < 1e-10:
        return 0.0
    rx, ry, rz = sat.position.position_teme_km
    h_mag = math.sqrt(hx * hx + hy * hy + hz * hz)
    nx_h, ny_h = nx / n_mag, ny / n_mag
    hx_h, hy_h, hz_h = hx / h_mag, hy / h_mag, hz / h_mag
    # t_hat = h_hat × n_hat （轨道面内垂直于升交点方向）
    tx = hy_h * 0.0 - hz_h * ny_h
    ty = hz_h * nx_h - hx_h * 0.0
    tz = hx_h * ny_h - hy_h * nx_h
    x_orb = rx * nx_h + ry * ny_h
    y_orb = rx * tx + ry * ty + rz * tz
    return math.degrees(math.atan2(y_orb, x_orb)) % 360.0


def _plane_id_gen1(sat: Satellite) -> int:
    """Shell 1 (Gen1-1)：按设计 RAAN 间隔 5° 将卫星归入 0..71 轨道面，使每面约 22 颗。

    优先使用 SGP4 传播后的 TEME 向量计算当前 RAAN（避免 TLE 历元差异导致偏差），
    若 TEME 数据不可用则回退到 TLE 中的 raan_deg。
    """
    prop_raan = _raan_from_teme(sat)
    raan = prop_raan if prop_raan is not None else (sat.tle.raan_deg % 360.0)
    return int(math.floor(raan / GEN1_1_RAAN_STEP_DEG + 0.5)) % GEN1_1_EXPECTED_NUM_PLANES


def _count_planes(
    satellites: list[Satellite],
    incl_tol: float,
    raan_tol: float,
) -> int:
    """在给定 incl_tol/raan_tol 下统计轨道面数量。"""
    keys: set[tuple[float, float]] = set()
    for s in satellites:
        keys.add(_plane_key(s, incl_tol, raan_tol))
    return len(keys)


def _find_raan_tol_for_planes(
    satellites: list[Satellite],
    incl_tol: float,
    expected: int,
    raan_lo: float = 0.5,
    raan_hi: float = 50.0,
    max_iters: int = 50,
) -> float:
    """二分查找使轨道面数等于 expected 的 raan_tol。raan_tol 越大面数越少。"""
    best_tol = (raan_lo + raan_hi) * 0.5
    best_diff = 999999
    for _ in range(max_iters):
        mid = (raan_lo + raan_hi) * 0.5
        cnt = _count_planes(satellites, incl_tol, mid)
        if cnt == expected:
            return mid
        if abs(cnt - expected) < best_diff:
            best_diff = abs(cnt - expected)
            best_tol = mid
        if cnt > expected:
            raan_lo = mid
        else:
            raan_hi = mid
        if raan_hi - raan_lo < 0.005:
            break
    # 若未精确命中，在 best_tol 附近微调尝试
    for delta in (0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05):
        t = max(0.5, best_tol + delta)
        if _count_planes(satellites, incl_tol, t) == expected:
            return t
    return best_tol


def build_isl_from_satellites(
    satellites: list[Satellite],
    incl_tol: float = 0.1,
    raan_tol: float = 2.0,
    expected_num_planes: int | None = GEN1_1_EXPECTED_NUM_PLANES,
) -> list[tuple[int, int]]:
    """
    根据卫星列表确定星间连接，返回 (edges, plane_sizes)：edges 为无向边列表且 catalog_number_a < catalog_number_b 去重，
    plane_sizes 为各轨道面（按排序后顺序）的卫星数量列表。

    规则：
    - 同轨道一前一后：按 (incl, raan) 分组得到轨道面，面内按 mean_anomaly_deg 排序后首尾相连成环。
    - 相邻轨道左右取最近：轨道面按 (incl, raan) 排序后，每个面与左邻面、右邻面各取 3D 距离最近的一颗星相连。
    - 每颗卫星最多 MAX_ISL_PER_SAT 条链路，添加边时若任一端已达上限则跳过。
    - expected_num_planes：Shell 1 (Gen1-1) 为 72；为 72 时按设计 RAAN 间隔 5° 分配（每面约 22 颗），否则用 incl_tol/raan_tol 取整并自动调整直至面数一致。
    """
    if not satellites:
        return [], []

    n = len(satellites)
    degree: dict[int, int] = {s.catalog_number: 0 for s in satellites}
    num_planes: int
    plane_keys_sorted: list
    planes: dict
    plane_index: dict

    if expected_num_planes == GEN1_1_EXPECTED_NUM_PLANES:
        # Shell 1：按设计 72 面、RAAN 间隔 5° 分配，每面约 22 颗
        planes = {p: [] for p in range(GEN1_1_EXPECTED_NUM_PLANES)}
        for i in range(n):
            pid = _plane_id_gen1(satellites[i])
            planes[pid].append(i)
        plane_keys_sorted = list(range(GEN1_1_EXPECTED_NUM_PLANES))
        num_planes = GEN1_1_EXPECTED_NUM_PLANES
        plane_index = {p: p for p in plane_keys_sorted}
    else:
        # 通用：按 (incl, raan) 取整分组
        planes = {}
        for i in range(n):
            key = _plane_key(satellites[i], incl_tol, raan_tol)
            if key not in planes:
                planes[key] = []
            planes[key].append(i)
        plane_keys_sorted = sorted(planes.keys(), key=lambda p: (p[0], p[1]))
        num_planes = len(plane_keys_sorted)
        if expected_num_planes is not None and num_planes != expected_num_planes:
            raan_tol = _find_raan_tol_for_planes(satellites, incl_tol, expected_num_planes)
            planes = {}
            for i in range(n):
                key = _plane_key(satellites[i], incl_tol, raan_tol)
                if key not in planes:
                    planes[key] = []
                planes[key].append(i)
            plane_keys_sorted = sorted(planes.keys(), key=lambda p: (p[0], p[1]))
            num_planes = len(plane_keys_sorted)
            if num_planes != expected_num_planes:
                raise ValueError(
                    f"Could not get {expected_num_planes} planes (got {num_planes}) after adjusting raan_tol. "
                    f"Try different incl_tol or pass expected_num_planes=None."
                )
        plane_index = {k: idx for idx, k in enumerate(plane_keys_sorted)}

    # 写回每颗卫星所属轨道面编号与沿轨角度位置
    for key in plane_keys_sorted:
        pk = plane_index[key]
        for idx in planes[key]:
            satellites[idx].plane_id = pk
            satellites[idx].orbit_pos_deg = _arg_of_latitude_from_teme(satellites[idx])

    edges: set[tuple[int, int]] = set()

    def add_edge(cat_a: int, cat_b: int) -> None:
        if cat_a == cat_b:
            return
        if degree.get(cat_a, 0) >= MAX_ISL_PER_SAT or degree.get(cat_b, 0) >= MAX_ISL_PER_SAT:
            return
        a, b = min(cat_a, cat_b), max(cat_a, cat_b)
        if (a, b) in edges:
            return
        edges.add((a, b))
        degree[cat_a] = degree.get(cat_a, 0) + 1
        degree[cat_b] = degree.get(cat_b, 0) + 1

    # 1) 同轨道一前一后：面内按 mean_anomaly 排序，相邻连边，首尾成环
    for key in plane_keys_sorted:
        indices = planes[key]
        if len(indices) <= 1:
            continue
        indices_sorted = sorted(
            indices,
            key=lambda idx: (satellites[idx].tle.mean_anomaly_deg % 360),
        )
        for k in range(len(indices_sorted)):
            i = indices_sorted[k]
            j = indices_sorted[(k + 1) % len(indices_sorted)]
            cat_i = satellites[i].catalog_number
            cat_j = satellites[j].catalog_number
            add_edge(cat_i, cat_j)

    # 2) 相邻轨道左右取最近：对每个轨道面，与左邻、右邻面各取距离当前星最近的一颗（72 面时首尾相接）
    wrap_planes = num_planes == GEN1_1_EXPECTED_NUM_PLANES
    for key in plane_keys_sorted:
        my_indices = planes[key]
        pk = plane_index[key]
        for idx in my_indices:
            sat = satellites[idx]
            cat_self = sat.catalog_number
            # 左邻轨
            left_pk = (pk - 1 + num_planes) % num_planes if wrap_planes else pk - 1
            if left_pk >= 0:
                left_key = plane_keys_sorted[left_pk]
                left_indices = planes[left_key]
                best_left: int | None = None
                best_d_left = float("inf")
                for o in left_indices:
                    d = _dist_km(sat, satellites[o])
                    if d < best_d_left:
                        best_d_left = d
                        best_left = o
                if best_left is not None:
                    add_edge(cat_self, satellites[best_left].catalog_number)
            # 右邻轨
            right_pk = (pk + 1) % num_planes if wrap_planes else pk + 1
            if right_pk < num_planes:
                right_key = plane_keys_sorted[right_pk]
                right_indices = planes[right_key]
                best_right: int | None = None
                best_d_right = float("inf")
                for o in right_indices:
                    d = _dist_km(sat, satellites[o])
                    if d < best_d_right:
                        best_d_right = d
                        best_right = o
                if best_right is not None:
                    add_edge(cat_self, satellites[best_right].catalog_number)

    plane_sizes = [len(planes[k]) for k in plane_keys_sorted]
    return sorted(edges), plane_sizes


def build_isl_nearest(
    satellites: list[Satellite],
    k: int = MAX_ISL_PER_SAT,
) -> list[tuple[int, int]]:
    """最近原则 ISL：每颗卫星连接距离最近的 k 颗，不限轨道面。

    双向确认：A 选 B 且 B 选 A 时才建立连接 → 度数上限自然满足。
    单向选择也加入（只要度数未满），提高连通性。
    """
    n = len(satellites)
    if n == 0:
        return [], []

    # 预计算所有卫星的 ECEF 坐标
    from group import _lat_lon_alt_to_ecef
    coords = []
    for s in satellites:
        p = s.position
        x, y, z = _lat_lon_alt_to_ecef(p.latitude_deg, p.longitude_deg, p.height_km)
        coords.append((x, y, z))

    # 每颗卫星找 k 个最近邻
    knn: list[list[int]] = []
    for i in range(n):
        xi, yi, zi = coords[i]
        dists = []
        for j in range(n):
            if j == i:
                continue
            xj, yj, zj = coords[j]
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            dists.append((d2, j))
        dists.sort()
        knn.append([j for _, j in dists[:k]])

    degree: dict[int, int] = {s.catalog_number: 0 for s in satellites}
    edges: set[tuple[int, int]] = set()

    def add_edge(idx_a: int, idx_b: int) -> bool:
        ca, cb = satellites[idx_a].catalog_number, satellites[idx_b].catalog_number
        if ca == cb:
            return False
        if degree[ca] >= k or degree[cb] >= k:
            return False
        a, b = min(ca, cb), max(ca, cb)
        if (a, b) in edges:
            return False
        edges.add((a, b))
        degree[ca] += 1
        degree[cb] += 1
        return True

    # 双向确认优先
    for i in range(n):
        for j in knn[i]:
            if i in knn[j]:
                add_edge(i, j)

    # 单向补充
    for i in range(n):
        for j in knn[i]:
            add_edge(i, j)

    # 写回轨道面信息 (按原始方法保持兼容)
    for i, s in enumerate(satellites):
        pid = _plane_id_gen1(s)
        s.plane_id = pid
        s.orbit_pos_deg = _arg_of_latitude_from_teme(s)

    plane_counts: dict[int, int] = {}
    for s in satellites:
        plane_counts[s.plane_id] = plane_counts.get(s.plane_id, 0) + 1
    plane_sizes = [plane_counts.get(p, 0)
                   for p in range(max(plane_counts.keys()) + 1)]

    return sorted(edges), plane_sizes


def build_isl_adjacency(
    satellites: list[Satellite],
    incl_tol: float = 0.1,
    raan_tol: float = 2.0,
) -> dict[int, list[int]]:
    """
    返回以 catalog_number 为键的邻接表：每个卫星的 ISL 邻居 catalog_number 列表。
    内部调用 build_isl_from_satellites，再转为邻接表。
    """
    edges, _ = build_isl_from_satellites(satellites, incl_tol, raan_tol)
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def update_satellite_isl_peers(
    satellites: list[Satellite],
    incl_tol: float = 0.1,
    raan_tol: float = 2.0,
    mode: str = "orbit",
) -> None:
    """根据当前卫星列表计算 ISL，并写回每颗星的 isl.connected_peers。

    mode: "orbit" 同轨+邻轨 (默认), "nearest" 最近原则
    """
    if mode == "nearest":
        edges, _ = build_isl_nearest(satellites)
    else:
        adj = build_isl_adjacency(satellites, incl_tol, raan_tol)
        for i, sat in enumerate(satellites):
            peers = adj.get(sat.catalog_number, [])
            sat.isl.connected_peers = peers
            sat.isl.active_links = len(peers)
        return

    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for i, sat in enumerate(satellites):
        peers = adj.get(sat.catalog_number, [])
        sat.isl.connected_peers = peers
        sat.isl.active_links = len(peers)

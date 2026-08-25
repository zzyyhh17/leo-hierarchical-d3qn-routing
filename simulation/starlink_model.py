"""
Starlink 卫星数据模型

定义卫星的核心属性结构, 便于后续扩展容量、负载、链路状态等业务字段.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TLEData:
    """TLE 轨道根数."""
    catalog_number: int = 0
    classification: str = ""
    launch_year: str = ""
    launch_number: str = ""
    launch_piece: str = ""
    epoch_year: int = 0
    epoch_day: float = 0.0
    epoch_utc: str = ""
    mean_motion_dot: float = 0.0
    bstar: float = 0.0
    element_set_number: int = 0
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    mean_motion_rev_per_day: float = 0.0
    rev_number: int = 0
    tle_line1: str = ""
    tle_line2: str = ""


@dataclass
class Position:
    """卫星位置 (某一时刻的快照)."""
    ref_time: str = ""
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    height_km: float = 0.0
    position_teme_km: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_teme_km_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    heading_deg: float = 0.0
    """地面轨迹航向角 (deg)，正北=0 顺时针递增；由 SGP4 传播时根据速度向量计算。"""


@dataclass
class ShellInfo:
    """壳层归属与运行状态."""
    shell: str = "unknown"
    status: str = "unknown"
    altitude_km: float = 0.0


@dataclass
class Capacity:
    """卫星通信容量与负载 (预留扩展)."""
    total_bandwidth_gbps: float = 20.0
    used_bandwidth_gbps: float = 0.0
    max_user_beams: int = 48
    active_user_beams: int = 0

    @property
    def load_ratio(self) -> float:
        if self.total_bandwidth_gbps <= 0:
            return 0.0
        return self.used_bandwidth_gbps / self.total_bandwidth_gbps

    @property
    def available_bandwidth_gbps(self) -> float:
        return max(0.0, self.total_bandwidth_gbps - self.used_bandwidth_gbps)


@dataclass
class ISLStatus:
    """星间链路状态 (预留扩展)."""
    max_links: int = 4
    active_links: int = 0
    connected_peers: list[int] = field(default_factory=list)


@dataclass
class Satellite:
    """Starlink 卫星完整模型.

    核心属性来自 TLE 数据, 扩展属性用于仿真业务逻辑.
    """
    name: str = ""
    tle: TLEData = field(default_factory=TLEData)
    position: Position = field(default_factory=Position)
    shell_info: ShellInfo = field(default_factory=ShellInfo)
    capacity: Capacity = field(default_factory=Capacity)
    isl: ISLStatus = field(default_factory=ISLStatus)
    available: bool = True
    plane_id: int = -1
    """当前卫星所属轨道面编号，0..num_planes-1；由 ISL 构建时赋值，未分配为 -1。"""
    orbit_pos_deg: float = 0.0
    """沿轨道的角度位置（纬度幅角 argument of latitude），0..360°；由 ISL 构建时赋值。"""
    group_id: int = -1
    """所属分组编号，0..num_groups-1；由分组算法赋值，未分配为 -1。"""

    @property
    def catalog_number(self) -> int:
        return self.tle.catalog_number

    def to_dict(self) -> dict:
        d = asdict(self)
        d["catalog_number"] = self.catalog_number
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Satellite:
        """从现有 JSON 结构 (starlink_gen1-1.json) 构建 Satellite 对象."""
        tle = TLEData(
            catalog_number=d.get("catalog_number", 0),
            classification=d.get("classification", ""),
            launch_year=d.get("launch_year", ""),
            launch_number=d.get("launch_number", ""),
            launch_piece=d.get("launch_piece", ""),
            epoch_year=d.get("epoch_year", 0),
            epoch_day=d.get("epoch_day", 0.0),
            epoch_utc=d.get("epoch_utc", ""),
            mean_motion_dot=d.get("mean_motion_dot", 0.0),
            bstar=d.get("bstar", 0.0),
            element_set_number=d.get("element_set_number", 0),
            inclination_deg=d.get("inclination_deg", 0.0),
            raan_deg=d.get("raan_deg", 0.0),
            eccentricity=d.get("eccentricity", 0.0),
            arg_perigee_deg=d.get("arg_perigee_deg", 0.0),
            mean_anomaly_deg=d.get("mean_anomaly_deg", 0.0),
            mean_motion_rev_per_day=d.get("mean_motion_rev_per_day", 0.0),
            rev_number=d.get("rev_number", 0),
            tle_line1=d.get("tle_line1", ""),
            tle_line2=d.get("tle_line2", ""),
        )

        pos = Position(
            ref_time=d.get("ref_time", ""),
            latitude_deg=d.get("latitude_deg", 0.0),
            longitude_deg=d.get("longitude_deg", 0.0),
            height_km=d.get("height_km", 0.0),
            position_teme_km=d.get("position_teme_km", [0.0, 0.0, 0.0]),
            velocity_teme_km_s=d.get("velocity_teme_km_s", [0.0, 0.0, 0.0]),
        )

        shell = ShellInfo(
            shell=d.get("shell", "unknown"),
            status=d.get("status", "unknown"),
            altitude_km=d.get("altitude_km", 0.0),
        )

        return cls(
            name=d.get("name", ""),
            tle=tle,
            position=pos,
            shell_info=shell,
            plane_id=d.get("plane_id", -1),
        )

    def __repr__(self) -> str:
        return (
            f"Satellite({self.name}, cat={self.catalog_number}, "
            f"shell={self.shell_info.shell}, status={self.shell_info.status}, "
            f"alt={self.shell_info.altitude_km:.1f}km, "
            f"load={self.capacity.load_ratio:.0%})"
        )


def load_satellites(path: str) -> list[Satellite]:
    """从 JSON 文件加载卫星列表."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Satellite.from_dict(d) for d in data.get("satellites", [])]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    data_path = Path(__file__).parent / "data" / "starlink_gen1-1.json"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    sats = load_satellites(str(data_path))
    print(f"Loaded {len(sats)} satellites\n")

    for sat in sats[:5]:
        print(sat)

    print(f"\n--- Example: modify capacity ---")
    s = sats[0]
    s.capacity.used_bandwidth_gbps = 8.5
    s.capacity.active_user_beams = 20
    s.isl.active_links = 3
    s.isl.connected_peers = [sats[1].catalog_number, sats[2].catalog_number, sats[3].catalog_number]
    print(s)
    print(f"  available bandwidth: {s.capacity.available_bandwidth_gbps:.1f} Gbps")
    print(f"  ISL peers: {s.isl.connected_peers}")

"""
Data loader：从 Starlink data 目录加载壳层卫星数据，返回 Satellite 列表。
支持按指定时间演化：用 SGP4 将 TLE 传播到目标时刻并更新卫星位置。
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

from starlink_model import Satellite, load_satellites

try:
    from sgp4.api import Satrec, jday
except ImportError:
    Satrec = None
    jday = None

# 默认数据目录：独立仿真包下的 starlink/data
DEFAULT_DATA_DIR = _GOOD.parent / "starlink" / "data"

# 地球平均半径 (km)，用于 TEME → 大地坐标
R_EARTH_KM = 6371.0


def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (rad), IAU 1982."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    return (gmst_sec % 86400) / 86400 * 2 * math.pi


def _teme_to_geodetic(x_km: float, y_km: float, z_km: float, jd: float, fr: float) -> tuple[float, float, float]:
    """TEME (km) → (lat_deg, lon_deg, alt_km)."""
    gmst = _gmst_rad(jd + fr)
    c, s = math.cos(gmst), math.sin(gmst)
    x_ecef = c * x_km + s * y_km
    y_ecef = -s * x_km + c * y_km
    z_ecef = z_km
    lon = math.degrees(math.atan2(y_ecef, x_ecef))
    r_xy = math.sqrt(x_ecef**2 + y_ecef**2)
    lat = math.degrees(math.atan2(z_ecef, r_xy))
    alt = math.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2) - R_EARTH_KM
    return lat, lon, alt


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间初始方位角 (deg)，正北=0 顺时针递增。输入为度。"""
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlon = lo2 - lo1
    x = math.sin(dlon) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


_HEADING_DT_S = 10.0  # 前向传播秒数，用于计算地面轨迹航向

MU_EARTH = 398600.4418  # km³/s²


def _is_ideal(sat: Satellite) -> bool:
    """检测是否为理想星座卫星（TLE 中 bstar=0、eccentricity=0）。"""
    return sat.tle.eccentricity == 0.0 and sat.name.startswith("IDEAL-")


def _keplerian_propagate(sat: Satellite, jd: float, fr: float, ref_iso: str) -> bool:
    """纯二体 Keplerian 圆轨道传播（无 J2），保持理想星座的完美均匀分布。

    从 TLE 历元到目标时刻，仅推进纬度幅角 u（圆轨道角速度 × Δt），
    RAAN、倾角、半长轴均保持不变。
    """
    tle = sat.tle
    mm = tle.mean_motion_rev_per_day
    if mm <= 0:
        return False

    n_rad_s = mm * 2.0 * math.pi / 86400.0
    a_km = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)

    epoch_yr = tle.epoch_year
    year = 2000 + epoch_yr if epoch_yr < 57 else 1900 + epoch_yr
    epoch_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(days=tle.epoch_day - 1)

    if jday is None:
        return False
    jd0, fr0 = jday(epoch_dt.year, epoch_dt.month, epoch_dt.day,
                     epoch_dt.hour, epoch_dt.minute,
                     epoch_dt.second + epoch_dt.microsecond / 1e6)
    dt_sec = (jd - jd0 + fr - fr0) * 86400.0

    incl_rad = math.radians(tle.inclination_deg)
    raan_rad = math.radians(tle.raan_deg)
    ma0_rad = math.radians(tle.mean_anomaly_deg)
    u_rad = ma0_rad + n_rad_s * dt_sec

    cos_u, sin_u = math.cos(u_rad), math.sin(u_rad)
    cos_O, sin_O = math.cos(raan_rad), math.sin(raan_rad)
    cos_i, sin_i = math.cos(incl_rad), math.sin(incl_rad)

    rx = a_km * (cos_O * cos_u - sin_O * sin_u * cos_i)
    ry = a_km * (sin_O * cos_u + cos_O * sin_u * cos_i)
    rz = a_km * (sin_u * sin_i)

    v_circ = math.sqrt(MU_EARTH / a_km)
    vx = v_circ * (-cos_O * sin_u - sin_O * cos_u * cos_i)
    vy = v_circ * (-sin_O * sin_u + cos_O * cos_u * cos_i)
    vz = v_circ * (cos_u * sin_i)

    lat, lon, alt = _teme_to_geodetic(rx, ry, rz, jd, fr)
    sat.position.ref_time = ref_iso
    sat.position.latitude_deg = round(lat, 4)
    sat.position.longitude_deg = round(lon, 4)
    sat.position.height_km = round(alt, 2)
    sat.position.position_teme_km = [round(rx, 3), round(ry, 3), round(rz, 3)]
    sat.position.velocity_teme_km_s = [round(vx, 6), round(vy, 6), round(vz, 6)]

    rx_fwd = rx + vx * _HEADING_DT_S
    ry_fwd = ry + vy * _HEADING_DT_S
    rz_fwd = rz + vz * _HEADING_DT_S
    fr_fwd = fr + _HEADING_DT_S / 86400.0
    lat_fwd, lon_fwd, _ = _teme_to_geodetic(rx_fwd, ry_fwd, rz_fwd, jd, fr_fwd)
    sat.position.heading_deg = round(_bearing_deg(lat, lon, lat_fwd, lon_fwd), 2)
    return True


def get_latest_shell_path(
    shell_name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    source: str = "real",
) -> Path | None:
    """返回 data 目录下最新的 starlink_{shell}*.json 文件（按修改时间）。

    source: "real" 仅真实数据, "ideal" 仅理想星座, "any" 不区分。
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None
    base = f"starlink_{shell_name.lower()}_"
    candidates = list(data_dir.glob(f"{base}*.json"))
    if source == "real":
        candidates = [p for p in candidates if "ideal" not in p.name]
    elif source == "ideal":
        candidates = [p for p in candidates if "ideal" in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_shell(
    shell_name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    source: str = "real",
) -> list[Satellite]:
    """加载指定壳层卫星列表（使用该壳层最新的 JSON 文件）。

    shell_name: 如 "Gen1-1", "gen1-1", "Gen2-1" 等。
    source: "real" / "ideal" / "any"。
    """
    path = get_latest_shell_path(shell_name, data_dir, source=source)
    if path is None:
        raise FileNotFoundError(
            f"No starlink_{shell_name.lower()}*.json ({source}) found in {data_dir}"
        )
    return load_satellites(str(path))


def load_gen11(data_dir: Path = DEFAULT_DATA_DIR) -> list[Satellite]:
    """加载 Gen1-1 壳层（便捷函数，等价于 load_shell('Gen1-1')）。"""
    return load_shell("Gen1-1", data_dir)


def _parse_ref_time(ref_time: datetime | str) -> datetime:
    """将 ref_time 转为 timezone-aware datetime (UTC)."""
    if isinstance(ref_time, datetime):
        dt = ref_time
    else:
        dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def evolve_to_time(
    satellites: list[Satellite],
    ref_time: datetime | str,
) -> list[Satellite]:
    """根据传入时间，用 SGP4 将已加载卫星的 TLE 传播到该时刻，更新每颗星的 position 并返回。

    ref_time: datetime 或 ISO 字符串（如 '2026-03-13T12:00:00Z'），未带时区则视为 UTC。
    依赖 sgp4，若未安装则抛出 ImportError。

    --------
    计算原理（简要）
    --------
    1) TLE（两行根数）
       每颗星有 tle_line1 / tle_line2，包含某一历元时刻的轨道根数（半长轴、偏心率、
       倾角、升交点赤经等）及平均运动、一阶导数等。TLE 对应的是「历元时刻」的轨道。

    2) 时间 → 儒略日
       将 ref_time 转为 UTC 儒略日 (jd, fr)，供 SGP4 使用。jday() 返回整数部分 jd 与
       小数部分 fr，满足 时刻 = jd + fr（天为单位）。

    3) SGP4 传播
       SGP4（Simplified General Perturbations 4）是 NORAD 等用的轨道外推模型：
       - 输入：TLE 解析出的轨道根数 + 目标时刻 (jd, fr)
       - 输出：该时刻在 TEME 坐标系下的位置 r (km)、速度 v (km/s)
       TEME（True Equator Mean Equinox）是地心、赤道面、春分点定义的惯性系，随地球
       自转与章动有约定定义，SGP4 直接给出在此系下的 (x,y,z)。

    4) TEME → 大地坐标 (lat, lon, alt)
       TEME 是地心惯性系，要得到经纬度需要加上「地球自转」：同一时刻 TEME 与地固系
       (ECEF) 的转换由该时刻的 Greenwich Mean Sidereal Time (GMST) 决定。
       _gmst_rad(jd+fr) 算出 GMST 弧度；用旋转矩阵 (绕 z 轴 -GMST) 把 (x,y,z)_TEME
       转为 (x,y,z)_ECEF；再由 ECEF 反算纬度、经度、地心距，减去地球半径得海拔 alt_km。
       即 _teme_to_geodetic(x,y,z, jd, fr) → (lat_deg, lon_deg, alt_km)。

    5) 结果写回
       将 ref_time、lat/lon/alt、position_teme_km、velocity_teme_km_s 写回每颗星的
       position，便于后续路由、可视化等使用。
    """
    if Satrec is None or jday is None:
        raise ImportError("evolve_to_time requires the sgp4 package: pip install sgp4")

    dt = _parse_ref_time(ref_time)
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6,
    )
    ref_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    success = 0

    for sat in satellites:
        try:
            if _is_ideal(sat):
                if _keplerian_propagate(sat, jd, fr, ref_iso):
                    success += 1
                continue

            satrec = Satrec.twoline2rv(sat.tle.tle_line1, sat.tle.tle_line2)
            e, r, v = satrec.sgp4(jd, fr)
            if e != 0:
                continue
            lat, lon, alt = _teme_to_geodetic(r[0], r[1], r[2], jd, fr)
            sat.position.ref_time = ref_iso
            sat.position.latitude_deg = round(lat, 4)
            sat.position.longitude_deg = round(lon, 4)
            sat.position.height_km = round(alt, 2)
            sat.position.position_teme_km = [round(r[0], 3), round(r[1], 3), round(r[2], 3)]
            sat.position.velocity_teme_km_s = [round(v[0], 6), round(v[1], 6), round(v[2], 6)]
            r_fwd = [r[i] + v[i] * _HEADING_DT_S for i in range(3)]
            fr_fwd = fr + _HEADING_DT_S / 86400.0
            lat_fwd, lon_fwd, _ = _teme_to_geodetic(r_fwd[0], r_fwd[1], r_fwd[2], jd, fr_fwd)
            sat.position.heading_deg = round(_bearing_deg(lat, lon, lat_fwd, lon_fwd), 2)
            success += 1
        except Exception:
            continue

    return satellites


def load_shell_at_time(
    shell_name: str,
    ref_time: datetime | str,
    data_dir: Path = DEFAULT_DATA_DIR,
    source: str = "real",
) -> list[Satellite]:
    """加载指定壳层并将所有卫星传播到 ref_time，返回位置已更新到该时刻的列表。"""
    sats = load_shell(shell_name, data_dir, source=source)
    return evolve_to_time(sats, ref_time)


def load_gen11_at_time(
    ref_time: datetime | str,
    data_dir: Path = DEFAULT_DATA_DIR,
    source: str = "real",
) -> list[Satellite]:
    """加载 Gen1-1 并传播到 ref_time。

    source: "real" 真实 TLE 数据, "ideal" 理想星座, "any" 最新文件。
    """
    return load_shell_at_time("Gen1-1", ref_time, data_dir, source=source)

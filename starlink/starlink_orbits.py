"""
Download Starlink TLE data from CelesTrak and save as JSON.
"""

import json
import sys
import urllib.request
import re
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
OUTPUT_DIR = Path(__file__).parent / "data"


def tle_epoch_to_datetime(epoch_year: int, epoch_day: float) -> str:
    year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(days=epoch_day - 1)
    return dt.isoformat()


def parse_tle_line1(line: str) -> dict:
    return {
        "catalog_number": int(line[2:7]),
        "classification": line[7].strip(),
        "launch_year": line[9:11].strip(),
        "launch_number": line[11:14].strip(),
        "launch_piece": line[14:17].strip(),
        "epoch_year": int(line[18:20]),
        "epoch_day": float(line[20:32]),
        "epoch_utc": tle_epoch_to_datetime(int(line[18:20]), float(line[20:32])),
        "mean_motion_dot": float(line[33:43]),
        "bstar": _parse_decimal_assumption(line[53:61]),
        "element_set_number": int(line[64:68].strip()),
    }


def parse_tle_line2(line: str) -> dict:
    return {
        "inclination_deg": float(line[8:16]),
        "raan_deg": float(line[17:25]),
        "eccentricity": float("0." + line[26:33].strip()),
        "arg_perigee_deg": float(line[34:42]),
        "mean_anomaly_deg": float(line[43:51]),
        "mean_motion_rev_per_day": float(line[52:63]),
        "rev_number": int(line[63:68].strip()),
    }


def _parse_decimal_assumption(s: str) -> float:
    """Parse TLE's implied-decimal-point notation like ' 12345-6' -> 0.12345e-6."""
    s = s.strip()
    if not s or s == "0" or s == "00000-0":
        return 0.0
    match = re.match(r"([+-]?)(\d+)([+-]\d)", s)
    if match:
        sign = -1 if match.group(1) == "-" else 1
        mantissa = float("0." + match.group(2))
        exponent = int(match.group(3))
        return sign * mantissa * (10 ** exponent)
    return 0.0


import math

MU_EARTH = 398600.4418        # km³/s²  (地球引力常数)
R_EARTH  = 6371.0             # km      (地球平均半径)

# FCC 批准的 Starlink 壳层定义: (name, altitude_km, inclination_deg, tolerance_km, tolerance_deg)
STARLINK_SHELLS = [
    ("Gen1-1",  550, 53.0,  30, 2.0),
    ("Gen1-2",  540, 53.2,  30, 2.0),
    ("Gen1-3",  570, 70.0,  30, 3.0),
    ("Gen1-4",  560, 97.6,  30, 2.0),
    ("Gen2-1",  525, 53.0,  30, 2.0),
    ("Gen2-2",  530, 43.0,  30, 3.0),
    ("Gen2-3",  535, 33.0,  30, 3.0),
    ("Gen2-4",  604, 148.0, 30, 3.0),
    ("Gen2-5",  614, 115.7, 30, 3.0),
]


def mean_motion_to_altitude_km(n_rev_per_day: float, inclination_deg: float = 53.0,
                                eccentricity: float = 0.0) -> float:
    """从 TLE 平均运动 (rev/day) 计算轨道高度 (km), 含 J2 修正."""
    J2 = 1.08263e-3
    omega = 2.0 * math.pi * n_rev_per_day / 86400.0
    a0 = (MU_EARTH / (omega ** 2)) ** (1.0 / 3.0)
    cos_i = math.cos(math.radians(inclination_deg))
    p = a0 / R_EARTH
    delta = 0.75 * J2 / (p ** 2) * (3.0 * cos_i ** 2 - 1.0) / (1.0 - eccentricity ** 2) ** 1.5
    a = a0 * (1.0 - delta / 3.0)
    return a - R_EARTH


def classify_shell(inclination_deg: float, altitude_km: float) -> tuple[str, str]:
    """根据倾角和高度匹配最接近的 Starlink 壳层.
    Returns (shell_name, status): status ∈ {'operational', 'raising', 'deorbiting', 'drifting', 'unknown'}.
    """
    best_name, best_alt, best_score = None, 0, float("inf")
    for name, alt, inc, tol_alt, tol_inc in STARLINK_SHELLS:
        # 倾角必须在合理范围内才纳入考虑
        if abs(inclination_deg - inc) > 5.0:
            continue
        score = (abs(inclination_deg - inc) / tol_inc) ** 2 + (abs(altitude_km - alt) / tol_alt) ** 2
        if score < best_score:
            best_name, best_alt, best_score = name, alt, score

    if best_name is None:
        return "unknown", "unknown"

    alt_diff = altitude_km - best_alt
    if abs(alt_diff) <= 30:
        return best_name, "operational"
    elif altitude_km < 350:
        return best_name, "deorbiting"
    elif alt_diff < -30:
        return best_name, "raising"
    else:
        return best_name, "drifting"


def enrich_with_shell(satellites: list[dict]) -> list[dict]:
    """为每颗卫星添加 altitude_km, shell, status 字段."""
    for sat in satellites:
        alt = mean_motion_to_altitude_km(sat["mean_motion_rev_per_day"],
                                         sat["inclination_deg"], sat["eccentricity"])
        sat["altitude_km"] = round(alt, 2)
        shell, status = classify_shell(sat["inclination_deg"], alt)
        sat["shell"] = shell
        sat["status"] = status
    return satellites


def download_tle(url: str = TLE_URL) -> str:
    print(f"Downloading TLE data from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    return data


def parse_tle(raw: str) -> list[dict]:
    lines = [l.rstrip() for l in raw.strip().splitlines() if l.strip()]
    satellites = []
    i = 0
    while i + 2 < len(lines):
        name_line = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            i += 1
            continue
        info = {"name": name_line.strip()}
        info.update(parse_tle_line1(line1))
        info.update(parse_tle_line2(line2))
        info["tle_line1"] = line1
        info["tle_line2"] = line2
        satellites.append(info)
        i += 3
    return satellites


def fetch_and_save(url: str = TLE_URL, output_dir: Path = OUTPUT_DIR) -> Path:
    """Download Starlink TLE data, parse it, and save as JSON. Returns the output file path."""
    raw = download_tle(url)
    satellites = parse_tle(raw)
    enrich_with_shell(satellites)
    print(f"Parsed {len(satellites)} Starlink satellites")

    shell_counts: dict[str, dict[str, int]] = {}
    for sat in satellites:
        shell = sat["shell"]
        status = sat["status"]
        if shell not in shell_counts:
            shell_counts[shell] = {}
        shell_counts[shell][status] = shell_counts[shell].get(status, 0) + 1
    for shell in sorted(shell_counts):
        parts = ", ".join(f"{s}={c}" for s, c in sorted(shell_counts[shell].items()))
        total = sum(shell_counts[shell].values())
        print(f"  {shell}: {total} ({parts})")

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"starlink_tle_{ts}.json"

    payload = {
        "source": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "count": len(satellites),
        "shell_counts": shell_counts,
        "satellites": satellites,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def get_latest_json(dir_path: Path) -> Path | None:
    """返回目录下修改时间最新的、以 starlink_tle_ 开头的 .json 文件；若无则返回 None。"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return None
    jsons = list(dir_path.glob("starlink_tle_*.json"))
    if not jsons:
        return None
    return max(jsons, key=lambda p: p.stat().st_mtime)


def filter_by_shell(input_path: Path, shell_name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """从已有的 JSON 文件中过滤指定壳层的卫星, 传播到当前时刻并另存."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [s for s in data["satellites"] if s["shell"] == shell_name]
    shell_def = next((s for s in STARLINK_SHELLS if s[0] == shell_name), None)

    ref_time = datetime.now(timezone.utc)
    propagate_positions(filtered, ref_time)

    payload = {
        "source": data["source"],
        "downloaded_at": data["downloaded_at"],
        "ref_time": ref_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shell": shell_name,
        "shell_info": {
            "altitude_km": shell_def[1],
            "inclination_deg": shell_def[2],
        } if shell_def else {},
        "count": len(filtered),
        "satellites": filtered,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = ref_time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"starlink_{shell_name.lower()}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Filtered {len(filtered)} satellites (shell={shell_name}) -> {out_path}")
    return out_path


from datetime import timedelta
from sgp4.api import Satrec, jday


def generate_czml(satellites: list[dict], duration_min: int = 120, step_sec: int = 30) -> list[dict]:
    """用 SGP4 传播 TLE，生成 Cesium CZML 格式数据."""
    now = datetime.now(timezone.utc)
    start_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now + timedelta(minutes=duration_min)
    end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    interval = f"{start_iso}/{end_iso}"

    czml: list[dict] = [{
        "id": "document",
        "name": "Starlink Constellation",
        "version": "1.0",
        "clock": {
            "interval": interval,
            "currentTime": start_iso,
            "multiplier": 60,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }]

    total_steps = duration_min * 60 // step_sec
    orbit_period_sec = 2 * math.pi * math.sqrt((R_EARTH + 550) ** 3 / MU_EARTH)

    propagate_positions(satellites, now)

    for sat in satellites:
        if "position_error" in sat:
            continue

        try:
            satrec = Satrec.twoline2rv(sat["tle_line1"], sat["tle_line2"])
        except Exception:
            continue

        cartesian = []
        for i in range(total_steps + 1):
            t_sec = i * step_sec
            dt = now + timedelta(seconds=t_sec)
            jd, fr = jday(dt.year, dt.month, dt.day,
                          dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
            e, r, _ = satrec.sgp4(jd, fr)
            if e != 0:
                continue
            cartesian.extend([t_sec, r[0] * 1000, r[1] * 1000, r[2] * 1000])

        if not cartesian:
            continue

        hue = (sat.get("raan_deg", 0) % 360) / 360
        r_c, g_c, b_c = _hsl_to_rgb(hue, 0.8, 0.7)

        packet: dict = {
            "id": str(sat["catalog_number"]),
            "name": sat["name"],
            "availability": interval,
            "position": {
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
                "referenceFrame": "INERTIAL",
                "epoch": start_iso,
                "cartesian": cartesian,
            },
            "point": {
                "color": {"rgba": [r_c, g_c, b_c, 255]},
                "pixelSize": 6,
                "outlineColor": {"rgba": [255, 255, 255, 80]},
                "outlineWidth": 1,
                "scaleByDistance": {
                    "nearFarScalar": [1e6, 2.5, 4e7, 0.8],
                },
            },
            "path": {
                "show": [{"interval": interval, "boolean": True}],
                "width": 0.5,
                "material": {
                    "solidColor": {
                        "color": {"rgba": [r_c, g_c, b_c, 35]},
                    }
                },
                "resolution": 120,
                "leadTime": orbit_period_sec / 2,
                "trailTime": orbit_period_sec / 2,
            },
            "properties": {
                "catalog_number": sat.get("catalog_number"),
                "shell": sat.get("shell"),
                "status": sat.get("status"),
                "inclination_deg": sat.get("inclination_deg"),
                "altitude_km": sat.get("height_km"),
                "latitude_deg": sat.get("latitude_deg"),
                "longitude_deg": sat.get("longitude_deg"),
            },
        }
        czml.append(packet)

    return czml


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (radians), IAU 1982 model."""
    t_ut1 = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * t_ut1 \
               + 0.093104 * t_ut1 ** 2 - 6.2e-6 * t_ut1 ** 3
    return (gmst_sec % 86400) / 86400 * 2 * math.pi


def _teme_to_geodetic(x_km: float, y_km: float, z_km: float, jd: float, fr: float):
    """TEME (km) → geodetic (lat_deg, lon_deg, alt_km)."""
    gmst = _gmst_rad(jd + fr)
    cos_g, sin_g = math.cos(gmst), math.sin(gmst)
    x_ecef = cos_g * x_km + sin_g * y_km
    y_ecef = -sin_g * x_km + cos_g * y_km
    z_ecef = z_km

    lon = math.degrees(math.atan2(y_ecef, x_ecef))
    r_xy = math.sqrt(x_ecef ** 2 + y_ecef ** 2)
    lat = math.degrees(math.atan2(z_ecef, r_xy))
    alt = math.sqrt(x_ecef ** 2 + y_ecef ** 2 + z_ecef ** 2) - R_EARTH
    return lat, lon, alt


def propagate_positions(satellites: list[dict], ref_time: datetime = None) -> list[dict]:
    """用 SGP4 将所有卫星传播到同一参考时刻，添加经纬度和位置信息."""
    if ref_time is None:
        ref_time = datetime.now(timezone.utc)

    jd, fr = jday(ref_time.year, ref_time.month, ref_time.day,
                  ref_time.hour, ref_time.minute,
                  ref_time.second + ref_time.microsecond / 1e6)

    ref_iso = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    success = 0

    for sat in satellites:
        try:
            satrec = Satrec.twoline2rv(sat["tle_line1"], sat["tle_line2"])
            e, r, v = satrec.sgp4(jd, fr)
            if e != 0:
                sat["position_error"] = f"SGP4 error code {e}"
                continue
            lat, lon, alt = _teme_to_geodetic(r[0], r[1], r[2], jd, fr)
            sat["ref_time"] = ref_iso
            sat["latitude_deg"] = round(lat, 4)
            sat["longitude_deg"] = round(lon, 4)
            sat["height_km"] = round(alt, 2)
            sat["position_teme_km"] = [round(r[0], 3), round(r[1], 3), round(r[2], 3)]
            sat["velocity_teme_km_s"] = [round(v[0], 6), round(v[1], 6), round(v[2], 6)]
            success += 1
        except Exception as ex:
            sat["position_error"] = str(ex)

    print(f"Propagated {success}/{len(satellites)} satellites to {ref_iso}")
    return satellites


def get_latest_shell_json(shell_name: str, dir_path: Path) -> Path | None:
    """返回目录下修改时间最新的、starlink_{shell}.json 或 starlink_{shell}_*.json；若无则返回 None。"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return None
    base = f"starlink_{shell_name.lower()}"
    # 带时间戳: starlink_gen1-1_20250121_123456.json；或旧版无时间戳: starlink_gen1-1.json
    candidates = list(dir_path.glob(f"{base}_*.json")) + list(dir_path.glob(f"{base}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_shell_data(shell_name: str = "Gen1-1", data_dir: Path = OUTPUT_DIR) -> dict:
    """加载指定壳层的缓存数据; 若不存在则先下载并过滤."""
    cache_path = get_latest_shell_json(shell_name, data_dir)
    if cache_path is None:
        all_path = fetch_and_save(output_dir=data_dir)
        filter_by_shell(all_path, shell_name, output_dir=data_dir)
        cache_path = get_latest_shell_json(shell_name, data_dir)
        assert cache_path is not None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tle_checksum(line: str) -> int:
    """TLE 行校验和：数字按值累加，'-' 算 1，其余忽略，取 mod 10。"""
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == '-':
            s += 1
    return s % 10


def _make_ideal_tle(
    cat: int,
    epoch_yr: int,
    epoch_day: float,
    incl_deg: float,
    raan_deg: float,
    ecc: float,
    argp_deg: float,
    ma_deg: float,
    mm_rev_day: float,
) -> tuple[str, str]:
    """为理想卫星生成合法的 TLE 两行字符串（无阻力、零导数）。"""
    ecc_s = f"{ecc:.7f}"[2:]  # "0.0001000" -> "0001000"
    l1 = (
        f"1 {cat:05d}U 26001A   "
        f"{epoch_yr:02d}{epoch_day:012.8f}"
        f"  .00000000  00000+0  00000+0 0  999"
    )
    l1 += str(_tle_checksum(l1))
    l2 = (
        f"2 {cat:05d}"
        f" {incl_deg:8.4f}"
        f" {raan_deg:8.4f}"
        f" {ecc_s}"
        f" {argp_deg:8.4f}"
        f" {ma_deg:8.4f}"
        f" {mm_rev_day:11.8f}"
        f"    0"
    )
    l2 += str(_tle_checksum(l2))
    return l1, l2


# Shell 1 (Gen1-1) Walker Delta 星座设计参数
SHELL1_ALT_KM = 550
SHELL1_INCL_DEG = 53.0
SHELL1_NUM_PLANES = 72
SHELL1_SATS_PER_PLANE = 22
SHELL1_TOTAL = SHELL1_NUM_PLANES * SHELL1_SATS_PER_PLANE  # 1584
SHELL1_RAAN_STEP = 360.0 / SHELL1_NUM_PLANES              # 5.0°
SHELL1_MA_STEP = 360.0 / SHELL1_SATS_PER_PLANE            # 16.3636°
SHELL1_WALKER_F = 1  # Walker 相位因子


def _keplerian_state(a_km: float, incl_rad: float, raan_rad: float, u_rad: float):
    """纯二体开普勒圆轨道 → ECI 位置/速度（无 J2 摄动，完美均匀）。

    a_km     : 半长轴 (km)
    incl_rad : 轨道倾角 (rad)
    raan_rad : 升交点赤经 (rad)
    u_rad    : 纬度幅角 = ω + ν (rad)，圆轨道即真近点角
    返回 (r_km[3], v_km_s[3])
    """
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

    return (rx, ry, rz), (vx, vy, vz)


def generate_ideal_shell1(
    ref_time: datetime | None = None,
    output_dir: Path = OUTPUT_DIR,
    cat_start: int = 80001,
) -> Path:
    """生成理想 Shell 1 (Gen1-1) Walker Delta 72/22/1 星座并保存为 JSON。

    - 72 个轨道面 × 22 颗/面 = 1584 颗卫星
    - 完美圆轨道 (ecc=0)，高度 550 km，倾角 53.0°
    - RAAN 间隔 5°，面内均匀分布，相邻面 Walker 相移 Δu = F × 360°/1584
    - 使用解析开普勒公式计算位置（无 J2 摄动），保证空间分布完美均匀
    - 同时生成合法 TLE 供后续 SGP4 传播使用
    """
    if ref_time is None:
        ref_time = datetime.now(timezone.utc)

    a_km = R_EARTH + SHELL1_ALT_KM
    n_rad_s = math.sqrt(MU_EARTH / a_km ** 3)
    mm_rev_day = n_rad_s * 86400.0 / (2.0 * math.pi)

    epoch_yr = ref_time.year % 100
    day_of_year = (ref_time - datetime(ref_time.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 1.0
    epoch_day = day_of_year

    walker_phase_step = SHELL1_WALKER_F * 360.0 / SHELL1_TOTAL  # ≈ 0.2273°

    jd, fr = jday(ref_time.year, ref_time.month, ref_time.day,
                  ref_time.hour, ref_time.minute,
                  ref_time.second + ref_time.microsecond / 1e6)
    ref_iso = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    incl_rad = math.radians(SHELL1_INCL_DEG)

    satellites: list[dict] = []

    for p in range(SHELL1_NUM_PLANES):
        raan_deg = (p * SHELL1_RAAN_STEP) % 360.0
        raan_rad = math.radians(raan_deg)
        for s in range(SHELL1_SATS_PER_PLANE):
            cat = cat_start + p * SHELL1_SATS_PER_PLANE + s
            ma_deg = (s * SHELL1_MA_STEP + p * walker_phase_step) % 360.0
            u_rad = math.radians(ma_deg)

            tle1, tle2 = _make_ideal_tle(
                cat=cat,
                epoch_yr=epoch_yr,
                epoch_day=epoch_day,
                incl_deg=SHELL1_INCL_DEG,
                raan_deg=raan_deg,
                ecc=0.0,
                argp_deg=0.0,
                ma_deg=ma_deg,
                mm_rev_day=mm_rev_day,
            )

            r, v = _keplerian_state(a_km, incl_rad, raan_rad, u_rad)
            lat, lon, alt = _teme_to_geodetic(r[0], r[1], r[2], jd, fr)

            sat = {
                "name": f"IDEAL-P{p:02d}-S{s:02d}",
                "catalog_number": cat,
                "classification": "U",
                "launch_year": f"{epoch_yr:02d}",
                "launch_number": "001",
                "launch_piece": "A",
                "epoch_year": epoch_yr,
                "epoch_day": epoch_day,
                "epoch_utc": ref_time.isoformat(),
                "mean_motion_dot": 0.0,
                "bstar": 0.0,
                "element_set_number": 999,
                "inclination_deg": SHELL1_INCL_DEG,
                "raan_deg": round(raan_deg, 4),
                "eccentricity": 0.0,
                "arg_perigee_deg": 0.0,
                "mean_anomaly_deg": round(ma_deg, 4),
                "mean_motion_rev_per_day": round(mm_rev_day, 8),
                "rev_number": 0,
                "tle_line1": tle1,
                "tle_line2": tle2,
                "altitude_km": SHELL1_ALT_KM,
                "shell": "Gen1-1",
                "status": "operational",
                "ref_time": ref_iso,
                "latitude_deg": round(lat, 4),
                "longitude_deg": round(lon, 4),
                "height_km": round(alt, 2),
                "position_teme_km": [round(r[0], 3), round(r[1], 3), round(r[2], 3)],
                "velocity_teme_km_s": [round(v[0], 6), round(v[1], 6), round(v[2], 6)],
            }
            satellites.append(sat)

    payload = {
        "source": "ideal_constellation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ref_time": ref_iso,
        "shell": "Gen1-1",
        "shell_info": {
            "altitude_km": SHELL1_ALT_KM,
            "inclination_deg": SHELL1_INCL_DEG,
        },
        "constellation": {
            "type": "Walker Delta",
            "planes": SHELL1_NUM_PLANES,
            "sats_per_plane": SHELL1_SATS_PER_PLANE,
            "phase_factor": SHELL1_WALKER_F,
        },
        "count": len(satellites),
        "satellites": satellites,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = ref_time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"starlink_gen1-1_ideal_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Generated ideal Shell 1: {SHELL1_NUM_PLANES} planes × {SHELL1_SATS_PER_PLANE} sats "
        f"= {len(satellites)} total -> {out_path}"
    )
    return out_path


import argparse


def main():
    parser = argparse.ArgumentParser(description="Starlink TLE data downloader & processor")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download TLE and save JSON")
    p_filter = sub.add_parser("filter", help="Filter by shell")
    p_filter.add_argument("--shell", default="Gen1-1")
    p_filter.add_argument(
        "--input",
        default=None,
        help="Input JSON path; if omitted, use the latest .json in data dir (by mtime)",
    )
    sub.add_parser("ideal", help="Generate ideal Shell 1 constellation (72x22)")

    args = parser.parse_args()

    if args.command == "download":
        fetch_and_save()
    elif args.command == "ideal":
        generate_ideal_shell1()
    elif args.command == "filter":
        if args.input is None:
            input_path = get_latest_json(OUTPUT_DIR)
            if input_path is None:
                print("No JSON file in data dir. Run 'download' first or pass --input <path>.", file=sys.stderr)
                sys.exit(1)
            print(f"Using latest: {input_path}")
        else:
            input_path = Path(args.input)
        filter_by_shell(input_path, args.shell)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Generate a publication figure for two real-TLE topology snapshots.

The snapshots use the same propagation and ISL-construction pipeline as
``eval_timevarying.py``.  Stable and switched links are distinguished by
catalog-number pairs so the figure remains meaningful as satellites move.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from data_loader import load_gen11_at_time
from isl import build_isl_from_satellites, update_satellite_isl_peers
from route import _build_isl_adj
from eval_timevarying import CANONICAL_REF_TIME, _topology_hash


def _parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _snapshot(at: datetime) -> dict:
    satellites = load_gen11_at_time(at, source="real")
    build_isl_from_satellites(satellites)
    update_satellite_isl_peers(satellites)
    adjacency = _build_isl_adj(satellites)
    positions = {
        sat.catalog_number: (
            float(sat.position.longitude_deg),
            float(sat.position.latitude_deg),
        )
        for sat in satellites
    }
    edges = {
        tuple(sorted((satellites[u].catalog_number, satellites[v].catalog_number)))
        for u, neighbors in adjacency.items()
        for v in neighbors
        if u < v
    }
    return {
        "time": at,
        "satellites": satellites,
        "positions": positions,
        "edges": edges,
        "topology_sha256": _topology_hash(adjacency),
    }


def _edge_segments(edges: set[tuple[int, int]], positions: dict) -> list:
    """Return line segments without drawing long artifacts across the dateline."""
    segments = []
    for a, b in edges:
        lon1, lat1 = positions[a]
        lon2, lat2 = positions[b]
        delta = lon2 - lon1
        if abs(delta) <= 180:
            segments.append([(lon1, lat1), (lon2, lat2)])
            continue

        # Unwrap the second longitude, find the crossing, and split at +/-180.
        lon2_unwrapped = lon2 - 360 if delta > 180 else lon2 + 360
        boundary = -180 if lon2_unwrapped < -180 else 180
        frac = (boundary - lon1) / (lon2_unwrapped - lon1)
        lat_cross = lat1 + frac * (lat2 - lat1)
        other_boundary = 180 if boundary == -180 else -180
        segments.append([(lon1, lat1), (boundary, lat_cross)])
        segments.append([(other_boundary, lat_cross), (lon2, lat2)])
    return segments


def _draw_snapshot(ax, snapshot: dict, stable: set, changed: set, changed_color: str,
                   changed_label: str, elapsed_min: int) -> None:
    pos = snapshot["positions"]
    ax.set_facecolor("#061522")
    ax.add_collection(LineCollection(
        _edge_segments(stable, pos),
        colors="#72b7e6",
        linewidths=0.18,
        alpha=0.16,
        zorder=1,
    ))
    ax.add_collection(LineCollection(
        _edge_segments(changed, pos),
        colors=changed_color,
        linewidths=0.50,
        alpha=0.68,
        zorder=2,
    ))
    lons = [value[0] for value in pos.values()]
    lats = [value[1] for value in pos.values()]
    ax.scatter(lons, lats, s=1.4, c="#e9f7ff", alpha=0.82,
               edgecolors="none", zorder=3)

    timestamp = snapshot["time"].strftime("%Y-%m-%d %H:%M:%S UTC")
    ax.set_title(
        f"({'a' if elapsed_min == 0 else 'b'})  $t={elapsed_min}$ min  |  {timestamp}\n"
        f"$N={len(snapshot['satellites']):,}$, ISLs={len(snapshot['edges']):,}, "
        f"{changed_label}={len(changed):,}",
        color="#102b3b",
        fontsize=9.5,
        pad=7,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Longitude (deg)", fontsize=8.5)
    ax.set_ylabel("Latitude (deg)", fontsize=8.5)
    ax.tick_params(labelsize=7.5, colors="#334e60")
    ax.grid(color="white", alpha=0.10, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_color("#5c7280")
        spine.set_linewidth(0.6)


def _validate_against_results(snapshot0: dict, snapshot1: dict, results_path: Path,
                              elapsed_min: int) -> None:
    if not results_path.exists():
        return
    with results_path.open(encoding="utf-8") as stream:
        results = json.load(stream)
    rows = {int(round(row["t_min"])): row for row in results["rows"]}
    for elapsed, snapshot in ((0, snapshot0), (elapsed_min, snapshot1)):
        row = rows.get(elapsed)
        if row is None:
            raise RuntimeError(f"No canonical result row found for t={elapsed} min")
        if row["n_edges"] != len(snapshot["edges"]):
            raise RuntimeError(
                f"ISL count mismatch at t={elapsed}: "
                f"{len(snapshot['edges'])} != {row['n_edges']}"
            )
        if row["topology_sha256"] != snapshot["topology_sha256"]:
            raise RuntimeError(f"Topology hash mismatch at t={elapsed} min")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", default=CANONICAL_REF_TIME)
    parser.add_argument("--elapsed-min", type=int, default=90)
    parser.add_argument(
        "--results",
        type=Path,
        default=_GOOD / "results_timevarying.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_GOOD / "fig8_real_tle_snapshots.pdf",
    )
    args = parser.parse_args()

    start = _parse_time(args.start_time)
    end = start + timedelta(minutes=args.elapsed_min)
    snapshot0 = _snapshot(start)
    snapshot1 = _snapshot(end)
    _validate_against_results(snapshot0, snapshot1, args.results, args.elapsed_min)

    stable = snapshot0["edges"] & snapshot1["edges"]
    removed = snapshot0["edges"] - snapshot1["edges"]
    added = snapshot1["edges"] - snapshot0["edges"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    _draw_snapshot(
        axes[0], snapshot0, stable, removed, "#ff9f43", "links absent at t=90", 0
    )
    _draw_snapshot(
        axes[1], snapshot1, stable, added, "#ff4f9a", "links absent at t=0",
        args.elapsed_min,
    )
    axes[1].set_ylabel("")

    legend = [
        Line2D([0], [0], color="#72b7e6", linewidth=1.3, alpha=0.7,
               label=f"Stable ISL ({len(stable):,})"),
        Line2D([0], [0], color="#ff9f43", linewidth=1.7,
               label=f"Removed by t=90 ({len(removed):,})"),
        Line2D([0], [0], color="#ff4f9a", linewidth=1.7,
               label=f"Added by t=90 ({len(added):,})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#607d8b",
               markeredgecolor="none", markersize=4,
               label="Active satellite"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.5, 0.005))
    fig.text(
        0.5,
        0.055,
        "Real Starlink Gen1-1 TLEs; SGP4 propagation; ISLs recomputed with "
        "the same pipeline as the zero-shot transfer experiment.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#425b6a",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.85, bottom=0.19, wspace=0.08)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated {args.out}")
    print(
        f"N={len(snapshot0['satellites'])}; stable={len(stable)}; "
        f"removed={len(removed)}; added={len(added)}; "
        f"ISLs={len(snapshot0['edges'])}->{len(snapshot1['edges'])}"
    )


if __name__ == "__main__":
    main()

"""Generate a spatial link-load comparison for the submission manuscript."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from dynamic_experiment import (
    CANONICAL_REF_TIME,
    ekey,
    evaluate,
    flows_fingerprint,
    load_topology,
    route_la_dijkstra,
    route_static,
    topology_fingerprint,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _edge_segments(edges: set[tuple[int, int]], satellites: list) -> list:
    """Return equirectangular segments without artifacts across the dateline."""
    segments = []
    for u, v in sorted(edges):
        p1 = satellites[u].position
        p2 = satellites[v].position
        lon1, lat1 = float(p1.longitude_deg), float(p1.latitude_deg)
        lon2, lat2 = float(p2.longitude_deg), float(p2.latitude_deg)
        delta = lon2 - lon1
        if abs(delta) <= 180:
            segments.append([(lon1, lat1), (lon2, lat2)])
            continue
        lon2_unwrapped = lon2 - 360 if delta > 180 else lon2 + 360
        boundary = -180 if lon2_unwrapped < -180 else 180
        fraction = (boundary - lon1) / (lon2_unwrapped - lon1)
        lat_cross = lat1 + fraction * (lat2 - lat1)
        other_boundary = 180 if boundary == -180 else -180
        segments.append([(lon1, lat1), (boundary, lat_cross)])
        segments.append([(other_boundary, lat_cross), (lon2, lat2)])
    return segments


def _edge_load(paths: list) -> dict[tuple[int, int], int]:
    load: dict[tuple[int, int], int] = {}
    for path in paths:
        if not path:
            continue
        for u, v in zip(path, path[1:]):
            edge = ekey(u, v)
            load[edge] = load.get(edge, 0) + 1
    return load


def _metric_matches(actual: dict, expected: dict) -> bool:
    keys = ("max_util", "drop_rate", "goodput")
    return all(actual[key] == expected[key] for key in keys)


def _draw(ax, satellites: list, edges: set, load: dict, cap: int,
          title: str, metric: dict, norm: Normalize):
    segments = []
    utilizations = []
    for edge in sorted(edges):
        edge_segments = _edge_segments({edge}, satellites)
        rho = load.get(edge, 0) / cap
        segments.extend(edge_segments)
        utilizations.extend([rho] * len(edge_segments))

    collection = LineCollection(
        segments,
        array=utilizations,
        cmap="turbo",
        norm=norm,
        linewidths=0.72,
        alpha=0.92,
        zorder=2,
    )
    ax.add_collection(collection)
    ax.scatter(
        [sat.position.longitude_deg for sat in satellites],
        [sat.position.latitude_deg for sat in satellites],
        s=1.25,
        color="#263c4a",
        alpha=0.55,
        edgecolors="none",
        zorder=3,
    )
    loss_pct = f"{metric['drop_rate'] * 100:.2f}".rstrip("0").rstrip(".")
    ax.set_title(
        f"{title}\n"
        f"peak $\\rho={metric['max_util']:.2f}$, "
        f"packet loss={loss_pct}\\%, "
        f"delivered={metric['goodput']:,}/2,000",
        fontsize=11.5,
        pad=7,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("Longitude (deg)", fontsize=10)
    ax.set_ylabel("Latitude (deg)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.18, linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_color("#71808a")
        spine.set_linewidth(0.6)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=_GOOD / "results_dynamic_main.json")
    parser.add_argument("--out", type=Path,
                        default=_GOOD / "fig8_load_distribution.pdf")
    parser.add_argument("--flows", type=int, default=2000)
    args = parser.parse_args()

    canonical = _load_json(args.results)
    params = canonical["params"]
    cap = int(params["cap"])
    tserv = float(params["tserv"])
    beta = float(params["beta"])
    seed = int(params["seed"])
    ref_time = params.get("ref_time") or CANONICAL_REF_TIME
    source = params.get("source", "ideal")

    satellites, adjacency, edges, propagation, _ecef = load_topology(
        source, ref_time
    )
    if topology_fingerprint(adjacency, propagation) != canonical["topology"]["sha256"]:
        raise RuntimeError("Topology does not match the canonical dynamic experiment")

    rng = random.Random(seed)
    flows = []
    while len(flows) < args.flows:
        src, dst = rng.randrange(len(satellites)), rng.randrange(len(satellites))
        if src != dst:
            flows.append((src, dst))
    canonical_row = next(
        row for row in canonical["congestion"] if row["K"] == args.flows
    )
    if flows_fingerprint(flows) != canonical_row["flows_sha256"]:
        raise RuntimeError("Flow manifest does not match the canonical experiment")

    sp_paths = route_static(adjacency, flows, lambda _u, _v: 1.0)
    la_paths = route_la_dijkstra(
        adjacency, propagation, flows, cap, tserv, beta
    )
    sp_metric = evaluate(sp_paths, propagation, cap, tserv)
    la_metric = evaluate(la_paths, propagation, cap, tserv)
    if not _metric_matches(sp_metric, canonical_row["SP-hop"]):
        raise RuntimeError("SP-hop metrics do not match the canonical result")
    if not _metric_matches(la_metric, canonical_row["LA-Dijkstra"]):
        raise RuntimeError("LA-Dijkstra metrics do not match the canonical result")

    norm = Normalize(vmin=0.0, vmax=3.0, clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True)
    collection = _draw(
        axes[0], satellites, edges, _edge_load(sp_paths), cap,
        "(a) SP-hop (load-blind)", sp_metric, norm,
    )
    _draw(
        axes[1], satellites, edges, _edge_load(la_paths), cap,
        "(b) LA-Dijkstra (load-aware)", la_metric, norm,
    )
    axes[1].set_ylabel("")

    colorbar = fig.colorbar(
        collection,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.17,
        aspect=45,
        ticks=[0, 0.5, 1, 2, 3],
    )
    colorbar.set_label(
        "Link utilization $\\rho$ (overload threshold: $\\rho=1$; "
        "values clipped at 3)",
        fontsize=10,
    )
    colorbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.30, wspace=0.08)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()

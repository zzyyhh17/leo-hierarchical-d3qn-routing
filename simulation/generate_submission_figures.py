"""Generate submission figures directly from the canonical experiment JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_congestion(results: dict, output: Path) -> None:
    rows = [r for r in results["congestion"] if r["K"] <= 2000]
    x = [r["K"] for r in rows]
    styles = {
        "SP-hop": ("o", "#1f77b4"),
        "SP-dist": ("s", "#d62728"),
        "LA-Dijkstra": ("^", "#2ca02c"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for name, (marker, color) in styles.items():
        axes[0].plot(
            x,
            [r[name]["drop_rate"] * 100 for r in rows],
            marker=marker,
            color=color,
            label=name,
        )
        axes[1].plot(
            x,
            [r[name]["max_util"] for r in rows],
            marker=marker,
            color=color,
            label=name,
        )

    axes[0].set_title("(a) Congestion packet loss")
    axes[0].set_xlabel("Concurrent flows $F$")
    axes[0].set_ylabel("Packet loss rate (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("(b) Peak link utilization")
    axes[1].set_xlabel("Concurrent flows $F$")
    axes[1].set_ylabel("Maximum link utilization")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="overload threshold $\\rho=1$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(results: dict, output: Path) -> None:
    styles = {
        "SP-hop": ("o", "#1f77b4", "SP-hop (load-blind)"),
        "overlay": ("s", "#ff7f0e", "D3QN-LA (inference superposition)"),
        "trained-rand": ("D", "#9467bd", "D3QN (random background)"),
        "trained-mf": ("v", "#17becf", "D3QN (cumulative multi-flow)"),
        "LA-Dijkstra": ("*", "#2ca02c", "LA-Dijkstra (centralized)"),
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, (marker, color, label) in styles.items():
        x = [r[key]["max_util"] for r in results["rows"]]
        y = [r[key]["goodput"] for r in results["rows"]]
        ax.plot(x, y, marker=marker, color=color, label=label)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Peak link utilization")
    ax.set_ylabel("Delivered flows (throughput)")
    ax.set_xlim(left=0.45)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", type=Path, default=Path("results_dynamic_main.json"))
    ap.add_argument("--final", type=Path, default=Path("results_final.json"))
    ap.add_argument("--congestion-out", type=Path, default=Path("fig5_congestion.pdf"))
    ap.add_argument("--tradeoff-out", type=Path, default=Path("fig6_tradeoff.pdf"))
    args = ap.parse_args()

    dynamic = load_json(args.dynamic)
    final = load_json(args.final)
    if dynamic["topology"]["sha256"] != final["topology"]["sha256"]:
        raise RuntimeError("Dynamic and final result files use different topology snapshots")
    for row in final["rows"]:
        base = next(r for r in dynamic["congestion"] if r["K"] == row["K"])
        if base["flows_sha256"] != row["flows_sha256"]:
            raise RuntimeError(f"Flow manifest mismatch at F={row['K']}")

    plot_congestion(dynamic, args.congestion_out)
    plot_tradeoff(final, args.tradeoff_out)
    print(f"Generated {args.congestion_out}")
    print(f"Generated {args.tradeoff_out}")


if __name__ == "__main__":
    main()

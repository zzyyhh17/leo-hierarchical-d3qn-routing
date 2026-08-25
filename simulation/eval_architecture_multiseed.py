"""Run and aggregate the controlled MLP vs Context Per-Action experiment."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


_GOOD = Path(__file__).resolve().parent
_CONTROLLED = _GOOD / "eval_architecture_controlled.py"
_NETWORKS = ("mlp", "context_per_action")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def sample_summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.fmean(values), 6),
        "sample_std": round(statistics.stdev(values), 6)
        if len(values) > 1 else None,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def exact_mcnemar_p(context_only: int, mlp_only: int) -> float:
    discordant = context_only + mlp_only
    if discordant == 0:
        return 1.0
    lower = min(context_only, mlp_only)
    tail_count = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail_count / (2 ** discordant))


def _status(record: dict, mode: str) -> bool:
    if mode == "greedy":
        return record["greedy_hops"] is not None
    if mode == "greedy_beam":
        return (
            record["greedy_hops"] is not None
            or record["beam_hops"] is not None
        )
    raise ValueError(f"unsupported inference mode: {mode}")


def _metric_values(metrics: dict, pair_count: int) -> dict[str, float]:
    return {
        "greedy_success_pct": 100.0 * metrics["greedy_success"] / pair_count,
        "greedy_conditional_stretch": metrics["greedy_stretch"]["mean"],
        "beam_success_pct": 100.0 * metrics["beam_success"] / pair_count,
        "beam_conditional_stretch": metrics["beam_stretch"]["mean"],
        "greedy_beam_success_pct": (
            100.0 * metrics["greedy_beam_success"] / pair_count
        ),
        "greedy_beam_selected_stretch": metrics["greedy_beam_stretch"]["mean"],
        "unresolved_after_greedy_beam": (
            pair_count - metrics["greedy_beam_success"]
        ),
    }


def aggregate_run_files(run_files: list[Path]) -> dict:
    if not run_files:
        raise ValueError("no run files supplied")
    runs = []
    for path in run_files:
        with path.open(encoding="utf-8") as f:
            runs.append((path, json.load(f)))

    topology_hashes = {run["topology"]["sha256"] for _, run in runs}
    pair_hashes = {run["pairs_sha256"] for _, run in runs}
    validation_hashes = {run["validation_pairs_sha256"] for _, run in runs}
    if len(topology_hashes) != 1:
        raise ValueError("topology hashes differ across seeds")
    if len(pair_hashes) != 1:
        raise ValueError("test-pair hashes differ across seeds")
    if len(validation_hashes) != 1:
        raise ValueError("validation-pair hashes differ across seeds")

    seeds = [int(run["params"]["seed"]) for _, run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed results supplied")
    pair_counts = {int(run["params"]["pairs"]) for _, run in runs}
    if len(pair_counts) != 1:
        raise ValueError("test-pair counts differ across seeds")
    pair_count = pair_counts.pop()

    aggregate = {}
    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    for network in _NETWORKS:
        if any(network not in run["results"] for _, run in runs):
            raise ValueError(f"missing {network} result in one or more seeds")
        network_rows = []
        for _, run in runs:
            seed = int(run["params"]["seed"])
            values = _metric_values(run["results"][network], pair_count)
            per_seed_metrics.setdefault(seed, {})[network] = values
            network_rows.append(values)
        aggregate[network] = {
            key: sample_summary([row[key] for row in network_rows])
            for key in network_rows[0]
        }
        aggregate[network]["parameter_count"] = runs[0][1]["results"][network][
            "parameter_count"
        ]

    paired_by_seed = []
    for path, run in runs:
        seed = int(run["params"]["seed"])
        mlp_records = run["pair_results"]["mlp"]
        context_records = run["pair_results"]["context_per_action"]
        if len(mlp_records) != pair_count or len(context_records) != pair_count:
            raise ValueError(f"pair-result count mismatch for seed {seed}")

        modes = {}
        for mode in ("greedy", "greedy_beam"):
            context_only = 0
            mlp_only = 0
            for mlp_record, context_record in zip(mlp_records, context_records):
                mlp_pair = (mlp_record["src"], mlp_record["dst"])
                context_pair = (context_record["src"], context_record["dst"])
                if mlp_pair != context_pair:
                    raise ValueError(
                        f"MLP/Context test-pair order differs for seed {seed}"
                    )
                mlp_ok = _status(mlp_record, mode)
                context_ok = _status(context_record, mode)
                context_only += int(context_ok and not mlp_ok)
                mlp_only += int(mlp_ok and not context_ok)
            modes[mode] = {
                "context_success_mlp_failure": context_only,
                "mlp_success_context_failure": mlp_only,
                "exact_two_sided_p": exact_mcnemar_p(context_only, mlp_only),
            }

        mlp_values = per_seed_metrics[seed]["mlp"]
        context_values = per_seed_metrics[seed]["context_per_action"]
        paired_by_seed.append(
            {
                "seed": seed,
                "run_file": str(path),
                "context_minus_mlp": {
                    key: round(context_values[key] - mlp_values[key], 6)
                    for key in mlp_values
                },
                "mcnemar": modes,
            }
        )

    delta_keys = paired_by_seed[0]["context_minus_mlp"]
    paired_delta_summary = {
        key: sample_summary(
            [row["context_minus_mlp"][key] for row in paired_by_seed]
        )
        for key in delta_keys
    }

    first = runs[0][1]
    return {
        "protocol": {
            "seeds": seeds,
            "episodes": first["params"]["episodes"],
            "expert_demos": first["params"]["expert_demos"],
            "validation_seed": first["params"]["validation_seed"],
            "validation_pairs": first["params"]["eval_pairs"],
            "test_seed": first["params"]["test_seed"],
            "test_pairs": pair_count,
            "beam_width": first["params"]["beam_width"],
            "networks": list(_NETWORKS),
        },
        "topology": first["topology"],
        "validation_pairs_sha256": next(iter(validation_hashes)),
        "pairs_sha256": next(iter(pair_hashes)),
        "run_files": [str(path) for path, _ in runs],
        "aggregate": aggregate,
        "paired_comparison": {
            "per_seed": paired_by_seed,
            "context_minus_mlp_summary": paired_delta_summary,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44"))
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--test-seed", type=int, default=456)
    ap.add_argument("--validation-seed", type=int, default=123)
    ap.add_argument("--validation-pairs", type=int, default=200)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--expert-demos", type=int, default=2000)
    ap.add_argument("--expert-margin-weight", type=float, default=0.25)
    ap.add_argument("--expert-margin", type=float, default=0.5)
    ap.add_argument("--expert-batch-size", type=int, default=32)
    ap.add_argument("--checkpoint-prefix", default="architecture_multiseed")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=_GOOD / "results_architecture_multiseed_runs",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_GOOD / "results_architecture_multiseed.json",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_files = [args.run_dir / f"seed_{seed}.json" for seed in args.seeds]
    if not args.aggregate_only:
        for seed, run_file in zip(args.seeds, run_files):
            if args.resume and run_file.exists():
                print(f"\n=== Seed {seed}: reusing {run_file} ===", flush=True)
                continue
            command = [
                sys.executable,
                str(_CONTROLLED),
                "--episodes", str(args.episodes),
                "--seed", str(seed),
                "--test-seed", str(args.test_seed),
                "--validation-seed", str(args.validation_seed),
                "--pairs", str(args.pairs),
                "--beam-width", str(args.beam_width),
                "--networks", ",".join(_NETWORKS),
                "--checkpoint-prefix", f"{args.checkpoint_prefix}_seed{seed}",
                "--expert-demos", str(args.expert_demos),
                "--expert-margin-weight", str(args.expert_margin_weight),
                "--expert-margin", str(args.expert_margin),
                "--expert-batch-size", str(args.expert_batch_size),
                "--eval-pairs", str(args.validation_pairs),
                "--out", str(run_file),
            ]
            print(f"\n=== Seed {seed}: starting controlled run ===", flush=True)
            subprocess.run(command, check=True)

    missing = [str(path) for path in run_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run files: {missing}")
    output = aggregate_run_files(run_files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved aggregate results to {args.out}", flush=True)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

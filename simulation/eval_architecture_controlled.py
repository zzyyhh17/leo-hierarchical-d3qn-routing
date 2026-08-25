"""Controlled architecture comparison on the same 52-dimensional local state."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import torch

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

from dynamic_experiment import (
    CANONICAL_REF_TIME,
    flows_fingerprint,
    load_topology,
    topology_fingerprint,
)
from dqn import train_intra_d3qn, intra_dqn_route, intra_dqn_beam_route
from dqn.env import IntraSatEnv
from dqn.agents import IntraD3QNAgent
from route import _dijkstra_shortest_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_valid(path, adj, src, dst) -> bool:
    return bool(
        path
        and path[0] == src
        and path[-1] == dst
        and all(v in adj.get(u, set()) for u, v in zip(path, path[1:]))
    )


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def generate_pairs(n_nodes: int, count: int, seed: int) -> list[tuple[int, int]]:
    """Generate a deterministic source-destination list without touching training RNG."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < count:
        src, dst = rng.randrange(n_nodes), rng.randrange(n_nodes)
        if src != dst:
            pairs.append((src, dst))
    return pairs


def evaluate(agent, env, adj, pairs, beam_width: int) -> tuple[dict, list[dict]]:
    records = []
    for pair_index, (src, dst) in enumerate(pairs):
        dij = _dijkstra_shortest_path(adj, src, dst)
        if not path_valid(dij, adj, src, dst):
            raise RuntimeError(f"Dijkstra failed for pair {pair_index}")
        dij_hops = len(dij) - 1

        greedy = intra_dqn_route(env, agent, src, dst)
        greedy_hops = len(greedy) - 1 if path_valid(greedy, adj, src, dst) else None
        beam = intra_dqn_beam_route(env, agent, src, dst, beam_width=beam_width)
        beam_hops = len(beam) - 1 if path_valid(beam, adj, src, dst) else None

        rec = {
            "pair_index": pair_index,
            "src": src,
            "dst": dst,
            "dijkstra_hops": dij_hops,
            "greedy_hops": greedy_hops,
            "beam_hops": beam_hops,
            "greedy_stretch": round(greedy_hops / dij_hops, 6)
            if greedy_hops is not None else None,
            "beam_stretch": round(beam_hops / dij_hops, 6)
            if beam_hops is not None else None,
        }
        for key in ("greedy_stretch", "beam_stretch"):
            if rec[key] is not None and rec[key] < 1.0 - 1e-9:
                raise RuntimeError(f"Impossible stretch for pair {pair_index}: {rec}")
        records.append(rec)

    greedy_records = [r for r in records if r["greedy_hops"] is not None]
    beam_records = [r for r in records if r["beam_hops"] is not None]
    combined_records = [
        r for r in records
        if r["greedy_hops"] is not None or r["beam_hops"] is not None
    ]
    combined_stretches = []
    for r in combined_records:
        selected_hops = (
            r["greedy_hops"]
            if r["greedy_hops"] is not None
            else r["beam_hops"]
        )
        combined_stretches.append(selected_hops / r["dijkstra_hops"])
    result = {
        "greedy_success": len(greedy_records),
        "greedy_avg_hops": round(
            statistics.fmean(r["greedy_hops"] for r in greedy_records), 2
        ) if greedy_records else None,
        "greedy_dijkstra_avg_hops_same_pairs": round(
            statistics.fmean(r["dijkstra_hops"] for r in greedy_records), 2
        ) if greedy_records else None,
        "greedy_stretch": summarize([r["greedy_stretch"] for r in greedy_records]),
        "beam_success": len(beam_records),
        "beam_avg_hops": round(
            statistics.fmean(r["beam_hops"] for r in beam_records), 2
        ) if beam_records else None,
        "beam_dijkstra_avg_hops_same_pairs": round(
            statistics.fmean(r["dijkstra_hops"] for r in beam_records), 2
        ) if beam_records else None,
        "beam_stretch": summarize([r["beam_stretch"] for r in beam_records]),
        "greedy_beam_success": len(combined_records),
        "beam_rescue": sum(
            1 for r in records
            if r["greedy_hops"] is None and r["beam_hops"] is not None
        ),
        "greedy_beam_stretch": summarize(combined_stretches),
    }
    return result, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-seed", type=int, default=123)
    ap.add_argument("--validation-seed", type=int, default=123)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--beam-width", type=int, default=5)
    ap.add_argument("--ref-time", default=CANONICAL_REF_TIME)
    ap.add_argument(
        "--networks",
        default="mlp,per_action",
        help="逗号分隔: mlp,per_action,context_per_action",
    )
    ap.add_argument("--checkpoint-prefix", default="submission_improved")
    ap.add_argument("--expert-demos", type=int, default=2000)
    ap.add_argument("--expert-margin-weight", type=float, default=0.25)
    ap.add_argument("--expert-margin", type=float, default=0.5)
    ap.add_argument("--expert-batch-size", type=int, default=32)
    ap.add_argument("--eval-pairs", type=int, default=200,
                    help="训练期间用于选择检查点的固定验证对数量")
    ap.add_argument("--no-hard-example-replay", action="store_true")
    ap.add_argument("--reuse-checkpoints", action="store_true",
                    help="跳过训练，加载已生成的受控检查点并重新评估")
    ap.add_argument(
        "--out",
        type=Path,
        default=_GOOD / "results_architecture_improved.json",
    )
    args = ap.parse_args()

    sats, adj, edges, dprop, _ecef = load_topology("ideal", args.ref_time)
    pairs = generate_pairs(len(sats), args.pairs, args.test_seed)
    validation_pairs = generate_pairs(
        len(sats), args.eval_pairs, args.validation_seed
    )

    results = {}
    pair_results = {}
    networks = [name.strip() for name in args.networks.split(",") if name.strip()]
    unsupported = set(networks) - {
        "mlp", "per_action", "context_per_action"
    }
    if unsupported:
        raise ValueError(f"Unsupported controlled networks: {sorted(unsupported)}")
    for network in networks:
        print(f"\n=== Controlled training: {network} ===", flush=True)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        checkpoint_name = (
            f"{args.checkpoint_prefix}_{network}_local52_best.pt"
        )
        checkpoint_path = _GOOD / "model" / checkpoint_name
        if args.reuse_checkpoints:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            env = IntraSatEnv(
                adj,
                sats,
                max_hops=ckpt["max_hops"],
                max_neighbors=ckpt["max_neighbors"],
                use_bfs=False,
            )
            agent = IntraD3QNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden=256,
                network=network,
                base_dim=env.base_dim,
                nb_feat_dim=env.per_nb_dim,
                device="cpu",
                use_per=True,
            )
            agent.q_net.load_state_dict(ckpt["state_dict"])
            agent.q_net.eval()
        else:
            agent, env = train_intra_d3qn(
                adj,
                sats,
                n_episodes=args.episodes,
                max_hops=60,
                log_interval=1000,
                device="cpu",
                update_every=4,
                network=network,
                expert_demos=args.expert_demos,
                curriculum=True,
                her=True,
                use_bfs=False,
                use_per=True,
                expert_margin_weight=args.expert_margin_weight,
                expert_margin=args.expert_margin,
                expert_batch_size=args.expert_batch_size,
                hard_example_replay=not args.no_hard_example_replay,
                eval_pair_count=args.eval_pairs,
                eval_pairs=validation_pairs,
                checkpoint_name=checkpoint_name,
            )
            if checkpoint_path.exists():
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            else:
                checkpoint = {
                    "state_dict": {
                        k: v.detach().cpu()
                        for k, v in agent.q_net.state_dict().items()
                    }
                }
            checkpoint.update(
                {
                    "state_dim": env.state_dim,
                    "action_dim": env.action_dim,
                    "base_dim": env.base_dim,
                    "nb_feat_dim": env.per_nb_dim,
                    "max_hops": env.max_hops,
                    "max_neighbors": env.max_neighbors,
                    "network": network,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "ref_time": args.ref_time,
                    "expert_demos": args.expert_demos,
                    "expert_margin_weight": args.expert_margin_weight,
                    "expert_margin": args.expert_margin,
                    "expert_batch_size": args.expert_batch_size,
                    "hard_example_replay": not args.no_hard_example_replay,
                    "eval_pairs": args.eval_pairs,
                    "validation_seed": args.validation_seed,
                    "validation_pairs_sha256": flows_fingerprint(validation_pairs),
                }
            )
            torch.save(checkpoint, checkpoint_path)
        metrics, details = evaluate(agent, env, adj, pairs, args.beam_width)
        metrics.update(
            {
                "state_dim": env.state_dim,
                "base_dim": env.base_dim,
                "per_neighbor_dim": env.per_nb_dim,
                "parameter_count": sum(p.numel() for p in agent.q_net.parameters()),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
        results[network] = metrics
        pair_results[network] = details
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    output = {
        "params": vars(args) | {"out": str(args.out)},
        "topology": {
            "n_sats": len(sats),
            "n_links": len(edges),
            "sha256": topology_fingerprint(adj, dprop),
        },
        "validation_pairs_sha256": flows_fingerprint(validation_pairs),
        "pairs_sha256": flows_fingerprint(pairs),
        "results": results,
        "pair_results": pair_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()

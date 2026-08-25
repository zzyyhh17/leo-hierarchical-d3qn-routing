"""时变拓扑泛化性最小验证：在一个轨道周期内取多张拓扑快照，
用同一个训练好的局部特征 D3QN（不重训）逐快照评估，看成功率/跳比是否稳定。

回答：纯局部特征策略能否"天然迁移"到不同时刻的拓扑（无需重算/重训）。
"""
from __future__ import annotations
import sys, time, random, json, argparse, hashlib, statistics
from pathlib import Path
from datetime import datetime, timedelta, timezone

_GOOD = Path(__file__).resolve().parent
if str(_GOOD) not in sys.path:
    sys.path.insert(0, str(_GOOD))

import torch
from data_loader import load_gen11_at_time
from isl import build_isl_from_satellites, update_satellite_isl_peers
from route import _build_isl_adj, _dijkstra_shortest_path
from dqn.env import IntraSatEnv
from dqn.agents import IntraD3QNAgent
from dqn.inference import intra_dqn_route, intra_dqn_beam_route


CANONICAL_REF_TIME = "2026-03-13T09:50:07Z"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _topology_hash(adj: dict[int, set[int]]) -> str:
    edges = sorted((u, v) for u, nbs in adj.items() for v in nbs if u < v)
    return _sha256_json(edges)


def _valid_path(path, adj, src, dst) -> bool:
    return bool(
        path
        and path[0] == src
        and path[-1] == dst
        and all(v in adj.get(u, set()) for u, v in zip(path, path[1:]))
    )


def _stretch_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _plot_summary(rows: list[dict], pairs: int, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    x = [r["t_min"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    axes[0].plot(x, [r["greedy_succ"] / pairs * 100 for r in rows],
                 marker="o", label="Greedy")
    axes[0].plot(x, [r["greedy_beam_succ"] / pairs * 100 for r in rows],
                 marker="s", label="Greedy + Beam")
    axes[0].set_xlabel("Elapsed time (min)")
    axes[0].set_ylabel("Routing success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, [r["greedy_stretch"]["mean"] for r in rows],
                 marker="o", label="Greedy (successful pairs)")
    axes[1].plot(x, [r["greedy_beam_stretch"]["mean"] for r in rows],
                 marker="s", label="Greedy + Beam (successful pairs)")
    axes[1].plot(x, [r["cascade_stretch"]["mean"] for r in rows],
                 marker="^", linestyle="--", label="Full cascade")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle=":",
                    label="Dijkstra optimum")
    axes[1].set_xlabel("Elapsed time (min)")
    axes[1].set_ylabel("Per-pair path stretch")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=int, default=10)
    ap.add_argument("--step_min", type=float, default=10.0, help="快照间隔(分钟)")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--source", choices=("real", "ideal"), default="real",
                    help="real用于真实TLE时变拓扑，ideal用于1584星理想Walker拓扑")
    ap.add_argument("--start-time", default=CANONICAL_REF_TIME,
                    help="首张快照UTC时间，默认固定为理想星座TLE历元")
    ap.add_argument("--ckpt", default="model/d3qn_per_action_best.pt")
    ap.add_argument("--out", default="results_timevarying.json")
    ap.add_argument("--figure-out", default="fig7_timevarying.pdf")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    base = base.astimezone(timezone.utc)
    times = [base + timedelta(minutes=args.step_min * i) for i in range(args.snapshots)]

    agent = None
    rows = []
    pair_results = []
    # 固定的"逻辑端点对"（index=同一颗卫星）跨快照复用，隔离拓扑效应
    print(f"时变拓扑泛化评估：{args.snapshots} 张快照，间隔 {args.step_min} 分钟，每张 {args.pairs} 对", flush=True)
    print(f"固定首快照：{base.isoformat()}，随机种子：{args.seed}", flush=True)
    t0 = time.time()
    for i, t in enumerate(times):
        sats = load_gen11_at_time(t, source=args.source)
        build_isl_from_satellites(sats)
        update_satellite_isl_peers(sats)
        adj = _build_isl_adj(sats)
        n = len(sats)
        n_edges = sum(len(v) for v in adj.values()) // 2
        env = IntraSatEnv(adj, sats, max_hops=ckpt["max_hops"],
                          max_neighbors=ckpt["max_neighbors"], use_bfs=False)
        if agent is None:
            agent = IntraD3QNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                                   hidden=256, network=ckpt["network"],
                                   base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim, device="cpu")
            agent.q_net.load_state_dict(ckpt["state_dict"]); agent.q_net.eval()

        rng = random.Random(args.seed)            # 每张快照同一组 index 对
        pairs = []
        while len(pairs) < args.pairs:
            s, d = rng.randrange(n), rng.randrange(n)
            if s != d:
                pairs.append((s, d))

        dij_hops = []
        snapshot_pairs = []
        for pair_index, (s, d) in enumerate(pairs):
            p = _dijkstra_shortest_path(adj, s, d)
            if not _valid_path(p, adj, s, d):
                raise RuntimeError(f"快照{i} 路由对{pair_index}的Dijkstra路径无效")
            d_hops = len(p) - 1
            dij_hops.append(d_hops)

            pg = intra_dqn_route(env, agent, s, d)
            g_hops = len(pg) - 1 if _valid_path(pg, adj, s, d) else None
            pb = None
            if g_hops is None:
                pb = intra_dqn_beam_route(env, agent, s, d, beam_width=5)
            b_hops = len(pb) - 1 if _valid_path(pb, adj, s, d) else None

            selected_hops = g_hops if g_hops is not None else b_hops
            mode = "greedy" if g_hops is not None else ("beam" if b_hops is not None else "dijkstra")
            cascade_hops = selected_hops if selected_hops is not None else d_hops
            rec = {
                "snapshot": i,
                "pair_index": pair_index,
                "src": s,
                "dst": d,
                "dijkstra_hops": d_hops,
                "greedy_hops": g_hops,
                "beam_hops": b_hops,
                "selected_mode": mode,
                "selected_hops": selected_hops,
                "cascade_hops": cascade_hops,
                "greedy_stretch": round(g_hops / d_hops, 6) if g_hops is not None else None,
                "greedy_beam_stretch": round(selected_hops / d_hops, 6) if selected_hops is not None else None,
                "cascade_stretch": round(cascade_hops / d_hops, 6),
            }
            for key in ("greedy_stretch", "greedy_beam_stretch", "cascade_stretch"):
                if rec[key] is not None and rec[key] < 1.0 - 1e-9:
                    raise RuntimeError(
                        f"快照{i} 路由对{pair_index}出现不可能的路径伸长 {key}={rec[key]}"
                    )
            snapshot_pairs.append(rec)
            pair_results.append(rec)

        greedy_records = [r for r in snapshot_pairs if r["greedy_hops"] is not None]
        combined_records = [r for r in snapshot_pairs if r["selected_hops"] is not None]
        g_succ = len(greedy_records)
        gb_succ = len(combined_records)
        dij_avg = sum(dij_hops) / len(dij_hops) if dij_hops else 0
        row = {
            "snapshot": i,
            "timestamp_utc": t.astimezone(timezone.utc).isoformat(),
            "t_min": round(args.step_min * i, 1),
            "n_edges": n_edges,
            "topology_sha256": _topology_hash(adj),
            "dij_avg_hops": round(dij_avg, 2),
            "greedy_succ": g_succ,
            "greedy_avg_hops": round(
                statistics.fmean(r["greedy_hops"] for r in greedy_records), 2
            ) if greedy_records else None,
            "greedy_baseline_avg_hops_same_pairs": round(
                statistics.fmean(r["dijkstra_hops"] for r in greedy_records), 2
            ) if greedy_records else None,
            "greedy_stretch": _stretch_summary(
                [r["greedy_stretch"] for r in greedy_records]
            ),
            "greedy_beam_succ": gb_succ,
            "greedy_beam_avg_hops": round(
                statistics.fmean(r["selected_hops"] for r in combined_records), 2
            ) if combined_records else None,
            "greedy_beam_baseline_avg_hops_same_pairs": round(
                statistics.fmean(r["dijkstra_hops"] for r in combined_records), 2
            ) if combined_records else None,
            "greedy_beam_stretch": _stretch_summary(
                [r["greedy_beam_stretch"] for r in combined_records]
            ),
            "cascade_succ": len(snapshot_pairs),
            "cascade_stretch": _stretch_summary(
                [r["cascade_stretch"] for r in snapshot_pairs]
            ),
        }
        rows.append(row)
        print(f"  快照{i:2d} (+{row['t_min']:5.0f}min) 边={n_edges} Dij跳={dij_avg:.1f} | "
              f"D3QN贪心成功={g_succ}/{args.pairs} 伸长={row['greedy_stretch']['mean']:.3f} | "
              f"+Beam={gb_succ}/{args.pairs} 伸长={row['greedy_beam_stretch']['mean']:.3f} | "
              f"三级={row['cascade_stretch']['mean']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # 稳定性汇总
    gs = [r["greedy_succ"] for r in rows]
    gbs = [r["greedy_beam_succ"] for r in rows]
    gr = [r["greedy_stretch"]["mean"] for r in rows]
    gbr = [r["greedy_beam_stretch"]["mean"] for r in rows]
    cr = [r["cascade_stretch"]["mean"] for r in rows]
    print(f"\n稳定性汇总（同一模型、不重训、跨 {args.snapshots} 张快照）：", flush=True)
    print(f"  贪心成功率: {min(gs)}—{max(gs)}/{args.pairs}（均值 {sum(gs)/len(gs):.0f}，极差 {max(gs)-min(gs)}）", flush=True)
    print(f"  贪心+Beam成功率: {min(gbs)}—{max(gbs)}/{args.pairs}（均值 {sum(gbs)/len(gbs):.0f}）", flush=True)
    print(f"  贪心条件路径伸长: {min(gr):.3f}—{max(gr):.3f}", flush=True)
    print(f"  贪心+Beam条件路径伸长: {min(gbr):.3f}—{max(gbr):.3f}", flush=True)
    print(f"  完整三级路径伸长: {min(cr):.3f}—{max(cr):.3f}", flush=True)

    pair_list = [(r["src"], r["dst"]) for r in pair_results if r["snapshot"] == 0]
    output = {
        "params": vars(args),
        "provenance": {
            "start_time_utc": base.isoformat(),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "pairs_sha256": _sha256_json(pair_list),
            "metric_definition": (
                "Path stretch is computed per route against Dijkstra on the same "
                "source-destination pair and the same topology snapshot."
            ),
        },
        "summary": {
            "greedy_success_range": [min(gs), max(gs)],
            "greedy_beam_success_range": [min(gbs), max(gbs)],
            "greedy_stretch_mean_range": [round(min(gr), 4), round(max(gr), 4)],
            "greedy_beam_stretch_mean_range": [round(min(gbr), 4), round(max(gbr), 4)],
            "cascade_stretch_mean_range": [round(min(cr), 4), round(max(cr), 4)],
        },
        "rows": rows,
        "pair_results": pair_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    _plot_summary(rows, args.pairs, Path(args.figure_out))
    print(f"\n结果已保存: {args.out}", flush=True)
    print(f"图已保存: {args.figure_out}", flush=True)


if __name__ == "__main__":
    main()

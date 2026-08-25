"""校验并复现多随机种子架构对比实验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


_SIMULATION = Path(__file__).resolve().parent
if str(_SIMULATION) not in sys.path:
    sys.path.insert(0, str(_SIMULATION))

from dynamic_experiment import (  # noqa: E402
    CANONICAL_REF_TIME,
    flows_fingerprint,
    load_topology,
    topology_fingerprint,
)
from eval_architecture_controlled import generate_pairs  # noqa: E402


SEEDS = (42, 43, 44)
NETWORKS = ("mlp", "context_per_action")
RESULT_DIR = _SIMULATION / "results_architecture_multiseed_runs"
AGGREGATE_RESULT = _SIMULATION / "results_architecture_multiseed.json"
MODEL_DIR = _SIMULATION / "model"
CONTROLLED_SCRIPT = _SIMULATION / "eval_architecture_controlled.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected(seed: int) -> dict:
    path = RESULT_DIR / f"seed_{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_aggregate_result() -> dict:
    if not AGGREGATE_RESULT.is_file():
        raise FileNotFoundError(f"缺少汇总结果文件: {AGGREGATE_RESULT}")
    with AGGREGATE_RESULT.open(encoding="utf-8") as stream:
        return json.load(stream)


def checkpoint_path(expected: dict, network: str) -> Path:
    recorded = Path(expected["results"][network]["checkpoint"])
    return MODEL_DIR / recorded.name


def normalized_results(results: dict) -> dict:
    """忽略移动复现包后必然变化的 checkpoint 绝对目录。"""
    normalized = {}
    for network, metrics in results.items():
        row = dict(metrics)
        row["checkpoint"] = Path(row["checkpoint"]).name
        normalized[network] = row
    return normalized


def success_pct(success: int, pair_count: int) -> float:
    return 100.0 * success / pair_count


def summary_text(summary: dict, digits: int) -> str:
    mean = summary["mean"]
    std = summary["sample_std"]
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def print_baseline_results() -> None:
    """展示保存的逐随机种子结果及多随机种子汇总结果。"""
    print("\n=== 基准结果：逐随机种子 ===")
    print("成功率单位为 %；伸长为相对 Dijkstra 的路径伸长。")
    for seed in SEEDS:
        expected = load_expected(seed)
        pair_count = int(expected["params"]["pairs"])
        print(f"\nseed={seed}（测试对={pair_count}）")
        for network in NETWORKS:
            metrics = expected["results"][network]
            model_name = "MLP" if network == "mlp" else "Context Per-Action"
            unresolved = pair_count - metrics["greedy_beam_success"]
            print(
                f"  {model_name}: "
                f"Greedy={success_pct(metrics['greedy_success'], pair_count):.2f}%"
                f" / stretch {metrics['greedy_stretch']['mean']:.4f}; "
                f"Beam={success_pct(metrics['beam_success'], pair_count):.2f}%"
                f" / stretch {metrics['beam_stretch']['mean']:.4f}; "
                "Greedy+Beam="
                f"{success_pct(metrics['greedy_beam_success'], pair_count):.2f}%"
                f" / stretch {metrics['greedy_beam_stretch']['mean']:.4f}; "
                f"unresolved={unresolved}; "
                f"parameters={metrics['parameter_count']}"
            )

    aggregate = load_aggregate_result()
    print("\n=== 基准结果：3 个随机种子均值 ± 样本标准差 ===")
    for network in NETWORKS:
        metrics = aggregate["aggregate"][network]
        model_name = "MLP" if network == "mlp" else "Context Per-Action"
        print(f"\n{model_name}（parameters={metrics['parameter_count']}）")
        print(
            "  Greedy success: "
            f"{summary_text(metrics['greedy_success_pct'], 2)}%"
        )
        print(
            "  Greedy conditional stretch: "
            f"{summary_text(metrics['greedy_conditional_stretch'], 4)}"
        )
        print(
            "  Beam success: "
            f"{summary_text(metrics['beam_success_pct'], 2)}%"
        )
        print(
            "  Beam conditional stretch: "
            f"{summary_text(metrics['beam_conditional_stretch'], 4)}"
        )
        print(
            "  Greedy+Beam success: "
            f"{summary_text(metrics['greedy_beam_success_pct'], 2)}%"
        )
        print(
            "  Greedy+Beam selected stretch: "
            f"{summary_text(metrics['greedy_beam_selected_stretch'], 4)}"
        )
        print(
            "  Unresolved after Greedy+Beam: "
            f"{summary_text(metrics['unresolved_after_greedy_beam'], 2)}"
        )

    delta = aggregate["paired_comparison"]["context_minus_mlp_summary"]
    print("\n=== Context Per-Action 相对 MLP 的基准差值 ===")
    print(
        "  Greedy success: "
        f"{delta['greedy_success_pct']['mean']:+.2f} percentage points"
    )
    print(
        "  Greedy conditional stretch: "
        f"{delta['greedy_conditional_stretch']['mean']:+.4f}"
    )
    print(
        "  Beam success: "
        f"{delta['beam_success_pct']['mean']:+.2f} percentage points"
    )
    print(
        "  Beam conditional stretch: "
        f"{delta['beam_conditional_stretch']['mean']:+.4f}"
    )
    print(
        "  Greedy+Beam success: "
        f"{delta['greedy_beam_success_pct']['mean']:+.2f} percentage points"
    )
    print(
        "  Greedy+Beam selected stretch: "
        f"{delta['greedy_beam_selected_stretch']['mean']:+.4f}"
    )
    print(
        "  Unresolved after Greedy+Beam: "
        f"{delta['unresolved_after_greedy_beam']['mean']:+.2f} pairs"
    )


def verify_static_inputs() -> None:
    expected_runs = [load_expected(seed) for seed in SEEDS]
    reference = expected_runs[0]
    params = reference["params"]

    topology_hashes = {run["topology"]["sha256"] for run in expected_runs}
    pair_hashes = {run["pairs_sha256"] for run in expected_runs}
    validation_hashes = {
        run["validation_pairs_sha256"] for run in expected_runs
    }
    if len(topology_hashes) != 1 or len(pair_hashes) != 1:
        raise AssertionError("三个随机种子的拓扑或测试对不一致")
    if len(validation_hashes) != 1:
        raise AssertionError("三个随机种子的验证对不一致")

    satellites, adjacency, _edges, delays, _ecef = load_topology(
        "ideal", params.get("ref_time", CANONICAL_REF_TIME)
    )
    actual_topology_hash = topology_fingerprint(adjacency, delays)
    if actual_topology_hash != reference["topology"]["sha256"]:
        raise AssertionError(
            "拓扑哈希不一致: "
            f"{actual_topology_hash} != {reference['topology']['sha256']}"
        )

    test_pairs = generate_pairs(
        len(satellites), params["pairs"], params["test_seed"]
    )
    if flows_fingerprint(test_pairs) != reference["pairs_sha256"]:
        raise AssertionError("测试对哈希不一致")

    validation_pairs = generate_pairs(
        len(satellites), params["eval_pairs"], params["validation_seed"]
    )
    if (
        flows_fingerprint(validation_pairs)
        != reference["validation_pairs_sha256"]
    ):
        raise AssertionError("验证对哈希不一致")

    for seed, expected in zip(SEEDS, expected_runs):
        for network in NETWORKS:
            checkpoint = checkpoint_path(expected, network)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"缺少权重文件: {checkpoint}")
            actual_hash = sha256_file(checkpoint)
            expected_hash = expected["results"][network]["checkpoint_sha256"]
            if actual_hash != expected_hash:
                raise AssertionError(
                    f"seed={seed} network={network} 权重哈希不一致: "
                    f"{actual_hash} != {expected_hash}"
                )

    print(
        "静态校验通过: "
        f"{len(satellites)} 颗卫星，拓扑/测试对/验证对一致，6 个权重哈希一致"
    )


def verify_full_inference() -> None:
    with tempfile.TemporaryDirectory(prefix="leo-route-verify-") as temp_dir:
        output_dir = Path(temp_dir)
        for seed in SEEDS:
            expected = load_expected(seed)
            params = expected["params"]
            output = output_dir / f"seed_{seed}.json"
            command = [
                sys.executable,
                str(CONTROLLED_SCRIPT),
                "--episodes", str(params["episodes"]),
                "--seed", str(seed),
                "--test-seed", str(params["test_seed"]),
                "--validation-seed", str(params["validation_seed"]),
                "--pairs", str(params["pairs"]),
                "--beam-width", str(params["beam_width"]),
                "--networks", params["networks"],
                "--checkpoint-prefix", params["checkpoint_prefix"],
                "--expert-demos", str(params["expert_demos"]),
                "--expert-margin-weight",
                str(params["expert_margin_weight"]),
                "--expert-margin", str(params["expert_margin"]),
                "--expert-batch-size", str(params["expert_batch_size"]),
                "--eval-pairs", str(params["eval_pairs"]),
                "--reuse-checkpoints",
                "--out", str(output),
            ]
            if params.get("no_hard_example_replay"):
                command.append("--no-hard-example-replay")
            subprocess.run(command, check=True)

            with output.open(encoding="utf-8") as stream:
                observed = json.load(stream)
            for key in ("topology", "pairs_sha256", "validation_pairs_sha256"):
                if observed[key] != expected[key]:
                    raise AssertionError(f"seed={seed} 的 {key} 不一致")
            if normalized_results(observed["results"]) != normalized_results(
                expected["results"]
            ):
                raise AssertionError(f"seed={seed} 的 results 不一致")
            if observed["pair_results"] != expected["pair_results"]:
                raise AssertionError(f"seed={seed} 的 pair_results 不一致")
            print(f"seed={seed} 完整推理结果与基准记录精确一致")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="重新执行三个随机种子的完整模型推理并逐项比较",
    )
    args = parser.parse_args()
    verify_static_inputs()
    print_baseline_results()
    if args.full:
        verify_full_inference()


if __name__ == "__main__":
    main()

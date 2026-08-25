# leo-hierarchical-d3qn-routing

Simulation codebase for the paper:

**A Per-Action Structured D3QN-Based Hierarchical Routing Algorithm for LEO Mega-Constellation Networks**
(*Applied Sciences*, MDPI, revised manuscript)

## Overview

Hierarchical inter-satellite routing for LEO mega-constellations:

- evolutionary-greedy domain partitioning (`routing/`);
- domain-level Dijkstra backbone + intra-domain Context Per-Action D3QN (`simulation/dqn/`);
- three-level inference fault tolerance (greedy → beam search → Dijkstra);
- load-aware dual-weight routing and dynamic-traffic evaluation (`simulation/`);
- controlled architecture comparison (flat MLP vs. Context Per-Action), including the
  three-seed replication (`simulation/eval_architecture_multiseed.py`).

## Layout

| Path | Content |
|---|---|
| `starlink/` | Starlink Gen1-1 constellation / orbit model (SGP4, Walker-Delta) |
| `routing/` | domain partitioning, hierarchical routing, shortest paths, metrics (+ unit tests) |
| `simulation/` | ISL model, D3QN training (`dqn/`), evaluation scripts, dynamic-load and real-TLE experiments |
| `experiments/` | benchmark drivers |

Key entry points:

- `simulation/dqn/train.py` — D3QN training (expert demonstrations, curriculum, HER, PER)
- `simulation/eval_architecture_controlled.py` — controlled MLP vs. Context Per-Action comparison
- `simulation/eval_architecture_multiseed.py` — three-seed (42/43/44) replication of the comparison
- `simulation/eval_load_aware.py`, `simulation/dynamic_experiment.py` — congestion / hotspot / failure experiments
- `simulation/eval_timevarying.py`, `simulation/generate_real_tle_snapshots.py` — real-TLE zero-shot transfer
- `simulation/generate_submission_figures.py`, `simulation/generate_load_distribution_figure.py` — figure generation

## Requirements

Python 3, PyTorch, NumPy, Matplotlib (see imports in each module).

## Reproducibility

All results in the paper can be independently reproduced by retraining with the
published code, configurations, and seeds (approximately 4 minutes per training run
on a desktop CPU):

- training seeds: 42, 43, 44; validation seed: 123; test seed: 456
- topology SHA-256: `bd8e048178733796fbbe8561507bdbc38d795834f60322f081c2567fe82f4676`
- validation-pair SHA-256: `3374fed5f6626a28ff890e07b227abe27726404b88c48883020a4d9db289a9e0`
- test-pair SHA-256: `b1f01df7af89bbaa7c31a39495c6e57911f129ee2a8fd50c91cc1f8bb8d4b2e4`

Processed result files are included in `results/` (the three-seed controlled
architecture comparison, including per-pair outcomes for all 1,000 test pairs per
seed), and the fixed ideal-Walker topology snapshot used in the experiments is at
`starlink/data/`. `simulation/verify_architecture_multiseed.py` re-checks the
published numbers against these files.
Trained model checkpoints are available from the corresponding author upon
reasonable request, as stated in the paper.

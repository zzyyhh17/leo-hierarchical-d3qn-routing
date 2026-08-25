from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteMetrics:
    hops: int
    distance_km: float
    propagation_latency_ms: float
    path_stretch: float | None = None


def evaluate_path(path: list[int], weight, baseline_cost: float | None = None) -> RouteMetrics:
    if not path: return RouteMetrics(-1, float("inf"), float("inf"), None)
    distance = sum(float(weight(u,v)) for u,v in zip(path,path[1:]))
    stretch = distance / baseline_cost if baseline_cost and baseline_cost > 0 else None
    return RouteMetrics(len(path)-1,distance,distance/299792.458*1000,stretch)

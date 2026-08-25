from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


Weight = Callable[[int, int], float]


@dataclass(frozen=True)
class RouteRequest:
    source: int
    target: int
    objective: str = "latency"
    inter_algorithm: str = "dijkstra_hops"
    intra_algorithm: str = "dijkstra_weighted"
    allow_global_fallback: bool = True
    max_group_size: int = 50


@dataclass
class TopologySnapshot:
    adjacency: dict[int, list[int]]
    groups: dict[int, list[int]]
    node_group: dict[int, int]
    weight: Weight = field(default=lambda _u, _v: 1.0)
    snapshot_id: str = ""
    learned_intra_router: Callable[[int,int,set[int],str],list[int] | None] | None = None

    def validate(self) -> None:
        nodes = set(self.adjacency)
        if set(self.node_group) - nodes:
            raise ValueError("分组中包含拓扑不存在的节点")
        for node, neighbors in self.adjacency.items():
            if any(neighbor not in nodes for neighbor in neighbors):
                raise ValueError(f"节点 {node} 包含无效邻居")


@dataclass
class RouteResult:
    success: bool
    path: list[int]
    group_path: list[int]
    cost: float
    algorithm: str
    fallback: str | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def hops(self) -> int:
        return max(0, len(self.path) - 1) if self.success else -1

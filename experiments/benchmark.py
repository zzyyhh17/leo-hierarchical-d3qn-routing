from __future__ import annotations

from routing import HierarchicalRouter, RouteRequest, TopologySnapshot
from routing.metrics import evaluate_path
from routing.shortest_path import dijkstra


def compare_with_global_baseline(snapshot: TopologySnapshot, pairs: list[tuple[int,int]]) -> list[dict]:
    """在同一快照和权重函数下公平比较二层路由与全局最短路。"""
    router=HierarchicalRouter(); rows=[]
    for source,target in pairs:
        baseline_path,baseline_cost=dijkstra(snapshot.adjacency,source,target,snapshot.weight)
        routed=router.route(snapshot,RouteRequest(source,target,allow_global_fallback=False))
        metrics=evaluate_path(routed.path,snapshot.weight,baseline_cost)
        rows.append({"source":source,"target":target,"success":routed.success,"hops":metrics.hops,
                     "distance_km":metrics.distance_km,"path_stretch":metrics.path_stretch,
                     "baseline_hops":len(baseline_path)-1 if baseline_path else -1,"searched_nodes":routed.diagnostics.get("searched_nodes",0)})
    return rows

from __future__ import annotations

from .domain import RouteRequest, RouteResult, TopologySnapshot
from .shortest_path import bidirectional_bfs, dijkstra


class HierarchicalRouter:
    """先确定组路径，再在组路径诱导子图内求卫星路径。"""

    def route(self, snapshot: TopologySnapshot, request: RouteRequest) -> RouteResult:
        snapshot.validate()
        if request.source not in snapshot.node_group or request.target not in snapshot.node_group:
            return self._failed("端点未分组")
        source_group, target_group = snapshot.node_group[request.source], snapshot.node_group[request.target]
        group_adjacency,group_weights = self._group_graph(snapshot)
        if request.inter_algorithm == "dijkstra_weighted":
            group_path,_=dijkstra(group_adjacency,source_group,target_group,lambda a,b:group_weights[(min(a,b),max(a,b))])
        elif request.inter_algorithm == "greedy_best_first":
            group_path=self._greedy_group_path(group_adjacency,source_group,target_group)
        else:
            group_path,_=dijkstra(group_adjacency,source_group,target_group,lambda _a,_b:1.0)
        if not group_path: return self._fallback(snapshot, request, [], "组间路由失败")
        allowed = {node for group in group_path for node in snapshot.groups.get(group, [])}
        if request.intra_algorithm in {"per_action_d3qn","per_action_d3qn_beam"}:
            if snapshot.learned_intra_router is None: return self._failed("D3QN 组内路由器未配置",group_path)
            path=snapshot.learned_intra_router(request.source,request.target,allowed,request.intra_algorithm)
        elif request.intra_algorithm == "bfs_hops":
            path,_=dijkstra(snapshot.adjacency,request.source,request.target,lambda _u,_v:1.0,allowed)
        elif request.intra_algorithm == "bidirectional_bfs":
            path=bidirectional_bfs(snapshot.adjacency,request.source,request.target,allowed)
        else:
            path,_=dijkstra(snapshot.adjacency, request.source, request.target, snapshot.weight, allowed)
        cost=sum(snapshot.weight(u,v) for u,v in zip(path or [],(path or [])[1:])) if path else float("inf")
        fallback = None
        if not path and request.allow_global_fallback:
            path, cost = dijkstra(snapshot.adjacency, request.source, request.target, snapshot.weight)
            fallback = "global_dijkstra"
        if not path: return self._failed("组路径诱导子图不可达", group_path)
        return RouteResult(True, path, group_path, cost, "hierarchical_dijkstra", fallback,
                           {"searched_nodes": len(allowed), "total_nodes": len(snapshot.adjacency)})

    @staticmethod
    def _group_graph(snapshot: TopologySnapshot):
        graph = {group: set() for group in snapshot.groups}
        weights={}
        for node, neighbors in snapshot.adjacency.items():
            source_group = snapshot.node_group.get(node)
            for neighbor in neighbors:
                target_group = snapshot.node_group.get(neighbor)
                if source_group is not None and target_group is not None and source_group != target_group:
                    graph[source_group].add(target_group); graph[target_group].add(source_group)
                    key=(min(source_group,target_group),max(source_group,target_group));weights[key]=min(weights.get(key,float("inf")),snapshot.weight(node,neighbor))
        return {group: sorted(neighbors) for group, neighbors in graph.items()},weights

    @staticmethod
    def _greedy_group_path(graph,source,target):
        path=[source];visited={source};current=source
        while current!=target:
            candidates=[node for node in graph[current] if node not in visited]
            if not candidates:return None
            scored=[]
            for candidate in candidates:
                _,distance=dijkstra(graph,candidate,target,lambda _a,_b:1.0)
                scored.append((distance,candidate))
            distance,current=min(scored)
            if distance==float("inf"):return None
            visited.add(current);path.append(current)
        return path

    def _fallback(self, snapshot, request, group_path, reason):
        if request.allow_global_fallback:
            path, cost = dijkstra(snapshot.adjacency, request.source, request.target, snapshot.weight)
            if path: return RouteResult(True, path, group_path, cost, "hierarchical_dijkstra", "global_dijkstra", {"reason": reason})
        return self._failed(reason, group_path)

    @staticmethod
    def _failed(reason: str, group_path: list[int] | None = None) -> RouteResult:
        return RouteResult(False, [], group_path or [], float("inf"), "hierarchical_dijkstra", diagnostics={"reason": reason})

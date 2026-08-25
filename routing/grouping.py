from __future__ import annotations

from collections import deque


def connected_groups(adjacency: dict[int, list[int]], max_size: int = 50) -> tuple[dict[int, list[int]], dict[int, int]]:
    """按拓扑邻接关系生成确定性的连通分组。"""
    if max_size < 1: raise ValueError("max_size 必须大于 0")
    unassigned = set(adjacency); groups: dict[int, list[int]] = {}; node_group: dict[int, int] = {}
    while unassigned:
        seed = min(unassigned); queue = deque([seed]); members = []
        while queue and len(members) < max_size:
            node = queue.popleft()
            if node not in unassigned: continue
            unassigned.remove(node); members.append(node)
            queue.extend(sorted(neighbor for neighbor in adjacency[node] if neighbor in unassigned))
        group_id = len(groups); groups[group_id] = members
        for node in members: node_group[node] = group_id
    return groups, node_group

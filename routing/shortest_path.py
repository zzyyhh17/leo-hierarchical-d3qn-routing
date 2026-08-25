from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Iterable


def dijkstra(adjacency: dict[int, list[int]], source: int, target: int,
             weight: Callable[[int, int], float], allowed: Iterable[int] | None = None) -> tuple[list[int] | None, float]:
    permitted = set(allowed) if allowed is not None else None
    if source not in adjacency or target not in adjacency or (permitted is not None and (source not in permitted or target not in permitted)):
        return None, float("inf")
    distances = {source: 0.0}; previous: dict[int, int] = {}; queue = [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances.get(node): continue
        if node == target: break
        for neighbor in adjacency[node]:
            if permitted is not None and neighbor not in permitted: continue
            edge_cost = float(weight(node, neighbor))
            if edge_cost < 0: raise ValueError("Dijkstra 不支持负权链路")
            candidate = cost + edge_cost
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate; previous[neighbor] = node; heapq.heappush(queue, (candidate, neighbor))
    if target not in distances: return None, float("inf")
    path = [target]
    while path[-1] != source: path.append(previous[path[-1]])
    path.reverse()
    return path, distances[target]


def bidirectional_bfs(adjacency: dict[int,list[int]],source:int,target:int,allowed=None):
    permitted=set(allowed) if allowed is not None else set(adjacency)
    if source not in permitted or target not in permitted:return None
    if source==target:return [source]
    left={source:None};right={target:None};left_q=deque([source]);right_q=deque([target])
    meeting=None
    while left_q and right_q and meeting is None:
        queue,seen,other=(left_q,left,right) if len(left_q)<=len(right_q) else (right_q,right,left)
        for _ in range(len(queue)):
            node=queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in permitted or neighbor in seen:continue
                seen[neighbor]=node;queue.append(neighbor)
                if neighbor in other:meeting=neighbor;break
            if meeting is not None:break
    if meeting is None:return None
    a=[];node=meeting
    while node is not None:a.append(node);node=left[node]
    a.reverse();node=right[meeting]
    while node is not None:a.append(node);node=right[node]
    return a

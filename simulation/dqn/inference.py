"""推理函数: greedy 路由 / beam search 路由。"""
from __future__ import annotations

import torch

from .env import IntraSatEnv


def intra_dqn_route(env: IntraSatEnv, agent, src: int, dst: int) -> list[int] | None:
    """用训练好的 agent greedy 路由，返回路径或 None。"""
    state = env.reset(src, dst)
    mask = env.valid_action_mask()
    path = [src]
    done = False
    while not done:
        action = agent.greedy_action(state, mask)
        state, reward, done = env.step(action)
        mask = env.valid_action_mask()
        path.append(env.current)
    if env.current == dst:
        return path
    return None


def intra_dqn_beam_route(env: IntraSatEnv, agent, src: int, dst: int,
                         beam_width: int = 3,
                         adaptive: bool = True,
                         max_revisit: int = 1,
                         revisit_penalty: float = 5.0) -> list[int] | None:
    """Beam search 推理：同时维护 beam_width 条候选路径，取最短到达路径。

    每步对每条候选展开所有合法动作，按 Q 值排序保留 top-beam_width 条。
    adaptive=True 时，在 Q 值分散或搜索深度增加时自动扩大 beam 宽度。
    max_revisit: 每个节点最多允许回访次数 (0=硬禁止, 1=允许回访一次)。
    revisit_penalty: 回访已访问节点时对 score 施加的惩罚值。
    """
    env.reset(src, dst)

    init_snap = (env.current, env.dest, env.hops,
                 frozenset(env.visited), tuple(env._neighbors))
    init_vc: dict[int, int] = {src: 1}

    beams: list[tuple[float, list[int], tuple, dict[int, int]]] = [
        (0.0, [src], init_snap, init_vc)]
    best_path: list[int] | None = None
    base_width = beam_width

    for _step in range(env.max_hops):
        if not beams:
            break
        candidates: list[tuple[float, list[int], tuple, dict[int, int]]] = []

        q_spread_sum = 0.0
        q_spread_cnt = 0

        for score, path, snap, vc in beams:
            cur, dest, hops, visited_fs, neighbors = snap
            env.current = cur
            env.dest = dest
            env.hops = hops
            env.visited = set(visited_fs)
            env._neighbors = list(neighbors)

            state = env._make_state()
            mask = env.valid_action_mask()
            valid_actions = mask.nonzero(as_tuple=True)[0]
            if len(valid_actions) == 0:
                continue

            with torch.no_grad():
                q = agent.q_net(state.unsqueeze(0).to(agent.device)).squeeze(0)
                q[~mask.to(agent.device)] = -1e9

            valid_q = q[mask.to(agent.device)]
            if adaptive and len(valid_q) >= 2:
                q_range = (valid_q.max() - valid_q.min()).item()
                q_spread_sum += q_range
                q_spread_cnt += 1

            for act_t in valid_actions:
                act = act_t.item()
                q_val = q[act].item()
                nb = env._neighbors[act]
                if nb < 0:
                    continue

                new_hops = hops + 1
                new_path = path + [nb]

                if nb == dest:
                    if best_path is None or len(new_path) < len(best_path):
                        best_path = new_path
                    continue

                if new_hops >= env.max_hops:
                    continue

                nb_visits = vc.get(nb, 0)
                if nb_visits > max_revisit:
                    continue

                penalty = revisit_penalty * nb_visits if nb_visits > 0 else 0.0

                new_visited = visited_fs | frozenset([nb])
                new_vc = vc.copy()
                new_vc[nb] = nb_visits + 1
                new_nbs = tuple(env._get_neighbors(nb))
                new_snap = (nb, dest, new_hops, new_visited, new_nbs)
                candidates.append((score - q_val + penalty,
                                   new_path, new_snap, new_vc))

        if not candidates:
            break

        if adaptive:
            avg_spread = (q_spread_sum / q_spread_cnt) if q_spread_cnt > 0 else 1.0
            depth_factor = 1 + _step // 10
            uncertainty_factor = 1 if avg_spread > 1.0 else 2
            cur_width = min(base_width * depth_factor * uncertainty_factor,
                            base_width * 4)
        else:
            cur_width = base_width

        candidates.sort(key=lambda x: x[0])
        beams = candidates[:cur_width]

        if best_path is not None and len(best_path) <= len(beams[0][1]) + 1:
            break

    return best_path

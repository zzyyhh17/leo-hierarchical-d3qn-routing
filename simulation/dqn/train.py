"""训练函数: D3QN / DQN / Enhanced D3QN / 并行训练。"""
from __future__ import annotations

import math
import pickle
import random
import time
from collections import deque

import torch

from .env import IntraSatEnv
from .agents import IntraDQNAgent, IntraD3QNAgent, EnhancedD3QNAgent
from .buffer import NStepBuffer
from .networks import IntraQNetwork, IntraDuelingQNetwork, EnhancedDuelingQNetwork


def _build_hop_neighborhoods(adj, n_sats, max_hop=15):
    """BFS 预计算每个节点 max_hop 跳内的邻居集合，用于训练时采样近距对。"""
    from collections import deque as _deque
    neighborhoods: list[list[int]] = [[] for _ in range(n_sats)]
    for src in range(n_sats):
        visited = {src}
        queue = _deque([(src, 0)])
        nbs = []
        while queue:
            cur, d = queue.popleft()
            if d > 0:
                nbs.append(cur)
            if d < max_hop:
                for nb in adj.get(cur, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, d + 1))
        neighborhoods[src] = nbs
    return neighborhoods


# ---------------------------------------------------------------------------
#  专家示范 + 课程学习 + HER 辅助函数
# ---------------------------------------------------------------------------

def _bfs_path(adj, src, dst, max_d=60):
    """BFS 最短路径，返回节点列表或 None。"""
    if src == dst:
        return [src]
    prev = {src: None}
    queue = deque([(src, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= max_d:
            continue
        for nb in sorted(adj.get(node, set())):
            if nb not in prev:
                prev[nb] = node
                if nb == dst:
                    path = []
                    cur = dst
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                queue.append((nb, d + 1))
    return None


ExpertExample = tuple[torch.Tensor, int, torch.Tensor]


def _replay_path_into_buffer(
    env,
    buffer,
    path,
    expert_examples: list[ExpertExample] | None = None,
):
    """将一条路径沿 env 重放，把每步 transition 存入 buffer，返回存入条数。"""
    if len(path) < 2:
        return 0
    src, dst = path[0], path[-1]
    state = env.reset(src, dst)
    count = 0
    for step_i in range(len(path) - 1):
        next_node = path[step_i + 1]
        nbs = env._get_neighbors(env.current)
        action = -1
        for a, nb in enumerate(nbs):
            if nb == next_node:
                action = a
                break
        if action < 0:
            break
        action_mask = env.valid_action_mask()
        if expert_examples is not None:
            expert_examples.append(
                (state.detach().clone(), action, action_mask.detach().clone())
            )
        next_state, reward, done = env.step(action)
        next_mask = env.valid_action_mask()
        buffer.push(state, action, reward, next_state, done, next_mask)
        count += 1
        if done:
            break
        state = next_state
    return count


def _prefill_expert_demos(
    adj,
    env,
    buffer,
    n_demos=2000,
    expert_examples: list[ExpertExample] | None = None,
):
    """用 BFS 最短路径预填充 replay buffer（专家示范）。"""
    all_idx = list(range(env.n_sats))
    total = 0
    paths_ok = 0
    for _ in range(n_demos):
        src, dst = random.sample(all_idx, 2)
        path = _bfs_path(adj, src, dst, max_d=env.max_hops)
        if path is None or len(path) < 2:
            continue
        n = _replay_path_into_buffer(
            env, buffer, path, expert_examples=expert_examples
        )
        if n > 0:
            paths_ok += 1
            total += n
    return paths_ok, total


def _her_relabel(env, path_taken, buffer, n_goals=2):
    """Hindsight Experience Replay：将失败轨迹重新标记为到中间节点的成功路径。

    从 path_taken 中随机选 n_goals 个中间节点作为虚拟目标，
    将轨迹前缀重放进 buffer。
    """
    if len(path_taken) < 3:
        return 0
    count = 0
    candidates = list(range(2, len(path_taken)))
    random.shuffle(candidates)
    for goal_idx in candidates[:n_goals]:
        fake_dst = path_taken[goal_idx]
        sub_path = path_taken[:goal_idx + 1]
        n = _replay_path_into_buffer(env, buffer, sub_path)
        count += n
    return count


def _curriculum_sample(ep, n_episodes, all_indices,
                       near_nbs, near_sources, mid_nbs, mid_sources):
    """渐进式课程学习采样：快速过渡到长距随机对，确保充分覆盖。"""
    progress = ep / n_episodes

    if progress < 0.10:
        r = random.random()
        if r < 0.5 and near_sources:
            src = random.choice(near_sources)
            return src, random.choice(near_nbs[src])
        if r < 0.7 and mid_sources:
            src = random.choice(mid_sources)
            return src, random.choice(mid_nbs[src])
    elif progress < 0.25:
        r = random.random()
        if r < 0.2 and near_sources:
            src = random.choice(near_sources)
            return src, random.choice(near_nbs[src])
        if r < 0.4 and mid_sources:
            src = random.choice(mid_sources)
            return src, random.choice(mid_nbs[src])
    else:
        r = random.random()
        if r < 0.05 and near_sources:
            src = random.choice(near_sources)
            return src, random.choice(near_nbs[src])
        if r < 0.15 and mid_sources:
            src = random.choice(mid_sources)
            return src, random.choice(mid_nbs[src])

    return tuple(random.sample(all_indices, 2))


def _hard_example_retrain(adj, env, agent, hard_pairs, update_every,
                          max_rounds=3,
                          expert_examples: list[ExpertExample] | None = None):
    """对失败的困难样本进行专家纠正训练。

    用 BFS 找到正确路径，重放进 buffer 并集中更新网络。
    返回 (纠正成功数, 注入转移数, 梯度更新次数)。
    """
    corrected = 0
    injected = 0
    grad_steps = 0
    for src, dst in hard_pairs:
        path = _bfs_path(adj, src, dst, max_d=env.max_hops)
        if path is None or len(path) < 2:
            continue
        for _ in range(max_rounds):
            n = _replay_path_into_buffer(
                env,
                agent.buffer,
                path,
                expert_examples=expert_examples,
            )
            injected += n
        corrected += 1
    n_updates = min(corrected * 8, 200)
    for _ in range(n_updates):
        loss = agent.update()
        if loss is not None:
            grad_steps += 1
    return corrected, injected, grad_steps


def train_intra_d3qn(adj, sats, n_episodes=5000, max_hops=50, log_interval=1000,
                     device="cpu", update_every: int = 4, network: str = "mlp",
                     expert_demos: int = 2000, curriculum: bool = True,
                     her: bool = True, use_bfs: bool = True,
                     use_per: bool = False,
                     expert_margin_weight: float = 0.0,
                     expert_margin: float = 0.5,
                     expert_batch_size: int = 32,
                     hard_example_replay: bool = False,
                     eval_pair_count: int = 50,
                     eval_pairs: list[tuple[int, int]] | None = None,
                     checkpoint_name: str | None = None):
    """训练组内 D3QN（Dueling Double DQN），返回 (agent, env)。

    改进:
      - expert_demos: 预填充 BFS 最短路径专家示范条数 (0=禁用)
      - curriculum: 渐进式课程学习 (短距 → 中距 → 全距)
      - her: Hindsight Experience Replay (失败轨迹重标记)
      - use_bfs: 是否使用 BFS 距离表特征 (False 则仅用几何特征)
      - use_per: 是否使用优先经验回放 (PER)
      - expert_margin_weight: 专家动作大间隔排序损失权重 (0=禁用)
      - hard_example_replay: 定期将近期贪心失败样本的 BFS 路径回灌
      - eval_pairs: 显式固定的检查点评估对；提供时优先于 eval_pair_count
    """
    from pathlib import Path as _Path
    model_dir = _Path(__file__).resolve().parent.parent / "model"
    model_dir.mkdir(exist_ok=True)

    max_nb = max(len(nbs) for nbs in adj.values()) if adj else 4
    max_nb = min(max_nb, 8)
    env = IntraSatEnv(adj, sats, max_hops=max_hops, max_neighbors=max_nb)
    estimated_grad_steps = n_episodes * max_hops // update_every
    eps_decay = max(n_episodes * 3 // 4, 3000)

    lr = 5e-4
    tau = 0.003
    agent = IntraD3QNAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        hidden=256, epsilon_decay=eps_decay,
        device=device, network=network,
        lr_T_max=estimated_grad_steps,
        lr=lr, target_tau=tau,
        base_dim=env.base_dim, nb_feat_dim=env.per_nb_dim,
        use_per=use_per,
    )
    n_params = sum(p.numel() for p in agent.q_net.parameters())
    flags = []
    if expert_demos > 0:
        flags.append(f"expert={expert_demos}")
    if curriculum:
        flags.append("curriculum")
    if her:
        flags.append("HER")
    if use_per:
        flags.append("PER")
    if expert_margin_weight > 0.0:
        flags.append(f"expert-rank={expert_margin_weight:g}")
    if hard_example_replay:
        flags.append("hard-replay")
    flags_str = " [" + ", ".join(flags) + "]" if flags else ""
    print(f"  IntraD3QN init: sats={len(sats)} max_nb={max_nb} max_hops={max_hops} "
          f"state_dim={env.state_dim} action_dim={env.action_dim} "
          f"network={network} params={n_params:,} device={device} "
          f"update_every={update_every}{flags_str}", flush=True)

    all_indices = list(range(len(sats)))

    # 预填充专家示范
    expert_examples: list[ExpertExample] = []
    if expert_demos > 0:
        n_demo = expert_demos
        print(f"  预填充专家示范 (BFS 最短路径 × {n_demo})...", flush=True)
        paths_ok, n_trans = _prefill_expert_demos(adj, env, agent.buffer,
                                                   n_demos=n_demo,
                                                   expert_examples=expert_examples)
        print(f"  专家示范: {paths_ok} 条路径 → {n_trans} 条转移入 buffer", flush=True)

    # 预计算分层邻域
    if curriculum:
        print(f"  预计算分层邻域 (≤5 跳, ≤15 跳)...", flush=True)
        near_nbs = _build_hop_neighborhoods(adj, len(sats), max_hop=5)
        mid_nbs = _build_hop_neighborhoods(adj, len(sats), max_hop=15)
        near_sources = [i for i in all_indices if len(near_nbs[i]) >= 1]
        mid_sources = [i for i in all_indices if len(mid_nbs[i]) >= 1]
    else:
        print(f"  预计算短距邻域 (≤15 跳)...", flush=True)
        mid_nbs = _build_hop_neighborhoods(adj, len(sats), max_hop=15)
        near_nbs = mid_nbs
        near_sources = mid_sources = [i for i in all_indices if len(mid_nbs[i]) >= 1]

    # 固定评估集
    if eval_pairs is None:
        eval_pairs = [
            tuple(random.sample(all_indices, 2)) for _ in range(eval_pair_count)
        ]
    else:
        eval_pairs = [(int(src), int(dst)) for src, dst in eval_pairs]
        if not eval_pairs:
            raise ValueError("eval_pairs must not be empty")
        if any(
            src == dst
            or src < 0
            or dst < 0
            or src >= len(sats)
            or dst >= len(sats)
            for src, dst in eval_pairs
        ):
            raise ValueError("eval_pairs contains an invalid source-destination pair")
    eval_interval = max(n_episodes // 20, 200)

    ep_success = deque(maxlen=500)
    ep_rewards: deque[float] = deque(maxlen=500)
    ep_lengths: deque[int] = deque(maxlen=500)
    recent_losses: deque[float] = deque(maxlen=500)
    her_count = 0
    t_start = time.time()
    last_log_time = t_start
    global_step = 0
    hard_failures: deque[tuple[int, int]] = deque(maxlen=256)

    best_eval_success = -1
    best_eval_avg = float("inf")
    best_ep = 0
    best_state_dict = None

    for ep in range(1, n_episodes + 1):
        # 采样 src/dst
        if curriculum:
            src, dst = _curriculum_sample(
                ep, n_episodes, all_indices,
                near_nbs, near_sources, mid_nbs, mid_sources)
        else:
            if random.random() < 0.5 and mid_sources:
                src = random.choice(mid_sources)
                dst = random.choice(mid_nbs[src])
            else:
                src, dst = random.sample(all_indices, 2)

        state = env.reset(src, dst)
        mask = env.valid_action_mask()
        done = False
        ep_reward = 0.0
        ep_len = 0
        path_taken = [src]

        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.valid_action_mask()
            agent.buffer.push(state, action, reward, next_state, done, next_mask)
            path_taken.append(env.current)
            global_step += 1
            if global_step % update_every == 0:
                expert_batch = None
                if expert_margin_weight > 0.0 and expert_examples:
                    expert_batch = random.sample(
                        expert_examples,
                        min(expert_batch_size, len(expert_examples)),
                    )
                loss = agent.update(
                    expert_batch=expert_batch,
                    expert_margin=expert_margin,
                    expert_weight=expert_margin_weight,
                )
                if loss is not None:
                    recent_losses.append(loss)
            ep_reward += reward
            ep_len += 1
            state = next_state
            mask = next_mask

        reached = env.current == env.dest
        ep_success.append(1.0 if reached else 0.0)
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)

        # HER：失败轨迹重标记
        if her and not reached and len(path_taken) >= 3:
            her_count += _her_relabel(env, path_taken, agent.buffer, n_goals=2)
        if hard_example_replay and not reached:
            hard_failures.append((src, dst))

        # 定期在固定评估集上评估，选最优模型
        if ep >= 500 and ep % eval_interval == 0:
            if hard_example_replay and hard_failures:
                unique_hard = list(dict.fromkeys(hard_failures))
                corrected, injected, hard_updates = _hard_example_retrain(
                    adj,
                    env,
                    agent,
                    unique_hard,
                    update_every,
                    max_rounds=2,
                    expert_examples=expert_examples,
                )
                hard_failures.clear()
                print(
                    f"  困难样本纠正: pairs={corrected} "
                    f"transitions={injected} updates={hard_updates}",
                    flush=True,
                )
            eval_succ = 0
            eval_hops = []
            for e_src, e_dst in eval_pairs:
                st = env.reset(e_src, e_dst)
                mk = env.valid_action_mask()
                for _ in range(env.max_hops):
                    a = agent.greedy_action(st, mk)
                    st, _, d = env.step(a)
                    mk = env.valid_action_mask()
                    if d:
                        break
                if env.current == env.dest:
                    eval_succ += 1
                    eval_hops.append(env.hops)
            eval_sr = eval_succ / len(eval_pairs)
            eval_avg = sum(eval_hops) / len(eval_hops) if eval_hops else env.max_hops
            if (
                eval_succ > best_eval_success
                or (
                    eval_succ == best_eval_success
                    and eval_avg < best_eval_avg
                )
            ):
                best_eval_success = eval_succ
                best_eval_avg = eval_avg
                best_ep = ep
                best_state_dict = {
                    k: v.cpu().clone()
                    for k, v in agent.q_net.state_dict().items()
                }

        now = time.time()
        if ep <= 5 or ep % log_interval == 0 or (now - last_log_time >= 30):
            sr = sum(ep_success) / len(ep_success) * 100
            avg_r = sum(ep_rewards) / len(ep_rewards)
            avg_l = sum(ep_lengths) / len(ep_lengths)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            elapsed = now - t_start
            eps_per_s = ep / elapsed if elapsed > 0 else 0
            best_info = (
                f" best@ep{best_ep}"
                f"(greedy={best_eval_success}/{len(eval_pairs)},"
                f"hops={best_eval_avg:.1f})"
                if best_ep > 0 else ""
            )
            her_info = f" HER={her_count}" if her else ""
            print(f"  IntraD3QN ep={ep:>5d}/{n_episodes} success={sr:.0f}% "
                  f"ε={agent.epsilon:.3f} lr={agent.lr:.1e} loss={avg_loss:.4f} "
                  f"reward={avg_r:.2f} steps={avg_l:.1f} "
                  f"buf={len(agent.buffer)} {eps_per_s:.1f}ep/s time={elapsed:.0f}s"
                  f"{best_info}{her_info}", flush=True)
            last_log_time = now

    if best_state_dict is not None:
        agent.q_net.load_state_dict({k: v.to(agent.device) for k, v in best_state_dict.items()})
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        ckpt_path = model_dir / (
            checkpoint_name or f"d3qn_{network}_best.pt"
        )
        torch.save({
            "state_dict": best_state_dict,
            "best_ep": best_ep,
            "best_eval_success": best_eval_success,
            "best_eval_pairs": len(eval_pairs),
            "best_eval_avg_hops": best_eval_avg,
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "max_hops": env.max_hops,
            "max_neighbors": env.max_neighbors,
            "network": network,
        }, ckpt_path)
        print(
              f"  ★ 已加载最优模型 (ep={best_ep}, "
              f"greedy={best_eval_success}/{len(eval_pairs)}, "
              f"hops={best_eval_avg:.1f}) "
              f"并保存到 {ckpt_path}", flush=True)
    else:
        print(f"  ⚠ 未找到满足条件的最优模型, 使用最终模型", flush=True)

    return agent, env


def train_intra_dqn(adj, sats, n_episodes=5000, max_hops=50, log_interval=1000,
                    device="cpu", update_every: int = 4):
    """训练组内（卫星级）DQN，返回 (agent, env)。"""
    max_nb = max(len(nbs) for nbs in adj.values()) if adj else 4
    max_nb = min(max_nb, 8)
    env = IntraSatEnv(adj, sats, max_hops=max_hops, max_neighbors=max_nb)
    agent = IntraDQNAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        hidden=128, epsilon_decay=max(n_episodes // 2, 2000),
        device=device,
    )
    print(f"  IntraDQN init: sats={len(sats)} max_nb={max_nb} max_hops={max_hops} "
          f"state_dim={env.state_dim} action_dim={env.action_dim} "
          f"device={device} update_every={update_every}", flush=True)

    all_indices = list(range(len(sats)))
    ep_success = deque(maxlen=500)
    ep_rewards: deque[float] = deque(maxlen=500)
    ep_lengths: deque[int] = deque(maxlen=500)
    recent_losses: deque[float] = deque(maxlen=500)
    t_start = time.time()
    last_log_time = t_start
    global_step = 0

    for ep in range(1, n_episodes + 1):
        src, dst = random.sample(all_indices, 2)
        state = env.reset(src, dst)
        mask = env.valid_action_mask()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.valid_action_mask()
            agent.buffer.push(state, action, reward, next_state, done, next_mask)
            global_step += 1
            if global_step % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    recent_losses.append(loss)
            ep_reward += reward
            ep_len += 1
            state = next_state
            mask = next_mask

        reached = env.current == env.dest
        ep_success.append(1.0 if reached else 0.0)
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)

        now = time.time()
        if ep <= 5 or ep % log_interval == 0 or (now - last_log_time >= 30):
            sr = sum(ep_success) / len(ep_success) * 100
            avg_r = sum(ep_rewards) / len(ep_rewards)
            avg_l = sum(ep_lengths) / len(ep_lengths)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            elapsed = now - t_start
            eps_per_s = ep / elapsed if elapsed > 0 else 0
            print(f"  IntraDQN ep={ep:>5d}/{n_episodes} success={sr:.0f}% "
                  f"ε={agent.epsilon:.3f} loss={avg_loss:.4f} "
                  f"reward={avg_r:.2f} steps={avg_l:.1f} "
                  f"buf={len(agent.buffer)} {eps_per_s:.1f}ep/s time={elapsed:.0f}s", flush=True)
            last_log_time = now

    return agent, env


def train_intra_enhanced_d3qn(adj, sats, n_episodes=5000, max_hops=50,
                               log_interval=1000, hidden=256, n_steps=3,
                               device="cpu"):
    """训练增强版组内 D3QN (NoisyNet + PER + N-step)，返回 (agent, env)。"""
    max_nb = max(len(nbs) for nbs in adj.values()) if adj else 4
    max_nb = min(max_nb, 8)
    env = IntraSatEnv(adj, sats, max_hops=max_hops, max_neighbors=max_nb)
    agent = EnhancedD3QNAgent(
        state_dim=env.state_dim, action_dim=env.action_dim,
        hidden=hidden, n_steps=n_steps, device=device,
    )
    print(f"  EnhancedD3QN init: sats={len(sats)} max_nb={max_nb} max_hops={max_hops} "
          f"state_dim={env.state_dim} action_dim={env.action_dim} "
          f"hidden={hidden} n_steps={n_steps} device={device}", flush=True)

    all_indices = list(range(len(sats)))
    ep_success = deque(maxlen=500)
    ep_rewards: deque[float] = deque(maxlen=500)
    ep_lengths: deque[int] = deque(maxlen=500)
    recent_losses: deque[float] = deque(maxlen=500)
    t_start = time.time()
    last_log_time = t_start

    for ep in range(1, n_episodes + 1):
        src, dst = random.sample(all_indices, 2)
        state = env.reset(src, dst)
        mask = env.valid_action_mask()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.valid_action_mask()
            agent.store(state, action, reward, next_state, done, next_mask)
            loss = agent.update()
            if loss is not None:
                recent_losses.append(loss)
            ep_reward += reward
            ep_len += 1
            state = next_state
            mask = next_mask

        reached = env.current == env.dest
        ep_success.append(1.0 if reached else 0.0)
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)

        now = time.time()
        if ep <= 5 or ep % log_interval == 0 or (now - last_log_time >= 30):
            sr = sum(ep_success) / len(ep_success) * 100
            avg_r = sum(ep_rewards) / len(ep_rewards)
            avg_l = sum(ep_lengths) / len(ep_lengths)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            elapsed = now - t_start
            eps_per_s = ep / elapsed if elapsed > 0 else 0
            print(f"  EnhancedD3QN ep={ep:>5d}/{n_episodes} success={sr:.0f}% "
                  f"loss={avg_loss:.4f} reward={avg_r:.2f} steps={avg_l:.1f} "
                  f"buf={len(agent.buffer)} train_steps={agent.steps} "
                  f"{eps_per_s:.1f}ep/s time={elapsed:.0f}s", flush=True)
            last_log_time = now

    return agent, env

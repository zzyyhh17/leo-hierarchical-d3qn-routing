"""并行训练: 多进程收集经验 + 主进程梯度更新。"""
from __future__ import annotations

import math
import multiprocessing as _mp
import pickle
import random
from collections import deque

import torch

from .env import IntraSatEnv
from .agents import IntraDQNAgent, IntraD3QNAgent, EnhancedD3QNAgent
from .networks import IntraQNetwork, IntraDuelingQNetwork, EnhancedDuelingQNetwork


def _collect_episodes_worker(args):
    """Worker 进程: 用冻结策略跑 n_eps 个 episode, 返回按 episode 分组的 transitions.

    所有 tensor 转为 numpy 返回, 避免 PyTorch 共享内存耗尽.
    """
    (algo, q_net_sd, epsilon, adj, sats_data,
     n_eps, max_hops, max_neighbors, state_dim, action_dim, hidden, seed) = args

    sats = pickle.loads(sats_data)
    env = IntraSatEnv(adj, sats, max_hops=max_hops, max_neighbors=max_neighbors)

    if algo == "dqn":
        q_net = IntraQNetwork(state_dim, action_dim, hidden)
    elif algo == "d3qn":
        q_net = IntraDuelingQNetwork(state_dim, action_dim, hidden)
    else:
        q_net = EnhancedDuelingQNetwork(state_dim, action_dim, hidden)
    q_net.load_state_dict(q_net_sd)

    use_noisy = (algo == "enhanced_d3qn")
    if use_noisy:
        q_net.train()
    else:
        q_net.eval()

    random.seed(seed)
    all_indices = list(range(len(sats)))
    episodes = []
    successes = []

    for _ in range(n_eps):
        if use_noisy:
            q_net.reset_noise()
        src, dst = random.sample(all_indices, 2)
        state = env.reset(src, dst)
        mask = env.valid_action_mask()
        done = False
        ep_trans = []
        while not done:
            if not use_noisy and random.random() < epsilon:
                valid = mask.nonzero(as_tuple=True)[0]
                action = valid[random.randint(0, len(valid) - 1)].item() if len(valid) > 0 else 0
            else:
                with torch.no_grad():
                    q = q_net(state.unsqueeze(0)).squeeze(0)
                    q[~mask] = -1e9
                    action = q.argmax().item()
            next_state, reward, done = env.step(action)
            next_mask = env.valid_action_mask()
            ep_trans.append((
                state.numpy(), action, reward,
                next_state.numpy(), done, next_mask.numpy()
            ))
            state = next_state
            mask = next_mask
        episodes.append(ep_trans)
        successes.append(1.0 if env.current == env.dest else 0.0)

    return episodes, successes


def train_intra_parallel(algo: str, adj, sats, n_episodes=5000, max_hops=50,
                          log_interval=1000, hidden=128, n_steps=3,
                          n_collectors=16, collect_eps=50, device="cpu"):
    """并行训练: 多进程收集经验 + 主进程梯度更新.

    架构:
      每轮: n_collectors 个 worker 各跑 collect_eps 个 episode → 合并经验 → 主进程做梯度更新
    """
    max_nb = max(len(nbs) for nbs in adj.values()) if adj else 4
    max_nb = min(max_nb, 8)
    env = IntraSatEnv(adj, sats, max_hops=max_hops, max_neighbors=max_nb)

    actual_hidden = 256 if algo == "enhanced_d3qn" else hidden
    if algo == "dqn":
        agent = IntraDQNAgent(
            state_dim=env.state_dim, action_dim=env.action_dim,
            hidden=actual_hidden, epsilon_decay=max(n_episodes // 2, 2000), device=device)
    elif algo == "d3qn":
        agent = IntraD3QNAgent(
            state_dim=env.state_dim, action_dim=env.action_dim,
            hidden=actual_hidden, epsilon_decay=max(n_episodes // 2, 2000), device=device)
    elif algo == "enhanced_d3qn":
        agent = EnhancedD3QNAgent(
            state_dim=env.state_dim, action_dim=env.action_dim,
            hidden=actual_hidden, n_steps=n_steps, device=device)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    sats_data = pickle.dumps(sats)
    ep_success = deque(maxlen=500)
    total_eps = 0
    eps_per_round = n_collectors * collect_eps
    n_rounds = math.ceil(n_episodes / eps_per_round)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_eps = max(n_episodes // 20, 100)

    def _calc_epsilon(ep_count: int) -> float:
        return epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-ep_count / epsilon_decay_eps)

    algo_names = {"dqn": "IntraDQN", "d3qn": "IntraD3QN", "enhanced_d3qn": "EnhancedD3QN"}
    name = algo_names[algo]

    pool = _mp.Pool(processes=n_collectors)

    try:
        for rnd in range(n_rounds):
            remaining = n_episodes - total_eps
            if remaining <= 0:
                break
            this_collect = min(collect_eps, math.ceil(remaining / n_collectors))

            q_net_sd = {k: v.cpu() for k, v in agent.q_net.state_dict().items()}
            epsilon = _calc_epsilon(total_eps) if algo != "enhanced_d3qn" else 0.0

            worker_args = [
                (algo, q_net_sd, epsilon, adj, sats_data, this_collect,
                 max_hops, max_nb, env.state_dim, env.action_dim, actual_hidden,
                 rnd * n_collectors + i)
                for i in range(n_collectors)
            ]

            results = pool.map(_collect_episodes_worker, worker_args)

            total_new_transitions = 0
            for episodes_batch, successes in results:
                for ep_trans in episodes_batch:
                    for s_np, a, r, ns_np, d, nm_np in ep_trans:
                        s = torch.from_numpy(s_np)
                        ns = torch.from_numpy(ns_np)
                        nm = torch.from_numpy(nm_np)
                        if algo == "enhanced_d3qn":
                            agent.store(s, a, r, ns, d, nm)
                        else:
                            agent.buffer.push(s, a, r, ns, d, nm)
                    total_new_transitions += len(ep_trans)
                ep_success.extend(successes)
                total_eps += len(successes)

            n_updates = max(1, total_new_transitions // agent.batch_size)
            for _ in range(n_updates):
                agent.update()

            cur_eps = _calc_epsilon(total_eps)
            if total_eps >= (total_eps // log_interval) * log_interval or rnd == n_rounds - 1:
                sr = sum(ep_success) / len(ep_success) * 100 if ep_success else 0
                buf_len = len(agent.buffer)
                if algo == "enhanced_d3qn":
                    print(f"  {name} ep={total_eps:>5d} success={sr:.0f}% "
                          f"buf={buf_len} steps={agent.steps}")
                else:
                    print(f"  {name} ep={total_eps:>5d} success={sr:.0f}% "
                          f"ε={cur_eps:.3f} buf={buf_len}")
    finally:
        pool.close()
        pool.join()

    return agent, env

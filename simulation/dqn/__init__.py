"""组内 DQN 路由模块。

子模块:
  env       - IntraSatEnv 卫星级路由环境
  buffer    - 经验回放缓冲区 (均匀 / PER / N-step)
  networks  - Q 网络架构 (MLP / Dueling / Transformer / Enhanced)
  agents    - Agent 实现 (DQN / D3QN / Enhanced D3QN)
  train     - 训练函数
  inference - 推理函数 (greedy / beam search)
"""

from .env import IntraSatEnv
from .buffer import IntraReplayBuffer, PrioritizedReplayBuffer, NStepBuffer, SumTree
from .networks import (
    IntraQNetwork, IntraDuelingQNetwork,
    PerActionDuelingQNetwork, ContextPerActionDuelingQNetwork,
    IntraTransformerDuelingQNetwork,
    NoisyLinear, EnhancedDuelingQNetwork,
)
from .agents import IntraDQNAgent, IntraD3QNAgent, EnhancedD3QNAgent
from .train import (
    train_intra_d3qn, train_intra_dqn, train_intra_enhanced_d3qn,
)
from .inference import intra_dqn_route, intra_dqn_beam_route
from .parallel import train_intra_parallel

__all__ = [
    "IntraSatEnv",
    "IntraReplayBuffer", "PrioritizedReplayBuffer", "NStepBuffer", "SumTree",
    "IntraQNetwork", "IntraDuelingQNetwork",
    "PerActionDuelingQNetwork", "ContextPerActionDuelingQNetwork",
    "IntraTransformerDuelingQNetwork",
    "NoisyLinear", "EnhancedDuelingQNetwork",
    "IntraDQNAgent", "IntraD3QNAgent", "EnhancedD3QNAgent",
    "train_intra_d3qn", "train_intra_dqn", "train_intra_enhanced_d3qn",
    "train_intra_parallel",
    "intra_dqn_route", "intra_dqn_beam_route",
]

"""Actor-critic heads for the two PPO uses in the benchmark.

StateActorCritic — flat-state expert (G1 demo generation): standard CleanRL
MLP actor-critic over the env's native action space, dense reward.

VisualActorCritic — sparse-reward fine-tuning of a trained BCPolicy: reuses
the BC trunk as the shared feature extractor, adds a value head and a
state-independent log_std. Masked canonical dims are excluded from log-prob
and entropy so exploration never pushes on joints the embodiment lacks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from xembench.constants import A_CANON
from xembench.models.bc_policy import BCPolicy


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class StateActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)

    def act(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.actor_mean(x) if deterministic else self.get_action_and_value(x)[0]


class VisualActorCritic(nn.Module):
    def __init__(self, bc: BCPolicy, init_logstd: float = -2.0):
        super().__init__()
        self.bc = bc
        self.value_head = layer_init(nn.Linear(bc.hidden_dim, 1), std=1.0)
        self.actor_logstd = nn.Parameter(torch.full((1, A_CANON), init_logstd))

    @classmethod
    def from_bc_checkpoint(cls, ckpt_path: str, vocab_size: int, **bc_kwargs):
        bc = BCPolicy(vocab_size, **bc_kwargs)
        bc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        return cls(bc)

    def get_value(self, obs: dict) -> torch.Tensor:
        return self.value_head(self.bc.encode(obs))

    def get_action_and_value(
        self, obs: dict, action_mask: torch.Tensor, action: torch.Tensor | None = None
    ):
        feat = self.bc.encode(obs)
        mean = torch.tanh(self.bc.head(feat))
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = (dist.sample() * action_mask).clamp(-1, 1)
        # masked dims contribute neither log-prob nor entropy
        logprob = (dist.log_prob(action) * action_mask).sum(-1)
        entropy = (dist.entropy() * action_mask).sum(-1)
        return action, logprob, entropy, self.value_head(feat)

    def act(self, obs: dict, action_mask: torch.Tensor, deterministic: bool = True):
        if deterministic:
            return torch.tanh(self.bc.head(self.bc.encode(obs))) * action_mask
        return self.get_action_and_value(obs, action_mask)[0]

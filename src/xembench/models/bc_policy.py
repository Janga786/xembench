"""BCPolicy: (image, proprio, instruction, embodiment) -> canonical action.

One network for every task and embodiment; the canonical action/proprio
layout plus the embodiment one-hot are what make cross-embodiment transfer a
pure adapter swap at evaluation time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from xembench.constants import A_CANON, NUM_EMBODIMENTS
from xembench.models.encoders import (
    VISION_ENCODERS,
    InstructionEncoder,
    ProprioEncoder,
)


class BCPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vision: str = "nature_cnn",
        d_vis: int = 256,
        d_lang: int = 128,
        d_prop: int = 128,
        hidden: int = 512,
    ):
        super().__init__()
        self.vision = VISION_ENCODERS[vision](d_vis)
        self.lang = InstructionEncoder(vocab_size, out_dim=d_lang)
        self.prop = ProprioEncoder(d_prop)
        fused = d_vis + d_lang + d_prop + NUM_EMBODIMENTS
        self.trunk = nn.Sequential(
            nn.Linear(fused, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, A_CANON)
        self.hidden_dim = hidden

    def encode(self, obs: dict) -> torch.Tensor:
        """-> fused features [B, hidden] (shared with the PPO ActorCritic)."""
        parts = [
            self.vision(obs["rgb"]),
            self.lang(obs["tokens"], obs["token_len"]),
            self.prop(obs["proprio"], obs["proprio_mask"]),
            F.one_hot(obs["embodiment_id"], NUM_EMBODIMENTS).float(),
        ]
        return self.trunk(torch.cat(parts, dim=-1))

    def forward(self, obs: dict) -> torch.Tensor:
        """-> canonical action [B, A_CANON] in [-1, 1]."""
        return torch.tanh(self.head(self.encode(obs)))


def masked_bc_loss(
    pred: torch.Tensor, target: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """SmoothL1 restricted to the embodiment's active canonical dims."""
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
    return (loss * action_mask).sum() / action_mask.sum().clamp(min=1)

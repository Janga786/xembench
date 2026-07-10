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
        chunk: int = 1,
    ):
        """chunk > 1 predicts the next `chunk` actions from one observation
        (action chunking, Zhao et al. 2023): open-loop execution within a
        chunk stops per-step imitation error from compounding — the failure
        mode that zeroed the precision tasks at chunk=1. chunk=1 is exactly
        the original single-step policy (checkpoint compatible)."""
        super().__init__()
        self.vision = VISION_ENCODERS[vision](d_vis)
        self.lang = InstructionEncoder(vocab_size, out_dim=d_lang)
        self.prop = ProprioEncoder(d_prop)
        fused = d_vis + d_lang + d_prop + NUM_EMBODIMENTS
        self.trunk = nn.Sequential(
            nn.Linear(fused, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.chunk = chunk
        self.head = nn.Linear(hidden, chunk * A_CANON)
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
        """-> canonical action(s) in [-1, 1]: [B, A_CANON] when chunk == 1,
        [B, chunk, A_CANON] otherwise."""
        out = torch.tanh(self.head(self.encode(obs)))
        if self.chunk > 1:
            return out.view(-1, self.chunk, A_CANON)
        return out


def masked_bc_loss(
    pred: torch.Tensor, target: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """SmoothL1 restricted to the embodiment's active canonical dims.
    Accepts [B, A] (single-step) or [B, chunk, A] (chunked) pred/target."""
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05)
    if loss.dim() == 3:
        action_mask = action_mask.unsqueeze(1)
    mask = action_mask.expand_as(loss)
    return (loss * mask).sum() / mask.sum().clamp(min=1)

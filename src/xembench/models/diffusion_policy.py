"""Diffusion policy — stretch goal (B6), interface stub.

Design (kept here so the trainer/eval code can grow a --policy diffusion
switch without touching anything else):

- Conditioning: the SAME encoders as BCPolicy (vision + instruction +
  proprio + embodiment one-hot), fused to a context vector.
- Denoiser: small 1D temporal U-Net over action chunks [H=8, A_CANON],
  FiLM-conditioned on the context vector.
- Training: DDPM, 100 steps, epsilon-prediction, masked MSE (action_mask).
- Inference: DDIM, 10 steps; execute the first k=4 actions of the chunk
  (receding horizon).
- Data: DemoDataset gains a chunked sampling mode returning [H, A_CANON]
  windows with zero-padding + a validity mask at episode ends.
"""


class DiffusionPolicy:  # pragma: no cover - placeholder
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Diffusion policy is the B6 stretch goal; see module docstring."
        )

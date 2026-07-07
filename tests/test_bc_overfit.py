"""BC learning sanity: loss must drop sharply on a trivially learnable
synthetic dataset (constant action per instruction). CPU, ~20s."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from xembench.constants import A_CANON, IMG_SIZE, P_CANON
from xembench.data.dataset import DemoDataset
from xembench.data.writer import DemoWriter
from xembench.language.tokenizer import get_tokenizer
from xembench.models.bc_policy import BCPolicy, masked_bc_loss
from xembench.utils.torch_utils import to_device


def test_bc_overfits_constant_actions(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    path = tmp_path / "demo.h5"
    target = rng.uniform(-0.8, 0.8, A_CANON).astype(np.float32)
    with DemoWriter(path, task="t", env_id="PickCubeLang-v1", embodiment="panda",
                    camera="base_camera", control_mode="pd_joint_delta_pos",
                    native_action_dim=8) as w:
        for ep in range(2):
            T = 8
            w.add_episode(
                rgb=rng.integers(0, 255, (T, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                proprio=rng.normal(size=(T, P_CANON)).astype(np.float32),
                action_canon=np.tile(target, (T, 1)),
                action_native=np.zeros((T, 8), dtype=np.float32),
                instruction="pick up the red cube and place it on the green target",
                success=True, seed=ep,
            )

    ds = DemoDataset([path], instruction_mode="canonical", preload=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = BCPolicy(vocab_size=len(get_tokenizer()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    it = iter(dl)
    for step in range(60):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = to_device(batch, torch.device("cpu"))
        loss = masked_bc_loss(model(batch), batch["action"], batch["action_mask"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first, last = np.mean(losses[:5]), np.mean(losses[-5:])
    assert last < 0.35 * first, f"BC failed to fit constants: {first:.4f} -> {last:.4f}"

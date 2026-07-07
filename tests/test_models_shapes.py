"""Model forward/backward shapes on fake batches (no simulator, CPU)."""

import torch

from xembench.constants import A_CANON, IMG_SIZE, L_MAX, P_CANON
from xembench.language.tokenizer import get_tokenizer
from xembench.models.ac_policy import StateActorCritic, VisualActorCritic
from xembench.models.bc_policy import BCPolicy, masked_bc_loss


def fake_batch(B=4):
    return dict(
        rgb=torch.randint(0, 255, (B, 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8),
        proprio=torch.randn(B, P_CANON),
        proprio_mask=torch.ones(B, P_CANON),
        tokens=torch.randint(1, 40, (B, L_MAX)),
        token_len=torch.randint(3, L_MAX, (B,)),
        embodiment_id=torch.randint(0, 2, (B,)),
    )


def test_bc_forward_and_grad():
    model = BCPolicy(vocab_size=len(get_tokenizer()))
    batch = fake_batch()
    out = model(batch)
    assert out.shape == (4, A_CANON)
    assert out.abs().max() <= 1.0  # tanh head
    mask = torch.zeros(4, A_CANON)
    mask[:, :8] = 1
    loss = masked_bc_loss(out, torch.rand(4, A_CANON), mask)
    loss.backward()
    assert model.head.weight.grad is not None


def test_masked_loss_ignores_inactive_dims():
    pred, target = torch.zeros(2, A_CANON), torch.ones(2, A_CANON)
    mask = torch.zeros(2, A_CANON)
    mask[:, 0] = 1
    full = masked_bc_loss(pred, target, torch.ones(2, A_CANON))
    one = masked_bc_loss(pred, target, mask)
    assert torch.isclose(full, one)  # per-active-dim mean is identical here
    target2 = target.clone()
    target2[:, 1:] = 99.0  # garbage outside the mask must not change the loss
    assert torch.isclose(masked_bc_loss(pred, target2, mask), one)


def test_visual_actor_critic():
    ac = VisualActorCritic(BCPolicy(vocab_size=len(get_tokenizer())))
    batch = fake_batch()
    mask = torch.zeros(4, A_CANON)
    mask[:, :8] = 1
    action, logprob, entropy, value = ac.get_action_and_value(batch, mask)
    assert action.shape == (4, A_CANON) and (action * (1 - mask) == 0).all()
    assert logprob.shape == (4,) and entropy.shape == (4,) and value.shape == (4, 1)
    det = ac.act(batch, mask)
    assert det.shape == (4, A_CANON)


def test_state_actor_critic():
    ac = StateActorCritic(state_dim=42, action_dim=25)
    x = torch.randn(6, 42)
    action, logprob, entropy, value = ac.get_action_and_value(x)
    assert action.shape == (6, 25) and value.shape == (6, 1)
    assert ac.act(x).shape == (6, 25)

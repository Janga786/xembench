"""Action chunking: dataset windows, model head, loss broadcast, execution."""

import h5py
import numpy as np
import pytest
import torch

from xembench.constants import A_CANON
from xembench.data.dataset import DemoDataset
from xembench.eval.runner import bc_policy_fn
from xembench.language.tokenizer import get_tokenizer
from xembench.models.bc_policy import BCPolicy, masked_bc_loss


@pytest.fixture()
def demo_file(tmp_path):
    """Minimal schema-compatible demo file: one 5-step panda episode whose
    action at step t is constant t (padding is then easy to assert)."""
    path = tmp_path / "demos.h5"
    T = 5
    with h5py.File(path, "w") as f:
        f.attrs.update(dict(embodiment="panda", env_id="PickCubeLang-v1",
                            task="pick_cube_arm", camera="base_camera",
                            control_mode="pd_joint_delta_pos",
                            schema_version=1, mani_skill_version="test"))
        ep = f.create_group("ep_00000")
        ep.attrs.update(dict(instruction="pick up the red cube",
                             success=True, episode_len=T, seed=0))
        ep.create_dataset("rgb", data=np.zeros((T, 128, 128, 3), np.uint8))
        ep.create_dataset("proprio", data=np.zeros((T, 64), np.float32))
        acts = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, A_CANON))
        ep.create_dataset("action_canon", data=acts)
        ep.create_dataset("action_native", data=np.zeros((T, 8), np.float32))
    return path


def test_chunk1_action_shape_unchanged(demo_file):
    ds = DemoDataset([demo_file], instruction_mode="canonical")
    assert ds[0]["action"].shape == (A_CANON,)


def test_chunk_window_and_repeat_last_padding(demo_file):
    ds = DemoDataset([demo_file], instruction_mode="canonical", chunk=4)
    assert len(ds) == 5  # one sample per timestep, unchanged
    a = ds[0]["action"]
    assert a.shape == (4, A_CANON)
    assert torch.equal(a[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # window starting at t=3 runs past the 5-step episode: pad = repeat last
    a = ds[3]["action"]
    assert torch.equal(a[:, 0], torch.tensor([3.0, 4.0, 4.0, 4.0]))


def test_chunked_policy_output_shape_and_loss():
    model = BCPolicy(vocab_size=len(get_tokenizer()), chunk=4)
    obs = dict(
        rgb=torch.zeros(2, 3, 128, 128, dtype=torch.uint8),
        proprio=torch.zeros(2, 64), proprio_mask=torch.ones(2, 64),
        tokens=torch.zeros(2, 32, dtype=torch.int64),
        token_len=torch.ones(2, dtype=torch.int64),
        embodiment_id=torch.zeros(2, dtype=torch.int64),
    )
    pred = model(obs)
    assert pred.shape == (2, 4, A_CANON)
    target = torch.zeros(2, 4, A_CANON)
    mask = torch.ones(2, A_CANON)
    assert torch.isfinite(masked_bc_loss(pred, target, mask))


def test_chunk1_checkpoint_shape_matches_original():
    m = BCPolicy(vocab_size=len(get_tokenizer()), chunk=1)
    assert m.head.out_features == A_CANON  # unchanged single-step head


def test_chunked_execution_queues_and_resets():
    class Fake:
        chunk = 3
        calls = 0

        def __call__(self, obs):
            self.calls += 1
            return torch.arange(3 * A_CANON, dtype=torch.float32).view(1, 3, A_CANON)

    fake = Fake()
    policy = bc_policy_fn(fake, torch.device("cpu"))
    obs = {"rgb": torch.zeros(1)}
    first = [policy(obs) for _ in range(3)]
    assert fake.calls == 1                       # one prediction fills the queue
    assert not torch.equal(first[0], first[1])   # distinct actions per step
    policy(obs)
    assert fake.calls == 2                       # queue empty -> re-predict
    policy.reset()
    policy(obs)
    assert fake.calls == 3                       # reset clears queue

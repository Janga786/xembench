"""Writer -> Dataset round-trip on synthetic episodes (no simulator)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xembench.constants import A_CANON, IMG_SIZE, L_MAX, P_CANON
from xembench.data.dataset import DemoDataset, split_by_episode
from xembench.data.writer import DemoWriter


@pytest.fixture()
def demo_file(tmp_path):
    path = tmp_path / "demo.h5"
    rng = np.random.default_rng(0)
    with DemoWriter(
        path, task="pick_cube_arm", env_id="PickCubeLang-v1", embodiment="panda",
        camera="base_camera", control_mode="pd_joint_delta_pos", native_action_dim=8,
    ) as w:
        for ep in range(4):
            T = 10 + ep
            w.add_episode(
                rgb=rng.integers(0, 255, (T, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                proprio=rng.normal(size=(T, P_CANON)).astype(np.float32),
                action_canon=rng.uniform(-1, 1, (T, A_CANON)).astype(np.float32),
                action_native=rng.uniform(-1, 1, (T, 8)).astype(np.float32),
                instruction="pick up the red cube and place it on the green target",
                success=ep != 3,  # one failed episode
                seed=ep,
            )
    return path


def test_success_filter_and_length(demo_file):
    ds = DemoDataset([demo_file], success_only=True)
    assert len(ds) == 10 + 11 + 12  # failed episode excluded
    ds_all = DemoDataset([demo_file], success_only=False)
    assert len(ds_all) == 10 + 11 + 12 + 13


def test_sample_schema(demo_file):
    ds = DemoDataset([demo_file], instruction_mode="canonical")
    s = ds[0]
    assert s["rgb"].shape == (3, IMG_SIZE, IMG_SIZE) and s["rgb"].dtype == torch.uint8
    assert s["proprio"].shape == (P_CANON,)
    assert s["action"].shape == (A_CANON,)
    assert s["action_mask"].sum() == 8  # panda
    assert s["tokens"].shape == (L_MAX,)
    assert s["embodiment_id"].item() == 0


def test_dataloader_batches_and_augmentation(demo_file):
    ds = DemoDataset([demo_file], instruction_mode="train", paraphrase_p=1.0)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    assert batch["rgb"].shape == (8, 3, IMG_SIZE, IMG_SIZE)
    assert batch["action"].shape == (8, A_CANON)
    # p=1.0 paraphrasing must produce at least one non-canonical token row
    canonical = DemoDataset([demo_file], instruction_mode="canonical")[0]["tokens"]
    assert any(not torch.equal(batch["tokens"][i], canonical) for i in range(8))


def test_split_by_episode_no_leak(demo_file):
    ds = DemoDataset([demo_file], success_only=True)
    train, val = split_by_episode(ds, val_fraction=0.34, seed=0)
    train_eps = {ds.index[i][:2] for i in train.indices}
    val_eps = {ds.index[i][:2] for i in val.indices}
    assert train_eps & val_eps == set()
    assert len(train) + len(val) == len(ds)

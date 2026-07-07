"""validate_demo_file catches the corruption modes that would poison training."""

import h5py
import numpy as np
import pytest

from xembench.constants import A_CANON, IMG_SIZE, P_CANON
from xembench.data.schema import validate_demo_file
from xembench.data.writer import DemoWriter


@pytest.fixture()
def valid_file(tmp_path):
    path = tmp_path / "demo.h5"
    rng = np.random.default_rng(0)
    with DemoWriter(path, task="pick_cube_arm", env_id="PickCubeLang-v1",
                    embodiment="panda", camera="base_camera",
                    control_mode="pd_joint_delta_pos", native_action_dim=8) as w:
        for ep in range(2):
            T = 6
            w.add_episode(
                rgb=rng.integers(0, 255, (T, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                proprio=rng.normal(size=(T, P_CANON)).astype(np.float32),
                action_canon=rng.uniform(-1, 1, (T, A_CANON)).astype(np.float32),
                action_native=rng.uniform(-1, 1, (T, 8)).astype(np.float32),
                instruction="pick up the red cube and place it on the green target",
                success=True, seed=ep,
            )
    return path


def test_valid_file_passes(valid_file):
    assert validate_demo_file(valid_file) == []


def test_detects_missing_file_attr(valid_file):
    with h5py.File(valid_file, "a") as f:
        del f.attrs["camera"]
    errors = validate_demo_file(valid_file)
    assert any("camera" in e for e in errors)


def test_detects_missing_dataset(valid_file):
    with h5py.File(valid_file, "a") as f:
        del f["ep_00000/action_canon"]
    errors = validate_demo_file(valid_file)
    assert any("ep_00000" in e and "action_canon" in e for e in errors)


def test_detects_shape_mismatch(valid_file):
    with h5py.File(valid_file, "a") as f:
        del f["ep_00001/proprio"]
        f["ep_00001/proprio"] = np.zeros((6, P_CANON + 1), dtype=np.float32)
    errors = validate_demo_file(valid_file)
    assert any("ep_00001/proprio" in e and "shape" in e for e in errors)


def test_detects_nonfinite_values(valid_file):
    with h5py.File(valid_file, "a") as f:
        data = f["ep_00000/action_canon"][:]
        data[0, 0] = np.nan
        del f["ep_00000/action_canon"]
        f["ep_00000/action_canon"] = data
    errors = validate_demo_file(valid_file)
    assert any("non-finite" in e for e in errors)


def test_detects_empty_file(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = 1
    errors = validate_demo_file(path)
    assert any("no ep_" in e for e in errors)

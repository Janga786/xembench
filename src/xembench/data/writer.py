"""DemoWriter: append validated episodes to a schema-conformant HDF5 file."""

import json
from pathlib import Path

import h5py
import numpy as np

import mani_skill

from xembench.constants import A_CANON, IMG_SIZE, P_CANON
from xembench.data.schema import SCHEMA_VERSION


class DemoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        task: str,
        env_id: str,
        embodiment: str,
        camera: str,
        control_mode: str,
        native_action_dim: int,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.native_action_dim = native_action_dim
        self.n = 0
        self.file.attrs.update(
            schema_version=SCHEMA_VERSION,
            task=task,
            env_id=env_id,
            embodiment=embodiment,
            camera=camera,
            control_mode=control_mode,
            mani_skill_version=mani_skill.__version__,
        )

    def add_episode(
        self,
        *,
        rgb: np.ndarray,
        proprio: np.ndarray,
        action_canon: np.ndarray,
        action_native: np.ndarray,
        instruction: str,
        success: bool,
        seed: int,
        slots: dict | None = None,
    ) -> str:
        T = action_canon.shape[0]
        assert rgb.shape == (T, IMG_SIZE, IMG_SIZE, 3) and rgb.dtype == np.uint8, rgb.shape
        assert proprio.shape == (T, P_CANON), proprio.shape
        assert action_canon.shape == (T, A_CANON), action_canon.shape
        assert action_native.shape == (T, self.native_action_dim), action_native.shape

        key = f"ep_{self.n:05d}"
        g = self.file.create_group(key)
        g.create_dataset(
            "rgb", data=rgb, compression="gzip", compression_opts=4,
            chunks=(1, IMG_SIZE, IMG_SIZE, 3),
        )
        g.create_dataset("proprio", data=proprio.astype(np.float32))
        g.create_dataset("action_canon", data=action_canon.astype(np.float32))
        g.create_dataset("action_native", data=action_native.astype(np.float32))
        g.attrs.update(
            instruction=instruction,
            success=bool(success),
            seed=int(seed),
            episode_len=int(T),
            slots=json.dumps(slots or {}),
        )
        self.n += 1
        return key

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

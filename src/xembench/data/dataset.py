"""DemoDataset: flat (episode, t) index over demo HDF5 files.

Each sample is one transition, model-ready (default torch collate works —
no strings leave the dataset). Paraphrase augmentation happens ONLINE per
sample with the split's synonym table, so one stored canonical instruction
yields many phrasings across epochs while heldout synonyms stay unseen.

HDF5 handles are opened lazily so each DataLoader worker (fork) gets its own;
``preload=True`` pulls everything into RAM instead (a 200-episode task file
is well under 1 GB).
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from xembench.embodiment import get_adapter
from xembench.language.paraphrase import paraphrase
from xembench.language.tokenizer import get_tokenizer


class DemoDataset(Dataset):
    def __init__(
        self,
        paths: list[str | Path],
        instruction_mode: str = "train",  # train (augment) | canonical
        paraphrase_p: float = 0.7,
        success_only: bool = True,
        preload: bool = False,
    ):
        assert instruction_mode in ("train", "canonical")
        self.paths = [str(p) for p in paths]
        self.instruction_mode = instruction_mode
        self.paraphrase_p = paraphrase_p
        self.preload = preload
        self.tokenizer = get_tokenizer()
        self._h5: dict[int, h5py.File] = {}
        self._cache: dict[tuple[int, str], dict] | None = {} if preload else None
        self._rng: np.random.Generator | None = None

        # index: (file_i, episode_key, t) + per-episode metadata
        self.index: list[tuple[int, str, int]] = []
        self.episode_meta: dict[tuple[int, str], dict] = {}
        self.file_meta: list[dict] = []
        for fi, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                self.file_meta.append(dict(f.attrs))
                adapter = get_adapter(f.attrs["embodiment"])
                for key in sorted(f.keys()):
                    ep = f[key]
                    if success_only and not ep.attrs["success"]:
                        continue
                    meta = dict(
                        instruction=ep.attrs["instruction"],
                        embodiment_id=adapter.embodiment_id,
                        action_mask=adapter.action_mask.numpy(),
                        proprio_mask=adapter.proprio_mask.numpy(),
                    )
                    self.episode_meta[(fi, key)] = meta
                    T = int(ep.attrs["episode_len"])
                    self.index += [(fi, key, t) for t in range(T)]
                    if self.preload:
                        self._cache[(fi, key)] = dict(
                            rgb=ep["rgb"][:],
                            proprio=ep["proprio"][:],
                            action_canon=ep["action_canon"][:],
                        )

    def __len__(self) -> int:
        return len(self.index)

    def _episode_arrays(self, fi: int, key: str) -> dict:
        if self.preload:
            return self._cache[(fi, key)]
        if fi not in self._h5:  # lazy per-process/worker open
            self._h5[fi] = h5py.File(self.paths[fi], "r")
        ep = self._h5[fi][key]
        return ep

    def _instruction_tokens(self, text: str) -> tuple[np.ndarray, int]:
        if self.instruction_mode == "train":
            if self._rng is None:
                # distinct stream per DataLoader worker and per epoch
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            text = paraphrase(text, self._rng, "train", self.paraphrase_p)
        return self.tokenizer.encode(text)

    def __getitem__(self, i: int) -> dict:
        fi, key, t = self.index[i]
        arrays = self._episode_arrays(fi, key)
        meta = self.episode_meta[(fi, key)]
        tokens, length = self._instruction_tokens(meta["instruction"])
        rgb = torch.from_numpy(np.asarray(arrays["rgb"][t]))  # [H, W, 3] uint8
        return dict(
            rgb=rgb.permute(2, 0, 1).contiguous(),
            proprio=torch.from_numpy(np.asarray(arrays["proprio"][t])),
            proprio_mask=torch.from_numpy(meta["proprio_mask"]),
            tokens=torch.from_numpy(tokens),
            token_len=torch.tensor(length, dtype=torch.int64),
            embodiment_id=torch.tensor(meta["embodiment_id"], dtype=torch.int64),
            action=torch.from_numpy(np.asarray(arrays["action_canon"][t])),
            action_mask=torch.from_numpy(meta["action_mask"]),
        )


def split_by_episode(
    dataset: DemoDataset, val_fraction: float = 0.1, seed: int = 0
) -> tuple[Subset, Subset]:
    """Train/val split at EPISODE granularity (no timestep leakage)."""
    episodes = sorted({(fi, key) for fi, key, _ in dataset.index})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_fraction))
    val_eps = set(map(tuple, episodes[:n_val]))
    train_idx = [i for i, (fi, key, _) in enumerate(dataset.index)
                 if (fi, key) not in val_eps]
    val_idx = [i for i, (fi, key, _) in enumerate(dataset.index)
               if (fi, key) in val_eps]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

"""On-disk demonstration schema (HDF5), shared by every demo source.

    data/demos/{embodiment}/{env_id}/{source}.h5
      attrs: schema_version, task, env_id, embodiment, camera, control_mode,
             mani_skill_version
      ep_00000/
        rgb            uint8   [T, IMG, IMG, 3]   (gzip-4, per-frame chunks)
        proprio        float32 [T, P_CANON]
        action_canon   float32 [T, A_CANON]
        action_native  float32 [T, K]
        attrs: instruction (canonical text), success (bool), seed (int),
               episode_len (int), slots (json)

T = number of actions; obs are pre-action (s_t, a_t) pairs, the terminal obs
is dropped. Paraphrasing is applied ONLINE at training time, never baked into
files — the stored instruction is always the canonical rendering.
"""

SCHEMA_VERSION = 1

REQUIRED_FILE_ATTRS = (
    "schema_version", "task", "env_id", "embodiment", "camera",
    "control_mode", "mani_skill_version",
)
REQUIRED_EPISODE_DATASETS = ("rgb", "proprio", "action_canon", "action_native")
REQUIRED_EPISODE_ATTRS = ("instruction", "success", "seed", "episode_len")


def demo_path(root, embodiment: str, env_id: str, source: str) -> str:
    return f"{root}/demos/{embodiment}/{env_id}/{source}.h5"


def validate_demo_file(path) -> list[str]:
    """Check a demo HDF5 against this schema; returns a list of problems
    (empty = valid). Used by tests and as a pre-training sanity check on
    freshly collected Phase B datasets."""
    import h5py
    import numpy as np

    from xembench.constants import A_CANON, IMG_SIZE, P_CANON

    errors: list[str] = []
    with h5py.File(path, "r") as f:
        for attr in REQUIRED_FILE_ATTRS:
            if attr not in f.attrs:
                errors.append(f"missing file attr {attr!r}")
        if int(f.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            errors.append(
                f"schema_version {f.attrs.get('schema_version')} != {SCHEMA_VERSION}"
            )
        episodes = [k for k in f.keys() if k.startswith("ep_")]
        if not episodes:
            errors.append("no ep_* groups")
        for key in episodes:
            ep = f[key]
            for name in REQUIRED_EPISODE_DATASETS:
                if name not in ep:
                    errors.append(f"{key}: missing dataset {name!r}")
            for attr in REQUIRED_EPISODE_ATTRS:
                if attr not in ep.attrs:
                    errors.append(f"{key}: missing attr {attr!r}")
            if any(e.startswith(key) for e in errors):
                continue  # shape checks would just cascade
            T = int(ep.attrs["episode_len"])
            checks = {
                "rgb": ((T, IMG_SIZE, IMG_SIZE, 3), np.uint8),
                "proprio": ((T, P_CANON), np.float32),
                "action_canon": ((T, A_CANON), np.float32),
            }
            for name, (shape, dtype) in checks.items():
                ds = ep[name]
                if tuple(ds.shape) != shape:
                    errors.append(f"{key}/{name}: shape {ds.shape} != {shape}")
                if ds.dtype != dtype:
                    errors.append(f"{key}/{name}: dtype {ds.dtype} != {dtype}")
            if ep["action_native"].shape[0] != T:
                errors.append(f"{key}/action_native: length != episode_len")
            for name in ("proprio", "action_canon", "action_native"):
                if not np.isfinite(ep[name][:]).all():
                    errors.append(f"{key}/{name}: non-finite values")
    return errors

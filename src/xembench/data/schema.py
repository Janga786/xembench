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


def demo_path(root, embodiment: str, env_id: str, source: str) -> str:
    return f"{root}/demos/{embodiment}/{env_id}/{source}.h5"

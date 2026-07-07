"""Slow integration tests that create real ManiSkill envs (pytest -m sim)."""

import pytest
import torch

from xembench.constants import A_CANON, IMG_SIZE, L_MAX, P_CANON
from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.language.parser import parse

CASES = [
    # task name, env_id, embodiment, camera, family, static slots
    ("pick_cube_arm", "PickCubeLang-v1", "panda", "base_camera", "pick_place", None),
    ("apple_bowl_g1", "UnitreeG1PlaceAppleInBowl-v1", "g1", "head_camera",
     "pick_place", {"color": "red", "object": "apple", "relation": "in",
                    "receptacle": "bowl"}),
    ("push_t_arm", "PushT-v1", "panda_stick", "base_camera", "tool_push", {}),
]


def _schema_ok(obs, B):
    assert obs["rgb"].shape == (B, 3, IMG_SIZE, IMG_SIZE) and obs["rgb"].dtype == torch.uint8
    assert obs["proprio"].shape == (B, P_CANON) and obs["proprio"].dtype == torch.float32
    assert obs["proprio_mask"].shape == (B, P_CANON)
    assert obs["tokens"].shape == (B, L_MAX) and obs["tokens"].dtype == torch.int64
    assert obs["token_len"].shape == (B,)
    assert obs["embodiment_id"].shape == (B,)
    assert len(obs["instruction"]) == B and all(obs["instruction"])


@pytest.mark.sim
@pytest.mark.parametrize("name,env_id,embodiment,camera,family,slots", CASES)
def test_wrapped_rollout(name, env_id, embodiment, camera, family, slots):
    adapter = get_adapter(embodiment)
    env = make_env(env_id, num_envs=1, obs_mode="rgb")
    adapter.validate_env(env)
    wrapped = UnifiedObsWrapper(env, adapter, camera, family, static_slots=slots)
    obs, _ = wrapped.reset(seed=0)
    _schema_ok(obs, 1)
    for _ in range(3):
        canon = torch.rand(1, A_CANON) * 0.2 - 0.1
        native = adapter.to_native_action(canon)
        obs, reward, terminated, truncated, info = wrapped.step(native)
    _schema_ok(obs, 1)
    wrapped.close()


@pytest.mark.sim
def test_instruction_names_the_actual_target_cube():
    """The rendered instruction must ground to the env's sampled target."""
    adapter = get_adapter("panda")
    env = make_env("PickCubeLang-v1", num_envs=1, obs_mode="rgb")
    wrapped = UnifiedObsWrapper(env, adapter, "base_camera", "pick_place")
    for seed in range(5):
        obs, _ = wrapped.reset(seed=seed)
        u = env.unwrapped
        expected = u.color_names[int(u.target_idx[0])]
        assert parse(obs["instruction"][0]).color == expected
    wrapped.close()


@pytest.mark.sim
def test_flat_state_present_for_collection_mode():
    adapter = get_adapter("panda")
    env = make_env("PickCubeLang-v1", num_envs=1, obs_mode="state_dict+rgb")
    wrapped = UnifiedObsWrapper(
        env, adapter, "base_camera", "pick_place", include_flat_state=True
    )
    obs, _ = wrapped.reset(seed=0)
    assert obs["flat_state"].ndim == 2 and obs["flat_state"].shape[0] == 1
    wrapped.close()

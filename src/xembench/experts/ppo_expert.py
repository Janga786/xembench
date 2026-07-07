"""Load a trained flat-state PPO expert as a collection policy.

Used by scripts/collect_demos.py (policy=ppo:<ckpt>): the collection env runs
obs_mode='state_dict+rgb', the wrapper exposes obs['flat_state'] with the
exact obs_mode='state' flattening the expert was trained on, and the expert
returns NATIVE actions.
"""

import torch

from xembench.models.ac_policy import StateActorCritic


def load_expert_policy(ckpt_path: str, env, adapter, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = StateActorCritic(ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    expected = env.action_space.shape[-1]
    assert ckpt["action_dim"] == expected, (
        f"expert action dim {ckpt['action_dim']} != env {expected} "
        f"(trained on a different task/embodiment?)"
    )

    @torch.no_grad()
    def policy(obs: dict) -> torch.Tensor:
        state = obs["flat_state"].to(device)
        assert state.shape[-1] == ckpt["state_dim"], (
            f"flat_state dim {state.shape[-1]} != expert's {ckpt['state_dim']} — "
            "collection env must expose the same extra obs as obs_mode='state'"
        )
        return model.act(state, deterministic=True).cpu()

    return policy

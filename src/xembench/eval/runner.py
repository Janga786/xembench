"""evaluate_policy: sequential-episode evaluation of a canonical-action policy.

The policy is a callable (unified obs dict) -> canonical action [B, A_CANON];
the runner translates through the eval embodiment's adapter, so running an
arm-trained policy on the humanoid is literally passing a different (env,
adapter) pair — nothing about the policy changes. num_envs=1 sequential
episodes: exact and identical on CPU (Mac) and GPU (Linux) sim.
"""

import time
from dataclasses import asdict, dataclass, field

import torch

from xembench.analysis.stats import wilson_ci
from xembench.analysis.taxonomy import classify_episode
from xembench.embodiment.base import EmbodimentAdapter

__all__ = ["EvalResult", "evaluate_policy", "bc_policy_fn", "wilson_ci"]


@dataclass
class EvalResult:
    n_episodes: int
    n_success: int
    success_rate: float
    ci_low: float
    ci_high: float
    avg_steps_to_success: float  # successes only; nan if none
    avg_steps_frac: float        # steps / max_steps, comparable across tasks
    # per-episode records for failure analysis; excluded from to_dict so the
    # aggregate CSV schema is unchanged
    episodes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("episodes")
        return d


@torch.no_grad()
def evaluate_policy(
    policy,
    wrapped_env,
    adapter: EmbodimentAdapter,
    n_episodes: int,
    max_steps: int,
    seed0: int = 10_000,
) -> EvalResult:
    successes, steps_when_success, episode_records = 0, [], []
    for ep in range(n_episodes):
        episode_seed = seed0 + ep
        t0 = time.perf_counter()
        obs, _ = wrapped_env.reset(seed=episode_seed)
        instruction = obs["instruction"][0] if "instruction" in obs else ""
        success, was_truncated, return_sum = False, True, 0.0
        for t in range(max_steps):
            canon = policy(obs)
            native = adapter.to_native_action(canon)
            obs, reward, terminated, truncated, info = wrapped_env.step(native)
            return_sum += float(torch.as_tensor(reward).reshape(-1)[0])
            success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
            if bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(
                torch.as_tensor(truncated).reshape(-1)[0]
            ):
                was_truncated = bool(torch.as_tensor(truncated).reshape(-1)[0])
                break
        steps = t + 1
        if success:
            successes += 1
            steps_when_success.append(steps)
        episode_records.append(dict(
            episode_idx=ep,
            episode_seed=episode_seed,
            success=int(success),
            steps=steps,
            max_steps=max_steps,
            failure_type=classify_episode(success, was_truncated),
            # schema-v2 optional fields (analysis/episodes.py)
            instruction=instruction,
            return_sum=round(return_sum, 6),
            elapsed_s=round(time.perf_counter() - t0, 3),
        ))

    lo, hi = wilson_ci(successes, n_episodes)
    mean_steps = (
        sum(steps_when_success) / len(steps_when_success)
        if steps_when_success else float("nan")
    )
    return EvalResult(
        n_episodes=n_episodes,
        n_success=successes,
        success_rate=successes / n_episodes if n_episodes else 0.0,
        ci_low=lo,
        ci_high=hi,
        avg_steps_to_success=mean_steps,
        avg_steps_frac=mean_steps / max_steps if steps_when_success else float("nan"),
        episodes=episode_records,
    )


def bc_policy_fn(model, device):
    """Wrap a BCPolicy into the runner's policy interface."""
    from xembench.utils.torch_utils import to_device

    def policy(obs: dict) -> torch.Tensor:
        batch = {k: v for k, v in obs.items() if torch.is_tensor(v)}
        return model(to_device(batch, device)).cpu()

    return policy

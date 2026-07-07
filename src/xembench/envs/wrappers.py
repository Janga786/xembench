"""UnifiedObsWrapper: ManiSkill obs -> the canonical benchmark observation.

Output obs dict (torch tensors, on the env's device):

    rgb            uint8   [B, 3, IMG, IMG]   task camera, channels-first
    proprio        float32 [B, P_CANON]       canonical qpos/qvel layout
    proprio_mask   float32 [B, P_CANON]
    tokens         int64   [B, L_MAX]         instruction token ids
    token_len      int64   [B]
    embodiment_id  int64   [B]
    instruction    list[str]                  raw text (logging / demo files)
    flat_state     float32 [B, S]             ONLY when the underlying env
                                              obs_mode includes state_dict
                                              (expert demo collection)

The wrapper handles OBSERVATIONS + INSTRUCTIONS only. Action translation is
explicit in runners (``env.step(adapter.to_native_action(a))``) so scripted
experts can keep acting natively on the same wrapped env.

Instructions are (re)sampled at episode starts: on reset() for all envs, and
after step() for any env that reported done (ManiSkillVectorEnv auto-resets
sub-envs before returning, so the unwrapped env state is already the new
episode's when we resample).
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from mani_skill.utils import common

from xembench.constants import IMG_SIZE, L_MAX
from xembench.embodiment.base import EmbodimentAdapter
from xembench.language.paraphrase import paraphrase
from xembench.language.templates import render
from xembench.language.tokenizer import get_tokenizer


def flatten_agent_extra(obs: dict) -> torch.Tensor:
    """The exact flattening ManiSkill uses for obs_mode='state'.

    Experts are trained on obs_mode='state'; during rgb demo collection the
    env runs obs_mode='state_dict+rgb' and this reproduces the identical
    state vector from the unflattened parts.
    """
    return common.flatten_state_dict(
        dict(agent=obs["agent"], extra=obs["extra"]), use_torch=True
    )


class UnifiedObsWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        adapter: EmbodimentAdapter,
        camera: str,
        family: str,
        static_slots: dict | None = None,
        instruction_mode: str = "canonical",  # canonical | train | heldout
        paraphrase_p: float = 0.7,
        language_seed: int = 0,
        include_flat_state: bool = False,
    ):
        super().__init__(env)
        assert instruction_mode in ("canonical", "train", "heldout")
        self.adapter = adapter
        self.camera = camera
        self.family = family
        self.static_slots = static_slots or {}
        self.instruction_mode = instruction_mode
        self.paraphrase_p = paraphrase_p
        self.include_flat_state = include_flat_state
        self.rng = np.random.default_rng(language_seed)
        self.tokenizer = get_tokenizer()
        self.instructions: list[str] = []
        self._tokens: torch.Tensor | None = None
        self._token_len: torch.Tensor | None = None

    # -- instructions ----------------------------------------------------------
    def _episode_slots(self) -> list[dict]:
        u = self.env.unwrapped
        if hasattr(u, "language_slots"):
            return u.language_slots()
        return [dict(self.static_slots)] * u.num_envs

    def _render_instruction(self, slots: dict) -> str:
        text = render(self.family, slots)
        if self.instruction_mode != "canonical":
            text = paraphrase(text, self.rng, self.instruction_mode, self.paraphrase_p)
        return text

    def _resample_instructions(self, env_idx: list[int] | None = None) -> None:
        slots = self._episode_slots()
        n = len(slots)
        if self._tokens is None or len(self.instructions) != n:
            self.instructions = [""] * n
            self._tokens = torch.zeros(n, L_MAX, dtype=torch.int64)
            self._token_len = torch.zeros(n, dtype=torch.int64)
        idx = range(n) if env_idx is None else env_idx
        for i in idx:
            text = self._render_instruction(slots[i])
            self.instructions[i] = text
            ids, length = self.tokenizer.encode(text)
            self._tokens[i] = torch.from_numpy(ids)
            self._token_len[i] = length

    # -- obs assembly ----------------------------------------------------------
    def _process_rgb(self, obs: dict) -> torch.Tensor:
        rgb = obs["sensor_data"][self.camera]["rgb"]  # [B, H, W, 3] uint8
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb.shape[-1] != IMG_SIZE or rgb.shape[-2] != IMG_SIZE:
            rgb = F.interpolate(
                rgb.float(), size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                align_corners=False,
            ).round().clamp(0, 255).to(torch.uint8)
        return rgb

    def _unify(self, obs: dict) -> dict:
        rgb = self._process_rgb(obs)
        device = rgb.device
        B = rgb.shape[0]
        proprio, pmask = self.adapter.extract_proprio(obs)
        out = dict(
            rgb=rgb,
            proprio=proprio.float(),
            proprio_mask=pmask.expand(B, -1).float(),
            tokens=self._tokens.to(device),
            token_len=self._token_len.to(device),
            embodiment_id=torch.full(
                (B,), self.adapter.embodiment_id, dtype=torch.int64, device=device
            ),
            instruction=list(self.instructions),
        )
        if self.include_flat_state:
            out["flat_state"] = flatten_agent_extra(obs)
        return out

    # -- gym API ----------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._resample_instructions()
        return self._unify(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
        if done.any():
            idx = torch.nonzero(done.reshape(-1)).squeeze(-1).tolist()
            self._resample_instructions(idx)
        return self._unify(obs), reward, terminated, truncated, info

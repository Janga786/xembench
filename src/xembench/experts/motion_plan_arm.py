"""Motion-planning demo collection for the arm Lang envs (LINUX ONLY: mplib).

Drives ManiSkill's Panda motion-planning solutions inside our Lang env
subclasses. The solutions manipulate `env.cube` / `env.cubeA` / `env.cubeB`;
the Lang envs expose those names as properties resolving to the CURRENT
episode's language-designated target(s), so the same solver code produces
color-grounded demonstrations.

    uv run python -m xembench.experts.motion_plan_arm \
        --set task=pick_cube_arm --set episodes=200

Run gate (B1): stats printed at the end must show ~100% success and the
instruction distribution covering all train colors.
"""

import sys

import numpy as np
import torch

from xembench.constants import CONTROL_MODE
from xembench.data.schema import demo_path
from xembench.data.writer import DemoWriter
from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.utils.config import common_parser, load_config, load_yaml, repo_root
from xembench.utils.seeding import seed_everything

# task -> ManiSkill motion-planning solver (imported lazily: mplib is
# Linux-only and these imports pull it in)
SOLVER_PATHS = {
    "pick_cube_arm": ("mani_skill.examples.motionplanning.panda.solutions.pick_cube",
                      "solve"),
    "stack_cube_arm": ("mani_skill.examples.motionplanning.panda.solutions.stack_cube",
                       "solve"),
    "pull_tool_arm": (
        "mani_skill.examples.motionplanning.panda.solutions.pull_cube_tool", "solve"),
}


class _Recorder:
    """Intercepts env.step to capture (unified obs, native action) pairs while
    the motion planner drives the env."""

    def __init__(self, wrapped: UnifiedObsWrapper):
        self.wrapped = wrapped
        self.frames, self.proprios, self.natives = [], [], []
        self.last_obs = None
        self._orig_step = wrapped.env.step

    def start(self, obs):
        self.frames, self.proprios, self.natives = [], [], []
        self.last_obs = obs
        self.wrapped.env.step = self._recording_step

    def _recording_step(self, action):
        o = self.last_obs
        self.frames.append(o["rgb"][0].permute(1, 2, 0).cpu().numpy())
        self.proprios.append(o["proprio"][0].cpu().numpy())
        self.natives.append(np.asarray(torch.as_tensor(action).reshape(-1),
                                       dtype=np.float32))
        result = self._orig_step(action)
        self.last_obs = self.wrapped._unify(result[0])
        return result

    def stop(self):
        self.wrapped.env.step = self._orig_step


def collect_mp(cfg: dict) -> int:
    import importlib

    tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
    task = tasks[cfg["task"]]
    module_name, fn_name = SOLVER_PATHS[cfg["task"]]
    solve = getattr(importlib.import_module(module_name), fn_name)

    adapter = get_adapter(task["embodiment"])
    env = make_env(task["env_id"], num_envs=1, obs_mode="rgb")
    wrapped = UnifiedObsWrapper(env, adapter, task["camera"], task["family"],
                                static_slots=task.get("slots"),
                                language_seed=cfg["seed"])
    recorder = _Recorder(wrapped)

    out = cfg.get("out") or demo_path(
        repo_root() / "data", task["embodiment"], task["env_id"], "motion_planning"
    )
    n_target, n_written, attempt = cfg["episodes"], 0, 0
    color_counts: dict[str, int] = {}
    with DemoWriter(out, task=cfg["task"], env_id=task["env_id"],
                    embodiment=task["embodiment"], camera=task["camera"],
                    control_mode=CONTROL_MODE,
                    native_action_dim=adapter.native_action_dim) as writer:
        while n_written < n_target and attempt < n_target * 5:
            seed = cfg["seed"] * 1_000_000 + attempt
            attempt += 1
            obs, _ = wrapped.reset(seed=seed)
            instruction = obs["instruction"][0]
            slots = wrapped._episode_slots()[0]
            recorder.start(obs)
            try:
                res = solve(env, seed=seed, debug=False, vis=False)
            except Exception as e:  # noqa: BLE001 - one bad plan shouldn't kill the run
                print(f"  attempt {attempt}: solver error {type(e).__name__}: {e}")
                recorder.stop()
                continue
            recorder.stop()
            success = bool(torch.as_tensor(res[-1]["success"]).reshape(-1)[0]) \
                if isinstance(res, tuple) else bool(
                    torch.as_tensor(env.unwrapped.evaluate()["success"]).reshape(-1)[0])
            if not success or not recorder.natives:
                continue
            native = torch.from_numpy(np.stack(recorder.natives))
            writer.add_episode(
                rgb=np.stack(recorder.frames).astype(np.uint8),
                proprio=np.stack(recorder.proprios),
                action_canon=adapter.to_canonical_action(native).numpy(),
                action_native=native.numpy(),
                instruction=instruction,
                success=True,
                seed=seed,
                slots=slots,
            )
            color = slots.get("color", "n/a")
            color_counts[color] = color_counts.get(color, 0) + 1
            n_written += 1
            if n_written % 10 == 0:
                print(f"  {n_written}/{n_target} ({n_written/attempt:.0%} solve rate)")

    wrapped.close()
    print(f"wrote {n_written} episodes -> {out}\ncolor distribution: {color_counts}")
    return 0 if n_written == n_target else 1


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "collect.yaml")
    seed_everything(cfg["seed"])
    return collect_mp(cfg)


if __name__ == "__main__":
    sys.exit(main())

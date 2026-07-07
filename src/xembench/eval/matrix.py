"""The evaluation grid: policies x tasks x conditions x embodiment transfer.

For every trained policy this produces:
- native cells: {canonical, heldout-paraphrase} x {train, unseen colors}
  (color axis only for the *Lang envs; stock scenes are fixed)
- transfer cells: the SAME network run on the partner task of its transfer
  pair with the partner embodiment's adapter (canonical instructions).
"""

import torch

from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.eval.runner import bc_policy_fn, evaluate_policy
from xembench.language.tokenizer import get_tokenizer
from xembench.models.bc_policy import BCPolicy
from xembench.utils.config import load_yaml, repo_root, resolve_path
from xembench.utils.torch_utils import auto_device


def _is_lang_env(env_id: str) -> bool:
    return "Lang" in env_id


def _make_wrapped(task: dict, instruction_cond: str, color_cond: str, seed: int):
    env_kwargs = {}
    if _is_lang_env(task["env_id"]) and color_cond == "unseen":
        env_kwargs["color_set"] = "unseen"
    env = make_env(task["env_id"], num_envs=1, obs_mode="rgb", **env_kwargs)
    adapter = get_adapter(task["embodiment"])
    wrapped = UnifiedObsWrapper(
        env, adapter, task["camera"], task["family"],
        static_slots=task.get("slots"),
        instruction_mode="canonical" if instruction_cond == "canonical" else "heldout",
        paraphrase_p=1.0,
        language_seed=seed,
    )
    return wrapped, adapter


def _load_policy(ckpt: str, vision: str, device) -> BCPolicy:
    model = BCPolicy(vocab_size=len(get_tokenizer()), vision=vision).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def _eval_cell(policy_fn, task: dict, episodes: int, instruction_cond: str,
               color_cond: str, seed: int, identity: dict) -> tuple[dict, list[dict]]:
    """-> (aggregate row fields, per-episode rows stamped with cell identity)."""
    wrapped, adapter = _make_wrapped(task, instruction_cond, color_cond, seed)
    result = evaluate_policy(
        policy_fn, wrapped, adapter, n_episodes=episodes,
        max_steps=task["max_steps"], seed0=100_000 * (seed + 1),
    )
    wrapped.close()
    episode_rows = [identity | rec for rec in result.episodes]
    return result.to_dict(), episode_rows


def run_matrix(cfg: dict) -> dict[str, list[dict]]:
    """-> {"cells": aggregate rows, "episodes": per-episode rows}."""
    registry = load_yaml(repo_root() / "configs" / "tasks.yaml")
    tasks = registry["tasks"]
    pairs = registry["transfer_pairs"]
    partner = {}
    for p in pairs:
        partner[p["a"]], partner[p["b"]] = p["b"], p["a"]

    device = auto_device(cfg.get("device", "auto"))
    rows: list[dict] = []
    episode_rows: list[dict] = []

    for spec in cfg["policies"]:
        for task_name in cfg["tasks"]:
            task = tasks[task_name]
            for seed in cfg["seeds"]:
                ckpt = resolve_path(spec["ckpt"].format(task=task_name, seed=seed))
                try:
                    model = _load_policy(ckpt, cfg.get("vision", "nature_cnn"), device)
                except FileNotFoundError:
                    if spec.get("optional", False):
                        continue
                    raise
                policy_fn = bc_policy_fn(model, device)
                base = dict(
                    policy_type=spec["type"], task=task_name,
                    train_embodiment=task["embodiment"], seed=seed, checkpoint=ckpt,
                )

                # native cells
                color_conds = (cfg["color_conds"] if _is_lang_env(task["env_id"])
                               else ["fixed"])
                for instr in cfg["instruction_conds"]:
                    for color in color_conds:
                        print(f"[eval] {spec['type']} {task_name} native "
                              f"{instr}/{color} seed{seed}")
                        identity = base | dict(
                            eval_task=task_name, eval_embodiment=task["embodiment"],
                            instruction_cond=instr, color_cond=color,
                            template_family=task["family"])
                        r, eps = _eval_cell(policy_fn, task, cfg["episodes_native"],
                                            instr, color, seed, identity)
                        rows.append(identity | r)
                        episode_rows += eps

                # transfer cell
                if cfg.get("transfer", True) and task_name in partner:
                    ptask_name = partner[task_name]
                    ptask = tasks[ptask_name]
                    print(f"[eval] {spec['type']} {task_name} -> {ptask_name} "
                          f"(transfer) seed{seed}")
                    identity = base | dict(
                        eval_task=ptask_name, eval_embodiment=ptask["embodiment"],
                        instruction_cond="canonical", color_cond="train",
                        template_family=ptask["family"])
                    r, eps = _eval_cell(policy_fn, ptask, cfg["episodes_transfer"],
                                        "canonical", "train", seed, identity)
                    rows.append(identity | r)
                    episode_rows += eps
    return {"cells": rows, "episodes": episode_rows}

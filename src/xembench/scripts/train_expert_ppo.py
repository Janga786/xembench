"""Train a flat-state PPO expert (dense reward) — the G1 demo source.

    uv run python -m xembench.scripts.train_expert_ppo \
        --config configs/ppo_expert_g1.yaml --set task=apple_bowl_g1
"""

import sys

from xembench.utils.config import common_parser, load_config, repo_root, resolve_path
from xembench.utils.seeding import seed_everything


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "ppo_expert_g1.yaml")
    seed_everything(cfg["seed"])
    if not cfg.get("out_dir"):
        cfg["out_dir"] = (f"{repo_root()}/data/checkpoints/"
                          f"expert_{cfg['task']}_seed{cfg['seed']}")
    cfg["out_dir"] = resolve_path(cfg["out_dir"])

    from xembench.training.ppo import train_ppo_state

    train_ppo_state(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

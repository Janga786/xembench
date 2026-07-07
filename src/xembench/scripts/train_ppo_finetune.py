"""Fine-tune a trained BC policy with sparse-reward PPO.

    uv run python -m xembench.scripts.train_ppo_finetune \
        --config configs/ppo_finetune.yaml --set task=pick_cube_arm \
        --set bc_ckpt=data/checkpoints/bc_pick_cube_arm_seed0/best.pt
"""

import sys

from xembench.utils.config import common_parser, load_config, repo_root, resolve_path
from xembench.utils.seeding import seed_everything


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "ppo_finetune.yaml")
    seed_everything(cfg["seed"])
    if not cfg.get("bc_ckpt"):
        cfg["bc_ckpt"] = (f"{repo_root()}/data/checkpoints/"
                          f"bc_{cfg['task']}_seed{cfg['seed']}/best.pt")
    cfg["bc_ckpt"] = resolve_path(cfg["bc_ckpt"])
    if not cfg.get("out_dir"):
        cfg["out_dir"] = (f"{repo_root()}/data/checkpoints/"
                          f"ppo_{cfg['task']}_seed{cfg['seed']}")
    cfg["out_dir"] = resolve_path(cfg["out_dir"])

    from xembench.training.ppo import train_ppo_visual

    train_ppo_visual(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

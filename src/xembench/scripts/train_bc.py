"""Train one BC policy for a (task, embodiment) pair.

    uv run python -m xembench.scripts.train_bc --config configs/bc.yaml \
        --set task=pick_cube_arm --seed 1

If cfg.demos is empty the demo files are discovered from the task's schema
directory (data/demos/{embodiment}/{env_id}/*.h5).
"""

import sys
from glob import glob

from xembench.utils.config import common_parser, load_config, load_yaml, repo_root
from xembench.utils.seeding import seed_everything


def resolve_demos(cfg: dict) -> list[str]:
    if cfg.get("demos"):
        return cfg["demos"]
    tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
    task = tasks[cfg["task"]]
    pattern = f"{repo_root()}/data/demos/{task['embodiment']}/{task['env_id']}/*.h5"
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no demo files match {pattern}")
    return paths


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "bc.yaml")
    seed_everything(cfg["seed"])
    cfg["demos"] = resolve_demos(cfg)
    if not cfg.get("out_dir"):
        cfg["out_dir"] = f"{repo_root()}/data/checkpoints/bc_{cfg['task']}_seed{cfg['seed']}"

    from xembench.training.bc_trainer import train_bc  # torch import deferred

    metrics = train_bc(cfg)
    print(f"done: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

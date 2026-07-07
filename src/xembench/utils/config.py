"""YAML configs with dotted-key CLI overrides.

Every pipeline script uses the same pattern:

    python -m xembench.scripts.train_bc --config configs/bc.yaml \
        --set task=pick_cube_arm --set train.lr=1e-4 --seed 3

Values after ``=`` are parsed as YAML, so numbers/bools/lists work naturally.
"""

import argparse
import copy
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def set_dotted(cfg: dict, dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        set_dotted(cfg, key.strip(), value)
    return cfg


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="dotted-key config override (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_config(args: argparse.Namespace, default_path: str | None = None) -> dict:
    path = args.config or default_path
    cfg = load_yaml(path) if path else {}
    cfg = apply_overrides(cfg, args.overrides)
    cfg["seed"] = args.seed
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def dump_run_config(cfg: dict, run_dir: str | Path) -> None:
    """Drop config + git SHA into an artifact directory for reproducibility."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(cfg)
    record["git_sha"] = git_sha()
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(record, f, sort_keys=False)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path) -> str:
    """Anchor relative paths at the repo root. SAPIEN's shader compilation can
    change the process cwd, so every path that crosses an env-creation
    boundary must be absolute."""
    p = Path(p)
    return str(p if p.is_absolute() else repo_root() / p)

"""Typed manifest + reproducibility capture for artifact packs.

The manifest is the pack's provenance record: exactly what produced these
numbers, from which inputs, on which code state. Synthetic data cannot be
packaged without a loud warning — validation enforces it.
"""

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from xembench.utils.config import repo_root, sanitize_path  # noqa: F401
# sanitize_path re-exported: it moved to utils.config so analysis-layer
# report generators can use it without importing the artifacts layer.

MANIFEST_SCHEMA_VERSION = 1

SYNTHETIC_WARNING = (
    "SYNTHETIC/TOY DATA: every metric in this pack was generated from "
    "labeled synthetic inputs to exercise the pipeline. Nothing here is an "
    "experimental result."
)


def git_state() -> tuple[str, bool]:
    """(commit hash, dirty flag); ('unknown', False) outside a repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root(), text=True,
            stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "unknown", False


def capture_environment() -> dict:
    """Reproducibility metadata. Deliberately no hostname/username."""
    env = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python": sys.version.split()[0],
    }
    for pkg in ("torch", "numpy", "pandas", "sapien", "mani_skill",
                "gymnasium", "h5py"):
        try:
            module = __import__(pkg)
            env[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[pkg] = "not installed"
    try:
        import torch

        env["torch_backends"] = {
            "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()),
        }
    except ImportError:
        pass
    return env


@dataclass
class PackManifest:
    project: str
    release_name: str
    synthetic: bool
    command: str
    inputs: list[str]
    schema_version: int = MANIFEST_SCHEMA_VERSION
    generated_utc: str = ""
    git_commit: str = ""
    git_dirty: bool = False
    environment: dict = field(default_factory=dict)
    configs_copied: list[str] = field(default_factory=list)
    metrics_files: list[str] = field(default_factory=list)
    figure_files: list[str] = field(default_factory=list)
    video_count: int = 0
    warning: str = ""
    notes: str = ""

    @classmethod
    def create(cls, *, release_name: str, synthetic: bool, inputs: list[str],
               notes: str = "", command: str | None = None) -> "PackManifest":
        commit, dirty = git_state()
        argv0 = Path(sys.argv[0]).name if sys.argv else ""
        return cls(
            project="xembench",
            release_name=release_name,
            synthetic=synthetic,
            command=command if command is not None
            else " ".join([argv0] + [sanitize_path(a) for a in sys.argv[1:]]),
            inputs=[sanitize_path(p) for p in inputs],
            generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=commit,
            git_dirty=dirty,
            environment=capture_environment(),
            warning=SYNTHETIC_WARNING if synthetic else "",
            notes=notes,
        )

    def validate(self) -> None:
        for name in ("project", "release_name", "generated_utc", "command"):
            if not getattr(self, name):
                raise ValueError(f"manifest field {name!r} must be non-empty")
        if not self.inputs:
            raise ValueError("manifest must list at least one input file")
        if self.synthetic and not self.warning:
            raise ValueError(
                "synthetic packs MUST carry a warning — refusing to label "
                "toy data as real")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} != "
                f"{MANIFEST_SCHEMA_VERSION}")

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PackManifest":
        data = json.loads(Path(path).read_text())
        manifest = cls(**data)
        manifest.validate()
        return manifest

"""Path discipline: SAPIEN chdirs during shader compilation, so everything
that outlives env creation must resolve to an absolute path."""

from pathlib import Path

from xembench.utils.config import repo_root, resolve_path


def test_resolve_path_anchors_relative_at_repo_root():
    p = resolve_path("data/checkpoints/foo")
    assert Path(p).is_absolute()
    assert p == str(repo_root() / "data/checkpoints/foo")


def test_resolve_path_passes_absolute_through():
    assert resolve_path("/tmp/x/y.pt") == "/tmp/x/y.pt"


def test_repo_root_is_the_repo():
    root = repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "xembench").is_dir()

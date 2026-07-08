"""Shared plumbing for flywheel CLIs: overwrite guards, path echo.

Every flywheel command follows the artifact-packer contract: fail loudly
with the fix in the message, never overwrite without --overwrite, sanitize
home paths in anything written into a report.
"""

from pathlib import Path


def guard_overwrite(path: str | Path, overwrite: bool, what: str) -> Path:
    """Refuse to clobber an existing non-empty output without --overwrite."""
    path = Path(path)
    exists = (path.exists() and
              (path.is_file() or (path.is_dir() and any(path.iterdir()))))
    if exists and not overwrite:
        raise FileExistsError(
            f"{what} {path} already exists — pass --overwrite to replace it "
            "(flywheel outputs are pure functions of their inputs, cheap to "
            "regenerate)")
    return path


def echo_written(paths) -> None:
    from xembench.utils.config import sanitize_path
    for p in paths:
        print(f"wrote {sanitize_path(p)}")

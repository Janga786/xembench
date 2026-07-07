"""Index benchmark videos into a markdown table.

Videos follow the naming convention (see artifacts/videos/README.md):

    task__embodiment__policy__split__seed__result.mp4
    e.g. pick_cube_arm__g1__bc__heldout__0__failure_grasp.mp4

    python -m xembench.analysis.video_index \
        --root artifacts/videos --out artifacts/videos/INDEX.md
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

FIELDS = ["task", "embodiment", "policy", "split", "seed", "result"]


@dataclass
class VideoEntry:
    path: Path
    task: str = "?"
    embodiment: str = "?"
    policy: str = "?"
    split: str = "?"
    seed: str = "?"
    result: str = "?"
    well_named: bool = False


def parse_name(path: Path) -> VideoEntry:
    parts = path.stem.split("__")
    if len(parts) != len(FIELDS):
        return VideoEntry(path=path)  # listed, flagged as non-conforming
    return VideoEntry(path=path, **dict(zip(FIELDS, parts)), well_named=True)


def build_index(root: Path) -> list[VideoEntry]:
    return [parse_name(p) for p in sorted(root.rglob("*.mp4"))]


def to_markdown(entries: list[VideoEntry], root: Path) -> str:
    lines = ["# Video index", "",
             f"{len(entries)} videos under `{root}` "
             "(regenerate: `python -m xembench.analysis.video_index`).", ""]
    if not entries:
        lines.append("_No videos yet — Phase B produces them "
                     "(docs/demo_video_plan.md)._")
        return "\n".join(lines) + "\n"
    lines += ["| " + " | ".join(FIELDS + ["file"]) + " |",
              "|" + "---|" * (len(FIELDS) + 1)]
    for e in entries:
        rel = e.path.relative_to(root)
        lines.append(
            f"| {e.task} | {e.embodiment} | {e.policy} | {e.split} | {e.seed} "
            f"| {e.result} | [{rel.name}]({rel}) |")
    bad = [e for e in entries if not e.well_named]
    if bad:
        lines += ["", f"**{len(bad)} file(s) do not follow the naming "
                      "convention** — rename to "
                      "`task__embodiment__policy__split__seed__result.mp4`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/videos")
    parser.add_argument("--out", default="artifacts/videos/INDEX.md")
    args = parser.parse_args(argv)

    from xembench.utils.config import resolve_path

    root = Path(resolve_path(args.root))
    out = Path(resolve_path(args.out))
    entries = build_index(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(entries, root))
    print(f"indexed {len(entries)} videos -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

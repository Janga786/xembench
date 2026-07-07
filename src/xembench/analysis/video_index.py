"""Index benchmark videos into markdown + JSON with grouping and demo planning.

Videos follow the naming convention (see artifacts/videos/README.md):

    task__embodiment__policy__split__seed__result.mp4
    e.g. pick_cube_arm__g1__bc__heldout__0__failure_grasp.mp4

    python -m xembench.analysis.video_index \
        --root artifacts/videos --out artifacts/videos/INDEX.md
    # also writes INDEX.json next to the markdown

The JSON payload groups videos by task/embodiment/policy/split/result and
suggests side-by-side demo pairings (cross-embodiment, matched policy+split)
and paraphrase sets (same task+embodiment+policy across splits) — planning
aids for docs/demo_video_plan.md, no video editing involved.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

FIELDS = ["task", "embodiment", "policy", "split", "seed", "result"]
INDEX_SCHEMA_VERSION = 1


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


def group_videos(entries: list[VideoEntry], root: Path) -> dict:
    """{axis: {value: [relative paths]}} over well-named videos, sorted."""
    groups: dict[str, dict[str, list[str]]] = {f: {} for f in FIELDS[:-2]}
    groups["result"] = {}
    for e in entries:
        if not e.well_named:
            continue
        rel = str(e.path.relative_to(root))
        for axis in groups:
            groups[axis].setdefault(getattr(e, axis), []).append(rel)
    return {axis: dict(sorted(vals.items())) for axis, vals in groups.items()}


def plan_side_by_side(entries: list[VideoEntry], root: Path) -> dict:
    """Demo-planning suggestions (no editing): cross-embodiment pairs with
    matched policy+split, and paraphrase sets varying only the split."""
    named = [e for e in entries if e.well_named]
    pairs = []
    for a, b in combinations(named, 2):
        if (a.embodiment != b.embodiment and a.policy == b.policy
                and a.split == b.split):
            pairs.append({
                "left": str(a.path.relative_to(root)),
                "right": str(b.path.relative_to(root)),
                "reason": f"cross-embodiment ({a.embodiment} vs {b.embodiment}), "
                          f"matched policy={a.policy} split={a.split}",
            })
    para: dict[tuple, list[str]] = {}
    for e in named:
        para.setdefault((e.task, e.embodiment, e.policy), []).append(
            str(e.path.relative_to(root)))
    paraphrase_sets = [
        {"task": k[0], "embodiment": k[1], "policy": k[2], "videos": sorted(v)}
        for k, v in sorted(para.items())
        if len({p.split for p in named
                if (p.task, p.embodiment, p.policy) == k}) > 1
    ]
    return {"cross_embodiment_pairs": pairs, "paraphrase_sets": paraphrase_sets}


def to_json(entries: list[VideoEntry], root: Path) -> dict:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "root": root.name,  # no absolute/home paths in artifacts
        "n_videos": len(entries),
        "videos": [
            {**{f: getattr(e, f) for f in FIELDS},
             "file": str(e.path.relative_to(root)),
             "well_named": e.well_named}
            for e in entries
        ],
        "groups": group_videos(entries, root),
        "demo_planning": plan_side_by_side(entries, root),
        "unparsed": [str(e.path.relative_to(root))
                     for e in entries if not e.well_named],
    }


def to_markdown(entries: list[VideoEntry], root: Path) -> str:
    from xembench.utils.config import sanitize_path

    lines = ["# Video index", "",
             f"{len(entries)} videos under `{sanitize_path(root)}` "
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
    parser.add_argument("--json-out", default=None,
                        help="default: <out> with .json suffix")
    args = parser.parse_args(argv)

    from xembench.utils.config import resolve_path

    root = Path(resolve_path(args.root))
    out = Path(resolve_path(args.out))
    json_out = Path(resolve_path(args.json_out)) if args.json_out \
        else out.with_suffix(".json")
    entries = build_index(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(entries, root))
    json_out.write_text(json.dumps(to_json(entries, root), indent=1))
    print(f"indexed {len(entries)} videos -> {out} + {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

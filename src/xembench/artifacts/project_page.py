"""Project-page assets: portfolio-ready summaries derived from a pack.

Everything is generated from the episode CSV + manifest — numbers can't
drift from the underlying data, and synthetic data stamps every artifact.
"""

import json
import math
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import is_transfer
from xembench.artifacts.figures import retention_data, success_by_group
from xembench.artifacts.manifest import PackManifest

OVERCLAIM_NOTE = (
    "> **Scope note:** report only what the data supports. Zero-shot "
    "transfer numbers are for the specific task pairs and conditions "
    "listed; this benchmark measures transfer, it does not claim to solve "
    "it."
)


def key_metrics(df: pd.DataFrame, synthetic: bool) -> dict:
    ret = retention_data(df)
    def _round(x: float) -> float | None:
        return round(float(x), 4) if x == x else None  # NaN -> None

    # itertuples: namedtuple attribute access is safe here (a pandas Series
    # would shadow 'drop' with the method — that bug bit once already)
    pairs = [
        {
            "pair": f"{r.task}→{r.eval_task}",
            "policy": r.policy_type,
            "sr_native": _round(r.sr_native),
            "sr_transfer": _round(r.sr_transfer),
            "drop": _round(r.drop),
            "retention": _round(r.retention),
        }
        for r in ret.itertuples()
    ]
    # paraphrase delta per (policy, task, color_cond): the comparison is only
    # meaningful with the color condition held fixed (docs/metrics.md)
    native = df[~is_transfer(df)]
    deltas = []
    for (policy, task, color), g in native.groupby(
            ["policy_type", "task", "color_cond"], sort=True):
        conds = g.groupby("instruction_cond")["success"].mean()
        if {"canonical", "heldout"} <= set(conds.index):
            deltas.append({
                "policy": policy, "task": task, "color_cond": color,
                "paraphrase_delta": round(
                    float(conds["canonical"] - conds["heldout"]), 4),
            })
    return {
        "synthetic": synthetic,
        "n_episodes": int(len(df)),
        "n_transfer_episodes": int(is_transfer(df).sum()),
        "transfer_pairs": pairs,
        "paraphrase_deltas": deltas,
    }


def result_cards(metrics: dict) -> list[dict]:
    """Small tiles for a portfolio page. Synthetic flag on every card."""
    cards = [{
        "title": "Episodes evaluated",
        "value": str(metrics["n_episodes"]),
        "subtitle": f"{metrics['n_transfer_episodes']} cross-embodiment",
        "synthetic": metrics["synthetic"],
    }]
    for p in metrics["transfer_pairs"]:
        retention = p["retention"]
        cards.append({
            "title": f"Transfer retention: {p['pair']}",
            "value": f"{retention:.0%}" if retention is not None else "n/a",
            "subtitle": (f"native {p['sr_native']:.0%} → transfer "
                         f"{p['sr_transfer']:.0%} ({p['policy']})"
                         if p["sr_native"] is not None else "no native cell"),
            "synthetic": metrics["synthetic"],
        })
    for d in metrics["paraphrase_deltas"]:
        cards.append({
            "title": f"Paraphrase robustness: {d['task']}",
            "value": f"{-d['paraphrase_delta']:+.0%}",
            "subtitle": f"heldout vs canonical instructions ({d['policy']})",
            "synthetic": metrics["synthetic"],
        })
    return cards


def _fmt(x: float | None, spec: str = ".0%") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return format(x, spec)


def project_summary_md(df: pd.DataFrame, manifest: PackManifest,
                       metrics: dict) -> str:
    banner = (f"> **{manifest.warning}**\n\n" if manifest.warning else "")
    emb = sorted(df["eval_embodiment"].unique())
    tasks = sorted(df["eval_task"].unique())
    policies = sorted(df["policy_type"].unique())
    pair_rows = "\n".join(
        f"| {p['pair']} | {p['policy']} | {_fmt(p['sr_native'])} | "
        f"{_fmt(p['sr_transfer'])} | {_fmt(p['drop'])} | {_fmt(p['retention'])} |"
        for p in metrics["transfer_pairs"]) or "| _no transfer cells_ | | | | | |"
    return f"""# xembench — results summary ({manifest.release_name})

{banner}**Thesis.** When language, task structure, and policy architecture are
shared, how much manipulation performance survives a change of robot body —
and where does transfer fail? One policy network speaks a canonical 21-dim
action interface; per-robot adapters translate to a Panda arm and a Unitree
G1 humanoid upper body (ManiSkill3). Transfer test = same weights, swapped
adapter.

**What was evaluated.** {metrics['n_episodes']} episodes
({metrics['n_transfer_episodes']} cross-embodiment) from
`{manifest.inputs[0]}` at commit `{manifest.git_commit[:12]}`.

- **Embodiments:** {', '.join(emb)}
- **Tasks:** {', '.join(tasks)}
- **Baselines:** {', '.join(policies)}

## Key metrics

| Pair | Policy | Native SR | Transfer SR | Drop | Retention |
|---|---|---|---|---|---|
{pair_rows}

<!-- TODO(Janga): once real Phase B numbers exist, add 2-3 sentences of
interpretation here: which direction transfers better, and whether failures
are grounding (wrong object) or motor (reach/grasp) — see failure_report.md -->

## Figures and videos

See [figure_index.md](figure_index.md) and [video_index.md](video_index.md).

## Important limitations

- Hand-designed canonical interface: kinematic aliasing between 7-DOF and
  5-DOF arms is real; normalized joint deltas are not metrically equivalent.
- Analogous (not identical) transfer tasks; camera viewpoint differs across
  embodiments (ablatable, not eliminated).
- Small closed-vocabulary language distribution; simulation only; two
  embodiments.

{OVERCLAIM_NOTE}
"""


def figure_index_md(figures: list[Path], pack_dir: Path,
                    synthetic: bool) -> str:
    lines = ["# Figure index", ""]
    if synthetic:
        lines += ["> All figures generated from SYNTHETIC toy data.", ""]
    pngs = [f for f in figures if f.suffix == ".png"]
    if not pngs:
        lines.append("_No figures (matplotlib unavailable at build time — "
                     "rebuild the pack where it is installed)._")
    for f in sorted(pngs):
        rel = f.relative_to(pack_dir)
        lines += [f"## {f.stem.replace('_', ' ')}", f"![{f.stem}](../{rel})",
                  f"(SVG: `../{rel.with_suffix('.svg')}`)", ""]
    return "\n".join(lines) + "\n"


def build_page_assets(df: pd.DataFrame, pack_dir: Path,
                      manifest: PackManifest, figures: list[Path],
                      video_payload: dict | None) -> list[Path]:
    out = pack_dir / "project_page_assets"
    out.mkdir(parents=True, exist_ok=True)
    metrics = key_metrics(df, manifest.synthetic)

    written = []

    def write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text)
        written.append(path)

    write("project_summary.md", project_summary_md(df, manifest, metrics))
    write("key_metrics.json", json.dumps(metrics, indent=1))
    write("result_cards.json", json.dumps(result_cards(metrics), indent=1))
    write("figure_index.md", figure_index_md(figures, pack_dir,
                                             manifest.synthetic))
    if video_payload and video_payload.get("n_videos"):
        vid_lines = ["# Video index", "",
                     f"{video_payload['n_videos']} videos "
                     "(see ../videos/video_index.json for grouping and "
                     "side-by-side demo planning)", ""]
        for v in video_payload["videos"]:
            vid_lines.append(f"- `{v['file']}` — {v['task']} / "
                             f"{v['embodiment']} / {v['split']} / {v['result']}")
        write("video_index.md", "\n".join(vid_lines) + "\n")
    else:
        write("video_index.md",
              "# Video index\n\n_No videos in this pack — Phase B records "
              "them (docs/demo_video_plan.md)._\n")
    return written

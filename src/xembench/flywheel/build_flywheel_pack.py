"""Build a flywheel artifact pack from a round manifest.

    python -m xembench.flywheel.build_flywheel_pack \
        --round-manifest artifacts/flywheel/toy_round/round_manifest.json \
        --out artifacts/releases/flywheel_toy_pack --synthetic

Reuses the existing artifact-pack machinery (PackManifest provenance,
checksum logic, path sanitization, synthetic rails) and adds the
flywheel-specific sections: dataset card, validation report + quality-issue
CSV, candidates, selection/collection plans, round comparison, model-card
placeholder, flywheel summary, and project-page assets.

Safety rails (same contract as xembench.artifacts.build_pack):
- a synthetic round manifest (or synthetic dataset card) without
  --synthetic is REFUSED;
- existing non-empty packs need --overwrite;
- no home paths in anything written.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from xembench.artifacts.build_pack import _checksums
from xembench.artifacts.manifest import PackManifest
from xembench.flywheel.schemas import FlywheelRoundManifest, load_json
from xembench.utils.config import resolve_path, sanitize_path

SYNTHETIC_LINE = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
                  "Every number in this pack is fake by construction.\n\n")

# (glob in the round directory, destination subdir in the pack)
SECTIONS = [
    ("*dataset_card*", "."),
    ("*validation_report*", "."),
    ("*failure_report*", "."),
    ("*model_card*", "."),          # placeholder and filled (real) cards
    ("*_vs_*", "."),                       # round comparisons
    ("*candidates*", "candidates"),
    ("*selection_plan*", "plans"),
    ("*collection_plan*", "plans"),
    ("*linux_commands.sh", "plans"),
    ("*eval*.csv", "raw"),
    ("*dataset_index*", "index"),
    ("TOY_DATA_README.txt", "raw"),
]


def _load_optional_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _collect_files(round_dir: Path, pack_dir: Path) -> list[Path]:
    copied = []
    for pattern, sub in SECTIONS:
        for src in sorted(round_dir.glob(pattern)):
            if not src.is_file():
                continue
            dest_dir = pack_dir if sub == "." else pack_dir / sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():  # first matching pattern wins
                shutil.copy2(src, dest)
                copied.append(dest)
    return copied


def _quality_issue_csv(pack_dir: Path) -> Path | None:
    report = next(iter(sorted(pack_dir.glob("*validation_report*.json"))),
                  None)
    if report is None:
        return None
    issues = _load_optional_json(report).get("issues", [])
    rows = [{k: v for k, v in i.items() if not k.startswith("_")}
            for i in issues]
    out = pack_dir / "data_quality_issues.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _round_stats(round_dir: Path, manifest: FlywheelRoundManifest) -> dict:
    """Pull headline numbers from the round's JSON files (all optional)."""
    stats: dict = {"round_id": manifest.round_id, "stage": manifest.stage}
    card = _load_optional_json(
        next(iter(sorted(round_dir.glob("*dataset_card*.json"))),
             round_dir / "missing"))
    if card:
        stats["dataset"] = {
            "total_episodes": card.get("total_episodes"),
            "successful_episodes": card.get("successful_episodes"),
            "failed_episodes": card.get("failed_episodes"),
            "tasks": card.get("tasks", []),
            "embodiments": card.get("embodiments", []),
        }
    validation = _load_optional_json(
        next(iter(sorted(round_dir.glob("*validation_report*.json"))),
             round_dir / "missing"))
    if validation:
        stats["validation"] = validation.get("summary", {})
    plan = _load_optional_json(Path(round_dir) / Path(
        manifest.selection_plan_path).name) if manifest.selection_plan_path \
        else {}
    if plan:
        stats["selection"] = {
            "strategy": plan.get("strategy"),
            "selected": len(plan.get("selected_ids", [])),
            "candidates_total": plan.get("candidates_total"),
            "budget_episodes": plan.get("budget_episodes"),
        }
    comparison = _load_optional_json(
        next(iter(sorted(round_dir.glob("*_vs_*.json"))),
             round_dir / "missing"))
    if comparison:
        stats["comparison"] = {
            "overall_delta": comparison.get("overall_delta"),
            "improvement_per_demo": comparison.get("improvement_per_demo"),
            "demos_added": comparison.get("demos_added"),
            "by_split": comparison.get("by_split", []),
        }
    return stats


def _summary_md(stats: dict, synthetic: bool) -> str:
    banner = SYNTHETIC_LINE if synthetic else ""
    lines = [f"# Flywheel round summary — {stats['round_id']}", "",
             banner + f"Round stage: **{stats['stage']}**.", ""]
    ds = stats.get("dataset")
    if ds:
        lines += [
            "## Dataset", "",
            f"- {ds['total_episodes']} episodes "
            f"({ds['successful_episodes']} success / "
            f"{ds['failed_episodes']} failure)",
            f"- tasks: {', '.join(ds['tasks'])}",
            f"- embodiments: {', '.join(ds['embodiments'])}", ""]
    val = stats.get("validation")
    if val:
        lines += [
            "## Validation", "",
            f"- records by status: "
            + (", ".join(f"{k}={v}" for k, v in
                         sorted(val.get("records_by_status", {}).items()))
               or "n/a"),
            f"- mean quality score: {val.get('mean_quality_score')}", ""]
    sel = stats.get("selection")
    if sel:
        lines += [
            "## Selection", "",
            f"- strategy **{sel['strategy']}**: {sel['selected']}/"
            f"{sel['candidates_total']} candidates under a budget of "
            f"{sel['budget_episodes']} episodes", ""]
    cmp_ = stats.get("comparison")
    if cmp_ and cmp_.get("overall_delta") is not None:
        lines += [
            "## Round-over-round comparison", "",
            f"- overall SR delta: **{cmp_['overall_delta']:+.4f}**"
            + (f" ({cmp_['demos_added']} demos added, "
               f"{cmp_['improvement_per_demo']:+.6f} per demo)"
               if cmp_.get("improvement_per_demo") is not None else ""),
            "", "| split | delta |", "|---|---|"]
        lines += [f"| {r['split']} | {r['delta']:+.3f} |"
                  for r in cmp_["by_split"]]
        lines.append("")
    lines += ["## Contents", "",
              "See `README.md` for the file inventory and `manifest.json` "
              "for provenance."]
    return "\n".join(lines) + "\n"


def _result_cards(stats: dict, synthetic: bool) -> list[dict]:
    cards = [{"title": "Flywheel round", "value": stats["round_id"],
              "subtitle": f"stage: {stats['stage']}",
              "synthetic": synthetic}]
    ds = stats.get("dataset")
    if ds:
        cards.append({
            "title": "Dataset episodes",
            "value": str(ds["total_episodes"]),
            "subtitle": f"{ds['successful_episodes']} success / "
                        f"{ds['failed_episodes']} failure",
            "synthetic": synthetic})
    val = stats.get("validation")
    if val and val.get("mean_quality_score") is not None:
        cards.append({
            "title": "Mean data quality score",
            "value": f"{val['mean_quality_score']:.3f}",
            "subtitle": ", ".join(
                f"{k}={v}" for k, v in
                sorted(val.get("records_by_status", {}).items())),
            "synthetic": synthetic})
    sel = stats.get("selection")
    if sel:
        cards.append({
            "title": "Demos requested",
            "value": str(sel["selected"]),
            "subtitle": f"of {sel['candidates_total']} candidates "
                        f"({sel['strategy']})",
            "synthetic": synthetic})
    cmp_ = stats.get("comparison")
    if cmp_ and cmp_.get("overall_delta") is not None:
        cards.append({
            "title": "SR delta after round",
            "value": f"{cmp_['overall_delta']:+.1%}",
            "subtitle": (f"{cmp_['improvement_per_demo']:+.4%} per demo"
                         if cmp_.get("improvement_per_demo") is not None
                         else "no demo count recorded"),
            "synthetic": synthetic})
    return cards


def _pack_readme(manifest: PackManifest, round_id: str) -> str:
    banner = f"> **{manifest.warning}**\n\n" if manifest.warning else ""
    return f"""# xembench flywheel pack — {manifest.release_name}

{banner}Flywheel round `{round_id}`, generated {manifest.generated_utc} at
commit `{manifest.git_commit[:12]}`{' (dirty tree)' if manifest.git_dirty
else ''} by `{manifest.command}`.

| File | What |
|---|---|
| `manifest.json` | provenance: environment, versions, inputs, git state |
| `round_manifest.json` | the flywheel round's file inventory |
| `flywheel_summary.md` | headline numbers for the round |
| `*dataset_card*` | dataset composition, sources, validation state |
| `*validation_report*` + `data_quality_issues.csv` | quality findings |
| `*failure_report*` | failure-taxonomy breakdown of the round's eval |
| `*_vs_*` | round-over-round comparison |
| `model_card_placeholder.md` | slot for real Linux training results |
| `candidates/` | failure-driven demo requests |
| `plans/` | selection plan + Linux command plan (nothing executed) |
| `raw/` | eval episode CSVs, verbatim |
| `index/` | dataset index (JSONL + episodes + summary CSV) |
| `project_page_assets/` | portfolio summary + result cards |
| `checksums.txt` | sha256 of every file above |

Packs are pure functions of the round directory — cheap to regenerate.
Design and interpretation guidance: docs/physical_ai_data_flywheel.md.
"""


def build_flywheel_pack(args: argparse.Namespace) -> Path:
    manifest_path = Path(resolve_path(args.round_manifest))
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"--round-manifest {manifest_path} does not exist. Generate a "
            "round first (e.g. `make flywheel-toy`).")
    round_manifest = load_json(FlywheelRoundManifest, manifest_path)
    round_dir = manifest_path.parent

    card = _load_optional_json(
        next(iter(sorted(round_dir.glob("*dataset_card*.json"))),
             round_dir / "missing"))
    data_is_synthetic = round_manifest.is_synthetic or \
        bool(card.get("synthetic"))
    if data_is_synthetic and not args.synthetic:
        raise SystemExit(
            "REFUSED: this round is marked synthetic (round manifest / "
            "dataset card) but --synthetic was not passed. Toy rounds "
            "cannot be packaged as real results.")

    pack_dir = Path(resolve_path(args.out))
    if pack_dir.exists() and any(pack_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{pack_dir} already exists and is not empty — pass "
                "--overwrite to replace it (packs are cheap to regenerate).")
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    copied = _collect_files(round_dir, pack_dir)
    shutil.copy2(manifest_path, pack_dir / "round_manifest.json")
    _quality_issue_csv(pack_dir)

    stats = _round_stats(round_dir, round_manifest)
    (pack_dir / "flywheel_summary.md").write_text(
        _summary_md(stats, args.synthetic))
    assets = pack_dir / "project_page_assets"
    assets.mkdir(exist_ok=True)
    (assets / "flywheel_summary.md").write_text(
        _summary_md(stats, args.synthetic))
    (assets / "flywheel_result_cards.json").write_text(
        json.dumps(_result_cards(stats, args.synthetic), indent=1) + "\n")

    manifest = PackManifest.create(
        release_name=args.release_name or pack_dir.name,
        synthetic=args.synthetic,
        inputs=[str(manifest_path)],
        notes=args.notes or "")
    manifest.metrics_files = sorted(
        str(p.relative_to(pack_dir)) for p in pack_dir.rglob("*.md")
        if p.is_file())
    (pack_dir / "README.md").write_text(
        _pack_readme(manifest, round_manifest.round_id))
    manifest.save(pack_dir / "manifest.json")
    _checksums(pack_dir)

    label = "SYNTHETIC" if args.synthetic else "real-data"
    print(f"built {label} flywheel pack -> {sanitize_path(pack_dir)}")
    print(f"  {len(copied) + 2} files collected from "
          f"{sanitize_path(round_dir)}")
    return pack_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round-manifest", required=True,
                        help="round_manifest.json of the round to package")
    parser.add_argument("--out", required=True,
                        help="pack directory, e.g. "
                             "artifacts/releases/flywheel_round1")
    parser.add_argument("--release-name", default=None,
                        help="default: basename of --out")
    parser.add_argument("--synthetic", action="store_true",
                        help="REQUIRED for toy/synthetic rounds; stamps "
                             "warnings everywhere")
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_flywheel_pack(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

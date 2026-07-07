"""Build a complete artifact pack from an episode-level eval CSV.

    python -m xembench.artifacts.build_pack \
        --eval-csv data/results/eval_episodes.csv \
        --out artifacts/releases/phase_b_v1 --release-name phase_b_v1 \
        --configs configs/presets/full_eval_matrix.yaml \
        --videos artifacts/videos

Output layout (see docs/artifact_packs.md):

    artifacts/releases/<name>/
      README.md  manifest.json  checksums.txt
      metrics_summary.md  transfer_matrix.{csv,md}  failure_report.md
      figures/  videos/  configs/  logs/  raw/  project_page_assets/

Safety rails:
- refuses to run on toy-stamped data without --synthetic (toy data can
  never masquerade as real results);
- --synthetic stamps the manifest, pack README, figures, and page assets;
- never overwrites an existing pack without --overwrite;
- fails loudly on missing inputs.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from xembench.analysis import failure_report, summarize_results
from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.analysis.transfer_matrix import compute_transfer, to_markdown
from xembench.analysis.video_index import build_index, to_json
from xembench.analysis.video_index import to_markdown as videos_md
from xembench.artifacts.figures import render_all
from xembench.artifacts.manifest import PackManifest, sanitize_path
from xembench.artifacts.project_page import build_page_assets
from xembench.utils.config import resolve_path

SUBDIRS = ["figures", "videos", "configs", "logs", "raw", "project_page_assets"]


def _pack_readme(manifest: PackManifest, video_count: int) -> str:
    banner = f"> **{manifest.warning}**\n\n" if manifest.warning else ""
    return f"""# xembench artifact pack — {manifest.release_name}

{banner}Generated {manifest.generated_utc} at commit
`{manifest.git_commit[:12]}`{' (dirty tree)' if manifest.git_dirty else ''}
from `{manifest.inputs[0]}` by `{manifest.command}`.

| File | What |
|---|---|
| `manifest.json` | provenance: environment, versions, inputs, git state |
| `metrics_summary.md` | success rates + CIs per eval cell |
| `transfer_matrix.csv` / `.md` | the headline cross-embodiment numbers |
| `failure_report.md` | failure-taxonomy breakdowns |
| `figures/` | PNG+SVG report figures |
| `videos/video_index.{{md,json}}` | indexed rollout videos ({video_count} found; mp4s are referenced, not copied) |
| `configs/` | config snapshots used for the runs |
| `logs/` | optional run logs |
| `raw/` | the input episode CSV, verbatim |
| `project_page_assets/` | portfolio-ready summaries, metric cards, indexes |
| `checksums.txt` | sha256 of every file above |

Regenerate anytime: packs are pure functions of the input CSV + configs.
Interpretation guidance: docs/how_to_write_results_section.md in the repo.
"""


def _checksums(pack_dir: Path) -> Path:
    lines = []
    for f in sorted(pack_dir.rglob("*")):
        if f.is_file() and f.name != "checksums.txt":
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{digest}  {f.relative_to(pack_dir)}")
    out = pack_dir / "checksums.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def build_pack(args: argparse.Namespace) -> Path:
    eval_csv = Path(resolve_path(args.eval_csv))
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"--eval-csv {eval_csv} does not exist. Run an evaluation "
            "(episodes_csv config key) or `make toy-report` for synthetic "
            "data first.")
    pack_dir = Path(resolve_path(args.out))
    if pack_dir.exists() and any(pack_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{pack_dir} already exists and is not empty — pass "
                "--overwrite to replace it (packs are cheap to regenerate).")
        shutil.rmtree(pack_dir)

    df = load_episodes(eval_csv)  # validates schema + failure labels
    data_is_toy = bool(toy_banner(df))
    if data_is_toy and not args.synthetic:
        raise SystemExit(
            "REFUSED: the input CSV is stamped as synthetic toy data "
            "(TOY_DATA checkpoint marker) but --synthetic was not passed. "
            "Toy data cannot be packaged as real results.")

    release = args.release_name or pack_dir.name
    for sub in SUBDIRS:
        (pack_dir / sub).mkdir(parents=True, exist_ok=True)

    # raw input, verbatim
    shutil.copy2(eval_csv, pack_dir / "raw" / "eval_episodes.csv")

    # metrics (existing analysis tooling, outputs redirected into the pack)
    summarize_results.main(["--input", str(eval_csv),
                            "--out", str(pack_dir / "metrics_summary.md")])
    failure_report.main(["--input", str(eval_csv),
                         "--out", str(pack_dir / "failure_report.md")])
    matrix = compute_transfer(df)
    matrix.to_csv(pack_dir / "transfer_matrix.csv", index=False)
    (pack_dir / "transfer_matrix.md").write_text(
        toy_banner(df) + to_markdown(matrix))

    # figures
    figures = render_all(df, pack_dir / "figures", synthetic=args.synthetic)

    # configs
    configs_copied = []
    for cfg in args.configs or []:
        src = Path(resolve_path(cfg))
        if not src.exists():
            raise FileNotFoundError(f"--configs {src} does not exist")
        shutil.copy2(src, pack_dir / "configs" / src.name)
        configs_copied.append(sanitize_path(src))

    # logs (optional)
    for log in args.logs or []:
        src = Path(resolve_path(log))
        if src.exists():
            shutil.copy2(src, pack_dir / "logs" / src.name)

    # videos: index + planning JSON; mp4s referenced, never copied
    video_payload = None
    video_count = 0
    if args.videos:
        vroot = Path(resolve_path(args.videos))
        entries = build_index(vroot) if vroot.exists() else []
        video_payload = to_json(entries, vroot) if vroot.exists() else None
        video_count = len(entries)
        (pack_dir / "videos" / "video_index.md").write_text(
            videos_md(entries, vroot) if vroot.exists()
            else "# Video index\n\n_video root not found_\n")
        if video_payload is not None:
            import json

            (pack_dir / "videos" / "video_index.json").write_text(
                json.dumps(video_payload, indent=1))

    manifest = PackManifest.create(
        release_name=release, synthetic=args.synthetic,
        inputs=[str(eval_csv)], notes=args.notes or "")
    manifest.configs_copied = configs_copied
    manifest.metrics_files = ["metrics_summary.md", "transfer_matrix.csv",
                              "transfer_matrix.md", "failure_report.md"]
    manifest.figure_files = [str(f.relative_to(pack_dir)) for f in figures]
    manifest.video_count = video_count

    build_page_assets(df, pack_dir, manifest, figures, video_payload)
    (pack_dir / "README.md").write_text(_pack_readme(manifest, video_count))
    manifest.save(pack_dir / "manifest.json")
    _checksums(pack_dir)

    label = "SYNTHETIC" if args.synthetic else "real-data"
    print(f"built {label} artifact pack -> {pack_dir}")
    print(f"  {len(figures)} figure files, {video_count} videos indexed, "
          f"{len(configs_copied)} configs copied")
    return pack_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-csv", required=True,
                        help="episode-level eval CSV (schema v1 or v2)")
    parser.add_argument("--out", required=True,
                        help="pack directory, e.g. artifacts/releases/<name>")
    parser.add_argument("--release-name", default=None,
                        help="default: basename of --out")
    parser.add_argument("--synthetic", action="store_true",
                        help="REQUIRED when the input is toy/synthetic data; "
                             "stamps warnings everywhere")
    parser.add_argument("--configs", action="append", default=[],
                        metavar="YAML", help="config file to snapshot (repeatable)")
    parser.add_argument("--logs", action="append", default=[],
                        metavar="FILE", help="log file to include (repeatable)")
    parser.add_argument("--videos", default=None,
                        help="video directory to index (mp4s referenced, not copied)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_pack(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

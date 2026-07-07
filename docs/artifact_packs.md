# Artifact packs — turning eval outputs into report/portfolio bundles

One command converts an episode-level eval CSV into a complete, checksummed,
provenance-stamped artifact pack. Packs are pure functions of their inputs:
cheap to regenerate, safe to delete, gitignored by default
(`artifacts/releases/`).

## The command (after Linux Phase B, step 13 of the runbook)

```bash
uv run python -m xembench.artifacts.build_pack \
  --eval-csv data/results/eval_episodes.csv \
  --out artifacts/releases/phase_b_v1 \
  --release-name phase_b_v1 \
  --configs configs/presets/full_eval_matrix.yaml \
  --configs configs/tasks.yaml \
  --videos artifacts/videos \
  --notes "first full Phase B eval, 3 seeds"
```

Mac rehearsal with synthetic data (identical code path, loud labels):

```bash
make artifact-pack-toy        # -> artifacts/releases/toy_pack/
```

## What a pack contains

```
artifacts/releases/<name>/
  README.md                  what/when/how, with synthetic banner if applicable
  manifest.json              provenance: git commit+dirty, env/package versions,
                             command, inputs, synthetic flag + warning
  metrics_summary.md         success rates + Wilson CIs per eval cell
  transfer_matrix.csv/.md    the headline cross-embodiment numbers
  failure_report.md          failure-taxonomy breakdowns
  figures/                   PNG+SVG: retention bars, success by embodiment,
                             failure counts, task x policy heatmap
  videos/video_index.{md,json}   indexed videos + side-by-side demo planning
                                 (mp4s referenced, never copied)
  configs/                   snapshots of the configs you passed
  logs/                      optional run logs (--logs)
  raw/eval_episodes.csv      the input, verbatim
  project_page_assets/       portfolio-ready: project_summary.md,
                             key_metrics.json, result_cards.json,
                             figure_index.md, video_index.md
  checksums.txt              sha256 of every file
```

## Safety rails (enforced, tested)

- Toy-stamped input **without `--synthetic` is refused** — synthetic data
  cannot be packaged as real results.
- `--synthetic` stamps the manifest warning, pack README, figure titles,
  and every project-page card.
- Existing non-empty packs are never overwritten without `--overwrite`.
- Missing inputs/configs fail loudly with the fix in the message.
- Manifests contain no usernames/home paths (`sanitize_path`).

## Using project_page_assets later

`project_summary.md` is written to be pasted into a portfolio page (thesis,
what was evaluated, key-metric table, limitations, overclaim note).
`result_cards.json` is machine-readable tiles (`title/value/subtitle/
synthetic`) for a website; `key_metrics.json` has the underlying numbers.
Writing the report itself: reports/templates/report_skeleton.md +
docs/how_to_write_results_section.md.

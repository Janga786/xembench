# [Report title] — results section skeleton

<!-- Fill from an artifact pack: artifacts/releases/<name>/. Every insert
point names its exact source file. Pair with
docs/how_to_write_results_section.md for the no-overclaiming rules.
Structure mirrors docs/report_outline.md sections 5-7. -->

## Results

### Native performance
INSERT: the native rows of `<pack>/metrics_summary.md` (task = eval_task),
grouped by policy. Quote success rates WITH their Wilson CIs, e.g.
"74% [65%, 82%]". Figure: `<pack>/figures/task_success_heatmap.png`.

Interpretation prompts (answer, don't speculate):
- Which tasks are saturated / floored? (CI touching 0% or 100%?)
- Did PPO fine-tuning beat BC on the same task, outside overlapping CIs?

### Cross-embodiment transfer (headline)
INSERT: `<pack>/transfer_matrix.md` verbatim as the main table.
Figure: `<pack>/figures/transfer_retention.png`.

Report BOTH directions of BOTH pairs; do not average directions. State
drop in percentage points and retention as a ratio. If a native SR is near
zero, say the transfer cell is uninformative rather than reporting a ratio.

### Language robustness
INSERT: paraphrase deltas from
`<pack>/project_page_assets/key_metrics.json` (`paraphrase_deltas`,
matched color condition) and the unseen-color rows of
`metrics_summary.md`.

### Efficiency
INSERT: `avg_steps_to_success` / `avg_steps_frac` columns from
`metrics_summary.md` — successes only; compare via the fraction, never raw
steps (budgets differ: 50 vs 100).

## Failure analysis
INSERT: `<pack>/failure_report.md` tables (by eval task, by train→eval
embodiment) + `<pack>/figures/failure_counts.png`.
INSERT: video montage links from `<pack>/videos/video_index.md` (use the
`demo_planning` block in video_index.json for side-by-side picks).

Answer: does transfer fail at grounding (wrong_object_selected) or motor
control (reach/grasp)? NOTE: if the run predates richer instrumentation,
most failures will read `timeout` — say that plainly, do not reinterpret.

## Limitations
Copy the limitations block from
`<pack>/project_page_assets/project_summary.md`, then add anything
run-specific (seeds, incomplete cells, expert quality bounds).

## Future work
Pull from docs/report_outline.md §8; keep only items the results actually
motivate.

## Reproducibility appendix
INSERT: `<pack>/manifest.json` fields (commit, dirty flag, versions,
command), `<pack>/configs/` snapshots, seed list, and the exact runbook
steps used (docs/phase_b_commands.md numbers).

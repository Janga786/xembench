# reports/

Generated analysis outputs (markdown/CSV summaries, small plots). Contents:

- `example_*` — committed documentation examples generated from **synthetic
  toy data** (`make toy-report`). Every number in them is fake by
  construction and banner-labeled as such.
- `summary.md`, `failures.md`, `transfer_matrix.{csv,md}` — real results,
  produced in Phase B by `uv run python scripts/make_report_assets.py`
  after the eval matrix has run. Not committed until they exist.
- `assets/` — plots (gitignored; regenerate on demand).

Regeneration is always cheap and deterministic from the episode CSVs in
`data/results/` — treat everything here as build artifacts, not sources.

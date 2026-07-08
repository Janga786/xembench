# How to write the flywheel results section (when real numbers exist)

Companion to [how_to_write_results_section.md](how_to_write_results_section.md)
(which covers the benchmark's transfer numbers). This page covers the
flywheel's data-centric claims. **Nothing can be written until a real
(non-synthetic) flywheel pack exists.**

## Source-of-truth rule

Every number comes from a file in `artifacts/releases/flywheel_round*/`
with `"synthetic": false` in its `manifest.json`:

| Claim | Source file |
|---|---|
| improvement per demonstration | `round0_vs_round1*.json` (`improvement_per_demo`) |
| per-split deltas | same (`by_split`) |
| dataset composition | `round0_dataset_card.json` |
| quality/issue statistics | `*validation_report.json`, `data_quality_issues.csv` |
| what was selected and why | `plans/round1_selection_plan.json` (`objective_terms`) |
| failure-mode shifts | `*failure_report.md` diffs between rounds |

If a claim has no source file, it does not go in the report.

## Order of presentation

1. **The comparison table first** (arms × splits, ΔSR ± CI) — readers
   should see targeted-vs-random before any narrative.
2. Improvement-per-demo, overall then per split. State N and the CI
   machinery (Wilson, from docs/metrics.md).
3. Cost-adjusted view if G1 episodes dominated the targeted arm's budget
   (per-episode AND per-cost-unit).
4. Validation story: what the noisy arm did, what filtering recovered,
   validation recall on injected corruption.
5. Failure-mode shifts: did the targeted labels actually shrink?

## Phrases to use / avoid

Use: "at this sample size", "on these six tasks / two embodiments",
"within/outside the 95% interval", "the targeted arm selected mostly
[labels], because [objective term]".

Avoid: "proves", "solves", "state-of-the-art", "significantly" without a
test, generalizations beyond the benchmark, and any sentence where a toy
number could be mistaken for a measured one.

## Nulls and negatives

"Targeted ≈ random at N=50" is a publishable observation about acquisition
heuristics at small budgets — report it with the same table and prominence
as a win would get, then say what budget/labels the next round changes.

## Checklist before publishing

- [ ] every number traces to a non-synthetic pack file
- [ ] CIs on every delta; deltas inside CIs labeled "no detectable effect"
- [ ] pre-registered design (docs/flywheel_experiments.md) deviations
      listed explicitly
- [ ] limitations section includes the coarse-label caveat if labels were
      coarse
- [ ] no home paths, no hostnames, no "public repo" phrasing while the
      repo is private

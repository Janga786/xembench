# How to turn an artifact pack into a results section (without overclaiming)

Input: one artifact pack (`docs/artifact_packs.md`). Template:
`reports/templates/report_skeleton.md`. This page is the editorial rules.

## The workflow

1. Build the pack from the final eval CSV; check `manifest.json` first —
   if `git_dirty: true` or the commit isn't the one you think, rerun before
   writing a single sentence.
2. Paste tables/figures at the skeleton's INSERT points. Never retype
   numbers by hand — copy from the generated files so text can't drift
   from data.
3. Write interpretation ONLY where the skeleton prompts for it, one claim
   per prompt, each claim traceable to a specific table row or figure.

## Rules that keep you honest

- **Every rate carries its CI.** "74% [65%, 82%]", not "74%". If two CIs
  overlap, you observed no difference — write exactly that.
- **Direction matters.** Report arm→G1 and G1→arm separately; they are
  different findings.
- **Denominators visible.** n=50 transfer cells are noisier than n=100
  native cells; say so when comparing them.
- **Failure labels are what they are.** Until richer instrumentation lands,
  most failures are `timeout` — "the policy failed to complete within
  budget" is the claim the data supports; "the policy couldn't grasp" is
  not, unless failure_report.md says grasp_failure.
- **Negative transfer needs a reference.** Only call transfer "negative"
  against the explicit reference defined in docs/metrics.md.
- **Synthetic stays synthetic.** If a pack's manifest has
  `"synthetic": true`, nothing in it may appear in a results section, even
  as a placeholder — use empty tables with "pending" instead.
- **Absence is a finding.** Cells that don't exist (missing checkpoints,
  skipped seeds) get listed as not-run, not silently omitted.
- **Scope sentence.** End the section with one sentence bounding the
  claims: two embodiments, six tasks, closed-vocabulary language, sim only.

## Phrases to prefer

- "Under condition X, we observe Y [CI]" over "the method achieves Y".
- "consistent with" over "proves"; "on this benchmark" over "in general".
- "transfer retained r% of native performance on pair P" over "the policy
  transfers".

## SOP / portfolio adaptation

For an SOP or portfolio page, keep exactly three numbers (from
`result_cards.json`): one native headline, one transfer retention, one
robustness delta — each with its scope in the same sentence. The
interesting story is the *measurement design* (canonical interface,
leak-proof splits, failure taxonomy), not the magnitude of any number.

# Flywheel application-facing summary

Honest, reusable blurbs for the Physical AI Data Flywheel extension.
**Ground rule (same as [application_summary.md](application_summary.md)):
no real flywheel round has run; everything below is written to be true
TODAY. The placeholder section is filled only from a real (non-synthetic)
flywheel pack.**

## 2-sentence pitch

On top of xembench I built a Physical AI Data Flywheel: a closed-loop
data-to-policy pipeline that converts benchmark failures into typed,
prioritized demonstration requests, selects what to collect next under an
episode/cost budget, validates every episode before it can be trained on,
and packages each round with full provenance. The designed experiment asks
a data-centric question — does failure-targeted collection beat random
collection per additional demonstration, especially on held-out language
and cross-embodiment evaluation — and the infrastructure reports whichever
answer the data gives.

## Resume bullet placeholders

- Built a failure-driven data-collection pipeline over a cross-embodiment
  manipulation benchmark: typed dataset/provenance schemas, a
  deterministic demonstration-quality validator with per-episode scoring,
  a failure-taxonomy→collection-request rule engine, and five budgeted
  data-selection strategies — all test-covered ([N] automated tests).
- Engineered trustworthy-data rails end to end: synthetic data is
  structurally prevented from being packaged as real results, every
  artifact carries git/environment provenance, and toy pipelines are
  byte-reproducible.
- [AFTER LINUX ONLY] Measured the value of failure-targeted collection:
  [X] pp success-rate gain per 50 demonstrations vs [Y] pp for random
  collection on held-out-language / cross-embodiment cells [CIs].

## SOP paragraph draft

Benchmarks tell you *that* a policy fails; they rarely tell you *what data
to collect about it*. After building xembench, I extended it into a data
flywheel that closes that loop: evaluation failures become typed candidate
demonstration requests, selection strategies choose what to collect under
an explicit budget, and a validation layer scores every episode — with
provenance and synthetic-data rails enforced by code rather than
discipline — before it can enter training. Designing it taught me to treat
the dataset, not the model, as the experimental variable: the pipeline's
unit of progress is a dataset delta with a measured effect on held-out
language grounding and cross-embodiment transfer. [After the Linux round,
add one sentence stating the measured improvement-per-demonstration
comparison, whichever way it came out.]

## What NOT to claim yet

- Any flywheel result: no improvement-per-demo numbers, no
  targeted-vs-random comparison, no validation-recall numbers. Toy values
  are hand-written constants encoding the hypothesis.
- That failure-targeted selection "works" — that is the open question.
- That the validator catches real-world corruption — it catches the
  deterministic classes it checks for, demonstrated on synthetic plants.
- VLA training/fine-tuning, real-robot data, or dataset release of any
  kind.
- Novelty over the data-centric/active-learning literature —
  frame as research infrastructure and an experiment platform.

## What CAN be claimed after the Linux round (from a real pack only)

From `artifacts/releases/flywheel_round1/` with `"synthetic": false` in
`manifest.json`:

- "Failure-targeted collection yielded [X] pp per 50 demos vs [Y] pp for
  random on [splits] [CIs]" — or the honest null: "at N=50 the difference
  was within confidence intervals."
- "Validation filtering recovered [Z]% of the corruption-induced drop and
  caught [R]% of injected corrupt episodes."
- "The round-1 dataset card / quality report documents [counts, issue
  distribution] across [tasks/embodiments]."

## How this complements the Booster K1 work and xembench

xembench answers "how much manipulation skill survives a body swap, and
where does it break?"; the flywheel answers "given where it breaks, what
data fixes it cheapest?". Together with the K1 humanoid research
(hardware-facing, real-robot constraints), the three legs cover benchmark
design, data-centric learning infrastructure, and physical deployment
context — one coherent story about making embodied learning claims
measurable. (K1 specifics stay in their own private repos.)

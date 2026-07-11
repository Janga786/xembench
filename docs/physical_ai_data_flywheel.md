# Physical AI Data Flywheel

**A trustworthy data-to-policy pipeline for language-grounded robot
manipulation, built on xembench.**

**Status: round 1 executed on real data (2026-07-09).** The full loop —
7,752-record dataset index, deep validation (zero criticals), 180
candidates from 6,155 real eval failures, failure-targeted vs random arms
at an equal 50-episode budget, arm-isolated retraining, and per-cell
CI-gated comparison — ran end to end on the RTX 5090. Result: an
informative null at this budget, plus a methodology finding (seed-matched
baselines are mandatory; pooled comparisons fabricate ±5–9 pt swings).
Round-1 pack: `artifacts/releases/flywheel_round1`; full account in
[../reports/phase_b_campaign_report.md](../reports/phase_b_campaign_report.md) §4.

## What it is

A closed-loop robot-learning data system that turns benchmark failures into
targeted new data:

```
        ┌──────────────────────────────────────────────────────────┐
        ▼                                                          │
  evaluation failures ─► failure analysis ─► candidate demo requests
        ▲                                            │
        │                                   targeted data selection
   policy training ◄─ dataset validation ◄─ demonstration collection
        │                                            ▲
        └── artifact / dataset / model reports ──────┘  (next round)
```

Every stage produces a typed, validated, provenance-stamped artifact
(`xembench.flywheel.schemas`), so a training run can never quietly consume
data nobody vetted, and a synthetic file can never masquerade as a real
result.

**Research question.** Does *failure-targeted* data collection improve
language-grounded manipulation performance and cross-embodiment transfer
more efficiently — per additional demonstration — than random additional
demonstrations? That is an empirical question the Linux experiment will
answer; this document describes the machinery, not an answer.

## Why it builds on xembench

xembench already provides everything a data flywheel needs to close its
loop under controlled conditions: a canonical 21-dim action interface with
per-robot adapters (so data collected for one body is interpretable for
another), language-conditioned tasks with provably leak-free held-out
paraphrase/color splits (so "targeted at grounding failures" is
measurable), a per-episode eval CSV schema with a closed failure taxonomy
(the flywheel's input), an HDF5 demo schema + validator (the flywheel's
collection format), and an artifact packer with synthetic-data rails (the
flywheel's output discipline). Building a separate system would mean
re-deriving all of that; building here means one repository where the
benchmark *generates* the failures and *measures* the fix.

## What is implemented in this Mac foundation

| Stage | Module (`xembench.flywheel.*`) | State |
|---|---|---|
| Schemas + JSON I/O | `schemas` | 9 typed schemas, validators, tests |
| Dataset index | `dataset_index`, `build_dataset_index` | eval-CSV + demo-HDF5 loaders, JSONL/CSV outputs, filtering |
| Validation | `validation`, `validate_dataset` | deterministic checks, quality scores, wraps the existing H5 validator |
| Candidate generation | `candidates`, `generate_candidates` | failure→request rule table, coarse-label fallback |
| Selection | `selection`, `select_candidates` | 5 deterministic strategies under episode/cost budgets |
| Dataset/model cards | `cards`, `build_dataset_card` | markdown + JSON; model card is a placeholder by design |
| Round comparison | `compare_rounds` | per-cell/per-split SR deltas, improvement-per-demo |
| Linux planning | `experiment_plan` | command plans generated, never executed |
| Toy pipeline | `toy_data`, `generate_toy_flywheel` | deterministic synthetic round exercising everything above |
| Artifact packs | `build_flywheel_pack` | reuses the existing packer's manifest/checksum/synthetic rails |

`make flywheel-help` lists the commands; `make flywheel-mac-check` runs the
whole Mac-side loop on toy data.

## What is intentionally deferred to Linux

Real demonstration collection (mplib motion planning, G1 PPO experts), BC
training on flywheel-selected data, real evaluation matrices, the actual
baseline/random/targeted comparison, video evidence, and any tuning of the
selection heuristics against measured outcomes. See
[flywheel_linux_plan.md](flywheel_linux_plan.md) for the ordered plan.

## What is NOT claimed (yet)

- **No evidence that failure-targeted selection works.** The toy data
  encodes the *hypothesis* (targeted > random > baseline; noisy hurts;
  filtering recovers) so the analysis stack can be tested against known
  structure — it is not a simulation and predicts nothing.
- No real dataset, no trained policy, no benchmark numbers, no VLA
  results, no real-robot data.
- The selection strategies are simple greedy/sampling algorithms with no
  optimality claims; expected-value numbers on candidates are heuristics.
- Quality scores are deterministic rule-based heuristics, not learned
  quality estimates.

## How this connects to the broader landscape

- **Cross-embodiment learning** — the flywheel inherits xembench's premise:
  transfer is measured, not assumed. Candidates from cross-embodiment
  failures are first-class (and weighted highest by default).
- **VLA-ready robot datasets** — the dataset index + provenance + card
  discipline mirrors what large vision-language-action data efforts need;
  the canonical action interface keeps collected data consumable across
  bodies. (Fine-tuning a VLA is explicitly future work, not a dependency.)
- **Data-centric AI for robotics** — the experiment treats *which data to
  collect next* as the variable, holding architecture and training recipe
  fixed.
- **Failure-driven data collection** — the candidate generator is a fixed,
  auditable rule table from failure taxonomy labels to collection specs,
  designed to be replaced by learned acquisition later without changing
  the surrounding pipeline.
- **Demonstration quality validation** — deterministic checks (finiteness,
  bounds, shapes, lengths, duplicates, leakage, provenance) score every
  episode before it can be trained on.
- **Reproducible robot-learning experiments** — every artifact carries git
  state, environment, inputs, and a synthetic flag; toy outputs are
  byte-reproducible.

## How this differs from "just train another policy"

Training another policy answers "is this architecture/recipe better?". The
flywheel holds the policy, architecture, and recipe *fixed* and varies only
the incremental data — its unit of progress is a **dataset delta with a
measured effect**, not a model. The deliverables are correspondingly
data-shaped: dataset cards, quality reports, selection plans, and
round-over-round comparisons, with model cards deliberately left as
placeholders until a real training round exists.

## The MVP experiment (planned, not run)

Five arms, identical BC recipe and budget, on the xembench eval matrix
(details + acceptance criteria: [flywheel_experiments.md](flywheel_experiments.md)):

1. **baseline** — train on the round-0 demo set.
2. **+random** — add N demos sampled uniformly from the same generators.
3. **+failure-targeted** — add N demos chosen by the
   `failure_targeted`/`cost_aware_failure_targeted` strategies from
   round-0 failures.
4. **+noisy** — add N deliberately corrupted demos (stress arm).
5. **+validated/filtered** — the noisy arm's data, filtered by the
   validation layer before training.

## The core metric

**Improvement per additional demonstration**: the episode-weighted
success-rate delta between rounds divided by demos added
(`xembench.flywheel.compare_rounds`), reported overall and per split —
with held-out-paraphrase, held-out-color, and cross-embodiment cells as
the cells that matter. Wilson CIs come from the existing metrics stack
([metrics.md](metrics.md)); a difference smaller than the interval is
reported as "no detectable effect", not rounded up to a win.

## Reading order

[flywheel_schemas.md](flywheel_schemas.md) — every schema and vocabulary ·
[flywheel_mac_plan.md](flywheel_mac_plan.md) — what this foundation
contains and how it was verified ·
[flywheel_linux_plan.md](flywheel_linux_plan.md) — the ordered real-data
plan · [flywheel_experiments.md](flywheel_experiments.md) — the MVP
experiment design ·
[how_to_write_flywheel_results_section.md](how_to_write_flywheel_results_section.md)
— reporting discipline once real numbers exist.

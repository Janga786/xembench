# Flywheel experiments — the MVP design (planned, not run)

**No experiment below has been executed. This page fixes the design now so
the Linux runs cannot drift into post-hoc storytelling.**

## Hypothesis

Failure-targeted data collection improves language-grounded manipulation
success — especially on held-out paraphrase, held-out color, and
cross-embodiment cells — more per additional demonstration than random
additional collection; corrupted data hurts; validation filtering recovers
most of the damage.

## Arms (all: identical BC recipe, seeds, and eval matrix)

| Arm | Extra data (+N episodes, same N everywhere) |
|---|---|
| `bc_baseline` | none — round-0 demos only |
| `bc_after_random_data` | uniform sample from the same generators, unconstrained conditions |
| `bc_after_failure_targeted_data` | `failure_targeted` (and, if G1 cost binds, `cost_aware_failure_targeted`) selection from round-0 failures |
| `bc_after_noisy_data` | random arm's data with injected corruption (action noise / mislabeled success / truncated episodes) |
| `bc_after_filtered_data` | noisy arm's data after `validate_dataset` filtering (drop `failed`/critical episodes) |

Start with N = 50; the toy pipeline uses the same number so the analysis
plumbing is already exercised at that scale.

## Controls and discipline

- Same BC config, steps, batch size, and seeds (0/1/2) across arms; only
  the added data differs.
- The random arm is drawn with a recorded seed and the SAME episode count
  and generator mix as the targeted arm (cost parity noted separately —
  targeted may skew toward expensive G1 episodes; report both
  per-episode and per-cost-unit numbers).
- Selection plans, dataset cards, and validation reports for every arm are
  committed to the round directory BEFORE training starts.
- Corruption in the noisy arm is scripted and seeded, and its dataset card
  says exactly what was corrupted.

## Metrics (from docs/metrics.md machinery)

1. **Improvement per additional demonstration** (core):
   episode-weighted ΔSR / N, overall and per flywheel split
   (`compare_rounds`).
2. Transfer drop / retention changes on the two transfer pairs.
3. Paraphrase and color deltas (did targeted data close held-out gaps?).
4. Failure-distribution shifts (did the targeted failure labels actually
   shrink?).
5. For the filtered arm: fraction of noisy episodes caught by validation
   (recall of the injected corruption).

## What counts as which outcome

- **Supports the hypothesis:** targeted ΔSR/demo exceeds random's, with
  non-overlapping Wilson CIs on at least the weighted splits.
- **Null / negative results are results.** "Targeted ≈ random at N=50" or
  "targeted helps natively but not cross-embodiment" get reported with the
  same prominence — the flywheel's value claim is the measurement loop,
  not a guaranteed win.
- Deltas inside the CIs are reported as "no detectable effect at this
  sample size", never rounded up.

## Threats to validity (acknowledged now)

- Coarse failure labels (pre-instrumentation) blunt targeting; report
  which labels drove selection.
- One benchmark, two embodiments, six tasks — no external-validity claims.
- The corruption model in the noisy arm is synthetic; it bounds what
  "filtering recovers" can mean.
- Selection heuristics were designed against the same taxonomy they
  target; the random arm is the guard against self-confirmation.

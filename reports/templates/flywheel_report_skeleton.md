# Physical AI Data Flywheel — report skeleton

<!-- Fill ONLY from a real (non-synthetic) flywheel pack. Every number gets
a source file; anything without one stays TODO. Companion guide:
docs/how_to_write_flywheel_results_section.md -->

## 1. Abstract

TODO — one paragraph: the loop, the experiment, the headline
improvement-per-demo numbers WITH CIs, one honest limitation.

## 2. Introduction

TODO — data, not architecture, as the variable; why targeted collection is
the question worth asking; why it needs a benchmark that measures transfer
and language generalization (xembench).

## 3. Related work (placeholders — verify citations before writing)

- Data-centric robot learning / dataset curation — TODO
- Failure-driven and active data collection — TODO
- Cross-embodiment datasets and transfer benchmarks — TODO
- Demonstration quality assessment — TODO

## 4. Method: the Physical AI Data Flywheel

TODO — the loop diagram; one paragraph per stage (index, validation,
candidates, selection, planning, packaging); what is rule-based vs learned
(all rule-based in this version).

## 5. Dataset schema and provenance

TODO — from docs/flywheel_schemas.md; emphasize the invariants (synthetic
flags, dim checks, leakage checks) and the sanitized-provenance discipline.

## 6. Validation and quality scoring

TODO — the check list, the scoring rule, and the measured issue
distribution from `data_quality_issues.csv` of the real round.

## 7. Failure-driven candidate generation

TODO — the rule table; which failure labels were actually available; the
candidate distribution of the real round (`round1_candidates.md`).

## 8. Data selection strategies

TODO — the five strategies and objective terms; the selection plans of the
real round; cost accounting (panda vs G1 episodes).

## 9. Training / evaluation loop

TODO — BC recipe (identical across arms), eval matrix configuration,
seeds; point at config snapshots in the pack.

## 10. Experiments

TODO — the five arms from docs/flywheel_experiments.md, exactly as
pre-registered there; note any deviations explicitly.

## 11. Results (placeholders)

| Arm | ΔSR overall | ΔSR heldout-para | ΔSR heldout-color | ΔSR cross-emb | ΔSR / demo |
|---|---|---|---|---|---|
| +random | TODO | TODO | TODO | TODO | TODO |
| +failure-targeted | TODO | TODO | TODO | TODO | TODO |
| +noisy | TODO | TODO | TODO | TODO | TODO |
| +filtered | TODO | TODO | TODO | TODO | TODO |

All cells: value ± Wilson CI, from `round0_vs_round1*.json`.

## 12. Failure analysis

TODO — did the targeted failure labels shrink? failure_report diffs per
arm; qualitative examples via episode_seed replays.

## 13. Limitations

TODO — start from the pre-registered threats-to-validity list in
docs/flywheel_experiments.md; add what actually bit.

## 14. Future work

TODO — richer failure instrumentation, learned acquisition, VLA-scale
datasets (currently out of scope), additional embodiments.

## 15. Appendix

TODO — full per-cell tables, dataset cards, validation reports, selection
plans, command logs (all already in the flywheel pack).

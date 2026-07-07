# Metrics and result schemas

Two CSVs are produced by evaluation (`xembench.eval.matrix` via
`python -m xembench.scripts.evaluate`); all analysis tooling
(`xembench.analysis`) consumes them. **No benchmark numbers exist yet** —
Phase B produces them; everything here defines how they will be computed.

## Aggregate CSV — `data/results/eval.csv` (one row per eval cell)

| Column | Meaning |
|---|---|
| `policy_type` | `bc`, `bc_ppo` (fine-tuned), later `diffusion` |
| `task` | task the policy was TRAINED on (registry name, e.g. `pick_cube_arm`) |
| `train_embodiment` | embodiment of the training task (`panda`, `panda_stick`, `g1`) |
| `eval_task` / `eval_embodiment` | where it was evaluated; differs from `task` on transfer cells |
| `instruction_cond` | `canonical` or `heldout` (heldout-synonym paraphrases, p=1.0) |
| `color_cond` | `train`, `unseen`, or `fixed` (stock scenes without color variation) |
| `n_episodes`, `n_success` | counts |
| `success_rate` | `n_success / n_episodes` |
| `ci_low`, `ci_high` | 95% Wilson interval (below) |
| `avg_steps_to_success` | mean steps over SUCCESSFUL episodes only; NaN if none |
| `avg_steps_frac` | `avg_steps_to_success / max_steps` — comparable across tasks with different budgets (50 vs 100) |
| `seed`, `checkpoint` | provenance |

## Per-episode CSV — `data/results/eval_episodes.csv` (one row per episode)

Schema owned by `xembench.analysis.episodes` (**schema v2**; core columns are
v1-stable, optional columns default-fill on read AND write, so v1 CSVs load
unchanged — that is the whole migration story. Validate single records
before writing with `validate_episode_record`).

Core columns (v1, never reordered/removed):

| Column | Meaning |
|---|---|
| all aggregate identity columns | `policy_type, task, train_embodiment, eval_task, eval_embodiment, instruction_cond, color_cond, seed, checkpoint` |
| `episode_idx` | 0-based index within the cell |
| `episode_seed` | exact env reset seed (reproduce any episode) |
| `success` | 0/1 |
| `steps` | env steps executed this episode |
| `max_steps` | episode budget |
| `failure_type` | one of the taxonomy labels below |

Optional columns (v2 — recorded by the runner where cheap, blank/NaN
otherwise):

| Column | Meaning |
|---|---|
| `instruction` | exact (possibly paraphrased) text the policy received |
| `template_family` | instruction family from the task registry |
| `return_sum` | episode reward sum (interpret against the env's reward mode) |
| `elapsed_s` | wall-clock seconds for the episode |
| `video_path` | recorded rollout if any (repo-relative; Phase B recording hook) |
| `config_path` | eval config snapshot if any (repo-relative) |

## Failure taxonomy (`xembench.analysis.taxonomy`)

Closed label set — the schema validator rejects anything else:

`success, language_grounding_failure, wrong_object_selected, reach_failure,
grasp_failure, object_dropped, placement_failure, collision, timeout,
adapter_action_mismatch, embodiment_reachability_limit, unstable_policy,
simulator_error, unknown_failure`

**Current auto-labeling is deliberately conservative.** The runner can only
derive three labels without extra instrumentation:

- `success` — env's `evaluate()` said so;
- `timeout` — episode truncated at budget without success;
- `unknown_failure` — terminated early without success (rare in these envs).

The richer labels (`wrong_object_selected`, `grasp_failure`, …) are defined
now so CSV schemas, reports, and tests are stable, but assigning them
requires reading env info flags (`is_grasped`, `is_obj_placed`, object/goal
distances) during evaluation. That instrumentation is Phase B work — see
`TODO(Janga)` in `xembench/analysis/taxonomy.py`. Reports render whatever
labels exist; they do not pretend finer labels are available.

## Formulas

**Success rate.** `SR = k / n` for k successes in n episodes.

**Wilson 95% CI** (used instead of the normal approximation because cells
have n=50–100 and SRs near 0 or 1):

```
center = (p + z²/2n) / (1 + z²/n)
half   = z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n),  z = 1.96
```

**Cross-embodiment transfer drop** — the headline metric. For a policy
trained on task T with transfer partner T′ (matched instruction/color
conditions):

```
drop(T→T′) = SR_native(T) − SR_transfer(T→T′)
```

Report both absolute drop (percentage points) and, where SR_native > 0, the
**retention ratio** `SR_transfer / SR_native`. Computed by
`python -m xembench.analysis.transfer_matrix`.

**Negative transfer.** Zero-shot transfer has no "before" state, so we use
the term only for comparisons against a reference policy on the target
task: transfer is *negative* if `SR_transfer(T→T′) <
SR_reference(T′)` where the reference is a language-and-task-matched policy
trained natively on T′ with the same data budget. A random-policy floor is
the weaker fallback reference. The analysis CLI reports raw numbers and
leaves the reference choice explicit.

**Paraphrase robustness.** `SR(canonical) − SR(heldout)`, same task and
policy. **Color generalization.** `SR(color=train) − SR(color=unseen)`,
Lang tasks only.

**Efficiency.** `avg_steps_to_success` over successes only (failures would
saturate at budget and mask differences); `avg_steps_frac` normalizes
across the 50 vs 100 step budgets. From episode CSVs, efficiency can be
recomputed with CIs by bootstrapping successful episodes.

**Failure distribution.** Share of each taxonomy label among failures, per
(task, embodiment, policy, condition) — `python -m
xembench.analysis.failure_report`.

## Planned but not implemented

- Subtask completion (reached / grasped / lifted / placed staging) — the
  env info flags exist; recording them per-step is Phase B instrumentation.
- Collision counts — SAPIEN contact queries, not currently recorded.
- Multi-seed variance — the matrix supports `seeds: [0,1,2]`; headline
  cells get 3 seeds in Phase B (plan B5).

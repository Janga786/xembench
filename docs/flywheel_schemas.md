# Flywheel schemas

Module: `xembench.flywheel.schemas` — plain dataclasses + explicit
validation (`validate() -> list[str]`, `[]` = valid), JSON round-trips via
`to_json_dict`/`from_json_dict`, files via `save_json`/`load_json` and
`write_jsonl`/`read_jsonl` (both validate on write AND read). Closed
vocabularies live in the same module; failure labels come from
`xembench.analysis.taxonomy` and are never redefined.

## Vocabularies

| Name | Values |
|---|---|
| `SOURCE_TYPES` | `sim_expert, sim_random, sim_policy, human_teleop, replay, synthetic_toy` |
| `FLYWHEEL_SPLITS` | `train_seen, heldout_paraphrase, heldout_color, cross_embodiment` |
| `SEVERITIES` | `info, warning, error, critical` |
| `ISSUE_CATEGORIES` | 17 categories: `missing_field, nan_value, action_out_of_bounds, action_shape_mismatch, observation_shape_mismatch, missing_images, language_leakage, duplicate_episode, failed_demo_labeled_success, embodiment_mismatch, suspicious_instruction, outlier_trajectory, short_episode, long_episode, missing_provenance, invalid_seed, unknown` |
| `CANDIDATE_STATUSES` | `proposed, selected, collected, rejected, deferred` |
| `SELECTION_STRATEGIES` | `random, stratified, failure_targeted, diversity_greedy, cost_aware_failure_targeted` |
| `VALIDATION_STATUSES` | `unvalidated, passed, passed_with_warnings, failed` |
| `ROUND_STAGES` | `indexed, validated, candidates, selected, planned, collected, trained, evaluated, packed` |

`derive_flywheel_split(task, eval_task, instruction_cond, color_cond)` maps
eval-CSV condition columns onto one split, with documented precedence:
cross-embodiment > heldout paraphrase > heldout color > train_seen.

## The nine schemas

**DemoProvenance** — where an episode came from: `source_type`,
`source_run_id`, `source_path` (sanitized), `generator_name/-version`,
`git_commit`, `config_path`, `created_at`, `machine` (coarse label, never a
hostname), `notes`, `is_synthetic`. Invariant: `source_type=synthetic_toy`
forces `is_synthetic=True` — toy data cannot be labeled real.

**DemoEpisodeMetadata** — one demonstration/rollout episode: identity
(`episode_id`, `task`, `embodiment`, `policy`, `split`), language
(`instruction`, `instruction_template_id`, `paraphrase_group`), conditions
(`object_condition`, `color_condition`, `seed`), outcome (`success`,
`failure_label`, `steps`, `return_value`), action layout (`action_space`,
`canonical_action_dim` — must equal 21, `native_action_dim` — must match
the embodiment), artifacts (`hdf5_path`, `video_path`), and a required
nested `provenance`.

**DatasetIndexRecord** — the compact per-episode index row: dataset/episode
IDs, task/embodiment/split, outcome, `instruction_hash`, conditions,
`num_steps`, payload flags (`has_rgb/-state/-canonical_actions/
-native_actions`), `quality_score` ∈ [0,1], `validation_status`, and
`provenance_id` (`<source_type>:<hash>` so the index stays filterable by
source without the sidecar).

**DataQualityIssue** — one validation finding: `issue_id`, `severity`,
`category`, `message`, `episode_id` ("" = dataset-level), `field`,
`suggested_fix`.

**CandidateDemoRequest** — "collect this next": target
(task/embodiment/split), language/visual conditions, `target_failure_label`,
`requested_policy_source` (a source type or `diagnostic`),
`estimated_cost` (> 0), `expected_value`/`priority` (heuristics, ≥ 0),
`reason`, free-form `constraints`, `seed`, `status`.

**SelectionPlan** — which candidates were chosen and why: `plan_id`,
`strategy`, budgets, `candidates_total`, `selected_ids` (unique, ≤ budget),
`objective_terms` (weights recorded verbatim), `expected_coverage`,
provenance of inputs, `created_at`, `notes`.

**FlywheelRoundManifest** — one loop iteration's file inventory:
`round_id`, `previous_round_id`, `stage`, paths to
index/eval-CSV/failure-report/candidates/plan/training-config/pack,
`created_at`, `git_commit`, `is_synthetic`, `notes`.

**DatasetCard** — human-facing dataset documentation: name/version,
intended use, composition (tasks/embodiments/splits/counts), source types
and collection methods, `validation_summary`, `known_limitations`,
`leakage_checks`, `provenance_summary`, ethics notes, `generated_at`,
`synthetic` (forced True when any `synthetic_toy` source is present;
success + failure counts must equal the total).

**ModelCardPlaceholder** — a slot for future Linux results: model/dataset/
policy identifiers, embodiments, `metrics` (**must stay empty while
`is_placeholder=True`** — enforced by the validator), limitations, failure
modes, intended/not-intended use.

## File conventions

| File | Producer | Content |
|---|---|---|
| `<name>.jsonl` | `build_dataset_index` | one `DatasetIndexRecord` per line |
| `<name>.episodes.jsonl` | same | full `DemoEpisodeMetadata` sidecar |
| `<name>.summary.csv` | same | flat pandas view of the index |
| `*_validation_report.md/.json` | `validate_dataset` | findings + summary |
| `*_candidates.jsonl/.csv/.md` | `generate_candidates` | requests + summary |
| `*_selection_plan.json/.md` | `select_candidates` | the plan |
| `*_collection_plan.md`, `*_linux_commands.sh` | `experiment_plan` | Linux plan (never executed) |
| `*_dataset_card.md/.json` | `build_dataset_card` | the card |
| `round_manifest.json` | round driver | `FlywheelRoundManifest` |

Every JSON object carries `_schema` and `_schema_version`
(`FLYWHEEL_SCHEMA_VERSION = 1`); `from_json_dict` rejects unknown fields so
schema drift fails loudly instead of silently.

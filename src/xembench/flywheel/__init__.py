"""Physical AI Data Flywheel: failure-driven data collection on top of xembench.

A closed-loop data-to-policy pipeline: evaluation failures -> failure
analysis -> candidate demo requests -> targeted data selection -> collection
-> dataset validation -> training -> evaluation -> artifact reports -> next
round. The Mac foundation (this package) implements schemas, the dataset
index, validation, candidate generation, selection strategies, Linux command
planning, and artifact packaging on toy data; real collection/training/
evaluation happen on the Linux machine (docs/flywheel_linux_plan.md).

No module here runs training or CUDA work. Every synthetic output is
loudly stamped SYNTHETIC — the same discipline as xembench.artifacts.

Entry points (all `python -m xembench.flywheel.<name>`):

    build_dataset_index    eval CSV / demo HDF5 -> dataset index (JSONL+CSV)
    validate_dataset       index -> quality issues, scores, report
    generate_candidates    eval failures -> candidate demo requests
    select_candidates      candidates + strategy + budget -> SelectionPlan
    build_dataset_card     index + validation -> dataset card (md+json)
    compare_rounds         two eval CSVs -> per-cell delta report
    generate_toy_flywheel  deterministic toy round (SYNTHETIC, Mac-safe)
    experiment_plan        SelectionPlan -> Linux command plan (not executed)
    build_flywheel_pack    round manifest -> checksummed artifact pack

Docs: docs/physical_ai_data_flywheel.md (design), docs/flywheel_schemas.md
(schema reference), docs/flywheel_mac_plan.md / docs/flywheel_linux_plan.md
(what runs where), docs/flywheel_experiments.md (the planned MVP experiment).
"""

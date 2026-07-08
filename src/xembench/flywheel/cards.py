"""Dataset cards and model-card placeholders (markdown + JSON).

A dataset card documents what a flywheel dataset IS — composition, sources,
validation state, known limitations — so a training run can never quietly
consume data nobody vetted. The model card is a placeholder by design: its
metrics stay empty until real Linux evaluation results exist (the schema
enforces this).

CLI: ``python -m xembench.flywheel.build_dataset_card``.
"""

from collections import Counter

from xembench.flywheel.dataset_index import source_type_of
from xembench.flywheel.schemas import (
    DatasetCard,
    DatasetIndexRecord,
    DemoEpisodeMetadata,
    ModelCardPlaceholder,
    utc_now_iso,
)

DEFAULT_LIMITATIONS = [
    "Simulation only (ManiSkill3); no real-robot data.",
    "Closed-vocabulary language distribution (see docs/language_splits.md).",
    "Failure labels beyond success/timeout/unknown_failure require Phase B "
    "instrumentation; coarse labels dominate until then.",
    "Quality scores are deterministic heuristics, not learned estimates.",
]


def build_dataset_card(records: list[DatasetIndexRecord],
                       episodes: list[DemoEpisodeMetadata] | None = None,
                       validation_summary: dict | None = None, *,
                       name: str, version: str,
                       intended_use: str = "",
                       generated_at: str | None = None) -> DatasetCard:
    source_types = sorted({source_type_of(r) for r in records
                           if r.provenance_id})
    methods = sorted({ep.policy for ep in episodes}) if episodes else []
    synthetic = "synthetic_toy" in source_types
    n_success = sum(1 for r in records if r.success)
    leakage = []
    if validation_summary:
        n_leak = validation_summary.get("issues_by_category", {}).get(
            "language_leakage", 0)
        leakage.append(
            f"train/heldout instruction overlap: "
            f"{'NONE found' if not n_leak else f'{n_leak} finding(s) — FIX'}")
    return DatasetCard(
        dataset_name=name,
        dataset_version=version,
        intended_use=intended_use or (
            "Behavior-cloning training and flywheel-round bookkeeping for "
            "the xembench benchmark. NOT a released dataset."),
        tasks=sorted({r.task for r in records}),
        embodiments=sorted({r.embodiment for r in records}),
        splits=sorted({r.split for r in records}),
        total_episodes=len(records),
        successful_episodes=n_success,
        failed_episodes=len(records) - n_success,
        source_types=source_types,
        collection_methods=methods,
        validation_summary=validation_summary or {},
        known_limitations=list(DEFAULT_LIMITATIONS),
        leakage_checks=leakage,
        provenance_summary=dict(Counter(source_type_of(r) for r in records
                                        if r.provenance_id)),
        ethical_or_safety_notes=(
            "Simulated tabletop manipulation; no human subjects, no "
            "personally identifying data. Paths in this card are "
            "repo-relative by construction."),
        generated_at=generated_at or utc_now_iso(),
        synthetic=synthetic,
    )


def card_markdown(card: DatasetCard) -> str:
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "Every episode in this dataset is synthetic.\n\n"
              if card.synthetic else "")
    vs = card.validation_summary
    validation = (
        f"{vs.get('n_records', '?')} records validated; status counts: "
        + (", ".join(f"{k}={v}" for k, v in
                     sorted(vs.get("records_by_status", {}).items()))
           or "n/a")
        + f"; mean quality score {vs.get('mean_quality_score', 'n/a')}"
        if vs else "_not validated yet — run "
        "`python -m xembench.flywheel.validate_dataset`_")
    lines = [
        f"# Dataset card — {card.dataset_name} v{card.dataset_version}", "",
        banner + f"Generated {card.generated_at}.", "",
        f"**Intended use.** {card.intended_use}", "",
        "## Composition", "",
        f"- episodes: {card.total_episodes} "
        f"({card.successful_episodes} success / "
        f"{card.failed_episodes} failure)",
        f"- tasks: {', '.join(card.tasks) or 'none'}",
        f"- embodiments: {', '.join(card.embodiments) or 'none'}",
        f"- splits: {', '.join(card.splits) or 'none'}",
        f"- source types: {', '.join(card.source_types) or 'none'}",
        f"- collection methods: "
        f"{', '.join(card.collection_methods) or 'n/a'}", "",
        "## Validation", "", validation, "",
        "## Leakage checks", "",
    ]
    lines += [f"- {c}" for c in card.leakage_checks] or ["- none run"]
    lines += ["", "## Provenance", ""]
    lines += [f"- {k}: {v} episode(s)"
              for k, v in sorted(card.provenance_summary.items())]
    lines += ["", "## Known limitations", ""]
    lines += [f"- {lim}" for lim in card.known_limitations]
    lines += ["", "## Ethics / safety", "", card.ethical_or_safety_notes]
    return "\n".join(lines) + "\n"


def build_model_card_placeholder(*, model_name: str,
                                 training_dataset: str = "",
                                 policy_type: str = "bc",
                                 embodiment_train: str = "",
                                 embodiment_eval: str = "",
                                 ) -> ModelCardPlaceholder:
    return ModelCardPlaceholder(
        model_name=model_name,
        training_dataset=training_dataset,
        policy_type=policy_type,
        embodiment_train=embodiment_train,
        embodiment_eval=embodiment_eval,
        metrics={},  # stays empty until real Linux results exist
        limitations=["No trained model exists yet — this card is a "
                     "placeholder for the Linux training round."],
        failure_modes=["(to be filled from the failure report of a real "
                       "evaluation)"],
        intended_use="Benchmark experiments inside xembench.",
        not_intended_use=("Deployment of any kind; the policy (once it "
                          "exists) will be a research artifact evaluated "
                          "in simulation only."),
        is_placeholder=True,
    )


def model_card_markdown(mc: ModelCardPlaceholder) -> str:
    lines = [
        f"# Model card (PLACEHOLDER) — {mc.model_name}", "",
        "> **PLACEHOLDER — no trained model, no metrics.** This card is "
        "generated so the flywheel pack has a stable slot for real Linux "
        "results; every metric field is intentionally empty.", "",
        f"- policy type: {mc.policy_type or 'tbd'}",
        f"- training dataset: {mc.training_dataset or 'tbd'}",
        f"- train embodiment: {mc.embodiment_train or 'tbd'}",
        f"- eval embodiment: {mc.embodiment_eval or 'tbd'}",
        f"- metrics: none (placeholder)", "",
        "## Limitations", "",
    ]
    lines += [f"- {x}" for x in mc.limitations]
    lines += ["", "## Failure modes", ""]
    lines += [f"- {x}" for x in mc.failure_modes]
    lines += ["", "## Intended use", "", mc.intended_use, "",
              "## Not intended for", "", mc.not_intended_use]
    return "\n".join(lines) + "\n"

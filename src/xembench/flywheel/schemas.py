"""Typed schemas for the Physical AI Data Flywheel.

Plain dataclasses + explicit validation (the repo's PackManifest pattern —
no pydantic dependency). Every schema has ``validate() -> list[str]``
returning problems ([] = valid, mirroring ``data.schema.validate_demo_file``)
and round-trips through JSON via ``to_json_dict`` / ``from_json_dict``.

Closed vocabularies are defined here once (source types, flywheel splits,
issue severities/categories, candidate statuses, selection strategies) so
CSV/JSONL files, reports, and tests can never drift. Failure labels are NOT
redefined — they come from ``xembench.analysis.taxonomy``.

Schema reference with field-by-field meaning: docs/flywheel_schemas.md.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from xembench.analysis.taxonomy import FAILURE_LABELS
from xembench.constants import (
    A_CANON,
    G1_NATIVE_ACTION_DIM,
    PANDA_NATIVE_ACTION_DIM,
    PANDA_STICK_NATIVE_ACTION_DIM,
)

FLYWHEEL_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------

# Where a demonstration came from. "synthetic_toy" exists so Mac-generated
# toy data is structurally incapable of masquerading as real collection.
SOURCE_TYPES = frozenset({
    "sim_expert", "sim_random", "sim_policy", "human_teleop", "replay",
    "synthetic_toy",
})

# Evaluation splits the flywheel reasons about. Derived from the eval CSV's
# (task vs eval_task, instruction_cond, color_cond) by derive_flywheel_split.
FLYWHEEL_SPLITS = ("train_seen", "heldout_paraphrase", "heldout_color",
                   "cross_embodiment")

SEVERITIES = ("info", "warning", "error", "critical")

ISSUE_CATEGORIES = frozenset({
    "missing_field", "nan_value", "action_out_of_bounds",
    "action_shape_mismatch", "observation_shape_mismatch", "missing_images",
    "language_leakage", "duplicate_episode", "failed_demo_labeled_success",
    "embodiment_mismatch", "suspicious_instruction", "outlier_trajectory",
    "short_episode", "long_episode", "missing_provenance", "invalid_seed",
    "unknown",
})

CANDIDATE_STATUSES = frozenset(
    {"proposed", "selected", "collected", "rejected", "deferred"})

# "diagnostic" = episodes collected to debug the pipeline (e.g. adapter
# action mismatch), never intended as training data.
POLICY_SOURCES = SOURCE_TYPES | {"diagnostic"}

SELECTION_STRATEGIES = ("random", "stratified", "failure_targeted",
                        "diversity_greedy", "cost_aware_failure_targeted")

VALIDATION_STATUSES = ("unvalidated", "passed", "passed_with_warnings",
                       "failed")

ROUND_STAGES = ("indexed", "validated", "candidates", "selected", "planned",
                "collected", "trained", "evaluated", "packed")

KNOWN_EMBODIMENTS = ("panda", "panda_stick", "g1")

NATIVE_ACTION_DIMS = {
    "panda": PANDA_NATIVE_ACTION_DIM,
    "panda_stick": PANDA_STICK_NATIVE_ACTION_DIM,
    "g1": G1_NATIVE_ACTION_DIM,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(text: str, length: int = 12) -> str:
    """Deterministic short content hash (episode/instruction identity)."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def derive_flywheel_split(task: str, eval_task: str, instruction_cond: str,
                          color_cond: str) -> str:
    """Map eval-CSV condition columns onto one flywheel split.

    Precedence (most specific first): cross-embodiment beats language/color
    conditions because the transfer axis is the benchmark's headline; among
    native cells, heldout paraphrase beats unseen color. Episodes that are
    both heldout-paraphrase and unseen-color are rare in the standard matrix
    and land in heldout_paraphrase by this rule — documented, deterministic.
    """
    if task != eval_task:
        return "cross_embodiment"
    if instruction_cond == "heldout":
        return "heldout_paraphrase"
    if color_cond == "unseen":
        return "heldout_color"
    return "train_seen"


# ---------------------------------------------------------------------------
# Validation plumbing shared by all schemas
# ---------------------------------------------------------------------------

def _check_in(problems: list[str], name: str, value, allowed) -> None:
    if value not in allowed:
        shown = ", ".join(sorted(str(a) for a in allowed))
        problems.append(f"{name} {value!r} not in {{{shown}}}")


def _check_nonempty_str(problems: list[str], name: str, value) -> None:
    if not isinstance(value, str) or not value:
        problems.append(f"{name} must be a non-empty string, got {value!r}")


def require_valid(obj) -> None:
    """Raise ValueError listing every problem; no-op when valid."""
    problems = obj.validate()
    if problems:
        raise ValueError(
            f"invalid {type(obj).__name__}: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# 1. DemoProvenance — where an episode came from
# ---------------------------------------------------------------------------

@dataclass
class DemoProvenance:
    source_type: str                 # one of SOURCE_TYPES
    source_run_id: str = ""          # collection run / eval run identifier
    source_path: str = ""            # sanitized origin (h5/csv), if any
    generator_name: str = ""         # e.g. "motion_plan_arm", "toy_data"
    generator_version: str = ""
    git_commit: str = ""
    config_path: str = ""
    created_at: str = ""             # ISO-8601 UTC
    machine: str = ""                # coarse ("mac", "linux-5090"), no hostname
    notes: str = ""
    is_synthetic: bool = False

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_in(problems, "source_type", self.source_type, SOURCE_TYPES)
        _check_nonempty_str(problems, "created_at", self.created_at)
        if self.source_type == "synthetic_toy" and not self.is_synthetic:
            problems.append(
                "source_type synthetic_toy requires is_synthetic=True — "
                "toy data can never be labeled real")
        return problems


# ---------------------------------------------------------------------------
# 2. DemoEpisodeMetadata — one demonstration episode
# ---------------------------------------------------------------------------

@dataclass
class DemoEpisodeMetadata:
    episode_id: str
    task: str                        # registry name (configs/tasks.yaml)
    embodiment: str                  # one of KNOWN_EMBODIMENTS
    policy: str                      # producer ("motion_planning", "ppo", ...)
    split: str                       # one of FLYWHEEL_SPLITS
    instruction: str
    instruction_template_id: str = ""
    paraphrase_group: str = ""       # "canonical" / "train" / "heldout"
    object_condition: str = ""       # e.g. "cube", "apple"
    color_condition: str = ""        # e.g. "red", or "train"/"unseen" set
    seed: int = -1
    success: bool = False
    failure_label: str = "unknown_failure"   # taxonomy label
    steps: int = 0
    return_value: float | None = None
    action_space: str = "pd_joint_delta_pos"
    canonical_action_dim: int = A_CANON
    native_action_dim: int = -1
    hdf5_path: str = ""              # sanitized, "" when metadata-only
    video_path: str = ""
    provenance: DemoProvenance | None = None

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "episode_id", self.episode_id)
        _check_nonempty_str(problems, "task", self.task)
        _check_nonempty_str(problems, "instruction", self.instruction)
        _check_in(problems, "embodiment", self.embodiment, KNOWN_EMBODIMENTS)
        _check_in(problems, "split", self.split, FLYWHEEL_SPLITS)
        _check_in(problems, "failure_label", self.failure_label,
                  FAILURE_LABELS)
        if self.success != (self.failure_label == "success"):
            problems.append("success flag and failure_label disagree")
        if not isinstance(self.seed, int) or self.seed < 0:
            problems.append(f"seed must be a non-negative int, got "
                            f"{self.seed!r}")
        if self.steps <= 0:
            problems.append(f"steps must be positive, got {self.steps}")
        if self.canonical_action_dim != A_CANON:
            problems.append(
                f"canonical_action_dim {self.canonical_action_dim} != "
                f"A_CANON ({A_CANON})")
        expected = NATIVE_ACTION_DIMS.get(self.embodiment)
        if expected is not None and self.native_action_dim != expected:
            problems.append(
                f"native_action_dim {self.native_action_dim} != {expected} "
                f"for embodiment {self.embodiment}")
        if self.provenance is None:
            problems.append("provenance is required")
        else:
            problems += [f"provenance: {p}" for p in self.provenance.validate()]
        return problems


# ---------------------------------------------------------------------------
# 3. DatasetIndexRecord — one row of the flywheel dataset index
# ---------------------------------------------------------------------------

@dataclass
class DatasetIndexRecord:
    dataset_id: str
    episode_id: str
    task: str
    embodiment: str
    split: str
    success: bool
    failure_label: str
    instruction_hash: str = ""
    object_condition: str = ""
    color_condition: str = ""
    action_space: str = "pd_joint_delta_pos"
    num_steps: int = 0
    has_rgb: bool = False
    has_state: bool = False
    has_canonical_actions: bool = False
    has_native_actions: bool = False
    quality_score: float = 1.0
    validation_status: str = "unvalidated"
    provenance_id: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "dataset_id", self.dataset_id)
        _check_nonempty_str(problems, "episode_id", self.episode_id)
        _check_nonempty_str(problems, "task", self.task)
        _check_in(problems, "embodiment", self.embodiment, KNOWN_EMBODIMENTS)
        _check_in(problems, "split", self.split, FLYWHEEL_SPLITS)
        _check_in(problems, "failure_label", self.failure_label,
                  FAILURE_LABELS)
        _check_in(problems, "validation_status", self.validation_status,
                  VALIDATION_STATUSES)
        if self.num_steps < 0:
            problems.append(f"num_steps must be >= 0, got {self.num_steps}")
        if not 0.0 <= self.quality_score <= 1.0:
            problems.append(
                f"quality_score {self.quality_score} outside [0, 1]")
        return problems


# ---------------------------------------------------------------------------
# 4. DataQualityIssue — one validation finding
# ---------------------------------------------------------------------------

@dataclass
class DataQualityIssue:
    issue_id: str
    severity: str                    # one of SEVERITIES
    category: str                    # one of ISSUE_CATEGORIES
    message: str
    episode_id: str = ""             # "" for dataset-level issues
    field: str = ""
    suggested_fix: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "issue_id", self.issue_id)
        _check_nonempty_str(problems, "message", self.message)
        _check_in(problems, "severity", self.severity, SEVERITIES)
        _check_in(problems, "category", self.category, ISSUE_CATEGORIES)
        return problems


# ---------------------------------------------------------------------------
# 5. CandidateDemoRequest — "collect this next" proposal
# ---------------------------------------------------------------------------

@dataclass
class CandidateDemoRequest:
    candidate_id: str
    task: str
    embodiment: str
    split: str                       # split whose failures motivated this
    instruction_family: str = ""
    object_condition: str = ""
    color_condition: str = ""
    target_failure_label: str = "unknown_failure"
    requested_policy_source: str = "sim_expert"   # POLICY_SOURCES
    estimated_cost: float = 1.0      # relative collection cost units
    expected_value: float = 0.0      # heuristic value estimate (not a claim)
    priority: float = 0.0
    reason: str = ""
    constraints: dict = field(default_factory=dict)
    seed: int = 0
    status: str = "proposed"

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "candidate_id", self.candidate_id)
        _check_nonempty_str(problems, "task", self.task)
        _check_nonempty_str(problems, "reason", self.reason)
        _check_in(problems, "embodiment", self.embodiment, KNOWN_EMBODIMENTS)
        _check_in(problems, "split", self.split, FLYWHEEL_SPLITS)
        _check_in(problems, "target_failure_label",
                  self.target_failure_label, FAILURE_LABELS)
        _check_in(problems, "requested_policy_source",
                  self.requested_policy_source, POLICY_SOURCES)
        _check_in(problems, "status", self.status, CANDIDATE_STATUSES)
        if self.estimated_cost <= 0:
            problems.append(
                f"estimated_cost must be positive, got {self.estimated_cost}")
        if self.expected_value < 0:
            problems.append(
                f"expected_value must be >= 0, got {self.expected_value}")
        if not isinstance(self.constraints, dict):
            problems.append("constraints must be a dict")
        return problems


# ---------------------------------------------------------------------------
# 6. SelectionPlan — which candidates were chosen, and why
# ---------------------------------------------------------------------------

@dataclass
class SelectionPlan:
    plan_id: str
    strategy: str                    # one of SELECTION_STRATEGIES
    budget_episodes: int
    budget_cost: float | None = None
    candidates_total: int = 0
    selected_ids: list[str] = field(default_factory=list)
    objective_terms: dict = field(default_factory=dict)  # term -> weight
    expected_coverage: dict = field(default_factory=dict)  # axis -> groups
    created_at: str = ""
    input_eval_csv: str = ""         # sanitized
    input_dataset_index: str = ""    # sanitized
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "plan_id", self.plan_id)
        _check_nonempty_str(problems, "created_at", self.created_at)
        _check_in(problems, "strategy", self.strategy, SELECTION_STRATEGIES)
        if self.budget_episodes <= 0:
            problems.append(
                f"budget_episodes must be positive, got "
                f"{self.budget_episodes}")
        if len(self.selected_ids) > self.budget_episodes:
            problems.append(
                f"{len(self.selected_ids)} selected > budget "
                f"{self.budget_episodes}")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            problems.append("selected_ids contains duplicates")
        if self.candidates_total < len(self.selected_ids):
            problems.append(
                f"candidates_total {self.candidates_total} < "
                f"{len(self.selected_ids)} selected")
        return problems


# ---------------------------------------------------------------------------
# 7. FlywheelRoundManifest — one loop iteration's file inventory
# ---------------------------------------------------------------------------

@dataclass
class FlywheelRoundManifest:
    round_id: str
    previous_round_id: str = ""      # "" for round 0
    stage: str = "indexed"           # furthest completed ROUND_STAGES entry
    dataset_index_path: str = ""
    eval_csv_path: str = ""
    failure_report_path: str = ""
    candidate_request_path: str = ""
    selection_plan_path: str = ""
    training_config_path: str = ""
    artifact_pack_path: str = ""
    created_at: str = ""
    git_commit: str = ""
    is_synthetic: bool = False
    notes: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "round_id", self.round_id)
        _check_nonempty_str(problems, "created_at", self.created_at)
        _check_in(problems, "stage", self.stage, ROUND_STAGES)
        return problems


# ---------------------------------------------------------------------------
# 8. DatasetCard — human-facing dataset documentation (md+json)
# ---------------------------------------------------------------------------

@dataclass
class DatasetCard:
    dataset_name: str
    dataset_version: str
    intended_use: str
    tasks: list[str] = field(default_factory=list)
    embodiments: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    source_types: list[str] = field(default_factory=list)
    collection_methods: list[str] = field(default_factory=list)
    validation_summary: dict = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    leakage_checks: list[str] = field(default_factory=list)
    provenance_summary: dict = field(default_factory=dict)
    ethical_or_safety_notes: str = ""
    generated_at: str = ""
    synthetic: bool = False          # any synthetic_toy source forces True

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "dataset_name", self.dataset_name)
        _check_nonempty_str(problems, "dataset_version", self.dataset_version)
        _check_nonempty_str(problems, "generated_at", self.generated_at)
        if self.total_episodes < 0:
            problems.append("total_episodes must be >= 0")
        if self.successful_episodes + self.failed_episodes \
                != self.total_episodes:
            problems.append(
                f"successful ({self.successful_episodes}) + failed "
                f"({self.failed_episodes}) != total ({self.total_episodes})")
        unknown = set(self.source_types) - SOURCE_TYPES
        if unknown:
            problems.append(f"unknown source_types {sorted(unknown)}")
        if "synthetic_toy" in self.source_types and not self.synthetic:
            problems.append(
                "dataset contains synthetic_toy episodes but synthetic "
                "flag is False — toy data can never be labeled real")
        return problems


# ---------------------------------------------------------------------------
# 9. ModelCardPlaceholder — filled AFTER real Linux training, never before
# ---------------------------------------------------------------------------

@dataclass
class ModelCardPlaceholder:
    model_name: str
    training_dataset: str = ""
    policy_type: str = ""            # "bc", "bc_ppo", later "diffusion"
    embodiment_train: str = ""
    embodiment_eval: str = ""
    metrics: dict = field(default_factory=dict)   # MUST stay empty until real
    limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    intended_use: str = ""
    not_intended_use: str = ""
    is_placeholder: bool = True

    def validate(self) -> list[str]:
        problems: list[str] = []
        _check_nonempty_str(problems, "model_name", self.model_name)
        if self.is_placeholder and self.metrics:
            problems.append(
                "placeholder model cards must not carry metrics — fill them "
                "only from real Linux evaluation results "
                "(set is_placeholder=False then)")
        return problems


# ---------------------------------------------------------------------------
# JSON / JSONL round-trip helpers
# ---------------------------------------------------------------------------

_NESTED_FIELDS = {DemoEpisodeMetadata: {"provenance": DemoProvenance}}


def to_json_dict(obj) -> dict:
    d = asdict(obj)
    d["_schema"] = type(obj).__name__
    d["_schema_version"] = FLYWHEEL_SCHEMA_VERSION
    return d


def from_json_dict(cls, data: dict):
    """Reconstruct a schema object; raises on unknown/missing fields."""
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown fields {sorted(unknown)} — schema "
            "changes belong in xembench.flywheel.schemas, never ad hoc")
    for name, sub_cls in _NESTED_FIELDS.get(cls, {}).items():
        if isinstance(data.get(name), dict):
            data[name] = from_json_dict(sub_cls, data[name])
    return cls(**data)


def save_json(obj, path: str | Path) -> Path:
    require_valid(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(obj), indent=1) + "\n")
    return path


def load_json(cls, path: str | Path):
    obj = from_json_dict(cls, json.loads(Path(path).read_text()))
    require_valid(obj)
    return obj


def write_jsonl(objs: list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for obj in objs:
        require_valid(obj)
        lines.append(json.dumps(to_json_dict(obj)))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def read_jsonl(cls, path: str | Path) -> list:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first (see python -m "
            "xembench.flywheel docstring for the pipeline order)")
    objs = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            obj = from_json_dict(cls, json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"{path}:{i + 1}: bad {cls.__name__}: {e}") from e
        require_valid(obj)
        objs.append(obj)
    return objs

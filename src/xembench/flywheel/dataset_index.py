"""Flywheel dataset index: one queryable inventory of every demo episode.

Builds episode metadata + compact index records from the artifacts xembench
already produces — per-episode eval CSVs (``xembench.analysis.episodes``)
and demo HDF5 files (``xembench.data.schema``) — plus flywheel toy data.
Three files per index (siblings of ``--out``):

    <name>.jsonl            DatasetIndexRecord per line (the compact index)
    <name>.episodes.jsonl   full DemoEpisodeMetadata per line (validation
                            input; carries nested provenance)
    <name>.summary.csv      pandas-friendly flat view of the index

``provenance_id`` is ``<source_type>:<hash>`` so the compact index remains
filterable by source without loading the episodes sidecar.

CLI: ``python -m xembench.flywheel.build_dataset_index``.
"""

import math
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.flywheel.schemas import (
    DatasetIndexRecord,
    DemoEpisodeMetadata,
    DemoProvenance,
    derive_flywheel_split,
    read_jsonl,
    stable_hash,
    utc_now_iso,
    write_jsonl,
)
from xembench.utils.config import sanitize_path


def episodes_sidecar(index_path: str | Path) -> Path:
    p = Path(index_path)
    return p.with_suffix(".episodes.jsonl")


def summary_sidecar(index_path: str | Path) -> Path:
    return Path(index_path).with_suffix(".summary.csv")


def _provenance_id(prov: DemoProvenance | None) -> str:
    if prov is None:  # "" -> validation flags missing_provenance
        return ""
    key = "|".join([prov.source_type, prov.source_run_id, prov.source_path,
                    prov.generator_name])
    return f"{prov.source_type}:{stable_hash(key)}"


def source_type_of(record: DatasetIndexRecord) -> str:
    """Recover the source type encoded in provenance_id ('' if absent)."""
    return record.provenance_id.split(":", 1)[0] if record.provenance_id else ""


# ---------------------------------------------------------------------------
# Episode metadata from the two existing xembench artifact formats
# ---------------------------------------------------------------------------

def episodes_from_eval_csv(csv_path: str | Path, *,
                           synthetic: bool = False,
                           machine: str = "",
                           created_at: str | None = None,
                           source_label: str | None = None,
                           ) -> list[DemoEpisodeMetadata]:
    """One DemoEpisodeMetadata per eval-CSV row (policy rollouts).

    Toy-stamped CSVs (TOY_DATA checkpoint marker) are forced to
    source_type=synthetic_toy regardless of the flag — toy provenance can
    never be dropped by forgetting an argument. ``source_label`` overrides
    the recorded source path (e.g. a basename for portable toy rounds).
    """
    df = load_episodes(csv_path)
    is_toy = bool(toy_banner(df)) or synthetic
    created = created_at or utc_now_iso()
    source = source_label or sanitize_path(csv_path)
    episodes = []
    for row in df.itertuples(index=False):
        prov = DemoProvenance(
            source_type="synthetic_toy" if is_toy else "sim_policy",
            source_run_id=str(row.checkpoint),
            source_path=source,
            generator_name="xembench.eval" if not is_toy
            else "xembench.flywheel.toy_data",
            git_commit="",
            created_at=created,
            machine=machine,
            is_synthetic=is_toy,
        )
        split = derive_flywheel_split(row.task, row.eval_task,
                                      row.instruction_cond, row.color_cond)
        identity = "|".join(str(v) for v in (
            row.policy_type, row.task, row.eval_task, row.eval_embodiment,
            row.instruction_cond, row.color_cond, row.seed, row.episode_idx,
            row.checkpoint))
        instruction = getattr(row, "instruction", "")
        if not isinstance(instruction, str) or not instruction:
            instruction = "(instruction not recorded in this CSV)"
        family = getattr(row, "template_family", "")
        if not isinstance(family, str):  # empty CSV cells read back as NaN
            family = ""
        video = getattr(row, "video_path", "")
        if not isinstance(video, str):
            video = ""
        ret = getattr(row, "return_sum", math.nan)
        episodes.append(DemoEpisodeMetadata(
            episode_id=f"ep_{stable_hash(identity)}",
            task=row.eval_task,
            embodiment=row.eval_embodiment,
            policy=row.policy_type,
            split=split,
            instruction=instruction,
            instruction_template_id=family,
            paraphrase_group=row.instruction_cond,
            color_condition=row.color_cond,
            seed=int(row.episode_seed),
            success=bool(row.success),
            failure_label=row.failure_type,
            steps=int(row.steps),
            return_value=None if ret != ret else float(ret),
            native_action_dim=_native_dim(row.eval_embodiment),
            video_path=video,
            provenance=prov,
        ))
    return episodes


def _native_dim(embodiment: str) -> int:
    from xembench.flywheel.schemas import NATIVE_ACTION_DIMS
    return NATIVE_ACTION_DIMS.get(embodiment, -1)


def episodes_from_demo_h5(h5_path: str | Path, *,
                          synthetic: bool = False,
                          machine: str = "",
                          created_at: str | None = None,
                          ) -> list[DemoEpisodeMetadata]:
    """One DemoEpisodeMetadata per ep_* group of a schema-v1 demo file."""
    import json as _json

    import h5py

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(
            f"demo file {path} does not exist — collect demos first "
            "(docs/phase_b_commands.md steps 5-10) or point --demo-h5 at an "
            "existing file")
    created = created_at or utc_now_iso()
    episodes = []
    with h5py.File(path, "r") as f:
        task = str(f.attrs.get("task", ""))
        embodiment = str(f.attrs.get("embodiment", ""))
        control_mode = str(f.attrs.get("control_mode", "pd_joint_delta_pos"))
        prov = DemoProvenance(
            source_type="synthetic_toy" if synthetic else "sim_expert",
            source_run_id=path.stem,
            source_path=sanitize_path(path),
            generator_name="xembench.data",
            created_at=created,
            machine=machine,
            is_synthetic=synthetic,
        )
        for key in sorted(k for k in f.keys() if k.startswith("ep_")):
            ep = f[key]
            slots = {}
            if "slots" in ep.attrs:
                try:
                    slots = _json.loads(ep.attrs["slots"])
                except (ValueError, TypeError):
                    slots = {}
            native_dim = (int(ep["action_native"].shape[1])
                          if "action_native" in ep else -1)
            success = bool(ep.attrs.get("success", False))
            episodes.append(DemoEpisodeMetadata(
                episode_id=f"ep_{stable_hash(sanitize_path(path) + key)}",
                task=task,
                embodiment=embodiment,
                policy=str(f.attrs.get("source", path.stem)),
                split="train_seen",  # demos are training data by definition
                instruction=str(ep.attrs.get("instruction", "")),
                paraphrase_group="canonical",  # files store canonical text
                object_condition=str(slots.get("object", "")),
                color_condition=str(slots.get("color", "")),
                seed=int(ep.attrs.get("seed", -1)),
                success=success,
                failure_label="success" if success else "unknown_failure",
                steps=int(ep.attrs.get("episode_len", 0)),
                action_space=control_mode,
                native_action_dim=native_dim,
                hdf5_path=sanitize_path(path),
                provenance=prov,
            ))
    return episodes


# ---------------------------------------------------------------------------
# Index records, writing, loading, filtering
# ---------------------------------------------------------------------------

def index_record(ep: DemoEpisodeMetadata, dataset_id: str,
                 ) -> DatasetIndexRecord:
    has_h5 = bool(ep.hdf5_path)
    return DatasetIndexRecord(
        dataset_id=dataset_id,
        episode_id=ep.episode_id,
        task=ep.task,
        embodiment=ep.embodiment,
        split=ep.split,
        success=ep.success,
        failure_label=ep.failure_label,
        instruction_hash=stable_hash(ep.instruction),
        object_condition=ep.object_condition,
        color_condition=ep.color_condition,
        action_space=ep.action_space,
        num_steps=ep.steps,
        has_rgb=has_h5,
        has_state=has_h5,
        has_canonical_actions=has_h5,
        has_native_actions=has_h5,
        provenance_id=_provenance_id(ep.provenance),
    )


def build_index(episodes: list[DemoEpisodeMetadata], dataset_id: str,
                ) -> list[DatasetIndexRecord]:
    return [index_record(ep, dataset_id) for ep in episodes]


def write_index(records: list[DatasetIndexRecord],
                episodes: list[DemoEpisodeMetadata],
                out_jsonl: str | Path) -> list[Path]:
    """Write the index triplet; returns the written paths."""
    out = Path(out_jsonl)
    written = [write_jsonl(records, out),
               write_jsonl(episodes, episodes_sidecar(out))]
    df = pd.DataFrame([{k: v for k, v in r.__dict__.items()}
                       for r in records])
    csv_path = summary_sidecar(out)
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    return written


def load_index(path: str | Path) -> list[DatasetIndexRecord]:
    return read_jsonl(DatasetIndexRecord, path)


def load_index_episodes(index_path: str | Path,
                        ) -> list[DemoEpisodeMetadata] | None:
    """The episodes sidecar if present (None when index-only)."""
    sidecar = episodes_sidecar(index_path)
    if not sidecar.exists():
        return None
    return read_jsonl(DemoEpisodeMetadata, sidecar)


def filter_records(records: list[DatasetIndexRecord], *,
                   task: str | None = None,
                   embodiment: str | None = None,
                   split: str | None = None,
                   success: bool | None = None,
                   failure_label: str | None = None,
                   source_type: str | None = None,
                   validation_status: str | None = None,
                   ) -> list[DatasetIndexRecord]:
    out = []
    for r in records:
        if task is not None and r.task != task:
            continue
        if embodiment is not None and r.embodiment != embodiment:
            continue
        if split is not None and r.split != split:
            continue
        if success is not None and r.success != success:
            continue
        if failure_label is not None and r.failure_label != failure_label:
            continue
        if source_type is not None and source_type_of(r) != source_type:
            continue
        if (validation_status is not None
                and r.validation_status != validation_status):
            continue
        out.append(r)
    return out

"""Flywheel CLI contracts: refusals, overwrite guards, pack structure,
path hygiene, and the Linux plan generator (which must execute nothing)."""

import json
import subprocess
from pathlib import Path

import pytest

from xembench.flywheel.build_dataset_card import main as card_main
from xembench.flywheel.build_dataset_index import main as index_main
from xembench.flywheel.build_flywheel_pack import main as pack_main
from xembench.flywheel.compare_rounds import main as compare_main
from xembench.flywheel.generate_candidates import main as candidates_main
from xembench.flywheel.generate_toy_flywheel import generate_toy_round
from xembench.flywheel.generate_toy_flywheel import main as toy_main
from xembench.flywheel.select_candidates import main as select_main
from xembench.flywheel.validate_dataset import main as validate_main
from xembench.flywheel.toy_data import write_toy_eval_csv


@pytest.fixture(scope="module")
def toy_round(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "toy_round"
    return generate_toy_round(out)


# --- build_dataset_index -------------------------------------------------------

def test_index_refuses_toy_csv_without_synthetic(tmp_path):
    csv = write_toy_eval_csv("bc_baseline", tmp_path / "eval.csv")
    with pytest.raises(SystemExit, match="REFUSED"):
        index_main(["--eval-csv", str(csv),
                    "--out", str(tmp_path / "idx.jsonl")])


def test_index_cli_writes_triplet_and_guards_overwrite(tmp_path):
    csv = write_toy_eval_csv("bc_baseline", tmp_path / "eval.csv")
    out = tmp_path / "idx.jsonl"
    argv = ["--eval-csv", str(csv), "--out", str(out), "--synthetic"]
    assert index_main(argv) == 0
    for suffix in [".jsonl", ".episodes.jsonl", ".summary.csv"]:
        assert out.with_suffix(suffix).exists()
    with pytest.raises(FileExistsError, match="--overwrite"):
        index_main(argv)
    assert index_main(argv + ["--overwrite"]) == 0


def test_index_requires_some_input(tmp_path, capsys):
    with pytest.raises(SystemExit):
        index_main(["--out", str(tmp_path / "idx.jsonl")])
    assert "--eval-csv or --demo-h5" in capsys.readouterr().err


# --- validate_dataset ----------------------------------------------------------

def test_validate_cli_writes_md_and_json(toy_round, tmp_path):
    out = tmp_path / "report.md"
    rc = validate_main(["--index",
                        str(toy_round / "round0_dataset_index.jsonl"),
                        "--out", str(out)])
    assert rc == 0  # warnings only, no criticals
    assert out.exists() and out.with_suffix(".json").exists()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["summary"]["synthetic"] is True


# --- generate_candidates / select_candidates ------------------------------------

def test_candidates_and_select_cli_roundtrip(toy_round, tmp_path):
    cands = tmp_path / "cands.jsonl"
    assert candidates_main(
        ["--eval-csv", str(toy_round / "round0_eval_episodes.csv"),
         "--dataset-index", str(toy_round / "round0_dataset_index.jsonl"),
         "--out", str(cands)]) == 0
    assert cands.exists()
    md = cands.with_suffix(".md").read_text()
    assert "SYNTHETIC TOY DATA" in md  # auto-detected from the TOY stamp

    plan = tmp_path / "plan.json"
    assert select_main(
        ["--candidates", str(cands), "--strategy", "failure_targeted",
         "--budget-episodes", "20", "--out", str(plan),
         "--synthetic"]) == 0
    payload = json.loads(plan.read_text())
    assert payload["strategy"] == "failure_targeted"
    assert len(payload["selected_ids"]) == 20
    with pytest.raises(FileExistsError, match="--overwrite"):
        select_main(["--candidates", str(cands), "--strategy", "random",
                     "--budget-episodes", "5", "--out", str(plan)])


# --- build_dataset_card ---------------------------------------------------------

def test_dataset_card_cli(toy_round, tmp_path):
    out = tmp_path / "card.md"
    assert card_main(
        ["--index", str(toy_round / "round0_dataset_index.jsonl"),
         "--validation",
         str(toy_round / "round0_validation_report.json"),
         "--out", str(out)]) == 0
    text = out.read_text()
    assert "SYNTHETIC TOY DATA" in text
    card = json.loads(out.with_suffix(".json").read_text())
    assert card["synthetic"] is True
    assert card["total_episodes"] == card["successful_episodes"] \
        + card["failed_episodes"]


# --- compare_rounds -------------------------------------------------------------

def test_compare_rounds_cli(toy_round, tmp_path):
    out = tmp_path / "cmp.md"
    assert compare_main(
        ["--before", str(toy_round / "round0_eval_episodes.csv"),
         "--after", str(toy_round /
                        "round1_eval_bc_after_failure_targeted_data.csv"),
         "--demos-added", "50", "--out", str(out)]) == 0
    text = out.read_text()
    assert "SYNTHETIC TOY DATA" in text
    assert "Improvement per additional demonstration" in text
    with pytest.raises(FileExistsError, match="--overwrite"):
        compare_main(
            ["--before", str(toy_round / "round0_eval_episodes.csv"),
             "--after", str(toy_round / "round0_eval_episodes.csv"),
             "--out", str(out)])


# --- generate_toy_flywheel ------------------------------------------------------

def test_toy_cli_guards_overwrite(tmp_path):
    out = tmp_path / "round"
    assert toy_main(["--out", str(out)]) == 0
    with pytest.raises(FileExistsError, match="--overwrite"):
        toy_main(["--out", str(out)])
    assert toy_main(["--out", str(out), "--overwrite"]) == 0


# --- experiment plan (inside the toy round) -------------------------------------

def test_linux_plan_generated_not_executed(toy_round):
    sh = toy_round / "round1_linux_commands.sh"
    md = toy_round / "round1_collection_plan.md"
    assert sh.exists() and md.exists()
    # syntax-valid bash, refuses to run on macOS, nothing was executed
    subprocess.run(["bash", "-n", str(sh)], check=True)
    text = sh.read_text()
    assert 'if [[ "$(uname)" == "Darwin" ]]' in text
    assert "motion_plan_arm" in text
    plan_md = md.read_text()
    for tag in ["Linux-only", "Mac-safe", "collection", "training",
                "evaluation", "reporting"]:
        assert tag in plan_md, f"missing {tag} tag in collection plan"
    # no demo files appeared: the plan really is a plan
    assert not list(toy_round.glob("**/*.h5"))


# --- build_flywheel_pack --------------------------------------------------------

def test_pack_refuses_synthetic_round_without_flag(toy_round, tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        pack_main(["--round-manifest",
                   str(toy_round / "round_manifest.json"),
                   "--out", str(tmp_path / "pack")])


def test_pack_structure_warnings_checksums(toy_round, tmp_path):
    out = tmp_path / "pack"
    argv = ["--round-manifest", str(toy_round / "round_manifest.json"),
            "--out", str(out), "--synthetic"]
    assert pack_main(argv) == 0
    for f in ["README.md", "manifest.json", "round_manifest.json",
              "flywheel_summary.md", "data_quality_issues.csv",
              "checksums.txt", "model_card_placeholder.md",
              "round0_dataset_card.md", "round0_validation_report.md",
              "round0_vs_round1.md",
              "candidates/round1_candidates.jsonl",
              "plans/round1_selection_plan.json",
              "plans/round1_linux_commands.sh",
              "raw/round0_eval_episodes.csv",
              "index/round0_dataset_index.jsonl",
              "project_page_assets/flywheel_summary.md",
              "project_page_assets/flywheel_result_cards.json"]:
        assert (out / f).exists(), f"missing {f}"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["synthetic"] and "SYNTHETIC" in manifest["warning"]
    assert "SYNTHETIC" in (out / "README.md").read_text()
    cards = json.loads(
        (out / "project_page_assets/flywheel_result_cards.json").read_text())
    assert cards and all(c["synthetic"] for c in cards)
    # checksums cover every file
    lines = (out / "checksums.txt").read_text().strip().splitlines()
    files = {line.split(maxsplit=1)[1] for line in lines}
    on_disk = {str(f.relative_to(out)) for f in out.rglob("*")
               if f.is_file() and f.name != "checksums.txt"}
    assert files == on_disk
    # overwrite guard
    with pytest.raises(FileExistsError, match="--overwrite"):
        pack_main(argv)
    assert pack_main(argv + ["--overwrite"]) == 0


def test_no_home_paths_in_round_or_pack(toy_round, tmp_path):
    home = str(Path.home())
    offenders = []
    for f in toy_round.iterdir():
        if f.is_file() and home in f.read_text(errors="ignore"):
            offenders.append(f.name)
    assert offenders == [], f"home paths leaked into: {offenders}"

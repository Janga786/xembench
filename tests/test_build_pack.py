"""Artifact pack builder: structure, safety rails, provenance."""

import json
from pathlib import Path

import pytest

from xembench.artifacts.build_pack import main as build_main
from xembench.artifacts.manifest import PackManifest
from xembench.scripts.generate_toy_results import generate


@pytest.fixture()
def toy_csv(tmp_path) -> Path:
    csv = tmp_path / "eval_episodes.csv"
    generate(csv, seed=0)
    return csv


def _build(toy_csv, out: Path, *extra: str) -> None:
    argv = ["--eval-csv", str(toy_csv), "--out", str(out),
            "--release-name", "test_pack", "--synthetic", *extra]
    assert build_main(argv) == 0


def test_pack_structure_complete(toy_csv, tmp_path):
    out = tmp_path / "pack"
    _build(toy_csv, out)
    for f in ["README.md", "manifest.json", "metrics_summary.md",
              "transfer_matrix.csv", "transfer_matrix.md",
              "failure_report.md", "checksums.txt",
              "raw/eval_episodes.csv",
              "project_page_assets/project_summary.md",
              "project_page_assets/key_metrics.json",
              "project_page_assets/result_cards.json",
              "project_page_assets/figure_index.md",
              "project_page_assets/video_index.md"]:
        assert (out / f).exists(), f"missing {f}"
    for sub in ["figures", "videos", "configs", "logs"]:
        assert (out / sub).is_dir()


def test_manifest_provenance_and_warning(toy_csv, tmp_path):
    out = tmp_path / "pack"
    _build(toy_csv, out)
    m = PackManifest.load(out / "manifest.json")
    assert m.synthetic and "SYNTHETIC" in m.warning
    assert m.release_name == "test_pack"
    assert m.git_commit != "unknown"
    assert m.environment["python"]
    assert "metrics_summary.md" in m.metrics_files
    # synthetic banner must reach the human-facing files too
    assert "SYNTHETIC" in (out / "README.md").read_text()
    assert "SYNTHETIC" in (out / "project_page_assets/project_summary.md").read_text()


def test_refuses_toy_data_without_synthetic_flag(toy_csv, tmp_path):
    argv = ["--eval-csv", str(toy_csv), "--out", str(tmp_path / "p")]
    with pytest.raises(SystemExit, match="REFUSED"):
        build_main(argv)


def test_no_silent_overwrite(toy_csv, tmp_path):
    out = tmp_path / "pack"
    _build(toy_csv, out)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _build(toy_csv, out)
    _build(toy_csv, out, "--overwrite")  # explicit overwrite OK


def test_missing_input_fails_clearly(tmp_path):
    argv = ["--eval-csv", str(tmp_path / "nope.csv"),
            "--out", str(tmp_path / "p"), "--synthetic"]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_main(argv)


def test_configs_copied_and_videos_indexed(toy_csv, tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    (vids / "pick_cube_arm__panda__bc__canonical__0__success.mp4").write_bytes(b"")
    (vids / "apple_bowl_g1__g1__bc__canonical__0__success.mp4").write_bytes(b"")
    cfg = tmp_path / "some_config.yaml"
    cfg.write_text("a: 1\n")
    out = tmp_path / "pack"
    _build(toy_csv, out, "--configs", str(cfg), "--videos", str(vids))

    assert (out / "configs/some_config.yaml").exists()
    payload = json.loads((out / "videos/video_index.json").read_text())
    assert payload["n_videos"] == 2
    assert len(payload["demo_planning"]["cross_embodiment_pairs"]) == 1
    m = PackManifest.load(out / "manifest.json")
    assert m.video_count == 2
    # mp4s are referenced, never copied into the pack
    assert not list((out / "videos").glob("*.mp4"))


def test_checksums_cover_all_files(toy_csv, tmp_path):
    out = tmp_path / "pack"
    _build(toy_csv, out)
    lines = (out / "checksums.txt").read_text().strip().splitlines()
    files = {line.split(maxsplit=1)[1] for line in lines}
    on_disk = {str(f.relative_to(out)) for f in out.rglob("*")
               if f.is_file() and f.name != "checksums.txt"}
    assert files == on_disk
    assert all(len(line.split(maxsplit=1)[0]) == 64 for line in lines)


def test_key_metrics_match_toy_construction(toy_csv, tmp_path):
    out = tmp_path / "pack"
    _build(toy_csv, out)
    metrics = json.loads((out / "project_page_assets/key_metrics.json").read_text())
    assert metrics["synthetic"] is True
    p2g = next(p for p in metrics["transfer_pairs"]
               if p["pair"] == "pick_cube_arm→apple_bowl_g1")
    assert p2g["sr_native"] == pytest.approx(0.80)
    assert p2g["retention"] == pytest.approx(0.375)
    # paraphrase delta: canonical 0.80 vs heldout 0.65 -> 0.15
    d = next(d for d in metrics["paraphrase_deltas"]
             if d["task"] == "pick_cube_arm")
    assert d["paraphrase_delta"] == pytest.approx(0.15)
    cards = json.loads((out / "project_page_assets/result_cards.json").read_text())
    assert all(c["synthetic"] for c in cards)

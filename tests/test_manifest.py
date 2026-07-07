"""Manifest schema, reproducibility capture, and synthetic-labeling rules."""

import json

import pytest

from xembench.artifacts.manifest import (
    MANIFEST_SCHEMA_VERSION,
    PackManifest,
    capture_environment,
    sanitize_path,
)
from xembench.utils.config import repo_root


def test_create_fills_provenance():
    m = PackManifest.create(release_name="t", synthetic=True,
                            inputs=["outputs/toy/eval_episodes.csv"])
    assert m.schema_version == MANIFEST_SCHEMA_VERSION
    assert m.project == "xembench"
    assert m.generated_utc and "T" in m.generated_utc
    assert m.git_commit and m.git_commit != ""
    assert m.environment["python"]
    assert "torch" in m.environment
    assert m.warning  # synthetic -> auto warning


def test_synthetic_without_warning_refused():
    m = PackManifest.create(release_name="t", synthetic=True, inputs=["x.csv"])
    m.warning = ""  # attempt to pass toy data off as real
    with pytest.raises(ValueError, match="refusing to label"):
        m.validate()


def test_real_pack_has_no_warning():
    m = PackManifest.create(release_name="t", synthetic=False, inputs=["x.csv"])
    assert m.warning == ""
    m.validate()


def test_requires_inputs():
    m = PackManifest.create(release_name="t", synthetic=False, inputs=[])
    with pytest.raises(ValueError, match="at least one input"):
        m.validate()


def test_save_load_roundtrip(tmp_path):
    m = PackManifest.create(release_name="rt", synthetic=True,
                            inputs=["a.csv"], notes="hello")
    path = m.save(tmp_path / "manifest.json")
    loaded = PackManifest.load(path)
    assert loaded == m


def test_no_user_paths_in_saved_manifest(tmp_path):
    m = PackManifest.create(
        release_name="t", synthetic=True,
        inputs=[str(repo_root() / "outputs/toy/eval_episodes.csv")])
    path = m.save(tmp_path / "manifest.json")
    text = path.read_text()
    assert "/Users/" not in text and "/home/" not in text
    assert json.loads(text)["inputs"] == ["outputs/toy/eval_episodes.csv"]


def test_sanitize_path_variants():
    assert sanitize_path(repo_root() / "configs/bc.yaml") == "configs/bc.yaml"
    assert sanitize_path("/tmp/other") == "/tmp/other"


def test_environment_capture_shape():
    env = capture_environment()
    assert {"os", "arch", "python", "torch"} <= set(env)
    assert "hostname" not in str(env).lower()

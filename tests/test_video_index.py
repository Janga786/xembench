"""Video index generator: naming convention parsing and markdown output."""

from pathlib import Path

from xembench.analysis.video_index import build_index, parse_name, to_markdown


def _touch(root: Path, name: str) -> Path:
    p = root / name
    p.write_bytes(b"")
    return p


def test_parse_wellformed_name(tmp_path):
    p = _touch(tmp_path, "pick_cube_arm__g1__bc__transfer__0__failure_grasp_failure.mp4")
    e = parse_name(p)
    assert e.well_named
    assert e.task == "pick_cube_arm"
    assert e.embodiment == "g1"
    assert e.policy == "bc"
    assert e.split == "transfer"
    assert e.seed == "0"
    assert e.result == "failure_grasp_failure"


def test_malformed_name_listed_but_flagged(tmp_path):
    _touch(tmp_path, "cool_video.mp4")
    entries = build_index(tmp_path)
    assert len(entries) == 1 and not entries[0].well_named
    md = to_markdown(entries, tmp_path)
    assert "do not follow the naming convention" in md


def test_empty_index(tmp_path):
    md = to_markdown(build_index(tmp_path), tmp_path)
    assert "No videos yet" in md


def test_index_recurses_and_sorts(tmp_path):
    (tmp_path / "raw").mkdir()
    _touch(tmp_path / "raw", "apple_bowl_g1__g1__bc__canonical__1__success.mp4")
    _touch(tmp_path, "pick_cube_arm__panda__bc__heldout__0__success.mp4")
    entries = build_index(tmp_path)
    assert len(entries) == 2
    md = to_markdown(entries, tmp_path)
    assert "apple_bowl_g1" in md and "pick_cube_arm" in md


def test_json_groups_and_planning(tmp_path):
    from xembench.analysis.video_index import to_json

    _touch(tmp_path, "pick_cube_arm__panda__bc__canonical__0__success.mp4")
    _touch(tmp_path, "apple_bowl_g1__g1__bc__canonical__0__success.mp4")
    _touch(tmp_path, "pick_cube_arm__panda__bc__heldout__1__success.mp4")
    _touch(tmp_path, "not_conforming.mp4")
    payload = to_json(build_index(tmp_path), tmp_path)

    assert payload["n_videos"] == 4
    assert payload["unparsed"] == ["not_conforming.mp4"]
    assert set(payload["groups"]["embodiment"]) == {"panda", "g1"}
    assert len(payload["groups"]["task"]["pick_cube_arm"]) == 2

    # cross-embodiment pair: matched policy+split, different bodies
    pairs = payload["demo_planning"]["cross_embodiment_pairs"]
    assert len(pairs) == 1
    assert "cross-embodiment" in pairs[0]["reason"]

    # paraphrase set: same task/embodiment/policy across >1 split
    sets_ = payload["demo_planning"]["paraphrase_sets"]
    assert len(sets_) == 1 and sets_[0]["task"] == "pick_cube_arm"
    assert len(sets_[0]["videos"]) == 2

    # no absolute paths leak into the payload
    import json as _json

    assert str(tmp_path) not in _json.dumps(payload)


def test_json_empty_dir(tmp_path):
    from xembench.analysis.video_index import to_json

    payload = to_json(build_index(tmp_path), tmp_path)
    assert payload["n_videos"] == 0
    assert payload["demo_planning"]["cross_embodiment_pairs"] == []

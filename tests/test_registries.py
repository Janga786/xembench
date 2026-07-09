"""Registries and configs must stay parseable and internally consistent."""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

EXPECTED_TASKS = {
    "pick_cube_arm", "stack_cube_arm", "push_t_arm", "pull_tool_arm",
    "apple_bowl_g1", "transport_box_g1",
}
REQUIRED_TASK_KEYS = {"env_id", "embodiment", "family", "camera",
                      "max_steps", "demo_source"}
KNOWN_EMBODIMENTS = {"panda", "panda_stick", "g1"}


def _load(rel: str) -> dict:
    with open(REPO / rel) as f:
        return yaml.safe_load(f)


def test_task_registry_lists_expected_tasks():
    tasks = _load("configs/tasks.yaml")["tasks"]
    assert set(tasks) == EXPECTED_TASKS
    for name, task in tasks.items():
        missing = REQUIRED_TASK_KEYS - set(task)
        assert not missing, f"{name}: missing keys {missing}"
        assert task["embodiment"] in KNOWN_EMBODIMENTS
        # horizons are per-task (sized to each task's demo-length p95 +
        # margin — see tasks.yaml comments); assert sanity, not magic values
        assert isinstance(task["max_steps"], int) and 0 < task["max_steps"] <= 500


def test_task_embodiments_resolve_to_adapters():
    from xembench.embodiment import get_adapter

    tasks = _load("configs/tasks.yaml")["tasks"]
    for name, task in tasks.items():
        adapter = get_adapter(task["embodiment"])  # raises KeyError if unknown
        assert adapter.native_action_dim > 0


def test_transfer_pairs_reference_real_tasks():
    reg = _load("configs/tasks.yaml")
    tasks = set(reg["tasks"])
    pairs = reg["transfer_pairs"]
    assert len(pairs) == 2
    assert sum(1 for p in pairs if p.get("primary")) == 1
    for p in pairs:
        assert p["a"] in tasks and p["b"] in tasks
        # a transfer pair must actually cross embodiments
        emb_a = reg["tasks"][p["a"]]["embodiment"]
        emb_b = reg["tasks"][p["b"]]["embodiment"]
        assert (emb_a == "g1") != (emb_b == "g1"), f"pair {p} does not cross bodies"


def test_full_eval_matrix_preset_covers_the_benchmark():
    cfg = _load("configs/presets/full_eval_matrix.yaml")
    assert set(cfg["tasks"]) == EXPECTED_TASKS
    assert {p["type"] for p in cfg["policies"]} == {"bc", "bc_ppo"}
    assert set(cfg["instruction_conds"]) == {"canonical", "heldout"}
    assert set(cfg["color_conds"]) == {"train", "unseen"}
    assert cfg["transfer"] is True
    assert cfg["episodes_csv"]  # per-episode records must be on for B5


def test_all_presets_parse_and_name_a_consumer():
    presets = sorted((REPO / "configs" / "presets").glob("*.yaml"))
    assert len(presets) >= 10
    for path in presets:
        cfg = _load(f"configs/presets/{path.name}")
        assert isinstance(cfg, dict) and cfg, f"{path.name} empty/invalid"
        header = path.read_text().splitlines()[0]
        assert header.startswith("# PRESET:"), f"{path.name} missing header"
        assert "Consumer:" in path.read_text(), f"{path.name} missing consumer note"

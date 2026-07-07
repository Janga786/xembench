# CI notes

`.github/workflows/ci.yml` runs the **pure unit subset** on every push/PR:
language stack, adapters, canonical metadata, registries, analysis tooling,
HDF5 schema validator, model shapes, path discipline. It installs CPU torch
explicitly and `pip install -e . --no-deps` — the simulator is intentionally
absent.

## Why the simulator suites are NOT in CI

- Installing `mani_skill`+`sapien` adds ~500 MB and, on Linux, this repo's
  own uv index gating would resolve **cu128 torch** (multi-GB, targeting the
  RTX 5090) — wasteful and slow on a CPU-only runner.
- The `-m sim` suite needs working Vulkan; GitHub runners have no GPU and
  headless CPU rendering through SAPIEN is possible but was not validated
  as *stable* on runners — a flaky CI is worse than a smaller one.
- `test_bc_overfit` and `test_dataset_roundtrip` would run (they only need
  torch+h5py) but sit on the edge of the runner budget; they stay local for
  now. TODO(Janga): move them into CI if runtimes prove acceptable.

Local equivalents (the real gates):

| Gate | Command | Machine |
|---|---|---|
| everything sim-free | `make test` | any |
| env integration | `make test-sim` | Mac/Linux with working Vulkan |
| full Phase A chain | `make smoke` | Mac/Linux |

## Keeping CI honest

The workflow lists test files explicitly. When adding a test module, decide:
pure (imports at most torch/h5py/pandas/yaml — add to the workflow list) or
simulator-bound (leave local; mark `@pytest.mark.sim` if it creates envs).
A module that imports `mani_skill` at import time cannot go on the list —
`xembench.data.writer` deliberately imports it lazily for this reason.

No lint/type-check job: no linter is configured in the repo (style is
"match surrounding code", see CONTRIBUTING.md). If one is ever adopted,
add it here as a separate fast job.

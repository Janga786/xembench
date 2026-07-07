# Contributing / working on xembench

Research repo, one primary developer, two machines (Mac dev seat, Linux
RTX 5090 for training). Optimize for reproducibility and honest results,
not enterprise ceremony.

## Setup

```bash
./scripts/setup_mac.sh      # macOS (dev/smoke)
./scripts/setup_linux.sh    # Linux 5090 (training)
uv run python scripts/check_env.py   # verify either
```

## Testing

```bash
make test        # fast sim-free suite — run before every commit
make test-sim    # creates real envs on CPU sim (~10 s)
make smoke       # full Phase A gate (~10-15 min) — run before pushing
                 # anything that touches envs/adapters/training
```

Conventions: sim-touching tests carry `@pytest.mark.sim` (excluded by
default via pyproject `addopts`). Tests that only need CSVs/torch/YAML stay
pure so CI can run them. No test may depend on trained checkpoints or
long episodes.

## Adding things

- **Task**: docs/tasks.md → "Adding a task" (env subclass → tasks.yaml →
  template/vocab → setup_check green).
- **Robot adapter**: docs/canonical_action_space.md → "Adding a new robot
  adapter" (subclass, pin joint order, register, extend the parametrized
  adapter tests).
- **Metric**: compute it in `xembench/analysis/` from the episode CSV if
  possible (post-hoc, testable on toy data); only touch `eval/runner.py`
  when the metric needs in-episode signals. Document in docs/metrics.md.
- **Failure label**: add to `analysis/taxonomy.FailureType` (never ad hoc
  strings in data), assign it in the runner, cover it in tests.
- **Config key**: read with `cfg.get(...)`; wrap paths in `resolve_path()`
  (SAPIEN chdirs — see docs/configs.md).

## Style

- Clean, boring Python; type hints on public function signatures.
- Comments explain design decisions, not restate code.
- Loud, actionable errors (say what to run to fix it).
- No formatter/linter is configured — match the surrounding code; keep
  modules small.
- `TODO(Janga):` marks deferred work; keep each one specific.

## Checklist before committing

1. `make test` green (and `make test-sim` if envs/wrappers changed).
2. No large binaries staged (`git status` — datasets/checkpoints/videos are
   gitignored; keep it that way).
3. New behavior documented (docs/ is part of the deliverable).
4. No fabricated numbers anywhere — synthetic data must be labeled
   (see `generate_toy_results` / `toy_banner`).
5. Commit messages state what changed and why.

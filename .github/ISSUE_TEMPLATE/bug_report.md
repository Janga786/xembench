---
name: Bug report
about: Something broke — env creation, training, eval, analysis, or packaging
labels: bug
---

## What happened

<!-- one paragraph; paste the exact error tail below -->

```
<error output>
```

## Reproduce

```bash
# exact command(s), including --set overrides and --seed
```

## Environment

Paste the output of:

```bash
uv run python scripts/check_env.py
```

## Context

- [ ] macOS (dev/smoke) or [ ] Linux (Phase B)
- git commit: `git rev-parse --short HEAD`
- Does `make test` pass? Does `make test-sim`?
- If simulator-related: does `uv run python -m xembench.scripts.setup_check` pass?

## Suspected area (optional)

envs / adapters / language / data / training / eval / analysis / artifacts

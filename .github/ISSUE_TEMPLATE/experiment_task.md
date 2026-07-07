---
name: Experiment task
about: A concrete training/eval run to execute (mostly Phase B, Linux)
labels: experiment
---

## Goal

<!-- what question does this run answer? one sentence -->

## Command(s)

```bash
# exact planned commands, preset + overrides
```

## Runbook step

<!-- which step of docs/phase_b_commands.md this belongs to, if any -->

## Gate / success criterion

<!-- the number or condition that decides pass/fail BEFORE running,
     e.g. ">=70% expert success on apple_bowl_g1" -->

## Expected outputs

- [ ] checkpoints: `data/checkpoints/...`
- [ ] eval CSVs: `data/results/...`
- [ ] demo files validated (`validate_demo_file`)
- [ ] seeds recorded

## After the run

- [ ] result recorded against the gate (pass/fail, no reinterpretation)
- [ ] config + git SHA confirmed in the run dir
- [ ] artifact pack rebuilt if this feeds the release

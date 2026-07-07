# Configuration guide

The config system is deliberately simple: plain YAML files loaded by each
script, with repeatable dotted overrides — no Hydra, no layering.

```bash
uv run python -m xembench.scripts.train_bc \
    --config configs/presets/bc_train_full.yaml \
    --set task=stack_cube_arm --set lr=1e-4 --seed 2
```

Values after `=` are parsed as YAML (numbers/bools/lists work). Every run
directory receives a `config.yaml` dump + git SHA (`utils/config.py:
dump_run_config`) so results are attributable.

**Paths**: SAPIEN changes the process working directory during shader
compilation, so any path used after env creation must be absolute. Scripts
route user paths through `resolve_path()` (anchors relative paths at the
repo root); trainers additionally `.resolve()` their out_dirs. If you add a
new script, do the same — there is a unit test for `resolve_path` but only
discipline protects new call sites.

## Base configs (`configs/*.yaml`) — script defaults

| File | Consumer script | Notes |
|---|---|---|
| `tasks.yaml` | everything | THE task registry: env ids, embodiments, cameras, template slots, budgets, demo sources, transfer pairs. Edit this to add tasks. |
| `bc.yaml` | `scripts.train_bc` | full-training defaults |
| `collect.yaml` | `scripts.collect_demos`, `experts.motion_plan_arm` | tiny defaults for pipeline checks |
| `ppo_expert_g1.yaml` | `scripts.train_expert_ppo` | G1 expert defaults (Linux-scale) |
| `ppo_finetune.yaml` | `scripts.train_ppo_finetune` | fine-tune defaults (Linux-scale) |
| `eval_matrix.yaml` | `scripts.evaluate` | full-grid defaults |

## Presets (`configs/presets/*.yaml`) — pick by intent

Each preset's header comment states its consumer command, machine
(Mac-safe vs Linux-only), expected runtime, and pass/fail gate.

| Preset | Intent | Machine |
|---|---|---|
| `bc_train_small` | does this demo file train at all? | Mac or Linux |
| `bc_train_full` | B3 production BC run per task | Linux (Mac works, slow) |
| `ppo_finetune_small` | fine-tune plumbing check | Mac or Linux |
| `ppo_finetune_full` | B4 production fine-tune | **Linux only** |
| `g1_expert_train` | B2 G1 expert (the de-risk item) | **Linux only** |
| `g1_expert_collection` | expert → 200 G1 demos | Linux recommended |
| `panda_demo_collection` | B1 motion-planning demos | **Linux only** (mplib) |
| `phase_a_eval` | tiny eval incl. one transfer cell | Mac or Linux |
| `full_eval_matrix` | B5 paper-table grid, 3 seeds | Linux recommended |
| `linux_gpu_smoke` | first run after setup_linux.sh | **Linux only** |

There is no `mac_smoke.yaml`: the Mac smoke is `make smoke`, which chains
several stages with inline `--set` overrides (see the Makefile — it is the
executable definition of the Phase A gate).

## Keys people actually tune

- **BC**: `steps`, `batch_size`, `vision` (`nature_cnn`|`resnet18`),
  `paraphrase_augment`, `augment` (DrQ shift), `preload`.
- **PPO expert**: `num_envs`, `total_timesteps`, `gamma`/`gae_lambda`
  (0.9/0.9 suit 50–100-step episodes), `reward_mode`.
- **PPO fine-tune**: `bc_reg_coef` (raise on collapse, lower on stall),
  `init_logstd`, `freeze_vision`, `step_penalty`, `num_envs` (VRAM-bound:
  rgb rollout buffers dominate).
- **Eval**: `episodes_native`/`episodes_transfer`, `seeds`,
  `instruction_conds`, `color_conds`, `episodes_csv` (per-episode records
  for failure analysis; omit to skip).

## Adding a config key

Read it with `cfg.get("key", default)` at the consumer; document it in the
preset headers if it is tunable; if it is a path, wrap it in
`resolve_path()`.

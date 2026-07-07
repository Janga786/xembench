# xembench — X-Embodiment Language-Grounded Manipulation Benchmark

**Status: Phase A.5 complete — pipeline built, smoke-verified, documented,
and hardened on macOS (v0.2). Phase B (actual training + benchmark numbers)
pending — requires the Linux / RTX 5090 machine. No benchmark results exist
yet; nothing in this repo claims any.**

## Thesis

When language, task structure, and policy architecture are all shared, how
much manipulation performance survives a change of robot body — and where
exactly does transfer fail? xembench turns that question into a measurable
systems problem: one language-conditioned policy network speaks a canonical
21-dimensional action interface, thin per-robot adapters translate to a
Franka Panda tabletop arm and a Unitree G1 humanoid upper body (ManiSkill3),
and the benchmark measures the success-rate drop under zero-shot
cross-embodiment evaluation — same weights, swapped adapter — alongside
held-out-paraphrase and unseen-color generalization.

## Why cross-embodiment transfer matters

Robot learning is accumulating large multi-robot datasets on the premise
that skills transfer across bodies. That premise is rarely *measured* under
controlled conditions: most benchmarks vary tasks while holding the robot
fixed. xembench holds language, tasks (analogous pairs), architecture, and
training recipe fixed and varies only the body — so a transfer failure is
attributable to embodiment, and the failure taxonomy says whether it broke
at grounding (wrong object) or at motor control (reach/grasp). The
benchmark deliberately does **not** claim to solve embodiment transfer; it
makes it auditable.

## What is implemented now (Phase A, smoke-verified on macOS)

- 6 tasks across 2 embodiments (ManiSkill3, CPU or GPU sim):
  `PickCubeLang-v1`, `StackCubeLang-v1` (ours: color-randomized,
  language-load-bearing), stock `PushT-v1`, `PullCubeTool-v1`, and G1
  `UnitreeG1PlaceAppleInBowl-v1`, `UnitreeG1TransportBox-v1`.
- Canonical 21-dim action space + Panda / panda_stick / G1 adapters with
  round-trip and live-env validation tests.
- Language stack with a provably leak-free train/heldout paraphrase split
  and disjoint train/unseen color sets (both enforced by tests).
- HDF5 demo pipeline (HF-replay conversion + live collection), BC trainer,
  CleanRL PPO (state-expert and visual fine-tune modes), evaluation matrix
  with per-episode records, failure-taxonomy analysis CLIs.
- The full path — train BC → evaluate natively → zero-shot on the other
  embodiment → CSV + transfer table — runs end-to-end in `make smoke`.

## What is NOT implemented / not run yet

- No demonstrations at scale, no trained policies, **no numbers** (Phase B).
- Motion-planning demo collection is written but untested (mplib is
  Linux-only).
- G1 PPO experts untrained (need GPU-parallel sim).
- Diffusion policy = design stub. Fine-grained failure labels beyond
  success/timeout need Phase B instrumentation.

## Quickstart — macOS (development/smoke; Apple Silicon, no CUDA)

```bash
./scripts/setup_mac.sh     # brew uv + molten-vk, uv sync, env check
make smoke                 # full Phase A gate (~10-15 min)
```

## Quickstart — Linux / RTX 5090 (Phase B)

```bash
./scripts/setup_linux.sh   # cu128 torch (sm_120), GPU + Vulkan checks, sim FPS bench
make phase-b-help          # prints the ordered Phase B command sequence
```

Then follow [docs/phase_b_commands.md](docs/phase_b_commands.md) step by
step (demo collection → G1 experts → BC → PPO → full eval matrix).
Docker alternative: `docker build -f docker/Dockerfile.cu128 -t xembench .`
and run with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`.

## Common commands

```bash
make test                  # fast sim-free suite (~10 s)
make test-sim              # integration tests, real envs on CPU sim (~10 s)
make check                 # create all 6 envs, validate pinned constants
make mac-check             # environment report (OS/torch/backends/ManiSkill)
make toy-report            # analysis pipeline on labeled synthetic data
make artifact-pack-toy     # full artifact pack from synthetic data (rehearses Phase B packaging)
make artifact-pack-help    # how to pack real Phase B results
make smoke                 # everything above chained + BC/PPO/eval smokes

uv run python -m xembench.scripts.setup_check
uv run python -m xembench.scripts.train_bc --config configs/bc.yaml --set task=pick_cube_arm
uv run python -m xembench.scripts.evaluate --config configs/eval_matrix.yaml
uv run python -m xembench.analysis.summarize_results --input data/results/eval_episodes.csv --out reports/summary.md
```

Every script accepts `--seed N` and repeatable `--set key=value` overrides;
run dirs receive a config dump + git SHA. Presets for common runs live in
`configs/presets/` (see [docs/configs.md](docs/configs.md)).

## Repository structure

```
src/xembench/
  constants.py     canonical action/proprio layout, pinned joint orders (source of truth)
  embodiment/      EmbodimentAdapter base + panda/panda_stick/g1 + registry
  envs/            factory, UnifiedObsWrapper, custom *Lang-v1 envs
  language/        templates, paraphrase split, tokenizer + vocab.json, parser
  data/            HDF5 schema/writer/validator, DemoDataset, HF converter
  experts/         motion-planning collector (Linux), PPO-expert loader
  models/          encoders, BCPolicy, actor-critics, diffusion stub
  training/        BC trainer, CleanRL PPO (expert + fine-tune modes)
  eval/            episode runner (Wilson CIs), eval matrix, table report
  analysis/        failure taxonomy, episode-CSV tools, transfer matrix
  scripts/         python -m xembench.scripts.<entrypoint>
configs/           tasks.yaml registry, per-stage defaults, presets/
docs/              architecture, canonical space, tasks, metrics, Phase B plan
tests/             fast suite + `-m sim` integration suite
reports/, outputs/, artifacts/   generated summaries / scratch / media (gitignored payloads)
```

## Expected outputs

| Producer | Output |
|---|---|
| demo collection | `data/demos/{embodiment}/{env_id}/{source}.h5` |
| training | `data/checkpoints/<run>/{best.pt,last.pt,config.yaml,tb/}` |
| evaluation | `data/results/eval.csv` (aggregate) + `data/results/eval_episodes.csv` (per episode) + printed tables |
| analysis | `reports/*.md`, `reports/*.csv` summaries |
| artifact packer | `artifacts/releases/<name>/` — manifest, metrics, figures, project-page assets, checksums ([docs/artifact_packs.md](docs/artifact_packs.md)) |

## Interpreting results

Aggregate rows carry `success_rate` with 95% Wilson CIs; `task` is the
training task, `eval_task` the evaluation task — rows where they differ are
transfer cells. The headline metric is the **transfer drop**
`SR_native − SR_transfer` for the pairs pick_cube_arm ↔ apple_bowl_g1 and
stack_cube_arm ↔ transport_box_g1; paraphrase robustness and unseen-color
deltas are the secondary axes. Full column definitions and formulas:
[docs/metrics.md](docs/metrics.md). Smoke-run CSVs contain ~0% success
rates by design (tiny data/training) — they verify plumbing, not policies.

## Documentation

[architecture](docs/architecture.md) ·
[canonical action space](docs/canonical_action_space.md) ·
[tasks](docs/tasks.md) · [language splits](docs/language_splits.md) ·
[metrics](docs/metrics.md) · [configs](docs/configs.md) ·
[Phase A status](docs/phase_a_status.md) ·
[Phase B plan](docs/phase_b_linux_plan.md) ·
[Phase B commands](docs/phase_b_commands.md) ·
[report outline](docs/report_outline.md) ·
[project page](docs/project_page.md) · [CONTRIBUTING](CONTRIBUTING.md)

## Current limitations

Hand-designed canonical interface (kinematic aliasing between bodies is
real and documented); analogous rather than identical transfer tasks;
camera viewpoint differs across embodiments (ablatable — the G1 exposes a
third-person camera too); small closed-vocabulary language distribution;
simulation only; two embodiments. See
[docs/canonical_action_space.md](docs/canonical_action_space.md) for the
full list and the adapter extension path (Isaac Lab / LeRobot / real
hardware are out of scope for now but the adapter seam is designed for
them).

## Citation

Placeholder — no write-up exists yet. If you build on the benchmark design,
cite this repository:

```bibtex
@misc{xembench2026,
  title  = {xembench: An X-Embodiment Language-Grounded Manipulation Benchmark},
  author = {Jangara},
  year   = {2026},
  note   = {https://github.com/Janga786/xembench (Phase A.5, pre-results)}
}
```

Built on [ManiSkill3](https://github.com/haosulab/ManiSkill) (Mu et al.) —
cite it for the simulator, tasks, and demo datasets.

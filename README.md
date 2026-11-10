# xembench: cross-embodiment language-conditioned manipulation in ManiSkill3

[![CI](https://github.com/Janga786/xembench/actions/workflows/ci.yml/badge.svg)](https://github.com/Janga786/xembench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

xembench is a simulation study and evaluation harness. It asks how much of a
language-conditioned manipulation policy survives when the same network is
moved from one robot body to another: a Franka Panda arm and the upper body
of a Unitree G1 humanoid, both in ManiSkill3. Both bodies share one 21-dim
canonical action interface, with a thin adapter per robot.

**Status, plainly.** The pipeline runs end to end, and I ran a 6,550-episode
baseline evaluation on one RTX 5090 in July 2026. The cross-embodiment
question itself is not answered yet. With the behavior-cloning (BC) baselines, at least one task in each transfer
pair scores 0% natively, so every transfer cell is 0% and no transfer drop
can be measured. Everything here is
simulation only; nothing ran on a physical robot.

## What I built

- Six tasks across three bodies (Panda, Panda with a stick tool, G1 upper
  body). Two are my language-wrapped versions of ManiSkill's PickCube and
  StackCube; the others are ManiSkill3 stock environments.
- A 21-dim canonical action space with per-robot adapters, round-trip tests,
  and live joint-order checks ([docs/canonical_action_space.md](docs/canonical_action_space.md)).
- A language stack with a train / held-out paraphrase split and disjoint
  train / unseen color sets, both enforced by tests.
- Demonstration generation: mplib motion planning for pick_cube and
  stack_cube, ManiSkill's published demos for PushT and PullCubeTool, and PPO experts I
  trained for the two G1 tasks.
- One BC policy network for all bodies (optional action chunking), a
  CleanRL-style PPO trainer, an evaluation matrix with 95% Wilson
  confidence intervals, a failure taxonomy, and a data-selection
  ("flywheel") module.
- 198 tests: 193 run without the simulator, 5 need ManiSkill. GitHub
  Actions runs a pure-unit subset.

## Results (simulation, BC baselines)

Native success, 3 seeds x 100 episodes = 300 episodes per cell, 95% Wilson
CI. Source: [reports/summary.md](reports/summary.md),
[reports/failures.md](reports/failures.md).

| Task | Body | Canonical instruction | Held-out paraphrase |
|---|---|---:|---:|
| push_t | Panda + stick | 45% [39, 51] | 46% [41, 52] |
| transport_box | G1 | 17% [13, 21] | 21% [17, 26] |
| pull_tool | Panda | 2% [1, 5] | 0% [0, 2] |
| pick_cube | Panda | 0% [0, 1] | 0% [0, 1] |
| stack_cube | Panda | 0% [0, 1] | 0% [0, 1] |
| apple_bowl | G1 | 0% [0, 1] | 0% [0, 1] |

- Cross-embodiment transfer (pick_cube <-> apple_bowl, stack_cube <->
  transport_box, both directions): 0% in all cells (150 episodes per BC
  cell, 50 per BC+PPO cell)
  ([reports/transfer_matrix.md](reports/transfer_matrix.md)). Because one
  side of each pair is already 0% natively, this says nothing yet about
  transfer.
- Held-out paraphrases did not lower success on the two tasks that work.
- PPO fine-tuning from the BC policies (pick, stack, apple_bowl; 1 seed)
  stayed at 0%.

## Follow-up: why pick_cube stays at 0%

Replaying the ground-truth demonstration actions open-loop through the same
adapter and environment succeeded 5 of 5 times, so the interface and data
are not the cause. Instrumented rollouts reach within 2-3 cm of the cube and
close the gripper, but do not complete the grasp.

I then tried single-seed probes of 25-50 episodes each. At this size, 4%
means 1 success in 25 episodes.

<img src="docs/media/pick_scaling.png" width="520" alt="pick_cube success vs number of demonstrations for single-step and chunked BC"/>

| Change (pick_cube probe) | Success |
|---|---:|
| single-step BC, 200 demos | 0% |
| 2,000 demos | 0% |
| action chunking k=16, 200 demos | 0% |
| temporal ensembling | 0% |
| action chunking k=8, 2,000 demos | 4% |
| k=8 plus gripper binarization or receding horizon | 4% |
| ResNet-18 encoder | 0% (overfits) |
| 10,000 demos, k=8 | 2% |
| dense-reward visual PPO fine-tune | 0.1% (training success at 460k steps; run stopped) |

Rerun at 3 seeds x 100 episodes, the k=8 / 2,000-demo recipe gave
pick_cube 0.2% and stack_cube 0.1%, so the 4% probe did not replicate. The
same recipe raised pull_tool from 1.3% (mean of canonical and held-out) to 14.7% (held-out paraphrase 15.3%).
These chunked-BC numbers come from
[reports/phase_b_campaign_report.md](reports/phase_b_campaign_report.md)
section 4b; their per-episode CSV is not committed.

My reading: with this setup (single-frame BC, 128 px images, MLP action
head), more data and different execution strategies did not fix
grasp-critical tasks. Sequence models such as ACT or diffusion policies are
the obvious next baseline. I have not tried them.

## Data-selection round 1

One round comparing failure-targeted vs random collection of 50 extra
episodes. Result: no difference (+0.11 vs -0.28 points, inside the noise).
A methods lesson came out of it: comparing a single-seed arm against a
pooled 3-seed baseline produced false swings of 5-9 points; seed-matched
comparison removed them. Details in the campaign report, section 4.

## Engineering problems found and fixed

Details in [reports/phase_b_campaign_report.md](reports/phase_b_campaign_report.md)
and [reports/b0_linux_prep_report.md](reports/b0_linux_prep_report.md).

1. Evaluation time limits were shorter than the demonstrations (50-step
   caps vs 66-105-step demos). Reset to the demo-length 95th percentile plus a
   margin and re-ran the whole matrix.
2. PhysX GPU collision memory demand grew as the policy learned contact
   (8M to 150M bytes); the 4M default silently dropped contacts at 2,048
   envs.
3. A PPO success statistic read 100% for a bad policy, because short
   windows contained only early-terminating successes. Fixed by pooling
   episodes across windows.
4. Staged reward bonuses caused PPO KL spikes (approx_kl 28.5) once
   successes began; added a target-KL epoch cutoff.
5. The G1 hand joints are mirrored in the URDF, so one constant opened one
   hand and closed the other. A joint-sweep check caught it before demo
   collection.

## What is and is not in this repo

Committed: all code, configs, tests, aggregated result tables in
[reports/](reports/), the two campaign reports, and the scaling figure.
Not committed: per-episode CSVs, demonstrations, checkpoints, run logs,
artifact packs, and rollout videos (the video index in
`artifacts/videos/` lists files that are not in the repo).
The numbers above can be checked against the committed tables, but not
re-derived from raw data without re-running.

## Running it

```bash
# Linux + NVIDIA (tested on RTX 5090 / CUDA 12.8)
./scripts/setup_linux.sh        # uv env, CUDA/Vulkan checks, sim FPS bench
make smoke                      # full pipeline check (~10 min)
make phase-b-help               # ordered training runbook

# macOS (development and smoke tests only)
./scripts/setup_mac.sh && make smoke
```

`make test` runs the simulator-free tests; `make test-sim` runs the 5
ManiSkill tests. Example:
`uv run python -m xembench.scripts.train_bc --config configs/presets/bc_train_full.yaml --set task=pick_cube_arm --set chunk=8`,
then
`uv run python -m xembench.scripts.evaluate --config configs/presets/full_eval_matrix.yaml`.
Evaluation writes `data/results/eval.csv`; rows where `task` and
`eval_task` differ are transfer cells. Metric definitions:
[docs/metrics.md](docs/metrics.md).

## Repository layout

```
src/xembench/
  constants.py     canonical action/proprio layout, pinned joint orders
  embodiment/      adapter base class, panda / panda_stick / g1, registry
  envs/            env factory, observation wrapper, *Lang-v1 envs
  language/        templates, paraphrase split, tokenizer, parser
  data/            HDF5 schema, writer, validator, dataset (chunked)
  experts/         mplib motion-planning collector, PPO-expert loader
  models/          encoders, BC policy (+ action chunking), actor-critics
  training/        BC trainer, PPO
  eval/            episode runner, evaluation matrix, Wilson CIs
  analysis/        failure taxonomy, episode CSVs, transfer matrix
  flywheel/        data-selection schemas, validation, candidates, rounds
configs/           task registry and presets
docs/              design notes, runbooks, and planning documents
reports/           campaign reports and generated result tables
```

## Limitations

- Simulation only (ManiSkill3 / SAPIEN); no hardware.
- Transfer is unmeasured: the paired tasks are at 0% natively.
- Transfer pairs are analogous tasks, not identical ones, and there are
  only two embodiments.
- Baselines are single-frame BC at 128 px with a closed vocabulary.
- The pick_cube intervention probes are single-seed and 25-50 episodes.
- Raw per-episode data and checkpoints are not in the repo.

## Credits

Built on [ManiSkill3](https://github.com/haosulab/ManiSkill) and SAPIEN,
including ManiSkill's stock environments and published
demonstrations; motion planning uses mplib; the PPO trainer follows CleanRL.
Action chunking follows Zhao et al., *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware* (ACT), 2023.

MIT License. See [LICENSE](LICENSE).

# Phase B Campaign Report — Linux / RTX 5090 · 2026-07-07 → 07-09

The complete Phase B execution: every dataset, model, and number the
benchmark now contains, how it was produced, what broke along the way, and
what the results honestly say. Companion documents:
[b0_linux_prep_report.md](b0_linux_prep_report.md) (environment bring-up),
[phase_b_commands.md](../docs/phase_b_commands.md) (the runbook executed),
[flywheel_linux_plan.md](../docs/flywheel_linux_plan.md) (round 1, executed
after Phase B). All work on branch `phase-b-linux-prep`; every number traces
to a checkpoint + config.yaml + git SHA per the repo's provenance rules.

## 1. What was produced

| Artifact | Count / size | Gate |
|---|---|---|
| Demonstrations | 6 tasks, 1,202 episodes (200/task arm+PushT, 201/task G1) | `validate_demo_file` clean on all 10 files |
| G1 PPO experts | 2 (apple_bowl, transport_box) + eval tooling | success measured empirically (below) |
| BC policies | 18 (6 tasks × 3 seeds), zero failed runs | val loss converged on all |
| PPO fine-tunes | 3 (pick, stack, apple_bowl; seed 0) | ran to budget; see findings |
| Eval matrix | 73 cells, 6,550 episodes (v2, corrected horizons) | every cell present |
| Reports + pack | `reports/*`, `artifacts/releases/phase_b_v1` | checksums + manifest, no path leaks |
| Flywheel round 0 index | 7,752 records validated deep | zero critical/error findings |
| Flywheel round 1 | 2 arms × 50-episode budget (below) | first real (non-synthetic) round |

## 2. Headline results (eval v2, corrected horizons)

Native success rate, mean over seeds and conditions (100 episodes/cell,
Wilson CIs in `data/results/eval.csv`):

| Task | BC | BC+PPO | Failure profile |
|---|---|---|---|
| push_t_arm | **45.7%** | — | robust to held-out paraphrases (46.3% vs 45.0%) |
| transport_box_g1 | **18.8%** | — | heldout 21.0% vs canonical 16.7% |
| pull_tool_arm | 1.3% | — | ~all timeout |
| pick_cube_arm | 0.0% | 0.0% | 100% timeout |
| stack_cube_arm | 0.0% | 0.0% | 100% timeout |
| apple_bowl_g1 | 0.0% | 0.0% | 100% timeout |

Cross-embodiment transfer (both directions of both pairs): 0% — expected,
since each pair has at least one 0% native side; the transfer machinery
itself is exercised and correct (verified end-to-end with the smoke cells).

The honest interpretation: **the benchmark works; the BC baselines are
weak on precision manipulation.** Tasks whose demonstrations give dense
state coverage under the policy's horizon (push_t's 200 full-length pushing
demos; transport's 201 stochastic-expert demos) yield meaningful success.
Tasks demanding millimeter grasp alignment (pick, stack, apple) defeat
128-px nature_cnn BC from 200 demos even with generous horizons — a
classic compounding-error result, and exactly the headroom a benchmark
should have at baseline.

## 3. Findings (in discovery order)

### 3.1 The eval horizons were impossible by construction (fixed)
The frozen configs capped pick/stack at 50 steps and pull at 100 — but the
canonical motion-planning demos run 66–105, 74–135, and 130–789 steps
respectively (the horizons were pinned on macOS before any demos existed).
Eval v1 measured 0% success with 100% timeout failures on exactly the
tasks whose demos exceed their horizon, and nonzero success on exactly the
tasks that fit. Fix (`b640f8e`): horizons = demo-length p95 + margin
(pick 100, stack 150, pull 300); eval and setup_check create envs at the
registry horizon and setup_check asserts the applied limit empirically.
v1 results archived at `data/results/archive/`. Re-running (v2) confirmed
the deeper truth: pick/stack stay ~0 even with time — horizon was
necessary, not sufficient; BC precision is the binding constraint.

### 3.2 G1 PPO experts plateau near 10% under staged rewards (5 attempts)
The B2 gate ("≥70% on apple_bowl") was not reachable. Root cause chain,
all reproducible from `data/logs/b2_*`:
- The dense reward jumps +4/+8 at stage boundaries; once a policy starts
  succeeding, the return scale shifts and unbounded PPO epochs produce
  KL detonations (observed approx_kl 28.5 mid-run) that wreck the policy.
- The documented fallback (γ 0.8 + normalized_dense + lr 1e-4, cold) was
  worse: 0.05% at 16M steps.
- A target-KL epoch cutoff (`e46261c`) contains the detonations but slows
  the climb; best run reached 13.6% rolling before shocks turned
  net-destructive.
- Measured honestly (deterministic, 1,000+ episodes,
  `scripts/eval_state_expert.py`): apple expert 8.5%, transport best.pt
  1.4% deterministic but **8.5% stochastic** — the transport policy's
  competence partly lives in its action noise, so demo collection samples
  the training distribution (`policy_deterministic=false`, `c5aabdd`).
- Demo collection is success-filtered, so weak experts cost collection
  time (6–8% kept over ~2,900 attempts/task), not dataset validity.

### 3.3 Stochastic-expert demos imitate better than lucky-deterministic ones
transport_box BC reached 18.8% from an 8.5%-stochastic expert's demos,
while apple_bowl BC reached 0% from a 6%-deterministic expert's demos —
consistent with stochastic successes covering a broader state
distribution (smoother BC targets) versus a thin manifold of lucky
deterministic rollouts. Worth a controlled experiment in Phase C.

### 3.4 PushT source actions exceed the normalized bound
The flywheel deep validator flagged all 200 PushT episodes:
|action_canon| up to 2.3 despite pd_joint_delta_pos being normalized to
[−1,1] — ManiSkill's HF demos record unclipped Gaussian actions. The BC
tanh head cannot reproduce >1 targets, a plausible contributor to the
45.7% ceiling. Documented, not patched (source-data property).

### 3.5 Sim-infra fixes required at 2048-env scale
PhysX GPU collision-stack overflow (demand grows as the policy learns
contact: 8.1M → 28.6M → 150M vs a 4.2M default) silently dropped contacts
until raised to 512MB (`fa553d3`, `a5181e8`). PPO's success stat needed
episode pooling across horizons — early-terminating windows contain only
successes and read 100% regardless of policy quality (`498aff9`,
`d0f7444`); best.pt selection was being gamed by exactly this.

### 3.6 B4 fine-tuning contributed nothing at these baselines
Sparse-reward PPO from a ~0%-success BC start has no signal to bootstrap
from (best 1.2% on pick). The B4 recipe presumes B3 baselines meaningfully
above zero; rerun it when they are. (Fine-tunes were also trained at the
old 50-step horizon; documented in their config.yaml.)

## 4. Flywheel round 1 (first real round)

Round-0 dataset indexed (7,752 records: 1,202 demos + 6,550 eval episodes)
and deep-validated with zero critical findings. Candidate generation from
the 6,155 v2 failures produced 180 candidates; two selection arms at an
equal 50-episode budget:
- **failure_targeted**: pick 20, stack 20, apple_bowl 10 (allocates by
  failure mass)
- **random**: pick 14, stack 12, transport 8, push_t 6, apple 6, pull 4

Per arm: collect with the round-0 generators (same provenance, arm-tagged
files, baselines protected — see `0b1c8f0`), retrain BC per task-with-new-
data (seed 0, single-seed screen; identical recipe), evaluate the full
6-task matrix (untouched tasks anchored on round-0 checkpoints), compare
per-cell against round 0 with Wilson CIs.

**Result: null at this budget — and cleanly measured.** Seed-0-matched
per-task deltas (native cells, 100 episodes each):

| Task | round0 (seed0) | targeted Δ | random Δ |
|---|---|---|---|
| pick_cube_arm | 0.0% | +0.5 | 0.0 |
| stack_cube_arm | 0.0% | 0.0 | 0.0 |
| apple_bowl_g1 | 0.0% | 0.0 | 0.0 |
| pull_tool_arm | 2.5% | 0.0 | −1.5 |
| push_t_arm | 41.0% | 0.0 | −0.5 |
| transport_box_g1 | 9.5% | 0.0 | −0.5 |

Overall episode-weighted deltas: targeted **+0.11 pts**, random
**−0.28 pts** — both inside the noise. Two methodological results worth
keeping:
- **The noise floor matters more than the point estimates.** Comparing a
  single-seed arm against the pooled 3-seed baseline produced spurious
  ±5–9 pt swings on *untouched anchor tasks* (transport read −9.3 before
  seed matching); seed-matched comparison collapses those to ≤1.5 pts.
  Seed-matched CSVs are the standard for future rounds
  (`round0seed0_vs_round1_*.md`).
- **The null is informative:** +10–25% more same-kind demos does not move
  compounding-error-limited 0% tasks — data volume is not the binding
  constraint at this scale, matching finding 3.3. Round-2 recommendations:
  target the improvable mid-range tasks (transport, push_t) where gradient
  exists, raise the budget an order of magnitude, or pair data addition
  with an intervention that attacks compounding error directly (e.g.
  DAgger-style relabeling or action-chunked heads) so the flywheel has a
  responsive system to steer.

The full machinery is validated on real data end-to-end: indexing (7,752
records), quality gates, failure-mass candidate targeting, budgeted
selection, baseline-isolated arm collection/training, per-cell CI
comparison, and packaging (`artifacts/releases/flywheel_round1`).

Flywheel foundation fixes made on the way: the episode schema rejected its
own seed=-1 unknown-provenance default (`7909652`), and the generated
collection commands would have overwritten round-0 baselines via
motion_plan_arm's hardcoded output name (`0b1c8f0`).

## 4b. The precision-manipulation intervention study (night 3)

Executing round 1's recommendation, a systematic attack on the 0%
precision baselines (pick_cube as the study task; 25–50-episode probes,
seed 0, canonical/train):

- **Pipeline exoneration:** open-loop replay of ground-truth demo actions
  through the exact canonical→adapter→delta-env path succeeds 5/5 — the
  interface and data are correct; the model is the ceiling.
- **Failure localization:** instrumented rollouts reach within 2–3 cm and
  command the gripper decisively; failures are last-inch grasp conversion.
- **Interventions:** action chunking implemented (k-step windows,
  repeat-last padding, queue/ensemble/receding-horizon execution,
  chunk=1 backward-compatible; `ad17566`, `d101c22`): k=16@200 demos 0%;
  k=8@2,000 demos **4%**; ensembling 0% (blurs gripper commitment);
  gripper binarization and receding horizon 4% (execution saturated);
  ResNet-18 encoder 0% (train 0.0002/val 0.0077 — overfits);
  **10,000 demos 2% — the scaling curve is flat**
  (`docs/media/pick_scaling.png`); dense-reward visual PPO fine-tune
  0.1% at 460k steps with recurring KL spikes (killed). Side effects:
  pull_tool 1.3% → **8%** (k=8 + 2,000 demos); stack_cube stayed 0%.
- **Conclusion:** with data volume, execution strategy, encoder capacity,
  and RL fine-tuning eliminated under a proven pipeline, the binding
  constraint is the policy class — single-frame MLP-head BC at 128 px vs
  the transformer sequence models (ACT-class) that solve this regime.
  That is the benchmark's quantified open problem; 11,600 additional
  clean demonstrations were collected and remain available for it.

## 5. Session engineering notes

- Two long stalls were self-inflicted and fixed procedurally: untracked
  `nohup` jobs without monitors (13h idle) → every long stage now runs as
  an idempotent marker-based driver script under a persistent monitor
  (`data/logs/*_driver.sh`); and a kill that took the `uv` wrapper but left
  the python child training — which later overwrote the best checkpoint of
  the night with a stale one (kill the python, verify with `nvidia-smi`).
- GPU sharing measured: state-PPO + BC concurrently starves BC ~5× while
  costing PPO ~10% — training stages are serialized in the drivers.
- Everything is idempotent and resumable: relaunch any driver; markers
  skip completed work.

## 6. Reproduction

```bash
# environment (one-time)
./scripts/setup_linux.sh
# phase B end-to-end (idempotent; markers under data/logs/)
bash scripts/campaign/overnight_driver.sh   # demos -> BC -> B4 -> eval -> pack
bash scripts/campaign/reeval_driver.sh      # eval v2 at corrected horizons
# flywheel round 1
bash scripts/campaign/round1_driver.sh
# night-3 scaling ladder
bash scripts/campaign/night3_driver.sh
```
Checkpoints: `data/checkpoints/` · results: `data/results/` · packs:
`artifacts/releases/{phase_b_v1,flywheel_round1}` · logs: `data/logs/`.

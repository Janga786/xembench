# Flywheel Linux plan — the first real round (RTX 5090)

Prerequisite: Phase B core is done through step 10 of
[phase_b_commands.md](phase_b_commands.md) (demos collected, G1 experts ≥70%,
BC baselines trained, full eval matrix run). The flywheel consumes those
outputs; it does not replace the runbook. Every command below is ordered;
gates are bold.

### F0. Index + validate the round-0 dataset — minutes · Mac-safe: yes
```bash
uv run python -m xembench.flywheel.build_dataset_index \
  --demo-h5 data/demos/panda/PickCubeLang-v1/motion_planning.h5 \
  --demo-h5 data/demos/panda/StackCubeLang-v1/motion_planning.h5 \
  --demo-h5 data/demos/g1/UnitreeG1PlaceAppleInBowl-v1/ppo_expert.h5 \
  --eval-csv data/results/eval_episodes.csv --machine linux-5090 \
  --out artifacts/flywheel/indexes/round0_dataset_index.jsonl
uv run python -m xembench.flywheel.validate_dataset \
  --index artifacts/flywheel/indexes/round0_dataset_index.jsonl --deep-h5 \
  --out artifacts/flywheel/validation/round0_validation_report.md \
  --update-index
uv run python -m xembench.flywheel.build_dataset_card \
  --index artifacts/flywheel/indexes/round0_dataset_index.jsonl \
  --validation artifacts/flywheel/validation/round0_validation_report.json \
  --out artifacts/flywheel/cards/round0_dataset_card.md
```
**Gate:** zero critical findings; investigate every error before training.

### F1. Candidates + selection from round-0 failures — seconds · Mac-safe: yes
```bash
uv run python -m xembench.flywheel.generate_candidates \
  --eval-csv data/results/eval_episodes.csv \
  --dataset-index artifacts/flywheel/indexes/round0_dataset_index.jsonl \
  --out artifacts/flywheel/candidates/round1_candidates.jsonl
for s in failure_targeted random; do
  uv run python -m xembench.flywheel.select_candidates \
    --candidates artifacts/flywheel/candidates/round1_candidates.jsonl \
    --strategy $s --budget-episodes 50 --seed 0 \
    --input-eval-csv data/results/eval_episodes.csv \
    --out artifacts/flywheel/plans/round1_selection_plan_$s.json
done
uv run python -m xembench.flywheel.experiment_plan \
  --plan artifacts/flywheel/plans/round1_selection_plan_failure_targeted.json \
  --candidates artifacts/flywheel/candidates/round1_candidates.jsonl \
  --out-dir artifacts/flywheel/plans --round-id round1
```
Note: with only coarse failure labels (timeout/unknown), candidates come
from the balanced fallback — still a valid experiment; richer labels
(roadmap Phase B TODO) sharpen the targeting.

### F2. Collect BOTH arms' data — ~1 h · LINUX-ONLY (mplib / G1 experts)
Run the generated `artifacts/flywheel/plans/round1_linux_commands.sh`
collection section for the targeted arm; collect the same episode COUNT
uniformly at random (same generators, unconstrained conditions) for the
random arm. **Gate:** `validate_dataset --deep-h5` clean on both new
datasets.

### F3. Train the arms — hours · LINUX-ONLY
One BC run per arm (baseline demos + arm's extra demos), identical recipe
(`configs/presets/bc_train_full.yaml`), same seeds. The noisy/filtered
arms ([flywheel_experiments.md](flywheel_experiments.md)) can join in a
second pass once the first three arms are green.

### F4. Evaluate + compare — hours · Linux recommended
```bash
# full matrix per arm -> data/results/round1_<arm>_eval_episodes.csv, then:
uv run python -m xembench.flywheel.compare_rounds \
  --before data/results/eval_episodes.csv \
  --after data/results/round1_failure_targeted_eval_episodes.csv \
  --demos-added 50 \
  --out artifacts/flywheel/comparisons/round0_vs_round1_targeted.md
```
**Gate:** report deltas WITH Wilson CIs; a delta inside the interval is
"no detectable effect".

### F5. Package the round — seconds · Mac-safe: yes
Write a `round_manifest.json` (stage `evaluated`, `is_synthetic=false`) in
the round directory, then:
```bash
uv run python -m xembench.flywheel.build_flywheel_pack \
  --round-manifest artifacts/flywheel/round1/round_manifest.json \
  --out artifacts/releases/flywheel_round1
```
First real (non-`--synthetic`) flywheel pack; fill the model card from the
eval results and update the roadmap.

## Known risks

- Coarse failure labels weaken targeting until the runner assigns richer
  labels (Phase B TODO in `analysis/taxonomy.py`) — report which labels
  actually drove selection.
- 50-demo deltas may sit inside CIs at n=50 episodes/cell; scale episode
  counts or budgets rather than over-reading small deltas.
- G1 collection cost (expert-dependent) can dominate; the
  `cost_aware_failure_targeted` strategy exists for exactly that case.

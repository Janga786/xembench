# Roadmap

## Phase A — pipeline (macOS) — ✅ completed (v0.1)

Six ManiSkill3 envs across both embodiments, canonical 21-dim action
interface + adapters, language stack with leak-proof splits, HDF5 demo
pipeline, BC trainer, CleanRL PPO (expert + fine-tune), eval matrix with
zero-shot transfer, `make smoke` green end to end.

## Phase A.5 — hardening & docs (macOS) — ✅ completed (v0.2, tag `v0.2-phase-a5`)

Full docs set (architecture, canonical space, tasks, language, metrics,
configs, Phase B runbook), config presets, failure taxonomy + per-episode
records + analysis CLIs, 80 tests, CI, example toy reports (banner-labeled).

## Phase A.6 — artifact packaging (macOS) — ✅ completed (v0.3, tag `v0.3-phase-a6`)

`xembench.artifacts`: build_pack CLI (metrics, figures, manifest with
git/env provenance, config snapshots, video index, project-page assets,
checksums), episode schema v2, video index v2 with demo planning,
report skeleton + results-writing guide. 115 tests.

## Phase B — training & evaluation (Linux / RTX 5090) — ⏳ NEXT

Follow `docs/phase_b_commands.md` steps 1–15 in order. Exact TODOs:

- [ ] **mplib collection shakedown** — `xembench.experts.motion_plan_arm`
      has never executed (mplib is Linux-only); validate at 5 episodes
      before 200 (runbook step 5)
- [ ] **G1 expert ≥70% gate** on `apple_bowl_g1` before any demo
      collection (step 9 — the schedule-risk item)
- [ ] **`G1_HAND_OPEN_SIGNS` verification** — hand-joint open/close signs
      in `constants.py` are assumed +1=open, unverified; check empirically
      if grasping underperforms
- [ ] **richer failure labels in the runner** — assign
      wrong_object_selected / reach / grasp / dropped / placement from env
      info flags (`TODO(Janga)` in `xembench/analysis/taxonomy.py`)
- [ ] **video-recording hook** in `xembench.scripts.evaluate`
      (`docs/demo_video_plan.md` TODO) so failure montages replay exact
      `episode_seed`s
- [ ] **BC/PPO full eval matrix** — 3 seeds, all cells
      (`configs/presets/full_eval_matrix.yaml`, step 13)
- [ ] **artifact pack from real results** — runbook step 14
      (`make artifact-pack-help`); first non-synthetic pack
- [ ] **diffusion policy — STRETCH, only after core results**
      (`models/diffusion_policy.py` stub; do not start before B5 is done)

## Phase C — report, videos, portfolio release

- Results section from the real artifact pack
  (`reports/templates/report_skeleton.md` +
  `docs/how_to_write_results_section.md`)
- Four demo videos per `docs/demo_video_plan.md`
- Project page from `project_page_assets/` (docs/project_page.md scaffold)
- Repo visibility decision + release tag with real numbers

## Phase D — stretch goals (unordered, all optional)

Camera ablation on the G1 (matched third-person vs egocentric) ·
IK/EE-space canonical variant · morphology-conditioned policies ·
additional embodiments via the adapter seam (Isaac Lab / LeRobot / real
hardware — design notes only today) · LLM-scale paraphrase splits ·
multi-seed variance study.

## Phase E — Physical AI Data Flywheel

Failure-driven data collection built on the benchmark
([physical_ai_data_flywheel.md](physical_ai_data_flywheel.md)).

**E.1 — Mac foundation — ✅ completed.** `xembench.flywheel`: typed
dataset/provenance schemas, dataset index (eval-CSV + demo-HDF5 loaders),
deterministic validation layer with quality scoring (wraps the existing H5
validator), failure→candidate rule engine, five budgeted selection
strategies, dataset/model cards, round comparison
(improvement-per-demonstration), Linux command planning (generated, never
executed), deterministic toy round, flywheel artifact packs on the
existing synthetic rails. `make flywheel-mac-check` green; toy data only —
**no flywheel results exist**.

**E.2 — first real round (Linux, after Phase B core)** — ⏳ blocked on B:

- [ ] index + validate the round-0 dataset (`--deep-h5`, zero criticals)
- [ ] candidates + failure_targeted/random selection from real eval
      failures ([flywheel_linux_plan.md](flywheel_linux_plan.md))
- [ ] collect both arms' data; validate before training
- [ ] MVP experiment: baseline / +random / +failure-targeted
      (+noisy / +filtered second pass) — pre-registered design in
      [flywheel_experiments.md](flywheel_experiments.md)
- [ ] compare rounds WITH CIs; first non-synthetic flywheel pack; fill
      the model card

**E.3 — later** — richer failure labels feeding sharper targeting (needs
the Phase B runner instrumentation above), learned acquisition baselines,
VLA-scale dataset export (explicitly out of scope today).

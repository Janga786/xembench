# Application-facing summary

Honest, reusable blurbs for applications, resume, portfolio, and SOP.
**Ground rule: real benchmark numbers do not exist until Linux Phase B
completes. Everything below is written to be true TODAY; the placeholder
section at the bottom is filled only from a real (non-synthetic) artifact
pack.**

## 2-sentence pitch

I built xembench, a simulation benchmark that measures how much of a
language-conditioned manipulation policy survives a change of robot body —
the same network drives both a Franka Panda arm and a Unitree G1 humanoid
upper body through a canonical 21-dimensional action interface with thin
per-robot adapters. Instead of assuming cross-embodiment transfer works, it
makes transfer measurable: held-out paraphrases, held-out object colors,
zero-shot embodiment swaps, and a closed failure taxonomy that says whether
grounding or motor control broke.

## 1-paragraph technical summary

xembench is a ManiSkill3-based benchmark comparing a 7-DOF Panda arm and a
Unitree G1 humanoid upper body on analogous language-conditioned tabletop
tasks. One policy network (CNN + LSTM instruction encoder + proprio MLP)
emits actions in a canonical 21-dim space with semantic slots (arm joints
proximal→distal, grasp synergies, torso/head); per-robot adapters (~40
lines each) translate to native joint-delta controls, so the zero-shot
transfer test is literally "same weights, swapped adapter". The language
stack has a provably leak-free train/heldout synonym split and disjoint
train/unseen color sets, both enforced by tests. Infrastructure: HDF5 demo
pipeline (motion-planning, RL-expert, and replay sources), BC + CleanRL-PPO
baselines, an evaluation matrix producing per-episode records with Wilson
CIs, failure-taxonomy analysis, and a one-command artifact packer that
emits manifests (git + environment provenance), figures, and project-page
assets. 115 tests; the full pipeline is smoke-verified end to end on macOS
CPU simulation; GPU training runs on an RTX 5090 Linux box (Phase B).

## Resume bullets

- Designed and built xembench, a cross-embodiment manipulation benchmark
  (ManiSkill3): one language-conditioned policy drives a Franka Panda arm
  and a Unitree G1 humanoid via a canonical 21-dim action interface with
  per-robot adapters, making zero-shot body transfer directly measurable.
- Implemented the full research pipeline solo: custom language-grounded
  environments, leak-proof paraphrase/color evaluation splits (test-
  enforced), HDF5 demonstration pipeline, behavior cloning and PPO
  baselines, and a per-episode evaluation harness with Wilson CIs and a
  closed failure taxonomy.
- Engineered for reproducibility: 115 automated tests + CI, config/git-SHA
  provenance on every run, one-command artifact packaging (metrics,
  figures, manifests, checksums), and a cross-platform workflow (macOS
  CPU-sim development, Linux RTX 5090 training).

## Portfolio-page summary

See docs/project_page.md (the scaffold) and, after Phase B,
`artifacts/releases/<name>/project_page_assets/` — `project_summary.md` and
`result_cards.json` are generated from real eval CSVs so page numbers can
never drift from the data. Until then the page states plainly that numbers
are pending.

## SOP paragraph draft

To understand how much of a learned manipulation skill belongs to the task
versus the body executing it, I built a benchmark that holds everything
else fixed — language instructions, task structure, network architecture,
training recipe — and varies only the robot. Making that comparison
rigorous forced concrete design decisions: a canonical action interface
with semantic joint slots so "the same policy on a different robot" is
well-defined; synonym splits that are provably disjoint so paraphrase
generalization cannot leak; a failure taxonomy so a transfer failure is
attributable to grounding or motor control rather than guessed at. Building
it solo — from simulator integration and adapter design through training
baselines and a reproducible evaluation harness — taught me that the hard
part of empirical robotics is rarely the model; it is constructing an
experiment whose numbers mean what they claim to mean. [After Phase B, add
1–2 sentences on what the measured transfer gap actually showed.]

## What NOT to claim yet

- Any success rate, transfer drop, retention ratio, or robustness delta —
  none exist; toy numbers are synthetic and banner-labeled.
- That policies "transfer" (or fail to) between the arm and the humanoid —
  unknown until B5.
- That the G1 experts/demonstrations are trainable at the ≥70% gate —
  that is the declared schedule risk, not a result.
- Sim-to-real anything; the project is simulation-only.
- Novelty claims against specific prior work not yet read carefully
  (docs/references.md is unverified pointers).

## What to claim after Phase B (fill from a real artifact pack ONLY)

From `artifacts/releases/<name>/project_page_assets/key_metrics.json` with
`"synthetic": false`:

- "Native success rates of [X]% (Panda pick-and-place) and [Y]% (G1
  apple-in-bowl) under canonical instructions [CIs]."
- "Zero-shot cross-embodiment transfer retained [R]% of native performance
  arm→humanoid and [R']% humanoid→arm on the primary pair."
- "Held-out paraphrases cost [D] points; unseen colors cost [D'] points."
- "The dominant transfer failure mode was [label] ([N]% of failures)" —
  only if richer failure labels were instrumented; otherwise report
  timeout-dominated failures as exactly that.

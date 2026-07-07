# xembench — When Language and Architecture Are Shared, How Much Skill Survives a Body Swap?

<!-- Project page scaffold. Everything in [brackets] is a placeholder to be
filled after Phase B produces real results. GitHub Pages ready: this file
renders as-is; if publishing, copy to a gh-pages docs site later. -->

**One shared language-conditioned policy stack. Two robot bodies — a Franka
Panda tabletop arm and a Unitree G1 humanoid upper body. The benchmark
measures exactly how much manipulation performance survives the swap, and
classifies where it fails.**

[Repo](../) · [Report — placeholder, see docs/report_outline.md] ·
[Videos — placeholder, see docs/demo_video_plan.md]

---

## Thesis

Cross-embodiment robot learning assumes skills transfer across bodies, but
that premise is rarely measured under controlled conditions. xembench holds
language, task structure, policy architecture, and training recipe fixed and
varies only the embodiment via a canonical 21-dim action interface + thin
per-robot adapters — so transfer drop is attributable, and the failure
taxonomy says whether grounding or motor control broke.

## Architecture

[ARCHITECTURE DIAGRAM — export the Mermaid diagram from
docs/architecture.md once finalized]

One `BCPolicy` (CNN + LSTM instruction encoder + proprio MLP → 21-dim
canonical action head). Per-robot adapters (~40 lines each) translate
canonical ↔ native controls. **Transfer test = same weights, swapped
adapter.**

## Benchmark table

[PLACEHOLDER — after Phase B (B5), generate with
`uv run python scripts/make_report_assets.py` and paste
`reports/transfer_matrix.md` + the headline rows of `reports/summary.md`.
Until then, see `reports/example_*.md` for the format on labeled synthetic
data — those numbers are fake by construction.]

| Policy | Task | Native SR | Transfer SR | Drop | Retention |
|---|---|---|---|---|---|
| _pending Phase B_ | | | | | |

## Key results

[PLACEHOLDER — three bullets, written only after B5:
1. headline transfer drop on the primary pair (both directions),
2. paraphrase robustness delta (canonical vs heldout),
3. unseen-color grounding delta.]

## Videos

[PLACEHOLDER — four embeds per docs/demo_video_plan.md: (1) same
instruction/two bodies, (2) three paraphrases/one body, (3) seen vs unseen
colors, (4) failure montage.]

## Failure analysis

[PLACEHOLDER — stacked failure-type distribution from
`reports/failures.md`, plus 2–3 sentences on the dominant transfer failure
mode once known.]

## What this project demonstrates

- **Benchmark design**: turning a vague premise ("skills transfer across
  robots") into a controlled, measurable experiment with leak-proof
  language splits and a closed failure taxonomy.
- **Systems engineering**: one codebase spanning simulator integration
  (ManiSkill3/SAPIEN incl. macOS MoltenVK), canonical interfaces with
  per-robot adapters, HDF5 data pipelines, BC + CleanRL-PPO training, and a
  reproducible eval matrix — 70+ tests, cross-platform (Mac dev / Linux
  RTX 5090 training).
- **Research honesty**: per-episode records with exact seeds, Wilson CIs,
  config+git-SHA provenance on every run, and no claimed numbers until the
  experiments have actually run.

## Status

Phase A.5 complete (v0.2): full pipeline smoke-verified end to end on
macOS, plus documentation, analysis tooling, and test hardening.
Phase B (demo collection, training, and the actual benchmark numbers on an
RTX 5090) pending. This page fills in as results land.

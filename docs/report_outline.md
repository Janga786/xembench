# Report / paper outline

Scaffold for the write-up. Every bullet says what to fill in — nothing here
is a claim. Numbers only exist after Phase B (Linux) runs.

## Abstract
- One sentence: benchmark measuring how much of a language-conditioned
  manipulation policy survives an embodiment change (Panda arm ↔ Unitree G1
  upper body) when everything else is held fixed.
- One sentence on the canonical 21-dim action interface + adapters.
- Headline numbers to insert: native SRs, transfer drops (primary pair),
  paraphrase and unseen-color deltas, dominant failure modes.

## 1. Introduction
- Framing: embodiment transfer as a *measurable systems problem*, not a
  general-intelligence claim. The question: when language, task structure,
  and architecture are shared, how much performance survives a body swap,
  and where does it fail?
- Why it matters: cross-embodiment data reuse (cf. Open X-Embodiment) needs
  attribution — which pipeline stages are embodiment-invariant?
- Contributions list: (1) benchmark + language splits, (2) canonical
  interface + adapters, (3) baseline results + failure taxonomy analysis.

## 2. Related work — placeholder
- Cross-embodiment learning: Open X-Embodiment / RT-X; CrossFormer;
  morphology-aware policies (e.g., joint-graph conditioning).
- Language-conditioned manipulation: CLIPort, RT-1/RT-2, VIMA, LangLfP.
- Benchmarks: ManiSkill3, CALVIN, RLBench, Meta-World, LIBERO.
- TODO(Janga): 1–2 sentences each on relation to this work; emphasize that
  most benchmarks vary tasks, not bodies, under fixed language.

## 3. Method
### 3.1 Environments and embodiments
- Table of 6 tasks (pull from docs/tasks.md): env, embodiment, camera,
  budget, success condition, demo source.
- The two transfer pairs and why analogous-task (not identical-task)
  pairing is the honest design (no G1 support in stock arm tasks).
- Deliberate asymmetries kept as measured variables: camera viewpoint,
  5 vs 7 DOF arms, gripper mechanism.
### 3.2 Canonical action interface
- Condense docs/canonical_action_space.md: semantic slots, masking,
  grasp synergy, round-trip invariant.
- Limitations paragraph: kinematic aliasing, normalized-delta scale,
  synergy lossiness.
### 3.3 Language instruction splits
- Templates, synonym tables, provable disjointness (cite the tests),
  unseen-color split, colors-never-paraphrased rule.
### 3.4 Baselines
- BC (masked SmoothL1, paraphrase augmentation, DrQ shift).
- BC + sparse PPO fine-tune (frozen vision, L2-to-frozen-BC).
- Diffusion policy: stretch — include only if B6 lands.
- Expert/demo provenance per task (motion planning vs PPO expert vs HF).

## 4. Evaluation protocol
- Grid: {policy} × {task} × {canonical, heldout} × {train, unseen colors}
  × {native, transfer}; episode counts and seeds; Wilson CIs.
- Metrics with formulas (pull from docs/metrics.md): SR, transfer drop,
  retention ratio, paraphrase/color deltas, steps-to-success, failure
  distribution; the negative-transfer reference definition.

## 5. Results — ALL TBD (Phase B)
- Table 1: native SRs per task × policy (orig vs paraphrased vs unseen).
- Table 2: transfer matrix — native vs transfer SR + drop, both directions
  of both pairs.
- Fig: failure-type distribution stacked bars, native vs transfer.
- Fig: camera ablation on the G1 (head cam vs matched third-person).
- TODO(Janga): fill from data/results/eval.csv via
  `python -m xembench.analysis.summarize_results` after B5.

## 6. Failure analysis
- Method: taxonomy + per-episode labels; qualitative video montage.
- Questions to answer: does transfer fail at grounding (wrong object) or
  at motor control (reach/grasp)? Do paraphrase failures differ in kind
  from color failures? Is the G1 hand synergy the bottleneck?

## 7. Limitations
- Hand-designed interface (see canonical doc's limitations — copy honestly).
- Analogous rather than identical transfer tasks; camera confound is
  ablated, not eliminated.
- Small language distribution; simulation only; two embodiments.

## 8. Future work
- IK/EE-space canonical variant; morphology-conditioned policies; Isaac
  Lab or real-hardware adapters; diffusion policy baseline; LLM-scale
  paraphrase splits; more embodiments (mobile base slots already reserved).

## 9. Reproducibility appendix
- Exact commands (docs/phase_b_commands.md), seeds, config dumps + git SHA
  captured per run dir, demo dataset stats tables, compute budget,
  hardware (RTX 5090 specifics), software versions from scripts/check_env.

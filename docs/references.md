# References scaffold

Bibliography skeleton for the report (docs/report_outline.md §2). Entries
are POINTERS, not citations — **verify every author list, year, venue, and
title before citing**; nothing below has been checked against the actual
papers. `[VERIFY]` marks required lookups.

## Cross-embodiment robot learning

- Open X-Embodiment / RT-X — large multi-robot dataset + cross-embodiment
  training. [VERIFY citation; Open X-Embodiment Collaboration]
- CrossFormer — single policy across embodiments. [VERIFY]
- Morphology-/embodiment-conditioned policies (joint-graph or morphology
  encodings; e.g. work descending from NerveNet / modular policies). [VERIFY
  which specific papers to cite]
- Relevance: most of this line varies data and architecture; xembench holds
  them fixed and varies only the body under controlled language.

## Vision-language-action models

- RT-1, RT-2 (Google) — language-conditioned manipulation at scale. [VERIFY]
- OpenVLA — open-source VLA. [VERIFY] (not a dependency here; related work
  only)
- CLIPort — language-grounded manipulation with CLIP features. [VERIFY]
- VIMA — multimodal prompts for manipulation. [VERIFY]

## Robot manipulation benchmarks

- CALVIN — long-horizon language-conditioned benchmark (single arm). [VERIFY]
- RLBench, Meta-World, LIBERO — task-generalization benchmarks, fixed
  embodiment. [VERIFY each]
- Relevance: benchmarks above vary tasks; xembench varies the robot body.

## ManiSkill / simulation

- ManiSkill3 (Mu et al. / Tao et al.?) — the simulator, tasks, and demo
  datasets this benchmark builds on. [VERIFY exact ManiSkill3 paper + the
  SAPIEN paper; cite both]
- SAPIEN — underlying simulator. [VERIFY]

## Humanoid manipulation

- Unitree G1 platform. [VERIFY what to cite: spec sheet or papers using G1
  in ManiSkill]
- Recent humanoid whole-/upper-body manipulation learning. [VERIFY current
  representative papers at writing time]

## Imitation learning

- Behavior cloning fundamentals (Pomerleau ALVINN lineage). [VERIFY]
- Diffusion Policy (Chi et al.) — relevant if the B6 stretch lands. [VERIFY]
- DrQ / random-shift augmentation for pixel policies (used by the BC
  trainer). [VERIFY which DrQ paper]

## Reinforcement learning

- PPO (Schulman et al., 2017). [VERIFY — likely arXiv:1707.06347]
- GAE (Schulman et al.). [VERIFY]
- CleanRL — implementation style the PPO follows. [VERIFY software citation]

## Evaluation and reproducibility

- Wilson score interval (Wilson, 1927). [VERIFY]
- Reproducibility checklists / reporting standards in RL (e.g. Henderson
  et al., "Deep RL that Matters"). [VERIFY]
- Statistical practice for small-n success rates in robotics evaluation.
  [VERIFY a suitable reference]

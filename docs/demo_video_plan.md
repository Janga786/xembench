# Demo video plan (Phase B / presentation)

No videos exist yet — they require trained policies. This documents exactly
what to record so Phase B captures the right footage the first time.
Storage + naming: `artifacts/videos/README.md`. ManiSkill's `RecordEpisode`
wrapper (or `render_mode="rgb_array"` + imageio) records at collection/eval
time; prefer recording during the B5 eval matrix run so videos correspond
to reported numbers (same seeds — the episode CSV stores `episode_seed`).

## The four final videos

### 1. Same instruction, two bodies (the thesis shot)
Side-by-side: Panda `pick_cube_arm` and G1 `apple_bowl_g1`, both executing
their native-trained BC policy under the same pick_place instruction family.
Caption: same network architecture, same language pipeline, different body.
Then append one *transfer* clip (arm-trained weights on the G1) — success or
instructive failure, honestly labeled.
- Record: 2 native successes + the best transfer episode and a
  representative transfer failure.

### 2. Same body, three paraphrases
One embodiment (Panda, `pick_cube_arm`), one scene seed, three instructions:
canonical, train-synonym, heldout-synonym phrasing. Shows the policy
responds to meaning, not a memorized string — or shows exactly where
heldout phrasing breaks it.
- Record: 3 episodes with fixed `episode_seed`, varying instruction_cond.

### 3. Seen vs unseen colors
`pick_cube_arm`: "pick up the red cube" (train color, expected success) vs
"pick up the yellow cube" (unseen color scene). Both outcomes are
informative; label them truthfully.
- Record: 2–3 pairs, matched scene layouts where possible.

### 4. Failure analysis montage
6–10 short clips, each titled with its taxonomy label (wrong_object_selected,
grasp_failure, timeout, ...), sampled from the episode CSV's failure rows.
This is the qualitative companion to `reports/failures.md`.
- Record: pull `episode_seed`s of failures from
  `data/results/eval_episodes.csv` and re-run those exact episodes with
  recording enabled.

## Practical notes

- Use the human render camera (`render_mode="rgb_array"`, 512×512) for
  presentation footage; the 128×128 policy view can be inset picture-in-
  picture to show what the network actually sees.
- Overlay the instruction text on every clip — the language conditioning is
  the story.
- Keep raw clips uncut in `raw/`; edit into the four finals in `edited/`.
- Re-run `python -m xembench.analysis.video_index` after adding files.

TODO(Janga): during B5, add `--record-video-dir` support to
`xembench.scripts.evaluate` (thread a `record_dir` through `make_env`'s
`render_mode` + ManiSkill RecordEpisode) — small, but needs a GPU-speed
sanity check, so it is Phase B work.

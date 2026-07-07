# artifacts/videos/

Benchmark rollout videos (gitignored payloads; this README, INDEX.md, and
directory structure are tracked).

```
raw/     straight-from-sim recordings (RecordEpisode / imageio dumps)
edited/  trimmed/composited cuts for the project page and report
```

## Naming convention (enforced by the indexer)

```
task__embodiment__policy__split__seed__result.mp4
```

- `task` — registry name (`pick_cube_arm`, `apple_bowl_g1`, ...)
- `embodiment` — the EVAL embodiment (`panda`, `panda_stick`, `g1`)
- `policy` — `bc`, `bc_ppo`, `expert`, `random`
- `split` — `canonical`, `heldout`, `unseen_color`, `transfer`
- `seed` — integer
- `result` — `success`, or `failure_<taxonomy label>` (e.g.
  `failure_grasp_failure`, see `xembench.analysis.taxonomy`)

Example: `pick_cube_arm__g1__bc__transfer__0__failure_wrong_object_selected.mp4`

## Index

```bash
uv run python -m xembench.analysis.video_index    # writes INDEX.md
```

Lists every mp4 with parsed fields and flags files that break the
convention. Run it after adding videos; commit INDEX.md, never the videos.

Planned final cuts: see [docs/demo_video_plan.md](../../docs/demo_video_plan.md).

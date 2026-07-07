## What & why

<!-- 2-3 sentences: the change and the reason -->

## Checklist

- [ ] `make test` green (and `make test-sim` if envs/adapters/wrappers changed)
- [ ] `make smoke` green if training/eval code changed
- [ ] no large binaries staged (datasets/checkpoints/videos stay gitignored)
- [ ] no fabricated numbers; synthetic data labeled (toy banners intact)
- [ ] docs updated if behavior/commands changed
- [ ] new config keys documented; paths go through `resolve_path()`
- [ ] new failure labels added via `analysis.taxonomy`, never ad hoc strings

## Test evidence

```
# paste the relevant pytest/make output tail
```

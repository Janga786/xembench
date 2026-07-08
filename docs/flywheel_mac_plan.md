# Flywheel Mac foundation — what was built and how it is verified

Everything in this plan runs on the MacBook (no CUDA, no mplib, no long sim
jobs) and touches ONLY synthetic toy data. It exists so the Linux session
spends its time on compute, not plumbing.

## Scope of the Mac foundation

1. **Schemas** (`xembench.flywheel.schemas`) — nine dataclass schemas with
   validators and JSON/JSONL round-trips ([flywheel_schemas.md](flywheel_schemas.md)).
2. **Dataset index** (`dataset_index`) — loaders from the two existing
   artifact formats (per-episode eval CSVs, schema-v1 demo HDF5), triplet
   output (index / episodes sidecar / CSV summary), filtering by
   task/embodiment/split/success/failure/source/status.
3. **Validation** (`validation`) — deterministic checks that wrap (never
   duplicate) `xembench.data.schema.validate_demo_file`, add metadata-level
   checks (duplicates, leakage, provenance, synthetic labeling, lengths,
   embodiment/task registry consistency, action bounds), and produce
   per-episode quality scores + statuses under a documented rule.
4. **Candidate generation** (`candidates`) — the failure→request rule
   table with coarse-label fallback (only success/timeout/unknown_failure
   exist before Phase B instrumentation).
5. **Selection** (`selection`) — five deterministic strategies; plans
   record objective terms and expected coverage.
6. **Cards** (`cards`) — dataset cards; model-card placeholders that
   refuse metrics until real results exist.
7. **Comparison** (`compare_rounds`) — per-cell/per-split SR deltas and
   improvement-per-demo.
8. **Linux planning** (`experiment_plan`) — command plans tagged
   Mac-safe/Linux-only; the generated shell script refuses to run on
   macOS; nothing is executed at generation time.
9. **Toy pipeline** (`toy_data`, `generate_toy_flywheel`) — a
   deterministic, loudly-labeled synthetic round exercising 1–8 end to
   end, byte-reproducible at a fixed commit.
10. **Packaging** (`build_flywheel_pack`) — flywheel packs reusing the
    existing packer's manifest, checksums, sanitization, and synthetic
    refusal rails.

## Verification (all Mac-safe)

```bash
make test                  # full fast suite (existing + flywheel)
make flywheel-test         # flywheel tests only
make flywheel-toy          # deterministic toy round -> artifacts/flywheel/toy_round
make flywheel-validate-toy # quality report on the toy index
make flywheel-plan-toy     # candidates + failure_targeted selection plan
make flywheel-pack-toy     # SYNTHETIC pack -> artifacts/releases/flywheel_toy_pack
make flywheel-mac-check    # all of the above, chained
```

Existing gates that must stay green (and do): `make test`, `make
artifact-pack-toy`, `make toy-report`. The flywheel adds no new
dependencies and does not touch the canonical action interface, adapters,
envs, training, or eval code paths.

## Deliberate non-goals on Mac

No real demo collection, no BC/PPO training, no eval matrices, no result
claims, no learned quality/selection models, no VLA fine-tuning, no Isaac
Lab / ManiSkill2 / quantum anything. The line between "toy" and "real" is
enforced by code (synthetic refusal rails), not by care.

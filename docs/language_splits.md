# Language: templates, paraphrases, and leakage prevention

Module: `src/xembench/language/`. The instruction distribution is fully
enumerable (templates × slot values × synonyms), which is a deliberate
choice: it makes the train/eval split *provably* leak-free rather than
probably so.

## Templates (`templates.py`)

One canonical template per task family, with `{slot}` holes:

| Family | Template |
|---|---|
| pick_place | `pick up the {color} {object} and place it {relation} the {receptacle}` |
| stack | `stack the {color} {object} on top of the {color2} {object}` |
| tool_push | `push the t-shaped block onto the red target` |
| tool_pull | `use the tool to pull the {color} {object} closer` |
| transport | `pick up the box and carry it to the other table` |

Slots come from the env at episode start (custom Lang envs implement
`language_slots()`, returning the actually-sampled target colors) or from
static slots in `configs/tasks.yaml` for fixed-scene stock envs. The same
family spans embodiments: "pick up the red cube…" (Panda) and "pick up the
red apple…" (G1) are one family with different groundings — that is the
transfer pair's premise.

## Paraphrases (`paraphrase.py`)

Seeded synonym substitution over rendered instructions, phrase-level,
longest-match-first, single pass (a substitution's output is never
re-substituted). Two tables:

- `TRAIN_SYNONYMS` — sampled online as augmentation during BC training
  (`DemoDataset(instruction_mode="train")`) and PPO fine-tuning.
- `HELDOUT_SYNONYMS` — used only by evaluation
  (`instruction_cond="heldout"`, applied with p=1.0).

Example: canonical *"pick up the red cube and place it on the green
target"* → train *"grab the red block and put it on the green target"* →
heldout *"hoist the red brick and deposit it on the green target"*.

**Color words are never paraphrased** — they carry the grounding, and the
parser + experts rely on them.

## Leakage prevention (tested, not just intended)

`tests/test_paraphrase.py` enforces, on every run of the fast suite:

1. `test_no_leakage_between_splits` — the value sets of the two tables are
   globally disjoint: no heldout synonym is ever a train synonym.
2. `test_heldout_absent_from_templates` — no heldout synonym appears
   literally in any canonical template (else it would be "seen" regardless
   of the tables).
3. `test_paraphrase_preserves_parse` — paraphrasing (both splits, p=1.0)
   never changes the parsed structure (verb/color/object/receptacle), so
   the heldout condition measures phrasing robustness, not task ambiguity.
4. `test_colors_never_paraphrased`.

`tests/test_tokenizer.py::test_vocab_covers_entire_instruction_space`
additionally proves every rendered + paraphrased instruction (both splits)
tokenizes without `<unk>` and fits `L_MAX` — heldout evaluation is a
distribution shift in *phrasing*, not an out-of-vocabulary artifact.

## Held-out colors (visual analogue of the paraphrase split)

`constants.py`: `TRAIN_COLORS = [red, green, blue]`,
`UNSEEN_COLORS = [yellow, purple, cyan]` (disjoint — tested). The custom
Lang envs bake one cube per color of the active set at scene-build time
(GPU-sim compatible); `color_set="unseen"` builds the eval scene. Training
data only ever contains train colors, so the unseen-color cell measures
color-word grounding generalization. Cube count stays 3 in both sets so the
visual shift is color only, not clutter.

## How to extend

- **New synonym**: add to exactly one table; the leakage tests will refuse
  overlaps. Rebuild the vocab: `uv run python -m xembench.language.tokenizer`
  (commits `vocab.json`).
- **New template/family**: add to `TEMPLATES` + `SLOT_VALUES`, rebuild
  vocab, extend `parser.py` keyword lists if new nouns/verbs appear, and add
  a case to `test_paraphrase.py::test_paraphrase_preserves_parse`.
- **Testing leakage after any language edit**: `uv run pytest
  tests/test_paraphrase.py tests/test_tokenizer.py -q` — this is the
  cheapest high-value check in the repo.

## Design limitations (honest)

- The synonym inventory is small and hand-built; heldout paraphrases are a
  *mild* linguistic shift, not free-form language. Scaling to LLM-generated
  paraphrases would require an actual held-out-by-construction procedure.
- The parser is a closed-vocabulary keyword matcher, sufficient for
  scripted experts and analysis — it is not a contribution.
- The LSTM encoder + 65-word vocab is deliberately minimal: the benchmark
  isolates *grounding under embodiment change*, not language modeling.

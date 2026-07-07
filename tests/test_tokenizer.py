"""Tokenizer: shapes, padding, and full-coverage of the instruction space."""

import itertools

import numpy as np

from xembench.constants import L_MAX
from xembench.language.paraphrase import SYNONYM_TABLES, paraphrase
from xembench.language.templates import SLOT_VALUES, TEMPLATES, render
from xembench.language.tokenizer import PAD, UNK, get_tokenizer


def test_encode_shape_and_padding():
    tok = get_tokenizer()
    ids, length = tok.encode("pick up the red cube")
    assert ids.shape == (L_MAX,) and ids.dtype == np.int64
    assert length == 5
    assert (ids[length:] == PAD).all()
    assert (ids[:length] != PAD).all()


def test_unknown_words_map_to_unk():
    tok = get_tokenizer()
    ids, _ = tok.encode("zorble the flurb")
    assert ids[0] == UNK and ids[2] == UNK  # 'the' is known


def _slot_fill(template: str) -> list[str]:
    """Render a template with every combination of its slot values."""
    import re
    slots = re.findall(r"\{([a-z0-9_]+)\}", template)
    if not slots:
        return [template]
    pools = [SLOT_VALUES[s] for s in slots]
    return [template.format(**dict(zip(slots, combo)))
            for combo in itertools.product(*pools)]


def test_vocab_covers_entire_instruction_space():
    """No rendered or paraphrased instruction may contain <unk> or overflow L_MAX."""
    tok = get_tokenizer()
    rng = np.random.default_rng(0)
    for family, template in TEMPLATES.items():
        for instr in _slot_fill(template):
            candidates = [instr] + [
                paraphrase(instr, rng, split, p=1.0)
                for split in SYNONYM_TABLES
                for _ in range(5)
            ]
            for text in candidates:
                ids, length = tok.encode(text)
                assert UNK not in ids[:length], f"<unk> in {text!r}"
                assert length < L_MAX, f"instruction overflows L_MAX: {text!r}"


def test_render_matches_templates():
    instr = render("stack", dict(color="red", object="cube", color2="blue"))
    assert instr == "stack the red cube on top of the blue cube"

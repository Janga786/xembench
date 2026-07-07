"""Paraphrase split hygiene: heldout phrasings must be genuinely unseen."""

import numpy as np

from xembench.language.paraphrase import (
    HELDOUT_SYNONYMS,
    TRAIN_SYNONYMS,
    paraphrase,
)
from xembench.language.parser import parse
from xembench.language.templates import TEMPLATES, render


def _all_values(table: dict) -> set[str]:
    return {phrase for phrases in table.values() for phrase in phrases}


def test_no_leakage_between_splits():
    train, heldout = _all_values(TRAIN_SYNONYMS), _all_values(HELDOUT_SYNONYMS)
    assert train & heldout == set(), f"leaked synonyms: {train & heldout}"


def test_heldout_absent_from_templates():
    # A heldout synonym appearing literally in a canonical template would be
    # "seen" in training data regardless of the split tables.
    template_text = " ".join(TEMPLATES.values())
    for phrase in _all_values(HELDOUT_SYNONYMS):
        assert phrase not in template_text, f"heldout {phrase!r} is in a template"


def test_deterministic_given_seed():
    instr = render("pick_place", dict(color="red", object="cube", relation="on",
                                      receptacle="green target"))
    a = paraphrase(instr, np.random.default_rng(7), "train")
    b = paraphrase(instr, np.random.default_rng(7), "train")
    assert a == b


def test_paraphrase_changes_text():
    instr = render("pick_place", dict(color="red", object="cube", relation="on",
                                      receptacle="green target"))
    rng = np.random.default_rng(0)
    variants = {paraphrase(instr, rng, "train", p=1.0) for _ in range(20)}
    assert len(variants) > 1
    assert all(v != instr for v in variants)


def test_paraphrase_preserves_parse():
    """The structured meaning must survive paraphrasing — both splits."""
    cases = [
        ("pick_place", dict(color="red", object="cube", relation="on",
                            receptacle="green target")),
        ("pick_place", dict(color="red", object="apple", relation="in",
                            receptacle="bowl")),
        ("stack", dict(color="blue", object="cube", color2="yellow")),
        ("tool_pull", dict(color="blue", object="cube")),
        ("tool_push", {}),
        ("transport", {}),
    ]
    rng = np.random.default_rng(123)
    for family, slots in cases:
        instr = render(family, slots)
        reference = parse(instr)
        for split in ("train", "heldout"):
            for _ in range(10):
                assert parse(paraphrase(instr, rng, split, p=1.0)) == reference


def test_colors_never_paraphrased():
    for table in (TRAIN_SYNONYMS, HELDOUT_SYNONYMS):
        for color in ["red", "green", "blue", "yellow", "purple", "cyan"]:
            assert color not in table

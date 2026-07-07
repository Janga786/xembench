"""Seeded synonym-substitution paraphrasing with a leakage-proof split.

TRAIN synonyms are sampled as data augmentation while training; HELDOUT
synonyms are reserved for evaluation ("paraphrase generalization"). The two
tables share keys but their value sets are disjoint — enforced by a unit test
— so a heldout phrasing is genuinely unseen at training time.

Color words are deliberately never paraphrased: they carry the grounding.
"""

import re

import numpy as np

# canonical phrase -> paraphrases sampled during TRAINING (augmentation)
TRAIN_SYNONYMS = {
    "pick up": ["grab", "lift", "take"],
    "place": ["put", "set"],
    "push": ["slide"],
    "pull": ["drag"],
    "carry": ["move", "bring"],
    "cube": ["block"],
    "box": ["crate"],
    "on top of": ["onto"],
    "closer": ["nearer"],
}

# canonical phrase -> paraphrases seen ONLY at evaluation time
HELDOUT_SYNONYMS = {
    "pick up": ["hoist", "fetch", "retrieve"],
    "place": ["deposit", "position"],
    "stack": ["pile"],
    "push": ["shove", "nudge"],
    "pull": ["tug"],
    "carry": ["transport", "haul"],
    "cube": ["brick", "cuboid"],
    "box": ["carton", "package"],
    "on top of": ["atop"],
    "closer": ["within reach"],
}

SYNONYM_TABLES = {"train": TRAIN_SYNONYMS, "heldout": HELDOUT_SYNONYMS}


def _compile(table: dict) -> re.Pattern:
    # Longest keys first so "pick up" wins over any single-word key; one pass
    # so a substitution's output is never re-substituted.
    keys = sorted(table.keys(), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_PATTERNS = {split: _compile(t) for split, t in SYNONYM_TABLES.items()}


def paraphrase(
    instruction: str,
    rng: np.random.Generator,
    split: str = "train",
    p: float = 0.7,
) -> str:
    """Replace each known phrase with a split-specific synonym w.p. ``p``."""
    table = SYNONYM_TABLES[split]

    def sub(match: re.Match) -> str:
        phrase = match.group(0)
        if rng.random() >= p:
            return phrase
        options = table[phrase]
        return options[rng.integers(len(options))]

    return _PATTERNS[split].sub(sub, instruction)

"""Instruction parser: rendered/paraphrased text -> structured slots.

Used by scripted experts (which cube does the instruction name?) and by
analysis code. The parser may consult BOTH synonym tables — it is a tooling
utility, not the learned policy, so "seeing" heldout synonyms is not leakage.
"""

import re
from dataclasses import dataclass

from xembench.language.paraphrase import SYNONYM_TABLES

COLORS = ["red", "green", "blue", "yellow", "purple", "cyan"]
OBJECTS = ["cube", "apple", "box", "t-shaped block", "tool"]
RECEPTACLES = ["green target", "bowl", "table"]
VERBS = ["pick up", "stack", "push", "pull", "carry", "place"]


@dataclass
class ParsedInstruction:
    verb: str | None = None
    color: str | None = None
    obj: str | None = None
    color2: str | None = None
    receptacle: str | None = None


def _canonicalize(text: str) -> str:
    """Map every known synonym (train or heldout) back to its canonical phrase."""
    text = " ".join(re.findall(r"[a-z0-9\-]+", text.lower()))
    pairs = []
    for table in SYNONYM_TABLES.values():
        for canonical, phrases in table.items():
            pairs += [(phrase, canonical) for phrase in phrases]
    # Longest synonyms first so multi-word phrases win over their substrings.
    for phrase, canonical in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", canonical, text)
    return text


def _first(text: str, candidates: list[str]) -> str | None:
    hits = [(text.find(c), c) for c in candidates if re.search(rf"\b{re.escape(c)}\b", text)]
    return min(hits)[1] if hits else None


def parse(instruction: str) -> ParsedInstruction:
    text = _canonicalize(instruction)

    # 'green target' is a receptacle, not a color grounding: mask it out
    # before color extraction.
    color_text = text.replace("green target", " ")
    found_colors = [
        (m.start(), m.group(0))
        for m in re.finditer(r"\b(" + "|".join(COLORS) + r")\b", color_text)
    ]
    found_colors.sort()

    return ParsedInstruction(
        verb=_first(text, VERBS),
        color=found_colors[0][1] if found_colors else None,
        obj=_first(text, OBJECTS),
        color2=found_colors[1][1] if len(found_colors) > 1 else None,
        receptacle=_first(text, RECEPTACLES),
    )

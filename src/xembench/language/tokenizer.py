"""Whitespace tokenizer over a small closed vocabulary.

The instruction distribution is fully enumerable (templates x slot values x
synonyms), so the vocab is built exhaustively once and committed as vocab.json
next to this file. Rebuild after editing templates/synonyms:

    uv run python -m xembench.language.tokenizer
"""

import json
import re
from pathlib import Path

import numpy as np

from xembench.constants import L_MAX
from xembench.language.paraphrase import SYNONYM_TABLES
from xembench.language.templates import SLOT_VALUES, TEMPLATES

PAD, UNK = 0, 1
_VOCAB_PATH = Path(__file__).parent / "vocab.json"
_WORD_RE = re.compile(r"[a-z0-9\-]+")


def normalize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _all_words() -> set[str]:
    words: set[str] = set()
    for template in TEMPLATES.values():
        words |= set(normalize(re.sub(r"\{[a-z0-9_]+\}", " ", template)))
    for values in SLOT_VALUES.values():
        for v in values:
            words |= set(normalize(v))
    for table in SYNONYM_TABLES.values():
        for phrases in table.values():
            for phrase in phrases:
                words |= set(normalize(phrase))
    return words


def build_vocab(path: Path = _VOCAB_PATH) -> dict:
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word in sorted(_all_words()):
        vocab[word] = len(vocab)
    with open(path, "w") as f:
        json.dump(vocab, f, indent=1)
    return vocab


class Tokenizer:
    def __init__(self, path: Path = _VOCAB_PATH):
        with open(path) as f:
            self.vocab: dict[str, int] = json.load(f)

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> tuple[np.ndarray, int]:
        """-> (ids int64 [L_MAX] zero-padded, true length)."""
        ids = [self.vocab.get(w, UNK) for w in normalize(text)][:L_MAX]
        length = len(ids)
        out = np.zeros(L_MAX, dtype=np.int64)
        out[:length] = ids
        return out, length


_tokenizer: Tokenizer | None = None


def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


if __name__ == "__main__":
    vocab = build_vocab()
    print(f"wrote {_VOCAB_PATH} ({len(vocab)} entries)")

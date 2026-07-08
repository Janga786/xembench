"""Build a dataset card (markdown + JSON) from an index + validation report.

    python -m xembench.flywheel.build_dataset_card \
        --index artifacts/flywheel/indexes/round0_dataset_index.jsonl \
        --validation artifacts/flywheel/validation/round0_validation_report.json \
        --out artifacts/flywheel/cards/round0_dataset_card.md

The card documents composition, sources, validation state, and limitations
(xembench.flywheel.cards). Synthetic datasets produce loudly-labeled cards —
the synthetic flag is derived from provenance, not trusted from flags.
"""

import argparse
import json
import sys
from pathlib import Path

from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.cards import build_dataset_card, card_markdown
from xembench.flywheel.dataset_index import load_index, load_index_episodes
from xembench.flywheel.schemas import save_json
from xembench.utils.config import resolve_path, sanitize_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True,
                        help="index JSONL from build_dataset_index")
    parser.add_argument("--validation", default=None,
                        help="validation report JSON (validate_dataset)")
    parser.add_argument("--out", required=True,
                        help="card markdown path (a .json sibling is "
                             "written)")
    parser.add_argument("--name", default=None,
                        help="dataset name; default: index basename")
    parser.add_argument("--version", default="0.1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    index_path = Path(resolve_path(args.index))
    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "dataset card")
    records = load_index(index_path)
    episodes = load_index_episodes(index_path)
    validation_summary = None
    if args.validation:
        payload = json.loads(
            Path(resolve_path(args.validation)).read_text())
        validation_summary = payload.get("summary", payload)

    card = build_dataset_card(
        records, episodes, validation_summary,
        name=args.name or index_path.stem, version=args.version)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card_markdown(card))
    save_json(card, out.with_suffix(".json"))
    print(f"wrote {sanitize_path(out)} and "
          f"{sanitize_path(out.with_suffix('.json'))}"
          + (" [SYNTHETIC]" if card.synthetic else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

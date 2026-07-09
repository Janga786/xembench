"""Generate candidate demo requests from evaluation failures.

    python -m xembench.flywheel.generate_candidates \
        --eval-csv outputs/eval_episodes.csv \
        [--dataset-index artifacts/flywheel/indexes/round0_dataset_index.jsonl] \
        --out artifacts/flywheel/candidates/round1_candidates.jsonl

Writes candidates as JSONL (+ .csv and .md siblings). The rule table lives
in xembench.flywheel.candidates; coarse failure labels (timeout /
unknown_failure — all the pre-Phase-B runner can assign) are handled by the
balanced-collection fallback. A dataset index, when given, boosts tasks that
are underrepresented in existing demos.
"""

import argparse
import sys

from xembench.analysis.episodes import load_episodes
from xembench.flywheel._cli import echo_written, guard_overwrite
from xembench.flywheel.candidates import (
    MAX_PER_GROUP_DEFAULT,
    eval_csv_is_synthetic,
    generate_candidates,
    write_candidates,
)
from xembench.flywheel.dataset_index import load_index
from xembench.utils.config import resolve_path, sanitize_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-csv", required=True,
                        help="episode-level eval CSV with failures")
    parser.add_argument("--dataset-index", default=None,
                        help="optional index JSONL for coverage boosting")
    parser.add_argument("--out", required=True,
                        help="candidates JSONL (.csv/.md siblings written)")
    parser.add_argument("--max-per-group", type=int,
                        default=MAX_PER_GROUP_DEFAULT,
                        help="cap on candidates per failure group")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    csv = resolve_path(args.eval_csv)
    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "candidates file")
    df = load_episodes(csv)
    index_records = (load_index(resolve_path(args.dataset_index))
                     if args.dataset_index else None)
    candidates = generate_candidates(df, index_records,
                                     max_per_group=args.max_per_group)
    if not candidates:
        print("no failures in the eval CSV — nothing to request; "
              "no files written")
        return 0
    synthetic = eval_csv_is_synthetic(csv)
    written = write_candidates(candidates, out, source=sanitize_path(csv),
                               synthetic=synthetic)
    echo_written(written)
    print(f"{len(candidates)} candidates from "
          f"{int((df['success'] == 0).sum())} failures"
          + (" [SYNTHETIC]" if synthetic else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

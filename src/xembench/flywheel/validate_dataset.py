"""Validate a flywheel dataset index; write a quality report.

    python -m xembench.flywheel.validate_dataset \
        --index artifacts/flywheel/indexes/round0_dataset_index.jsonl \
        --out artifacts/flywheel/validation/round0_validation_report.md \
        [--deep-h5] [--update-index]

Runs the deterministic quality checks (xembench.flywheel.validation) using
the .episodes.jsonl sidecar when present; --deep-h5 additionally opens every
referenced HDF5 file through the existing xembench.data.schema validator.
Writes <out> (markdown) plus a .json sibling; --update-index rewrites the
index with quality scores and validation statuses applied.
"""

import argparse
import sys

from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.dataset_index import (
    load_index,
    load_index_episodes,
    write_index,
)
from xembench.flywheel.validation import (
    validate_index,
    write_validation_report,
)
from xembench.utils.config import resolve_path, sanitize_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True,
                        help="index JSONL from build_dataset_index")
    parser.add_argument("--out", required=True,
                        help="markdown report path (a .json sibling is "
                             "written too)")
    parser.add_argument("--deep-h5", action="store_true",
                        help="also open every referenced HDF5 file")
    parser.add_argument("--update-index", action="store_true",
                        help="rewrite the index with scores/statuses "
                             "applied")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    index_path = resolve_path(args.index)
    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "validation report")

    records = load_index(index_path)
    episodes = load_index_episodes(index_path)
    if episodes is None:
        print("note: no .episodes.jsonl sidecar found — running "
              "index-level checks only")
    result = validate_index(records, episodes, deep_h5=args.deep_h5)
    md, js = write_validation_report(result, out, sanitize_path(index_path))
    print(f"wrote {sanitize_path(md)} and {sanitize_path(js)}")
    s = result.summary
    print(f"{s['n_records']} records: "
          + (", ".join(f"{k}={v}" for k, v in
                       sorted(s["records_by_status"].items())) or "none")
          + f"; {len(result.issues)} issues")

    if args.update_index:
        write_index(result.records, episodes or [], index_path)
        print(f"updated {sanitize_path(index_path)} with quality scores")
    # Non-zero exit on critical findings so shell pipelines stop.
    return 1 if s["issues_by_severity"].get("critical") else 0


if __name__ == "__main__":
    sys.exit(main())

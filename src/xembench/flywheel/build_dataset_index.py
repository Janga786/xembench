"""Build a flywheel dataset index from eval CSVs and/or demo HDF5 files.

    python -m xembench.flywheel.build_dataset_index \
        --eval-csv outputs/eval_episodes.csv \
        --demo-h5 data/demos/panda/PickCubeLang-v1/motion_planning.h5 \
        --out artifacts/flywheel/indexes/round0_dataset_index.jsonl \
        [--synthetic]

Writes three sibling files: the index (<out>), full episode metadata
(<out stem>.episodes.jsonl), and a flat CSV summary (<out stem>.summary.csv).
Toy-stamped eval CSVs are REFUSED without --synthetic — the same rail as
the artifact packer.
"""

import argparse
import sys

from xembench.flywheel._cli import echo_written, guard_overwrite
from xembench.flywheel.candidates import eval_csv_is_synthetic
from xembench.flywheel.dataset_index import (
    build_index,
    episodes_from_demo_h5,
    episodes_from_eval_csv,
    write_index,
)
from xembench.utils.config import resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-csv", action="append", default=[],
                        metavar="CSV",
                        help="episode-level eval CSV (repeatable)")
    parser.add_argument("--demo-h5", action="append", default=[],
                        metavar="H5",
                        help="schema-v1 demo HDF5 file (repeatable)")
    parser.add_argument("--out", required=True,
                        help="index JSONL path (two siblings are written)")
    parser.add_argument("--dataset-id", default=None,
                        help="default: basename of --out")
    parser.add_argument("--synthetic", action="store_true",
                        help="REQUIRED when any input is toy/synthetic data")
    parser.add_argument("--machine", default="",
                        help="coarse machine label for provenance "
                             "(e.g. mac, linux-5090); never a hostname")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not args.eval_csv and not args.demo_h5:
        parser.error("provide at least one --eval-csv or --demo-h5")
    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "dataset index")

    episodes = []
    for csv in args.eval_csv:
        csv = resolve_path(csv)
        if eval_csv_is_synthetic(csv) and not args.synthetic:
            raise SystemExit(
                f"REFUSED: {csv} is stamped as synthetic toy data but "
                "--synthetic was not passed. Toy data cannot be indexed "
                "as real demonstrations.")
        episodes += episodes_from_eval_csv(csv, synthetic=args.synthetic,
                                           machine=args.machine)
    for h5 in args.demo_h5:
        episodes += episodes_from_demo_h5(resolve_path(h5),
                                          synthetic=args.synthetic,
                                          machine=args.machine)

    dataset_id = args.dataset_id or out.stem
    records = build_index(episodes, dataset_id)
    written = write_index(records, episodes, out)
    echo_written(written)
    print(f"indexed {len(records)} episodes as dataset {dataset_id!r}"
          + (" [SYNTHETIC]" if args.synthetic else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

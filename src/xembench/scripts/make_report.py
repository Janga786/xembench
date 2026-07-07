"""Re-print the summary tables from the accumulated results CSV."""

import sys

import pandas as pd

from xembench.eval.report import print_table
from xembench.utils.config import repo_root


def main() -> int:
    csv = sys.argv[1] if len(sys.argv) > 1 else repo_root() / "data/results/eval.csv"
    df = pd.read_csv(csv)
    print_table(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())

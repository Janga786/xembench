"""Artifact packaging: turn eval outputs into report/portfolio-ready bundles.

The one command that matters:

    python -m xembench.artifacts.build_pack --eval-csv <episodes.csv> \
        --out artifacts/releases/<name> --release-name <name> [--synthetic]

Everything here is simulator-free and Mac/CI-safe; it consumes CSVs and
config files, never the simulator.
"""

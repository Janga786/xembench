#!/bin/bash
# B5 v2: full eval matrix at corrected horizons + reports + artifact pack.
# Idempotent via markers; safe to relaunch.
cd /home/janga/xembench || exit 1
M=/home/janga/xembench/data/logs
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if [ ! -f "$M/.b5v2" ]; then
  say B5V2_EVAL_START
  if uv run python -u -m xembench.scripts.evaluate \
       --config configs/presets/full_eval_matrix.yaml > "$M/b5v2_eval.log" 2>&1; then
    touch "$M/.b5v2"; say B5V2_EVAL_DONE
  else
    say B5V2_EVAL_FAIL
  fi
fi

if [ ! -f "$M/.reportsv2" ] && [ -f "$M/.b5v2" ]; then
  say REPORTSV2_START
  if uv run python scripts/make_report_assets.py --input data/results/eval_episodes.csv \
       > "$M/b5v2_reports.log" 2>&1 \
     && uv run python -m xembench.artifacts.build_pack \
       --eval-csv data/results/eval_episodes.csv \
       --out artifacts/releases/phase_b_v1 --release-name phase_b_v1 \
       --configs configs/presets/full_eval_matrix.yaml --configs configs/tasks.yaml \
       --videos artifacts/videos --overwrite >> "$M/b5v2_reports.log" 2>&1; then
    touch "$M/.reportsv2"; say REPORTSV2_DONE
  else
    say REPORTSV2_FAIL
  fi
fi
say V2_ALL_DONE

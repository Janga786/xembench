#!/bin/bash
# Night-4: matrix-grade chunked-BC numbers. Trains bc_chunk (k=8, baseline
# + 2k-demo sets) for the three motion-planning arm tasks x 3 seeds, then
# evaluates the full-matrix conditions for those policies (3 seeds, fresh
# CSVs). Idempotent via markers in data/logs.
cd /home/janga/xembench || exit 1
M=/home/janga/xembench/data/logs
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

declare -A DEMOS=(
  [pick_cube_arm]='["data/demos/panda/PickCubeLang-v1/motion_planning.h5","data/demos/panda/PickCubeLang-v1/motion_planning_xl.h5"]'
  [stack_cube_arm]='["data/demos/panda/StackCubeLang-v1/motion_planning.h5","data/demos/panda/StackCubeLang-v1/motion_planning_xl.h5"]'
  [pull_tool_arm]='["data/demos/panda/PullCubeTool-v1/motion_planning.h5","data/demos/panda/PullCubeTool-v1/motion_planning_xl.h5"]'
)

say N4_TRAIN_START
for t in pick_cube_arm stack_cube_arm pull_tool_arm; do
  for s in 0 1 2; do
    if [ -f "$M/.n4_bc_${t}_${s}" ]; then say "BC_SKIP $t seed$s"; continue; fi
    say "BC_START $t seed$s"
    if uv run python -m xembench.scripts.train_bc \
         --config configs/presets/bc_train_full.yaml --set task=$t --seed $s \
         --set chunk=8 --set steps=100000 --set demos="${DEMOS[$t]}" \
         --set out_dir=data/checkpoints/bc_chunk_${t}_seed${s} \
         >> "$M/n4_bc.log" 2>&1; then
      touch "$M/.n4_bc_${t}_${s}"; say "BC_DONE $t seed$s"
    else say "BC_FAIL $t seed$s"; fi
  done
done
say N4_TRAIN_DONE

if [ ! -f "$M/.n4_eval" ]; then
  say N4_EVAL_START
  if uv run python -u -m xembench.scripts.evaluate \
       --config configs/presets/full_eval_matrix.yaml \
       --set tasks='[pick_cube_arm, stack_cube_arm, pull_tool_arm]' \
       --set policies='[{type: bc_chunk, ckpt: "data/checkpoints/bc_chunk_{task}_seed{seed}/best.pt", chunk: 8}]' \
       --set csv=data/results/eval_chunk.csv \
       --set episodes_csv=data/results/eval_chunk_episodes.csv \
       > "$M/n4_eval.log" 2>&1; then
    touch "$M/.n4_eval"; say N4_EVAL_DONE
  else say N4_EVAL_FAIL; fi
fi
say N4_ALL_DONE

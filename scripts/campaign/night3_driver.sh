#!/bin/bash
# Night-3 driver: data-scaling ladder + chunked fleet upgrades. Idempotent.
cd /home/janga/xembench || exit 1
M=/home/janga/xembench/data/logs
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

PICKD=data/demos/panda/PickCubeLang-v1
STACKD=data/demos/panda/StackCubeLang-v1
PULLD=data/demos/panda/PullCubeTool-v1

mp(){ uv run python -m xembench.experts.motion_plan_arm \
        --config configs/presets/panda_demo_collection.yaml \
        --set task=$1 --set episodes=$2 --set source=$3 --seed $4; }

# ---------- S1: collections (parallel, CPU) ----------
say S1_COLLECT_START
[ -f "$M/.n3_pickxxl" ]  || ( mp pick_cube_arm 8000 motion_planning_xxl 5 \
    > "$M/n3_pick_xxl.log" 2>&1 && touch "$M/.n3_pickxxl" ) &
[ -f "$M/.n3_stackxl" ]  || ( mp stack_cube_arm 1800 motion_planning_xl 3 \
    > "$M/n3_stack_xl.log" 2>&1 && touch "$M/.n3_stackxl" ) &
[ -f "$M/.n3_pullxl" ]   || ( mp pull_tool_arm 1800 motion_planning_xl 3 \
    > "$M/n3_pull_xl.log" 2>&1 && touch "$M/.n3_pullxl" ) &
wait
say S1_COLLECT_DONE

bc_run(){  # name task chunk steps demos-json
  local name=$1 t=$2 k=$3 steps=$4 demos=$5
  [ -f "$M/.n3_bc_$name" ] && { say "BC_SKIP $name"; return 0; }
  say "BC_START $name"
  if uv run python -m xembench.scripts.train_bc \
       --config configs/presets/bc_train_full.yaml --set task=$t --seed 0 \
       --set chunk=$k --set steps=$steps --set demos="$demos" \
       --set out_dir=data/checkpoints/bc_$name >> "$M/n3_bc.log" 2>&1; then
    touch "$M/.n3_bc_$name"; say "BC_DONE $name"
  else say "BC_FAIL $name"; fi
}
ev(){  # name spec
  say "EVAL_START $1"
  uv run python -u -m xembench.scripts.evaluate \
    --set tasks="[$2]" --set seeds='[0]' --set episodes_native=25 \
    --set episodes_transfer=0 --set transfer=false \
    --set instruction_conds='[canonical]' --set color_conds='[train]' \
    --set policies="[$3]" \
    --set csv=data/results/night3_probe.csv \
    --set episodes_csv=data/results/night3_probe_episodes.csv \
    >> "$M/n3_eval.log" 2>&1 \
    && grep -h "| $1 " "$M/n3_eval.log" | tail -1 || say "EVAL_FAIL $1"
}

# ---------- S2: stack/pull chunked @2k ----------
if [ -f "$M/.n3_stackxl" ]; then
  bc_run chunk8xl_stack_cube_arm_seed0 stack_cube_arm 8 100000 \
    "[\"$STACKD/motion_planning.h5\",\"$STACKD/motion_planning_xl.h5\"]"
  ev n3stack stack_cube_arm "{type: n3stack, ckpt: \"data/checkpoints/bc_chunk8xl_stack_cube_arm_seed0/best.pt\", chunk: 8}"
fi
if [ -f "$M/.n3_pullxl" ]; then
  bc_run chunk8xl_pull_tool_arm_seed0 pull_tool_arm 8 100000 \
    "[\"$PULLD/motion_planning.h5\",\"$PULLD/motion_planning_xl.h5\"]"
  ev n3pull pull_tool_arm "{type: n3pull, ckpt: \"data/checkpoints/bc_chunk8xl_pull_tool_arm_seed0/best.pt\", chunk: 8}"
fi

# ---------- S3: pick @10k (the scaling point) ----------
if [ -f "$M/.n3_pickxxl" ]; then
  bc_run chunk8xxl_pick_cube_arm_seed0 pick_cube_arm 8 250000 \
    "[\"$PICKD/motion_planning.h5\",\"$PICKD/motion_planning_xl.h5\",\"$PICKD/motion_planning_xxl.h5\"]"
  ev n3pick10k pick_cube_arm "{type: n3pick10k, ckpt: \"data/checkpoints/bc_chunk8xxl_pick_cube_arm_seed0/best.pt\", chunk: 8}"
fi
say N3_ALL_DONE

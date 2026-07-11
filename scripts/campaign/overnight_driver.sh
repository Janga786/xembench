#!/bin/bash
# Overnight Phase B completion driver — idempotent (markers in data/logs),
# safe to relaunch anytime; every stage skips work whose marker exists.
cd /home/janga/xembench || exit 1
R=/home/janga/xembench
M=$R/data/logs
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# ---------- Stage 1: transport demo collection (3 parts, parallel) ----------
say STAGE1_TRANSPORT_COLLECT_START
for i in 1 2 3; do
  if [ ! -f "$M/.collect_transport_$i" ]; then
    ( uv run python -u -m xembench.scripts.collect_demos --seed $((i*7)) \
        --set task=transport_box_g1 \
        --set policy=ppo:$R/data/checkpoints/expert_transport_box_g1_seed0/best.pt \
        --set policy_deterministic=false --set episodes=67 \
        --set source=ppo_expert_part$i --set max_attempts=1600 \
        > "$M/b2b_transport_collect_part$i.log" 2>&1 \
      && touch "$M/.collect_transport_$i" ) &
  fi
done
wait
say STAGE1_TRANSPORT_COLLECT_DONE
uv run python - <<'PY' 2>&1 | tail -4
from xembench.data.schema import validate_demo_file as v
import glob, h5py
tot = 0
for p in sorted(glob.glob('data/demos/g1/UnitreeG1TransportBox-v1/ppo_expert_part*.h5')):
    with h5py.File(p) as f: n = len(f.keys())
    tot += n
    print(p.split('/')[-1], n, v(p) or 'VALID')
print('transport demos total:', tot)
PY

# ---------- Stage 2+4: BC training (arm remainder, then G1) ----------
bc_run(){
  local t=$1 s=$2
  [ -f "$M/.bc_${t}_${s}" ] && { say "BC_SKIP $t seed$s"; return 0; }
  say "BC_START $t seed$s"
  if uv run python -m xembench.scripts.train_bc --config configs/presets/bc_train_full.yaml \
       --set task=$t --seed $s >> "$M/b3_bc_all.log" 2>&1 \
     || uv run python -m xembench.scripts.train_bc --config configs/presets/bc_train_full.yaml \
       --set task=$t --seed $s >> "$M/b3_bc_all.log" 2>&1; then
    touch "$M/.bc_${t}_${s}"; say "BC_DONE $t seed$s"
  else
    say "BC_FAIL $t seed$s"
  fi
}
say STAGE2_ARM_BC_START
for t in pick_cube_arm stack_cube_arm pull_tool_arm push_t_arm; do
  for s in 0 1 2; do bc_run $t $s; done
done
say STAGE2_ARM_BC_DONE
say STAGE4_G1_BC_START
for t in apple_bowl_g1 transport_box_g1; do
  for s in 0 1 2; do bc_run $t $s; done
done
say STAGE4_G1_BC_DONE

# ---------- Stage 5: B4 PPO fine-tunes (seed 0, 3 tasks) ----------
say STAGE5_B4_START
for t in pick_cube_arm stack_cube_arm apple_bowl_g1; do
  if [ -f "$M/.b4_${t}" ]; then say "B4_SKIP $t"; continue; fi
  say "B4_START $t"
  if uv run python -u -m xembench.scripts.train_ppo_finetune \
       --config configs/presets/ppo_finetune_full.yaml --set task=$t \
       >> "$M/b4_finetune.log" 2>&1; then
    touch "$M/.b4_${t}"; say "B4_DONE $t"
  else
    say "B4_FAIL $t"
  fi
done
say STAGE5_B4_DONE

# ---------- Stage 6: B5 full eval matrix ----------
if [ ! -f "$M/.b5" ]; then
  say B5_EVAL_START
  if uv run python -u -m xembench.scripts.evaluate \
       --config configs/presets/full_eval_matrix.yaml > "$M/b5_eval.log" 2>&1; then
    touch "$M/.b5"; say B5_EVAL_DONE
  else
    say B5_EVAL_FAIL
  fi
fi

# ---------- Stage 7: reports + real artifact pack ----------
if [ ! -f "$M/.reports" ] && [ -f "$M/.b5" ]; then
  say REPORTS_START
  if uv run python scripts/make_report_assets.py --input data/results/eval_episodes.csv \
       > "$M/b5_reports.log" 2>&1 \
     && uv run python -m xembench.artifacts.build_pack \
       --eval-csv data/results/eval_episodes.csv \
       --out artifacts/releases/phase_b_v1 --release-name phase_b_v1 \
       --configs configs/presets/full_eval_matrix.yaml --configs configs/tasks.yaml \
       --videos artifacts/videos --overwrite >> "$M/b5_reports.log" 2>&1; then
    touch "$M/.reports"; say REPORTS_DONE
  else
    say REPORTS_FAIL
  fi
fi
say ALL_DONE

#!/bin/bash
# Flywheel round 1 driver (F2-F5) — idempotent via markers in data/logs.
# Design: two arms at equal 50-episode budget (failure_targeted vs random),
# BC retrained per task-with-new-data (seed 0; single-seed round-1 screen),
# full 6-task eval matrix per arm (untouched tasks use round-0 checkpoints),
# per-cell comparison vs round 0 with Wilson CIs, first real flywheel pack.
cd /home/janga/xembench || exit 1
R=/home/janga/xembench
M=$R/data/logs
CK=$R/data/checkpoints
say(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

PICK=data/demos/panda/PickCubeLang-v1/motion_planning.h5
STACK=data/demos/panda/StackCubeLang-v1/motion_planning.h5
PULL=data/demos/panda/PullCubeTool-v1/motion_planning.h5
PUSH=data/demos/panda_stick/PushT-v1/hf_replay.h5
APPLED=data/demos/g1/UnitreeG1PlaceAppleInBowl-v1
TRAND=data/demos/g1/UnitreeG1TransportBox-v1
APPLE_CKPT=$CK/expert_apple_bowl_g1_seed0/try1_last_23pct.pt
TRAN_CKPT=$CK/expert_transport_box_g1_seed0/best.pt

mp(){ uv run python -m xembench.experts.motion_plan_arm \
        --config configs/presets/panda_demo_collection.yaml \
        --set task=$1 --set episodes=$2 --set source=$3; }

# ---------- S1: targeted arm collection (pick 20, stack 20, apple 10) ------
if [ ! -f "$M/.r1_s1" ]; then
  say R1_S1_TARGETED_COLLECT_START
  { mp pick_cube_arm 20 flywheel_timeout \
    && mp stack_cube_arm 20 flywheel_timeout \
    && uv run python -m xembench.scripts.collect_demos --seed 101 \
         --set task=apple_bowl_g1 --set policy=ppo:$APPLE_CKPT \
         --set episodes=10 --set source=flywheel_timeout --set max_attempts=600 \
    && touch "$M/.r1_s1" && say R1_S1_DONE; } >> "$M/r1_collect.log" 2>&1 \
    || say R1_S1_FAIL
fi

# ---------- S2: random arm collection (pick 14, stack 12, pull 4, ---------
# ----------     apple 6, transport 8, push 6) ------------------------------
if [ ! -f "$M/.r1_s2" ]; then
  say R1_S2_RANDOM_COLLECT_START
  { mp pick_cube_arm 14 flywheel_random \
    && mp stack_cube_arm 12 flywheel_random \
    && mp pull_tool_arm 4 flywheel_random \
    && uv run python -m xembench.scripts.collect_demos --seed 102 \
         --set task=apple_bowl_g1 --set policy=ppo:$APPLE_CKPT \
         --set episodes=6 --set source=flywheel_random --set max_attempts=400 \
    && uv run python -m xembench.scripts.collect_demos --seed 103 \
         --set task=transport_box_g1 --set policy=ppo:$TRAN_CKPT \
         --set policy_deterministic=false --set episodes=8 \
         --set source=flywheel_random --set max_attempts=400 \
    && uv run python -m mani_skill.trajectory.replay_trajectory \
         --traj-path $HOME/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 \
         -o rgb --save-traj --count 216 -b physx_cpu --use-env-states \
    && uv run python -m xembench.data.convert_ms_demos \
         --input $HOME/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5 \
         --task push_t_arm --source flywheel_tmp --max-episodes 216 \
    && uv run python - <<'PY' \
    && touch "$M/.r1_s2" && say R1_S2_DONE; } >> "$M/r1_collect.log" 2>&1 \
    || say R1_S2_FAIL
import h5py, shutil
src = 'data/demos/panda_stick/PushT-v1/flywheel_tmp.h5'
dst = 'data/demos/panda_stick/PushT-v1/flywheel_random.h5'
with h5py.File(src) as f:
    eps = sorted(f.keys())
    # round 0 used the first 200 replayed episodes; the arm takes the next 6
    new = eps[200:206]
    assert len(new) == 6, f"need 6 new episodes, got {len(new)} of {len(eps)}"
    with h5py.File(dst, 'w') as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        for i, e in enumerate(new):
            f.copy(e, g, name=f'ep_{i:05d}')
import os; os.remove(src)
print('wrote', dst)
PY
fi

# ---------- S3: index + deep-validate the arm files -------------------------
if [ ! -f "$M/.r1_s3" ] && [ -f "$M/.r1_s1" ] && [ -f "$M/.r1_s2" ]; then
  say R1_S3_VALIDATE_START
  { uv run python -m xembench.flywheel.build_dataset_index \
      --demo-h5 data/demos/panda/PickCubeLang-v1/flywheel_timeout.h5 \
      --demo-h5 data/demos/panda/StackCubeLang-v1/flywheel_timeout.h5 \
      --demo-h5 $APPLED/flywheel_timeout.h5 \
      --demo-h5 data/demos/panda/PickCubeLang-v1/flywheel_random.h5 \
      --demo-h5 data/demos/panda/StackCubeLang-v1/flywheel_random.h5 \
      --demo-h5 data/demos/panda/PullCubeTool-v1/flywheel_random.h5 \
      --demo-h5 $APPLED/flywheel_random.h5 \
      --demo-h5 $TRAND/flywheel_random.h5 \
      --demo-h5 data/demos/panda_stick/PushT-v1/flywheel_random.h5 \
      --machine linux-5090 --overwrite \
      --out artifacts/flywheel/indexes/round1_collected_index.jsonl \
    && uv run python -m xembench.flywheel.validate_dataset \
      --index artifacts/flywheel/indexes/round1_collected_index.jsonl --deep-h5 \
      --out artifacts/flywheel/validation/round1_validation_report.md \
      --update-index --overwrite \
    && touch "$M/.r1_s3" && say R1_S3_DONE; } >> "$M/r1_validate.log" 2>&1 \
    || say R1_S3_FAIL
fi

# ---------- S4: per-arm checkpoint layouts + BC retraining ------------------
bc_arm(){  # arm task demos-yaml-list
  local arm=$1 t=$2 demos=$3
  [ -f "$M/.r1_bc_${arm}_${t}" ] && { say "R1_BC_SKIP $arm $t"; return 0; }
  say "R1_BC_START $arm $t"
  if uv run python -m xembench.scripts.train_bc \
       --config configs/presets/bc_train_full.yaml --set task=$t --seed 0 \
       --set demos="$demos" \
       --set out_dir=$CK/round1_$arm/bc_${t}_seed0 >> "$M/r1_bc.log" 2>&1; then
    touch "$M/.r1_bc_${arm}_${t}"; say "R1_BC_DONE $arm $t"
  else say "R1_BC_FAIL $arm $t"; fi
}
if [ -f "$M/.r1_s3" ]; then
  say R1_S4_BC_START
  mkdir -p $CK/round1_targeted $CK/round1_random
  # untouched tasks reuse round-0 checkpoints via symlink
  for t in pull_tool_arm push_t_arm transport_box_g1; do
    ln -sfn $CK/bc_${t}_seed0 $CK/round1_targeted/bc_${t}_seed0
  done
  bc_arm targeted pick_cube_arm "[\"$PICK\",\"data/demos/panda/PickCubeLang-v1/flywheel_timeout.h5\"]"
  bc_arm targeted stack_cube_arm "[\"$STACK\",\"data/demos/panda/StackCubeLang-v1/flywheel_timeout.h5\"]"
  bc_arm targeted apple_bowl_g1 "[\"$APPLED/ppo_expert_part1.h5\",\"$APPLED/ppo_expert_part2.h5\",\"$APPLED/ppo_expert_part3.h5\",\"$APPLED/flywheel_timeout.h5\"]"
  bc_arm random pick_cube_arm "[\"$PICK\",\"data/demos/panda/PickCubeLang-v1/flywheel_random.h5\"]"
  bc_arm random stack_cube_arm "[\"$STACK\",\"data/demos/panda/StackCubeLang-v1/flywheel_random.h5\"]"
  bc_arm random pull_tool_arm "[\"$PULL\",\"data/demos/panda/PullCubeTool-v1/flywheel_random.h5\"]"
  bc_arm random push_t_arm "[\"$PUSH\",\"data/demos/panda_stick/PushT-v1/flywheel_random.h5\"]"
  bc_arm random apple_bowl_g1 "[\"$APPLED/ppo_expert_part1.h5\",\"$APPLED/ppo_expert_part2.h5\",\"$APPLED/ppo_expert_part3.h5\",\"$APPLED/flywheel_random.h5\"]"
  bc_arm random transport_box_g1 "[\"$TRAND/ppo_expert_part1.h5\",\"$TRAND/ppo_expert_part2.h5\",\"$TRAND/ppo_expert_part3.h5\",\"$TRAND/flywheel_random.h5\"]"
  say R1_S4_BC_DONE
fi

# ---------- S5: full-matrix eval per arm (seed 0) ---------------------------
eval_arm(){
  local arm=$1
  [ -f "$M/.r1_eval_${arm}" ] && { say "R1_EVAL_SKIP $arm"; return 0; }
  say "R1_EVAL_START $arm"
  if uv run python -u -m xembench.scripts.evaluate \
       --config configs/presets/full_eval_matrix.yaml \
       --set seeds='[0]' \
       --set policies="[{type: bc, ckpt: \"data/checkpoints/round1_$arm/bc_{task}_seed{seed}/best.pt\"}]" \
       --set csv=data/results/round1_${arm}_eval.csv \
       --set episodes_csv=data/results/round1_${arm}_eval_episodes.csv \
       >> "$M/r1_eval_${arm}.log" 2>&1; then
    touch "$M/.r1_eval_${arm}"; say "R1_EVAL_DONE $arm"
  else say "R1_EVAL_FAIL $arm"; fi
}
if [ -f "$M/.r1_s3" ]; then
  eval_arm targeted
  eval_arm random
fi

# ---------- S6: compare + manifest + first real flywheel pack ---------------
if [ ! -f "$M/.r1_s6" ] && [ -f "$M/.r1_eval_targeted" ] && [ -f "$M/.r1_eval_random" ]; then
  say R1_S6_COMPARE_START
  { for arm in targeted random; do
      uv run python -m xembench.flywheel.compare_rounds \
        --before data/results/eval_episodes.csv \
        --after data/results/round1_${arm}_eval_episodes.csv \
        --demos-added 50 --overwrite \
        --out artifacts/flywheel/comparisons/round0_vs_round1_${arm}.md
    done \
    && uv run python - <<'PY' \
    && uv run python -m xembench.flywheel.build_flywheel_pack \
         --round-manifest artifacts/flywheel/round1/round_manifest.json \
         --out artifacts/releases/flywheel_round1 --overwrite \
         --notes "first real flywheel round: failure_targeted vs random, 50-episode budget" \
    && touch "$M/.r1_s6" && say R1_S6_DONE; } >> "$M/r1_pack.log" 2>&1 \
    || say R1_S6_FAIL
import json, shutil, subprocess, datetime, os
rd = 'artifacts/flywheel/round1'
os.makedirs(rd, exist_ok=True)
files = {
 'dataset_index_path': 'artifacts/flywheel/indexes/round1_collected_index.jsonl',
 'eval_csv_path': 'data/results/round1_targeted_eval_episodes.csv',
 'failure_report_path': 'artifacts/flywheel/validation/round1_validation_report.md',
 'candidate_request_path': 'artifacts/flywheel/candidates/round1_candidates.jsonl',
 'selection_plan_path': 'artifacts/flywheel/plans/round1_selection_plan_failure_targeted.json',
}
extras = ['artifacts/flywheel/comparisons/round0_vs_round1_targeted.md',
          'artifacts/flywheel/comparisons/round0_vs_round1_random.md',
          'artifacts/flywheel/plans/round1_selection_plan_random.json',
          'artifacts/flywheel/cards/round0_dataset_card.md',
          'data/results/round1_random_eval_episodes.csv']
manifest = {
 'round_id': 'round1', 'previous_round_id': 'round0', 'stage': 'evaluated',
 'training_config_path': '', 'artifact_pack_path': '',
 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'git_commit': subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
 'is_synthetic': False,
 'notes': ('First real flywheel round on the RTX 5090: failure_targeted vs '
           'random arms at a 50-episode budget, BC retrained per task with '
           'new data (seed 0), full 6-task matrix per arm.'),
 '_schema': 'FlywheelRoundManifest', '_schema_version': 1,
}
for key, src in files.items():
    dst = os.path.join(rd, os.path.basename(src))
    shutil.copy(src, dst)
    manifest[key] = os.path.basename(src)
for src in extras:
    shutil.copy(src, os.path.join(rd, os.path.basename(src)))
json.dump(manifest, open(os.path.join(rd, 'round_manifest.json'), 'w'), indent=1)
print('round1 manifest + files staged')
PY
fi
say R1_ALL_DONE

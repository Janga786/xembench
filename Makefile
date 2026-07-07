# xembench — common entrypoints. `make smoke` is the Phase-A gate.

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
# SAPIEN bundles a Vulkan loader but no ICD on macOS; point it at MoltenVK.
export VK_ICD_FILENAMES ?= /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export VK_DRIVER_FILES ?= /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
endif

MS_DEMOS := $(HOME)/.maniskill/demos/PickCube-v1/motionplanning
SLICE := data/demos/panda/PickCube-v1/hf_replay_check.h5
SLOTS := {"color": "red", "object": "cube", "relation": "on", "receptacle": "green target"}

.PHONY: check test test-sim data-slice bc-smoke eval-smoke ppo-smoke smoke clean-smoke \
        mac-check toy-report video-index phase-b-help docs \
        artifact-pack-toy artifact-pack-help

check:
	uv run python -m xembench.scripts.setup_check

test:
	uv run pytest -q

test-sim:
	uv run pytest -m sim -q

mac-check:
	uv run python scripts/check_env.py

# Full analysis pipeline on labeled synthetic data (fast, any machine);
# refreshes the committed reports/example_* documentation examples.
toy-report:
	uv run python scripts/make_report_assets.py --toy

video-index:
	uv run python -m xembench.analysis.video_index

# Full artifact pack from deterministic SYNTHETIC data (loudly labeled) —
# proves the Phase B packaging path end to end on any machine.
artifact-pack-toy:
	uv run python -m xembench.scripts.generate_toy_results
	uv run python -m xembench.artifacts.build_pack \
	  --eval-csv outputs/toy/eval_episodes.csv \
	  --out artifacts/releases/toy_pack --release-name toy_pack \
	  --synthetic --overwrite \
	  --configs configs/tasks.yaml --configs configs/presets/full_eval_matrix.yaml \
	  --videos artifacts/videos \
	  --notes "deterministic toy pack from make artifact-pack-toy"

artifact-pack-help:
	@cat docs/artifact_packs.md

phase-b-help:
	@cat docs/phase_b_commands.md

docs:
	@echo "docs/ index:" && ls -1 docs/*.md | sed 's/^/  /'
	@echo "start with README.md, then docs/architecture.md"

$(SLICE):
	uv run python -m mani_skill.utils.download_demo PickCube-v1
	uv run python -m mani_skill.trajectory.replay_trajectory \
	  --traj-path $(MS_DEMOS)/trajectory.h5 \
	  -o rgb -c pd_joint_delta_pos --save-traj --count 10
	uv run python -m xembench.data.convert_ms_demos \
	  --input $(MS_DEMOS)/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5 \
	  --task pick_cube_arm --env-id-override PickCube-v1 \
	  --source hf_replay_check --slots '$(SLOTS)' --max-episodes 10

data-slice: $(SLICE)

bc-smoke: data-slice
	uv run python -m xembench.scripts.train_bc \
	  --set demos='[$(SLICE)]' --set steps=400 --set batch_size=64 \
	  --set eval_every=100 --set num_workers=0 --set preload=true \
	  --set out_dir=data/checkpoints/bc_overfit_check

eval-smoke:
	uv run python -m xembench.scripts.evaluate \
	  --set tasks='[pick_cube_arm]' --set episodes_native=3 --set episodes_transfer=2 \
	  --set instruction_conds='[canonical]' --set color_conds='[train]' \
	  --set policies='[{type: bc, ckpt: "data/checkpoints/bc_overfit_check/best.pt"}]' \
	  --set csv=data/results/eval_smoke.csv \
	  --set episodes_csv=data/results/eval_smoke_episodes.csv

ppo-smoke:
	uv run python -m xembench.scripts.train_expert_ppo \
	  --set task=pick_cube_arm --set num_envs=1 --set num_steps=32 \
	  --set total_timesteps=512 --set num_minibatches=4 --set update_epochs=2 \
	  --set device=cpu --set out_dir=data/checkpoints/ppo_smoke
	uv run python -m xembench.scripts.train_ppo_finetune \
	  --set task=pick_cube_arm --set bc_ckpt=data/checkpoints/bc_overfit_check/best.pt \
	  --set num_envs=1 --set num_steps=25 --set total_timesteps=100 \
	  --set num_minibatches=1 --set update_epochs=2 --set device=cpu \
	  --set out_dir=data/checkpoints/ppo_ft_smoke

smoke: check test test-sim bc-smoke eval-smoke ppo-smoke
	@echo "=== PHASE A SMOKE GREEN ==="

clean-smoke:
	rm -rf data/checkpoints/bc_overfit_check data/checkpoints/ppo_smoke \
	  data/checkpoints/ppo_ft_smoke data/results/eval_smoke.csv \
	  data/results/eval_smoke_episodes.csv data/checkpoints/linux_gpu_smoke

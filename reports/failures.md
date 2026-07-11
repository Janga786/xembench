# Failure breakdown

Source: `data/results/eval_episodes.csv` — 6550 episodes, 6155 failures.

Labels beyond success/timeout/unknown_failure/simulator_error require Phase B instrumentation (see docs/metrics.md); their absence in real runs is expected for now.

## By evaluation task

| eval_task        |   success |   timeout |   total |
|:-----------------|----------:|----------:|--------:|
| apple_bowl_g1    |         0 |      1000 |    1000 |
| pick_cube_arm    |         0 |      1800 |    1800 |
| pull_tool_arm    |         8 |       592 |     600 |
| push_t_arm       |       274 |       326 |     600 |
| stack_cube_arm   |         0 |      1750 |    1750 |
| transport_box_g1 |       113 |       687 |     800 |

## By evaluation embodiment

| eval_embodiment   |   success |   timeout |   total |
|:------------------|----------:|----------:|--------:|
| g1                |       113 |      1687 |    1800 |
| panda             |         8 |      4142 |    4150 |
| panda_stick       |       274 |       326 |     600 |

## By policy

| policy_type   |   success |   timeout |   total |
|:--------------|----------:|----------:|--------:|
| bc            |       395 |      5005 |    5400 |
| bc_ppo        |         0 |      1150 |    1150 |

## By instruction / color split

| instruction_cond   | color_cond   |   success |   timeout |   total |
|:-------------------|:-------------|----------:|----------:|--------:|
| canonical          | fixed        |       192 |      1108 |    1300 |
| canonical          | train        |         0 |      1550 |    1550 |
| canonical          | unseen       |         0 |       800 |     800 |
| heldout            | fixed        |       203 |      1097 |    1300 |
| heldout            | train        |         0 |       800 |     800 |
| heldout            | unseen       |         0 |       800 |     800 |

## By train -> eval embodiment

| train_embodiment   | eval_embodiment   |   success |   timeout |   total |
|:-------------------|:------------------|----------:|----------:|--------:|
| g1                 | g1                |       113 |      1287 |    1400 |
| g1                 | panda             |         0 |       350 |     350 |
| panda              | g1                |         0 |       400 |     400 |
| panda              | panda             |         8 |      3792 |    3800 |
| panda_stick        | panda_stick       |       274 |       326 |     600 |

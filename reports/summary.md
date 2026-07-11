# Evaluation summary

Source: `data/results/eval_episodes.csv` — 6550 episodes, 750 of them cross-embodiment transfer.

| policy_type   | task             | eval_task        | eval_embodiment   | instruction_cond   | color_cond   |   n_episodes | success       | steps   | top_failure   |
|:--------------|:-----------------|:-----------------|:------------------|:-------------------|:-------------|-------------:|:--------------|:--------|:--------------|
| bc            | apple_bowl_g1    | apple_bowl_g1    | g1                | canonical          | fixed        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | apple_bowl_g1    | apple_bowl_g1    | g1                | heldout            | fixed        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | apple_bowl_g1    | pick_cube_arm    | panda             | canonical          | train        |          150 | 0% [0%,2%]    | -       | timeout:150   |
| bc            | pick_cube_arm    | apple_bowl_g1    | g1                | canonical          | train        |          150 | 0% [0%,2%]    | -       | timeout:150   |
| bc            | pick_cube_arm    | pick_cube_arm    | panda             | canonical          | train        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | pick_cube_arm    | pick_cube_arm    | panda             | canonical          | unseen       |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | pick_cube_arm    | pick_cube_arm    | panda             | heldout            | train        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | pick_cube_arm    | pick_cube_arm    | panda             | heldout            | unseen       |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | pull_tool_arm    | pull_tool_arm    | panda             | canonical          | fixed        |          300 | 2% [1%,5%]    | 203     | timeout:293   |
| bc            | pull_tool_arm    | pull_tool_arm    | panda             | heldout            | fixed        |          300 | 0% [0%,2%]    | 168     | timeout:299   |
| bc            | push_t_arm       | push_t_arm       | panda_stick       | canonical          | fixed        |          300 | 45% [39%,51%] | 41      | timeout:165   |
| bc            | push_t_arm       | push_t_arm       | panda_stick       | heldout            | fixed        |          300 | 46% [41%,52%] | 38      | timeout:161   |
| bc            | stack_cube_arm   | stack_cube_arm   | panda             | canonical          | train        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | stack_cube_arm   | stack_cube_arm   | panda             | canonical          | unseen       |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | stack_cube_arm   | stack_cube_arm   | panda             | heldout            | train        |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | stack_cube_arm   | stack_cube_arm   | panda             | heldout            | unseen       |          300 | 0% [0%,1%]    | -       | timeout:300   |
| bc            | stack_cube_arm   | transport_box_g1 | g1                | canonical          | train        |          150 | 0% [0%,2%]    | -       | timeout:150   |
| bc            | transport_box_g1 | stack_cube_arm   | panda             | canonical          | train        |          150 | 0% [0%,2%]    | -       | timeout:150   |
| bc            | transport_box_g1 | transport_box_g1 | g1                | canonical          | fixed        |          300 | 17% [13%,21%] | 59      | timeout:250   |
| bc            | transport_box_g1 | transport_box_g1 | g1                | heldout            | fixed        |          300 | 21% [17%,26%] | 54      | timeout:237   |
| bc_ppo        | apple_bowl_g1    | apple_bowl_g1    | g1                | canonical          | fixed        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | apple_bowl_g1    | apple_bowl_g1    | g1                | heldout            | fixed        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | apple_bowl_g1    | pick_cube_arm    | panda             | canonical          | train        |           50 | 0% [0%,7%]    | -       | timeout:50    |
| bc_ppo        | pick_cube_arm    | apple_bowl_g1    | g1                | canonical          | train        |           50 | 0% [0%,7%]    | -       | timeout:50    |
| bc_ppo        | pick_cube_arm    | pick_cube_arm    | panda             | canonical          | train        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | pick_cube_arm    | pick_cube_arm    | panda             | canonical          | unseen       |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | pick_cube_arm    | pick_cube_arm    | panda             | heldout            | train        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | pick_cube_arm    | pick_cube_arm    | panda             | heldout            | unseen       |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | stack_cube_arm   | stack_cube_arm   | panda             | canonical          | train        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | stack_cube_arm   | stack_cube_arm   | panda             | canonical          | unseen       |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | stack_cube_arm   | stack_cube_arm   | panda             | heldout            | train        |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | stack_cube_arm   | stack_cube_arm   | panda             | heldout            | unseen       |          100 | 0% [0%,4%]    | -       | timeout:100   |
| bc_ppo        | stack_cube_arm   | transport_box_g1 | g1                | canonical          | train        |           50 | 0% [0%,7%]    | -       | timeout:50    |

Columns: `task` = training task, `eval_task` = where evaluated (differing values = transfer cell); success shows a 95% Wilson CI; `steps` averages successful episodes only. See docs/metrics.md for formulas.

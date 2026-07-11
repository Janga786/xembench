# Transfer matrix

`drop = SR_native - SR_transfer`; `retention = SR_transfer / SR_native`.

| policy_type   | task             | eval_task        | train_embodiment   | eval_embodiment   | instruction_cond   | color_cond   |   n_native | sr_native   |   n_transfer | sr_transfer   | drop   | retention   |
|:--------------|:-----------------|:-----------------|:-------------------|:------------------|:-------------------|:-------------|-----------:|:------------|-------------:|:--------------|:-------|:------------|
| bc            | apple_bowl_g1    | pick_cube_arm    | g1                 | panda             | canonical          | train        |          0 | -           |          150 | 0%            | -      | -           |
| bc            | pick_cube_arm    | apple_bowl_g1    | panda              | g1                | canonical          | train        |        300 | 0%          |          150 | 0%            | 0%     | -           |
| bc            | stack_cube_arm   | transport_box_g1 | panda              | g1                | canonical          | train        |        300 | 0%          |          150 | 0%            | 0%     | -           |
| bc            | transport_box_g1 | stack_cube_arm   | g1                 | panda             | canonical          | train        |          0 | -           |          150 | 0%            | -      | -           |
| bc_ppo        | apple_bowl_g1    | pick_cube_arm    | g1                 | panda             | canonical          | train        |          0 | -           |           50 | 0%            | -      | -           |
| bc_ppo        | pick_cube_arm    | apple_bowl_g1    | panda              | g1                | canonical          | train        |        100 | 0%          |           50 | 0%            | 0%     | -           |
| bc_ppo        | stack_cube_arm   | transport_box_g1 | panda              | g1                | canonical          | train        |        100 | 0%          |           50 | 0%            | 0%     | -           |

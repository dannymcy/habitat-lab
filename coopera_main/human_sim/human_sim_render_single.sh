#!/bin/bash

# ------------------------------
# CONFIGURE PARAMETERS HERE
# ------------------------------

# conda activate /hdd2/kai/Dynamic_Human_Robot_Value_Alignments/env
# chmod +x ./coopera_main/human_sim/human_sim_render_single.sh
# ./coopera_main/human_sim/human_sim_render_single.sh

# Scene (sd) indices
SCENES=(0)

# Human / profile (i) indices
PROFILES=(0 1 2 3 4 5 6 7 8 9)

# Day (d) indices
DAYS=(0 1 2 3 4)

# Hour (j) indices (0-12 for 9am-9pm)
HOURS=(0 1 2 3 4 5 6 7 8 9 10 11 12)

GPU_ID=0   # GPU ID

# ------------------------------
# MAIN LOOP
# ------------------------------

for sd in "${SCENES[@]}"; do
  for human in "${PROFILES[@]}"; do
    for day in "${DAYS[@]}"; do
      for hour in "${HOURS[@]}"; do

        echo "Running sd=$sd, human=$human, day=$day, hour=$hour"

        CUDA_VISIBLE_DEVICES="3" python coopera_main/human_sim/human_sim_render.py \
            --gpu-id "$GPU_ID" \
            --scene-indices "$sd" \
            --profile-indices "$human" \
            --day-indices "$day" \
            --hour-indices "$hour" \
            --use-gpt-human True \
            --collab-type 1 \
            --max-days 5

        echo "--------------------------------------"
      done
    done
  done
done
#!/bin/bash

# ======================================================
# PARALLEL HUMAN SIMULATION SCRIPT
# ======================================================
# This script runs human behavior simulations across multiple scenes in parallel.
# Idealy only for GPT API.
# Each scene processes 10 human profiles sequentially, but different scenes
# run simultaneously to speed up the overall computation.

# ------------------------------
# CONFIGURE PARAMETERS HERE
# ------------------------------

# conda activate /hdd2/kai/Dynamic_Human_Robot_Value_Alignments/env
# chmod +x ./coopera_main/human_sim/human_sim_llm_inference_parallel.sh
# ./coopera_main/human_sim/human_sim_llm_inference_parallel.sh

# Scene (sd) indices
SCENES=(0 1 2 3 4)

# Human / profile (i) indices
PROFILES=(0 1 2 3 4 5 6 7 8 9)

GPU_ID=0   # GPU ID

# ------------------------------
# MAIN LOOP - PARALLEL SCENES
# ------------------------------

for sd in "${SCENES[@]}"; do
  (
    for human in "${PROFILES[@]}"; do
      echo "Running sd=$sd, human=$human"

      CUDA_VISIBLE_DEVICES="0" python coopera_main/human_sim/human_sim.py \
          --gpu-id "$GPU_ID" \
          --scene-indices "$sd" \
          --profile-indices "$human" \
          --use-gpt-human True \
          --start-logic-human True \
          --collab-type 2

      echo "--------------------------------------"
    done
  ) &  # Run entire inner loop in background
done

wait  # Wait for all scenes to complete
echo "All scenes processed in parallel!"
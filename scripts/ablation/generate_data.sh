#!/bin/bash

# Ensure three arguments are provided
# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
flamingo=${4:-"flamingo_9B"}

if [ -z "$task" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 <task> <dataset> [gpu_ids]"
    exit 1
fi

which python

declare -A configs

# Baseline:
configs["BASELINE"]="beam_size=5 candidate_set_num=64"

# Different beam sizes:
configs["BEAMSIZE:1"]="beam_size=1 candidate_set_num=64"
configs["BEAMSIZE:10"]="beam_size=10 candidate_set_num=64"

# Different candidate set nums:
configs["CANDIDATE:32"]="beam_size=5 candidate_set_num=32"
configs["CANDIDATE:128"]="beam_size=5 candidate_set_num=128"

# Different candidate set methods:
configs["TEXT-SIM"]="beam_size=5 candidate_set_num=64 candidate_set_method=text-sim"
configs["IMAGE-SIM"]="beam_size=5 candidate_set_num=64 candidate_set_method=image-sim"

# Different sample nums:
configs["1w SAMPLE_NUM"]="beam_size=5 candidate_set_num=64 sample_num=10000"

for key in "${!configs[@]}"; do
    echo "=====================BEGIN TO $key====================="
    python generate_data.py \
        ${configs[$key]} \
        few_shot_num=2 \
        sample_num=5000 \
        gpu_ids="${gpu_ids}" \
        task=${task} \
        dataset=${dataset} \
        flamingo=${flamingo}
done

#!/bin/bash

# Ensure three arguments are provided
# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}

if [ -z "$task" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 <task> <dataset> [gpu_ids]"
    exit 1
fi


which python

# Baseline:
echo "=====================BEGIN TO BASELINE====================="
python generate_data.py \
    beam_size=5 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO BEAMSIZE:1====================="
python generate_data.py \
    beam_size=1 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO BEAMSIZE:10====================="
python generate_data.py \
    beam_size=10 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO CANDIDATE:128====================="
python generate_data.py \
    beam_size=5 \
    few_shot_num=2 \
    candidate_set_num=128 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO TEXT-SIM====================="
python generate_data.py \
    beam_size=5 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="text-sim" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO IMAGE-SIM====================="
python generate_data.py \
    beam_size=5 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="image-sim" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

echo "=====================BEGIN TO 1w SAMPLE_NUM===================="
python generate_data.py \
    beam_size=5 \
    few_shot_num=2 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="image-sim" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset}

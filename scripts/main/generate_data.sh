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

command1_pid=2566156

# 循环检查command1的状态，直到它完成​
while ps -p $command1_pid > /dev/null; do
  echo "command1 not done"
  sleep 60
done

# Baseline:
echo "=====================BEGIN TO BASELINE====================="
python generate_data.py \
    beam_size=5 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset} \
    few_shot_num=4
#!/bin/bash

# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device_num=${3:-1}

which python

# Define a function to run the train.py script with the given parameters
run_train() {
    local data_file=$1
    local val_step=$2
    local ex_name_suffix=$3

    local ex_name_prefix="ab_${task}"

    python train.py train=query_img_ice_text \
        data_files="${data_file}" \
        epochs=20 \
        val_step=${val_step} \
        ex_name="${ex_name_prefix}_${ex_name_suffix}" \
        device_num=${device_num} \
        dataset=${dataset} \
        task=${task}
}

if [ "${task}" == "vqa" ]; then
    # VQA mode
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "beam_size"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" 160 "10000sample_num"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "random_sample"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "random_sample_128candidate_set_num"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "image_sim_method"
    run_train "vqa-vqav2--OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "not_only_yloss"

elif [ "${task}" == "caption" ]; then
    # Caption mode
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" 160 "10000sample_num"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "random_sample"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "random_sample_128candidate_set_num"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "image_sim_method"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "beamsize"
    # run_train "caption-coco2017--OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "not_only_yloss"

else
    echo "Invalid task. Please choose 'vqa' or 'caption'."
    exit 1
fi

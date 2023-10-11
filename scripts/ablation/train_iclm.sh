#!/bin/bash

# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device_num=${3:-1}
iclm_model=${4:-query_img_ice_text}

which python

# Define a function to run the train.py script with the given parameters
run_train() {
    local data_file=$1
    local val_step=$2
    local ex_name_suffix=$3

    local ex_name_prefix="ab_${task}"

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
        python train.py train="${iclm_model}" \
            data_files="${data_file}" \
            epochs=20 \
            val_step=${val_step} \
            ex_name="${ex_name_prefix}_${ex_name_suffix}_${iclm_model}" \
            device_num=${device_num} \
            dataset=${dataset} \
            task=${task}

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
        python train.py train="${iclm_model}" \
            data_files="${data_file}" \
            epochs=20 \
            val_step=${val_step} \
            ex_name="${ex_name_prefix}_${ex_name_suffix}_non_norm_${iclm_model}" \
            device_num=${device_num} \
            dataset=${dataset} \
            task=${task} \
            train.iclm_model.norm=false
    fi
}

if [ "${task}" == "vqa" ]; then
    # VQA mode
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:1-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "1beam"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "10beam"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "128candidate"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "text-sim"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "img-sim"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "1wanchors"

elif [ "${task}" == "caption" ]; then
    # Caption mode
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:1-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "1beam"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "10beam"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "128candidate"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "text-sim"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "img-sim"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "1wanchors"
else
    echo "Invalid task. Please choose 'vqa' or 'caption'."
    exit 1
fi

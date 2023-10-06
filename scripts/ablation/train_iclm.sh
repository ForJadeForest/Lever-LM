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
    local iclm_model=$4

    local ex_name_prefix="ab_${task}"
    echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
    python train.py train="${iclm_model}" \
        data_files="${data_file}" \
        epochs=20 \
        val_step=${val_step} \
        ex_name="${ex_name_prefix}_${ex_name_suffix}_${iclm_model}" \
        device_num=${device_num} \
        dataset=${dataset} \
        task=${task}
}

if [ "${task}" == "vqa" ]; then
    # VQA mode
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline" "${iclm_model}"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "beam_size" "${iclm_model}"
    # run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" 160 "10000sample_num" "${iclm_model}"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "random_sample" "${iclm_model}"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "random_sample_128candidate_set_num" "${iclm_model}"
    run_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "image_sim_method" "${iclm_model}"
    

elif [ "${task}" == "caption" ]; then
    # Caption mode
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline" "${iclm_model}"
    # run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" 160 "10000sample_num" "${iclm_model}"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "random_sample" "${iclm_model}"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random_sample_candidate-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "random_sample_128candidate_set_num" "${iclm_model}"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "image_sim_method" "${iclm_model}"
    run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "beamsize" "${iclm_model}"
    run_train "caption-coco2017-cider_versionOpenFlamingo-9B-vitl-mpt7b-text-beam_size:5-few_shot:2-candidate_set_num:64.json" 80 "cider_score" "${iclm_model}"

else
    echo "Invalid task. Please choose 'vqa' or 'caption'."
    exit 1
fi

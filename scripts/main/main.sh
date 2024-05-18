#!/bin/bash

# Ensure three arguments are provided
# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
flamingo=${4:-flamingo_9B}

if [ -z "$task" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 <task> <dataset> [gpu_ids]"
    exit 1
fi


which python


echo "=====================BEGIN TO BASELINE====================="
python generate_data.py \
    beam_size=5 \
    candidate_set_num=64 \
    sample_num=5000 \
    candidate_set_method="random" \
    gpu_ids="${gpu_ids}" \
    task=${task} \
    dataset=${dataset} \
    few_shot_num=4 \
    flamingo=${flamingo}


run_train() {
    local data_file=$1
    local val_step=$2
    local ex_name_suffix=$3

    local ex_name_prefix="main_${task}"

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${lever_lm_model}==========" 
        python train.py train="${lever_lm_model}" \
            data_files="${data_file}" \
            epochs=20 \
            val_step=${val_step} \
            ex_name="${ex_name_prefix}_${ex_name_suffix}_${lever_lm_model}" \
            device_num=${device_num} \
            dataset=${dataset} \
            task=${task}

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${lever_lm_model}==========" 
        python train.py train="${lever_lm_model}" \
            data_files="${data_file}" \
            epochs=20 \
            val_step=${val_step} \
            ex_name="${ex_name_prefix}_${ex_name_suffix}_freeze_adapter_non_norm_${lever_lm_model}" \
            device_num=${device_num} \
            dataset=${dataset} \
            task=${task} \
            train.lever_lm_model.norm=false \
            train.lever_lm_model.freeze_prefix_list="[img_model,sen_model]" \
            train.lever_lm_model.adpter=true
    fi
}
if ["${flamingo}" == "flamingo_9B"]; then
    flamingo_name="OpenFlamingo-9B-vitl-mpt7b"
elif ["${flamingo}" == "flamingo_9B_v1"]; then
    flamingo_name="OpenFlamingo-9B-deprecated"
elif ["${flamingo}" == "flamingo_3B"]; then
    flamingo_name="OpenFlamingo-3B-vitl-mpt1b"
fi

run_train "${task}-${dataset}-only_y_loss-${flamingo_name}-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "${flamingo}"



# Define a function to run the train.py script with the given parameters
run_inference() {
    local ex_name_suffix=$1
    local ex_name_prefix="main_${task}"
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${lever_lm_model}==========" 
        python inference_flamingo_fast.py   train="${lever_lm_model}" \
                                            ex_name="${ex_name_prefix}_${ex_name_suffix}_${lever_lm_model}" \
                                            dataset=${dataset} \
                                            task=${task}\
                                            inference_bs=${inference_bs}\
                                            test_lever_lm=true\
                                            flamingo=${flamingo}
    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${lever_lm_model}==========" 
        python inference_flamingo_fast.py   train="${lever_lm_model}" \
                                            ex_name="${ex_name_prefix}_${ex_name_suffix}_freeze_adapter_non_norm_${lever_lm_model}" \
                                            dataset=${dataset} \
                                            task=${task} \
                                            inference_bs=${inference_bs} \
                                            test_lever_lm=true \
                                            flamingo=${flamingo} \
                                            train.lever_lm_model.norm=false \
                                            train.lever_lm_model.freeze_prefix_list="[img_model,sen_model]" \
                                            train.lever_lm_model.adpter=true
    fi
}
run_inference "${flamingo}"
#!/bin/bash

# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
lever_lm_model=${4:-query_img_ice_text}
inference_bs=${5:-4}
flamingo=${6:-flamingo_9B}


which python
export CUDA_VISIBLE_DEVICES="${device}"

# Define a function to run the train.py script with the given parameters
run_inference() {
    local ex_name_suffix=$1
    local ex_name_prefix="ab_${task}"
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
                                            ex_name="${ex_name_prefix}_${ex_name_suffix}_non_norm_freeze_adapter_${lever_lm_model}" \
                                            dataset=${dataset} \
                                            task=${task}\
                                            inference_bs=${inference_bs}\
                                            test_lever_lm=true\
                                            flamingo=${flamingo} \
                                            train.lever_lm_model.norm=false \
                                            train.lever_lm_model.freeze_prefix_list="[img_model,sen_model]" \
                                            train.lever_lm_model.adpter=true
    fi
}

run_inference "baseline"
run_inference "1beam"
run_inference "10beam"
run_inference "128candidate"
run_inference "text-sim"
run_inference "img-sim"
run_inference "1wanchors"

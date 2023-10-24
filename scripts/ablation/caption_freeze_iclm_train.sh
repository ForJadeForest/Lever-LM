#!/bin/bash

# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device_num=${3:-1}
iclm_model=${4:-query_img_ice_text}

which python
export CUDA_VISIBLE_DEVICES="2,3"
# Define a function to run the train.py script with the given parameters
run_train() {
    local data_file=$1
    local val_step=$2
    local ex_name_suffix=$3

    local ex_name_prefix="ab_${task}"


    echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
    python train.py train="${iclm_model}" \
        data_files="${data_file}" \
        epochs=20 \
        val_step=${val_step} \
        ex_name="${ex_name_prefix}_${ex_name_suffix}_non_norm_freeze_adapter_${iclm_model}" \
        device_num=${device_num} \
        dataset=${dataset} \
        task=${task} \
        train.iclm_model.norm=false \
        train.iclm_model.freeze_prefix_list="[img_model,sen_model]" \
        train.iclm_model.adpter=true
    
}


run_inference() {
    local ex_name_suffix=$1
    local ex_name_prefix="ab_${task}"

    echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
    python inference_flamingo_fast.py   train="${iclm_model}" \
                                        ex_name="${ex_name_prefix}_${ex_name_suffix}_non_norm_freeze_adapter_${iclm_model}" \
                                        dataset=${dataset} \
                                        task=${task}\
                                        inference_bs=4\
                                        test_iclm=true\
                                        train.iclm_model.norm=false \
                                        train.iclm_model.freeze_prefix_list="[img_model,sen_model]" \
                                        train.iclm_model.adpter=true
}



run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "baseline"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:1-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "1beam"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:10-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "10beam"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:128-sample_num:5000.json" 80 "128candidate"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-text-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "text-sim"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-image-sim-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json" 80 "img-sim"
run_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:10000.json" 160 "1wanchors"


export CUDA_VISIBLE_DEVICES="0" && run_inference "baseline" &
sleep 120
export CUDA_VISIBLE_DEVICES="1" && run_inference "1beam" &
sleep 120
export CUDA_VISIBLE_DEVICES="2" && run_inference "10beam" &
sleep 120
export CUDA_VISIBLE_DEVICES="3" && run_inference "128candidate" &
wait 

export CUDA_VISIBLE_DEVICES="1" && run_inference "text-sim" &
sleep 120
export CUDA_VISIBLE_DEVICES="2" && run_inference "img-sim" &
sleep 120
export CUDA_VISIBLE_DEVICES="3" && run_inference "1wanchors" &

wait
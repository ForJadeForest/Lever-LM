#!/bin/bash
device_num=${1:-1}
which python
unset CUDA_VISIBLE_DEVICES

check_command_status() {
    local pid=$1
    # 循环检查command1的状态，直到它完成
    while ps -p $pid > /dev/null; do
        echo "command $pid not done"
        sleep 60
    done
}

pids=(22533 13950)

for pid in "${pids[@]}"; do
    check_command_status $pid
done



# Define a function to run the train.py script with the given parameters
run_vqa_train() {
    local data_file=$1
    local val_step=$2
    local ex_name_suffix=$3
    local ex_name_prefix="ab_${task}"


    echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
    python train.py train="query_img_text_ice_img_text" \
        data_files="${data_file}" \
        epochs=20 \
        val_step=${val_step} \
        ex_name="${ex_name_prefix}_${ex_name_suffix}_${iclm_model}" \
        device_num=${device_num} \
        dataset="vqav2_local_sub" \
        task="vqa"
}

run_caption_train(){
    echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
    python train.py train="query_img_ice_img_text" \
        data_files="${data_file}" \
        epochs=20 \
        val_step=${val_step} \
        ex_name="${ex_name_prefix}_${ex_name_suffix}_freeze_adapter_non_norm_${iclm_model}" \
        device_num=${device_num} \
        dataset="coco2017" \
        task="caption" \
        train.iclm_model.norm=false \
        train.iclm_model.freeze_prefix_list="[img_model,sen_model]" \
        train.iclm_model.adpter=true
}


run_inference() {
    local ex_name_suffix=$1
    local ex_name_prefix="ab_${task}"
    local task=$2
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
        python inference_flamingo_fast.py   train="${iclm_model}" \
                                            ex_name="${ex_name_prefix}_${ex_name_suffix}_${iclm_model}" \
                                            dataset=vqav2_local_sub \
                                            task=${task} \
                                            inference_bs=${inference_bs} \
                                            test_iclm=true

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICLM: ${iclm_model}==========" 
        python inference_flamingo_fast.py   train="${iclm_model}" \
                                            ex_name="${ex_name_prefix}_${ex_name_suffix}_non_norm_freeze_adapter_${iclm_model}" \
                                            dataset=coco2017 \
                                            task=${task}\
                                            inference_bs=${inference_bs}\
                                            test_iclm=true\
                                            train.iclm_model.norm=false \
                                            train.iclm_model.freeze_prefix_list="[img_model,sen_model]" \
                                            train.iclm_model.adpter=true
    fi
}


run_caption_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:1000.json" 80 "1kanchors"
run_caption_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:2000.json" 80 "2kanchors"
run_caption_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:3000.json" 80 "3kanchors"
run_caption_train "caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:4000.json" 80 "4kanchors"





run_vqa_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:1000.json" 80 "1kanchors"
run_vqa_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:2000.json" 80 "2kanchors"
run_vqa_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:3000.json" 80 "3kanchors"
run_vqa_train "vqa-vqav2-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:4000.json" 80 "4kanchors"


export CUDA_VISIBLE_DEVICES="0" && run_inference "1kanchors" "vqa" &
sleep 120
export CUDA_VISIBLE_DEVICES="1" && run_inference "2kanchors" "vqa" &
sleep 120
export CUDA_VISIBLE_DEVICES="2" && run_inference "3kanchors" "vqa" &
sleep 120
export CUDA_VISIBLE_DEVICES="3" && run_inference "4kanchors" "vqa" &


export CUDA_VISIBLE_DEVICES="0" && run_inference "1kanchors" "caption" &
sleep 120
export CUDA_VISIBLE_DEVICES="1" && run_inference "2kanchors" "caption" &
sleep 120
export CUDA_VISIBLE_DEVICES="2" && run_inference "3kanchors" "caption" &
sleep 120
export CUDA_VISIBLE_DEVICES="3" && run_inference "4kanchors" "caption" &
# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
lever_lm=${4:-query_img_icd_img_text}
inference_bs=${5:-4}


run_inference() {
    local ex_name_prefix="main_${task}"
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-LeverLM: ${lever_lm}==========" 
        python icl_inference.py train="${lever_lm}" \
                                ex_name="${ex_name_prefix}_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task}\
                                inference_bs=${inference_bs}\
                                test_lever_lm=true

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-LeverLM: ${lever_lm}==========" 
        python icl_inference.py train="${lever_lm}" \
                                ex_name="${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                inference_bs=${inference_bs} \
                                test_lever_lm=true\
                                train.lever_lm.norm=false \
                                train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                                train.lever_lm.adapter=true
    fi
}
run_inference
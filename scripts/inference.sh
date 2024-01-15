# Set default values
task=${1:-caption}
dataset=${2:-coco2017}
device=${3:-0}
icd_lm=${4:-query_img_icd_img_text}
inference_bs=${5:-4}


run_inference() {
    local ex_name_prefix="main_${task}"
    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICDLM: ${icd_lm}==========" 
        python icl_inference.py train="${icd_lm}" \
                                ex_name="${ex_name_prefix}_${icd_lm}" \
                                dataset=${dataset} \
                                task=${task}\
                                inference_bs=${inference_bs}\
                                test_icd_lm=true

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}_${ex_name_suffix}-ICDLM: ${icd_lm}==========" 
        python icl_inference.py train="${icd_lm}" \
                                ex_name="${ex_name_prefix}_freeze_adapter_non_norm_${icd_lm}" \
                                dataset=${dataset} \
                                task=${task} \
                                inference_bs=${inference_bs} \
                                test_icd_lm=true\
                                train.icd_lm.norm=false \
                                train.icd_lm.freeze_prefix_list="[img_model,sen_model]" \
                                train.icd_lm.adpter=true
    fi
}
run_inference
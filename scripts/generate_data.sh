task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}

if [ -z "$task" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 <task> <dataset> [gpu_ids]"
    exit 1
fi


python generate_data.py beam_size=5 \
                        sampler.candidate_num=64 \
                        sample_num=5000 \
                        gpu_ids="${gpu_ids}" \
                        task=${task} \
                        dataset=${dataset} \
                        few_shot_num=2
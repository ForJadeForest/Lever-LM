data_files="caption-coco2017-idefics_9B-RandSampler-construct_order:leftbeam_size:5-few_shot:2-candidate_num:64-sample_num:5000.json"

python train.py data_files=${data_files} \
                ex_name="idefics_caption_tiny_lever-lm" \
                train.lever_lm.lm_config.n_layer=1 \
                train.lever_lm.lm_config.n_embd=128 \
                num_workers=8


python train.py data_files=${data_files} \
                ex_name="idefics_caption_tiny_large_emb_lever-lm" \
                train.lever_lm.lm_config.n_layer=1 \
                train.lever_lm.lm_config.n_embd=512 \
                num_workers=8


python train.py data_files=${data_files} \
                ex_name="idefics_caption_large_lever-lm" \
                train.lever_lm.lm_config.n_layer=4 \
                train.lever_lm.lm_config.n_embd=512 \
                num_workers=8

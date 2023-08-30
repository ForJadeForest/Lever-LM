python generate_data.py --model_name "open_flamingo" \
                        --train_coco_dataset_root "/data/share/pyz/data/mscoco/mscoco2017/train2017" \
                        --train_coco_annotation_file "/data/share/pyz/data/mscoco/mscoco2017/annotations/captions_train2017.json" \
                        --lang_encoder_path "anas-awadalla/mpt-7b" \
                        --tokenizer_path "anas-awadalla/mpt-7b" \
                        --flamingo_checkpoint_path "/data/share/pyz/checkpoint/openflamingo/OpenFlamingo-9B-vitl-mpt7b"\
                        --cross_attn_every_n_layers 4 \
                        --hf_root "OpenFlamingo-9B-vitl-mpt7b" \
                        --sim_method "caption"\
                        --sim_model_type "openai/clip-vit-large-patch14"\
                        --query_set_batch_size "128" \
                        --query_top_k 64 \
                        --beam_size 16 \
                        --few_shot_num 2 \
                        --batch_size 32 \
                        --device "cuda" \
                        --precision "bf16" \
                        --result_dir "/home/pyz32/code/iclm/result" \
                        --gpu_ids 0 1 2 3 \
                        --sample_num 10000
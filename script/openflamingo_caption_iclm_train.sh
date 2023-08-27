python open_flamingo_caption/iclm_train.py  --model_name "open_flamingo" \
                                            --lang_encoder_path "anas-awadalla/mpt-7b" \
                                            --tokenizer_path "anas-awadalla/mpt-7b" \
                                            --flamingo_checkpoint_path "/data/share/pyz/checkpoint/openflamingo/OpenFlamingo-9B-vitl-mpt7b"\
                                            --cross_attn_every_n_layers 4 \
                                            --hf_root "OpenFlamingo-9B-vitl-mpt7b" \
                                            --n_layers 4 \
                                            --vocab_size 118287\
                                            --n_embd 4096 \
                                            --n_head 16 \
                                            --device "cuda:2" \
                                            --epochs 40 \
                                            --train_ratio 0.9 \
                                            --batch_size 128 \
                                            --lr 1e-5 \
                                            --warm_up_ratio 0.05 \
                                            --precisio "bf16" \
                                            --num_workers 8 \
                                            --val_coco_dataset_root "/data/share/pyz/data/mscoco/val2017" \
                                            --val_coco_annotation_file "/data/share/pyz/data/mscoco/annotations/captions_val2017.json" \
                                            --dataset_encoder_bs 8 \
                                            --ex_name "random_init_image_flamingo" \
                                            --result_dir "/home/pyz32/code/ICLM/open_flamingo_caption/result" \
                                            --data_files "open_flamingo-coco-image-beam_size:8-few_shot:2-query_top_k:50.json"\
                                            --pool_method "mean"




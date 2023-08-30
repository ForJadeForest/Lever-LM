#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python sen_img_encode_iclm_lstm_train_fabric.py \
    --n_layers 2 \
    --vocab_size 118287 \
    --n_embd 512 \
    --n_head 16 \
    --device "cuda" \
    --device_num 4 \
    --epochs 50 \
    --train_ratio 0.9 \
    --batch_size 64 \
    --lr 1e-3 \
    --warm_up_ratio 0.05 \
    --precisio "32" \
    --num_workers 8 \
    --train_coco_dataset_root "/data/share/pyz/data/mscoco/mscoco2017/train2017" \
    --train_coco_annotation_file "/data/share/pyz/data/mscoco/mscoco2017/annotations/captions_train2017.json" \
    --result_dir "/home/pyz32/code/iclm/result" \
    --data_files "open_flamingo-coco-caption-beam_size:8-few_shot:4-query_top_k:50_by_train_ds.json" \
    --ex_name "sen_img_encode_iclm_lstm"


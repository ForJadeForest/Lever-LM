import json
import os
import random
from time import sleep
from typing import Dict, List

import hydra
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig
from PIL import Image
from torch.multiprocessing import spawn
from tqdm import tqdm

from src.metrics.info_score import get_info_score
from src.utils import (
    encode_image,
    encode_text,
    init_flamingo,
    load_coco_ds,
    load_vqa_train_ds,
    recall_sim_feature,
)


def get_caption_prompt(single_caption=None) -> str:
    return f"<image>Output:{single_caption if single_caption is not None else ''}{'<|endofchunk|>' if single_caption is not None else ''}"


def get_vqa_prompt(question, answer=None) -> str:
    return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"


def load_feature_cache(cfg, cache_path, encoding_method, coco_dataset, data_key):
    if os.path.exists(cache_path):
        features = torch.load(cache_path)
    else:
        data_list = [d[data_key] for d in coco_dataset]
        features = encoding_method(
            data_list, cfg.device, cfg.sim_model_type, cfg.candidate_set_encode_bs
        )
        torch.save(features, cache_path)
    return features


def beam_filter(score_list, data_id_list, beam_size):
    score_list = torch.tensor(score_list)
    score_value, indices = torch.topk(score_list, beam_size)
    return score_value.tolist(), [data_id_list[idx] for idx in indices]


@torch.inference_mode()
def generate_single_sample_ice(
    model,
    tokenizer,
    image_processor,
    test_data: Dict,
    cfg: DictConfig,
    candidate_set: Dataset,
    autocast_context,
    device,
    few_shot_num=4,
    batch_size=32,
    beam_size=5,
):
    # 构建test sample prompt
    if cfg.task.task_name == 'vqa':
        prompt_method = get_vqa_prompt
    elif cfg.task.task_name == 'caption':
        prompt_method = get_caption_prompt

    test_text_input = {k: test_data[k] for k in cfg.task.text_fields}
    test_data_text = prompt_method(**test_text_input)

    test_data_image = test_data[cfg.task.img_field]
    test_data_id = test_data['idx']

    # 构建candidate set
    candidateidx2data = {
        data['idx']: {
            'text_input': prompt_method(**{k: data[k] for k in cfg.task.text_fields}),
            'image': data[cfg.task.img_field],
            'idx': data['idx'],
        }
        for data in candidate_set
    }
    test_data_id_list = [[test_data_id]]

    for _ in range(few_shot_num):
        new_test_data_id_list = []
        new_test_score_list = []
        for test_data_id_seq in test_data_id_list:
            # 避免添加重复的结果 将已经添加的进行过滤
            filtered_candidateidx2data = candidateidx2data.copy()
            if len(test_data_id_list) >= 2:
                filter_id_list = test_data_id_seq[:-1]
                for i in filter_id_list:
                    filtered_candidateidx2data.pop(i)
                

            # 构建已经选好的ice + 测试样本的输入
            ice_id_seq = test_data_id_seq[:-1]
            lang_x = [candidateidx2data[idx]['text_input'] for idx in ice_id_seq] + [
                test_data_text
            ]
            image_x = [candidateidx2data[idx]['image'] for idx in ice_id_seq] + [
                test_data_image
            ]

            filtered_idx_list = sorted(list(filtered_candidateidx2data.keys()))
            info_score = get_info_score(
                model,
                tokenizer,
                image_processor,
                device,
                ice_join_char=cfg.task.ice_join_char,
                lang_x=lang_x,
                image_x=image_x,
                candidate_set=filtered_candidateidx2data,
                batch_size=batch_size,
                autocast_context=autocast_context,
            )

            # 选出最高的InfoScore
            scores, indices = info_score.topk(beam_size)
            indices = indices.tolist()
            indices = list(
                map(
                    lambda x: filtered_candidateidx2data[filtered_idx_list[x]]['idx'],
                    indices,
                )
            )
            scores = scores.tolist()

            for idx, score in zip(indices, scores):
                new_test_data_id_list.append([idx, *test_data_id_seq])
                new_test_score_list.append(score)

        new_test_score_list, new_test_data_id_list = beam_filter(
            new_test_score_list, new_test_data_id_list, beam_size
        )
        test_data_id_list = new_test_data_id_list
    return {
        test_data_id: {'id_list': test_data_id_list, 'score_list': new_test_score_list}
    }


def gen_data(
    rank,
    lang_encoder_path,
    tokenizer_path,
    flamingo_checkpoint_path,
    cross_attn_every_n_layers,
    hf_root,
    precision,
    sample_data,
    train_ds,
    candidate_set_idx,
    save_path,
    cfg,
):
    world_size = len(cfg.gpu_ids)
    process_device = f'cuda:{cfg.gpu_ids[rank]}'

    subset_size = len(sample_data) // world_size
    subset_start = rank * subset_size
    subset_end = (
        subset_start + subset_size if rank != world_size - 1 else len(sample_data)
    )
    subset = sample_data.select(range(subset_start, subset_end))
    sub_cand_set_idx = candidate_set_idx[subset_start:subset_end]

    # load several models will cost large memory at the same time.
    # use sleep to load one by one.
    sleep(cfg.sleep_time * rank)
    model, image_processor, tokenizer, autocast_context = init_flamingo(
        lang_encoder_path,
        tokenizer_path,
        flamingo_checkpoint_path,
        cross_attn_every_n_layers,
        hf_root,
        precision,
        process_device,
    )
    tokenizer.pad_token = tokenizer.eos_token

    final_res = {}
    sub_res_basename = (
        os.path.basename(save_path).split('.')[0]
        + f'_rank:{rank}_({subset_start}, {subset_end}).json'
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)

    for i, test_data in enumerate(tqdm(subset, disable=(rank != 0))):
        candidate_set = train_ds.select(sub_cand_set_idx[i])
        res = generate_single_sample_ice(
            model,
            tokenizer,
            image_processor,
            test_data,
            cfg,
            candidate_set,
            device=process_device,
            few_shot_num=cfg.few_shot_num,
            autocast_context=autocast_context,
            batch_size=cfg.bs,
            beam_size=cfg.beam_size,
        )
        final_res.update(res)
    with open(save_path, 'w') as f:
        json.dump(final_res, f)
    return final_res


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data.yaml"
)
def main(cfg: DictConfig):
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)
    cache_dir = os.path.join(cfg.result_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_dir = os.path.join(cfg.result_dir, 'generated_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_proc_save_dir = os.path.join(save_dir, 'sub_proc_data')
    if not os.path.exists(sub_proc_save_dir):
        os.makedirs(sub_proc_save_dir)

    candidate_method = (
        'random_sample_candidate' if cfg.random_sample_candidate_set else cfg.sim_method
    )

    save_file_name = (
        f'{cfg.task.task_name}-{cfg.flamingo.hf_root}-{cfg.dataset.name}-{candidate_method}-'
        f'beam_size:{cfg.beam_size}-few_shot:{cfg.few_shot_num}-'
        f'candidate_set_num:{cfg.candidate_set_num}.json'
    )

    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    # 加载数据集
    if cfg.task.task_name == 'caption':
        train_ds = load_coco_ds(cfg, split='train')
    elif cfg.task.task_name == 'vqa':
        train_ds = load_vqa_train_ds(cfg)
    else:
        raise ValueError(f'{cfg.task.task_name=} error, should in ["caption", "vqa"]')

    # sample from train idx
    sample_index = random.sample(range(0, len(train_ds)), cfg.sample_num)
    sample_data = train_ds.select(sample_index)

    # get the candidate set
    if cfg.random_sample_candidate_set:
        candidate_set_idx = []
        for s_idx in sample_index:
            random_candidate_set = random.sample(
                range(0, len(train_ds)), cfg.candidate_set_num
            )
            while s_idx in random_candidate_set:
                random_candidate_set = random.sample(
                    list(range(0, len(train_ds))), cfg.candidate_set_num
                )
            candidate_set_idx.append(random_candidate_set)

    else:
        # pre-calculate the cache feature for knn search
        if cfg.sim_method == 'text':
            encoding_method = encode_text
            data_key = cfg.task.sim_text_field
        elif cfg.sim_method == 'image':
            encoding_method = encode_image
            data_key = cfg.task.sim_img_field
        else:
            raise ValueError('the sim_method error')
        sim_model_name = cfg.sim_model_type.split('/')[-1]
        train_cache_path = os.path.join(
            cache_dir,
            f'{cfg.task.task_name}-{cfg.dataset.name}-'
            f'{cfg.sim_method}-{sim_model_name}-feature.pth',
        )
        train_feature = load_feature_cache(
            cfg, train_cache_path, encoding_method, train_ds, data_key
        )
        test_feature = train_feature[sample_index]
        _, candidate_set_idx = recall_sim_feature(
            test_feature, train_feature, top_k=cfg.candidate_set_num + 1
        )

        candidate_set_idx = candidate_set_idx[:, 1:]

    spawn(
        gen_data,
        args=(
            cfg.flamingo.lang_encoder_path,
            cfg.flamingo.tokenizer_path,
            cfg.flamingo.flamingo_checkpoint_path,
            cfg.flamingo.cross_attn_every_n_layers,
            cfg.flamingo.hf_root,
            cfg.precision,
            sample_data,
            train_ds,
            candidate_set_idx,
            sub_save_path,
            cfg,
        ),
        nprocs=len(cfg.gpu_ids),
        join=True,
    )

    world_size = len(cfg.gpu_ids)
    subset_size = len(sample_data) // world_size
    total_data = {}
    for rank in range(world_size):
        subset_start = rank * subset_size
        subset_end = (
            subset_start + subset_size if rank != world_size - 1 else len(sample_data)
        )
        sub_res_basename = (
            os.path.basename(save_path).split('.')[0]
            + f'_rank:{rank}_({subset_start}, {subset_end}).json'
        )
        sub_save_path = sub_save_path.replace(
            os.path.basename(sub_save_path), sub_res_basename
        )
        with open(sub_save_path, 'r') as f:
            data = json.load(f)
        total_data.update(data)

    with open(save_path, 'w') as f:
        json.dump(total_data, f)


if __name__ == '__main__':
    load_dotenv()
    main()

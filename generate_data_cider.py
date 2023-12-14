import json
import os
import random
from time import sleep
from typing import Dict, List

import hydra
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from openicl import PromptTemplate
from PIL import Image
from torch.multiprocessing import spawn
from tqdm import tqdm

from src.load_ds_utils import load_coco_ds, load_vqav2_ds
from src.metrics.cider_calculator import get_cider_score
from src.utils import encode_image, encode_text, init_flamingo, recall_sim_feature

from .utils import load_ds


def load_feature_cache(cfg, cache_path, encoding_method, train_ds, data_key):
    if os.path.exists(cache_path):
        features = torch.load(cache_path)
    else:
        features = encoding_method(
            train_ds,
            data_key,
            cfg.device,
            cfg.sim_model_type,
            cfg.candidate_set_encode_bs,
        )
        torch.save(features, cache_path)
    return features


def beam_filter(score_list, data_id_list, beam_size):
    score_list = torch.tensor(score_list)
    score_value, indices = torch.topk(score_list, beam_size)
    return score_value.tolist(), [data_id_list[idx] for idx in indices]


@torch.inference_mode()
def generate_single_sample_icd(
    model,
    tokenizer,
    image_processor,
    test_data: Dict,
    cfg: DictConfig,
    candidate_set: Dataset,
    autocast_context,
    device,
):
    template = PromptTemplate(
        cfg.task.template,
        column_token_map=dict(cfg.task.column_token_map),
        icd_token=cfg.task.icd_token,
    )

    # 构建test sample prompt
    test_data_text = template.generate_item(
        test_data, output_field=cfg.task.output_column
    )

    test_data_image = test_data[cfg.task.image_field]
    test_data_id = test_data['idx']

    # 构建candidate set
    candidateidx2data = {
        data['idx']: {
            'text_input': template.generate_item(data),
            'image': data[cfg.task.image_field],
            'idx': data['idx'],
            'image_id': data['image_id'],
        }
        for data in candidate_set
    }
    test_data_id_list = [[test_data_id]]

    for _ in range(cfg.few_shot_num):
        new_test_data_id_list = []
        new_test_score_list = []
        for test_data_id_seq in test_data_id_list:
            # 避免添加重复的结果 将已经添加的进行过滤
            filtered_candidateidx2data = candidateidx2data.copy()
            if len(test_data_id_seq) >= 2:
                filter_id_list = test_data_id_seq[:-1]
                for i in filter_id_list:
                    filtered_candidateidx2data.pop(i)

            # 构建已经选好的icd + 测试样本的输入
            icd_id_seq = test_data_id_seq[:-1]
            lang_x = [candidateidx2data[idx]['text_input'] for idx in icd_id_seq] + [
                test_data_text
            ]
            image_x = [candidateidx2data[idx]['image'] for idx in icd_id_seq] + [
                test_data_image
            ]

            filtered_idx_list = sorted(list(filtered_candidateidx2data.keys()))
            info_score = get_cider_score(
                model,
                tokenizer,
                image_processor,
                device,
                icd_join_char=cfg.task.icd_join_char,
                lang_x=lang_x,
                image_x=image_x,
                candidate_set=filtered_candidateidx2data,
                batch_size=cfg.batch_size,
                train_ann_path=cfg.dataset.train_coco_annotation_file,
                gen_kwargs=cfg.task.gen_args,
                autocast_context=autocast_context,
            )

            # 选出最高的InfoScore
            scores, indices = info_score.topk(cfg.beam_size)
            indices = indices.tolist()
            indices = list(
                map(
                    lambda x: filtered_idx_list[x],
                    indices,
                )
            )
            scores = scores.tolist()

            for idx, score in zip(indices, scores):
                new_test_data_id_list.append([idx, *test_data_id_seq])
                new_test_score_list.append(score)

        new_test_score_list, new_test_data_id_list = beam_filter(
            new_test_score_list, new_test_data_id_list, cfg.beam_size
        )
        test_data_id_list = new_test_data_id_list
    return {
        test_data_id: {'id_list': test_data_id_list, 'score_list': new_test_score_list}
    }


def gen_data(
    rank,
    cfg,
    sample_data,
    train_ds,
    candidate_set_idx,
    save_path,
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
        cfg.flamingo.lang_encoder_path,
        cfg.flamingo.tokenizer_path,
        cfg.flamingo.flamingo_checkpoint_dir,
        cfg.flamingo.cross_attn_every_n_layers,
        cfg.flamingo.hf_root,
        cfg.precision,
        process_device,
        cfg.flamingo.load_from_local,
    )

    final_res = {}
    sub_res_basename = (
        os.path.basename(save_path).split('.')[0]
        + f'_rank:{rank}_({subset_start}, {subset_end}).json'
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)
    if os.path.exists(save_path):
        final_res.update(json.load(open(save_path)))
        logger.info(
            f'Rank: {rank} reloading data from {save_path}, begin from {len(final_res)}'
        )
    if len(final_res) == subset_size:
        logger.info(f'Rank: {rank} task is Done.')
        return

    subset = subset.select(range(len(final_res), len(subset)))
    for i, test_data in enumerate(
        tqdm(
            subset,
            disable=(rank != 0),
            total=subset_size,
            initial=len(final_res),
            ncols=100,
        ),
    ):
        candidate_set = train_ds.select(sub_cand_set_idx[i])
        res = generate_single_sample_icd(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            test_data=test_data,
            cfg=cfg,
            candidate_set=candidate_set,
            device=process_device,
            autocast_context=autocast_context,
        )
        final_res.update(res)

        with open(save_path, 'w') as f:
            json.dump(final_res, f)
    return


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data_cider.yaml"
)
def main(cfg: DictConfig):
    logger.info(f'{cfg=}')
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

    save_file_name = (
        f'{cfg.task.task_name}-{cfg.dataset.name}-cider_version'
        f'{cfg.flamingo.hf_root}-{cfg.candidate_set_method}-'
        f'beam_size:{cfg.beam_size}-few_shot:{cfg.few_shot_num}-'
        f'candidate_set_num:{cfg.candidate_set_num}-sample_num:{cfg.sample_num}.json'
    )

    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    # 加载数据集
    train_ds = load_ds(cfg, 'train')
    # sample from train idx
    anchor_set_cache_filename = os.path.join(
        cache_dir, f'{cfg.dataset.name}-sample_num:{cfg.sample_num}.json'
    )

    candidate_set_cache_filename = os.path.join(
        cache_dir,
        f'{cfg.dataset.name}-sample_num:{cfg.sample_num}-'
        f'candidate_set_num:{cfg.candidate_set_num}-method:{cfg.candidate_set_method}.json',
    )

    if os.path.exists(anchor_set_cache_filename):
        logger.info('the anchor_set_cache_filename exists, loding...')
        anchor_idx_list = json.load(open(anchor_set_cache_filename, 'r'))
    else:
        anchor_idx_list = random.sample(range(0, len(train_ds)), cfg.sample_num)
        with open(anchor_set_cache_filename, 'w') as f:
            logger.info(f'save {anchor_set_cache_filename}...')
            json.dump(anchor_idx_list, f)
    anchor_data = train_ds.select(anchor_idx_list)

    if os.path.exists(candidate_set_cache_filename):
        logger.info('the candidate set cache exists, loding...')
        candidate_set_idx = json.load(open(candidate_set_cache_filename, 'r'))
        candidate_set_idx = {int(k): v for k, v in candidate_set_idx.items()}
    else:
        candidate_set_idx = {}
        if cfg.candidate_set_method == 'random':
            for s_idx in anchor_idx_list:
                random_candidate_set = random.sample(
                    range(0, len(train_ds)), cfg.candidate_set_num
                )
                while s_idx in random_candidate_set:
                    random_candidate_set = random.sample(
                        list(range(0, len(train_ds))), cfg.candidate_set_num
                    )
                candidate_set_idx[s_idx] = random_candidate_set
        else:
            # pre-calculate the cache feature for knn search
            if cfg.candidate_set_method == 'text-sim':
                encoding_method = encode_text
                data_key = cfg.task.sim_text_field
            elif cfg.candidate_set_method == 'image-sim':
                encoding_method = encode_image
                data_key = cfg.task.sim_image_field
            else:
                raise ValueError('the candidate_set_method error')
            sim_model_name = cfg.sim_model_type.split('/')[-1]
            train_cache_path = os.path.join(
                cache_dir,
                f'{cfg.task.task_name}-{cfg.dataset.name}-'
                f'{cfg.candidate_set_method}-{sim_model_name}-feature.pth',
            )
            train_feature = load_feature_cache(
                cfg, train_cache_path, encoding_method, train_ds, data_key
            )
            test_feature = train_feature[anchor_idx_list]
            _, sim_sample_idx = recall_sim_feature(
                test_feature, train_feature, top_k=cfg.candidate_set_num + 1
            )

            sim_sample_idx = sim_sample_idx[:, 1:].tolist()
            candidate_set_idx = {
                idx: cand for idx, cand in zip(anchor_idx_list, sim_sample_idx)
            }
        with open(candidate_set_cache_filename, 'w') as f:
            logger.info(f'save {candidate_set_cache_filename}...')
            json.dump(candidate_set_idx, f)
    candidate_set_idx = [candidate_set_idx[k] for k in anchor_idx_list]

    spawn(
        gen_data,
        args=(
            cfg,
            anchor_data,
            train_ds,
            candidate_set_idx,
            sub_save_path,
        ),
        nprocs=len(cfg.gpu_ids),
        join=True,
    )

    world_size = len(cfg.gpu_ids)
    subset_size = len(anchor_data) // world_size
    total_data = {}
    for rank in range(world_size):
        subset_start = rank * subset_size
        subset_end = (
            subset_start + subset_size if rank != world_size - 1 else len(anchor_data)
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
        logger.info(f'load the data from {sub_save_path}, the data length: {len(data)}')
        total_data.update(data)
    with open(save_path, 'w') as f:
        json.dump(total_data, f)
    logger.info(f'save the final data to {save_path}')


if __name__ == '__main__':
    logger.info('begin load env variables')
    load_dotenv()
    main()

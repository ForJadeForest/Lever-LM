import json
import logging
import os
import random
from time import sleep
from typing import Dict, List

import hydra
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig
from openicl import PromptTemplate
from PIL import Image
from torch.multiprocessing import spawn
from tqdm import tqdm

from src.load_ds_utils import load_coco_ds, load_vqav2_ds
from src.metrics.info_score import get_info_score
from src.utils import encode_image, encode_text, init_flamingo, recall_sim_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Remove all default handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3],
)
handler.setFormatter(formatter)
logger.addHandler(handler)


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
):
    template = PromptTemplate(
        cfg.task.template,
        column_token_map=dict(cfg.task.column_token_map),
        ice_token=cfg.task.ice_token,
    )

    # 构建test sample prompt
    test_data_text = template.generate_item(test_data)

    test_data_image = test_data[cfg.task.image_field]
    test_data_id = test_data['idx']

    # 构建candidate set
    candidateidx2data = {
        data['idx']: {
            'text_input': template.generate_item(data),
            'image': data[cfg.task.image_field],
            'idx': data['idx'],
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
                batch_size=cfg.batch_size,
                autocast_context=autocast_context,
                only_y_loss=cfg.only_y_loss,
                split_token=cfg.split_token,
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
    cur_idx = 0
    sub_res_basename = (
        os.path.basename(save_path).split('.')[0]
        + f'_rank:{rank}_({subset_start}, {subset_end}).json'
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)
    if os.path.exists(save_path):
        final_res.update(json.load(open(save_path)))
        cur_idx = final_res['cur_idx']
        logger.info(
            f'Rank: {rank} reloading data from {save_path}, begin from {cur_idx + 1}'
        )
    if len(final_res) == subset_size:
        logger.info(f'Rank: {rank} task is Done.')
        return

    subset = subset.select(range(cur_idx + 1, len(subset)))
    for i, test_data in enumerate(
        tqdm(subset, disable=(rank != 0), total=subset_size, initial=cur_idx + 1),
    ):
        candidate_set = train_ds.select(sub_cand_set_idx[i])
        res = generate_single_sample_ice(
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
        cur_idx += 1
        final_res['cur_idx'] = cur_idx
        with open(save_path, 'w') as f:
            json.dump(final_res, f)
    return


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
        f'{cfg.task.task_name}-{cfg.dataset.name}-{"only_y_loss" if cfg.only_y_loss else ""}-'
        f'{cfg.flamingo.hf_root}-{candidate_method}-'
        f'beam_size:{cfg.beam_size}-few_shot:{cfg.few_shot_num}-'
        f'candidate_set_num:{cfg.candidate_set_num}.json'
    )

    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    # 加载数据集
    if cfg.task.task_name == 'caption':
        train_ds = load_coco_ds(cfg, split='train')
    elif cfg.task.task_name == 'vqa':
        train_ds = load_vqav2_ds(cfg, split='train')
    else:
        raise ValueError(f'{cfg.task.task_name=} error, should in ["caption", "vqa"]')

    # sample from train idx
    idx_cache_filename = (
        f'{cfg.dataset.name}-{cfg.sample_num}-'
        f'{cfg.candidate_set_num}-{candidate_method}.json'
    )

    sample_data_idx_cache = os.path.join(cache_dir, idx_cache_filename)
    if os.path.exists(sample_data_idx_cache):
        sample_cache_metainfo = json.load(open(sample_data_idx_cache, 'r'))
        sample_index = sample_cache_metainfo['sample_index']
        candidate_set_idx = sample_cache_metainfo['candidate_set_idx']
        sample_data = train_ds.select(sample_index)
    else:
        sample_cache_metainfo = dict()
        sample_index = random.sample(range(0, len(train_ds)), cfg.sample_num)
        sample_cache_metainfo['sample_index'] = sample_index
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
                data_key = cfg.task.sim_image_field
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

            candidate_set_idx = candidate_set_idx[:, 1:].tolist()
        sample_cache_metainfo['candidate_set_idx'] = candidate_set_idx
        with open(sample_data_idx_cache, 'w') as f:
            logger.info(f'save {sample_data_idx_cache}...')
            json.dump(sample_cache_metainfo, f)

    spawn(
        gen_data,
        args=(
            cfg,
            sample_data,
            train_ds,
            candidate_set_idx,
            sub_save_path,
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
        logger.info(f'load the data from {sub_save_path}, the data length: {len(data)}')
        total_data.update(data)
    total_data.pop('cur_idx')
    with open(save_path, 'w') as f:
        json.dump(total_data, f)
    logger.info(f'save the final data to {save_path}')


if __name__ == '__main__':
    load_dotenv()
    main()

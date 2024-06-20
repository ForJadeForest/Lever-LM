import json
import os
import random
import sys
from time import sleep
from typing import Dict, List

import hydra
import more_itertools
import torch
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from torch.multiprocessing import spawn
from tqdm import tqdm

from open_mmicl.interface import FlamingoInterface
from lever_lm.utils import init_interface
from utils import load_ds


@torch.inference_mode()
def generate_single_sample_icd(
    interface: FlamingoInterface,
    test_data: Dict,
    train_ds: Dataset,
    cfg: DictConfig,
    candidate_seq_idx_list: List,
):
    test_data_id = test_data["idx"]

    candidate_seq_data_list = [
        [train_ds[i] for i in icd_seq] for icd_seq in candidate_seq_idx_list
    ]

    test_lang_x_input = interface.gen_ice_prompt(test_data, add_image_token=True)

    prompts = interface.transfer_prompts([test_data], is_last_for_generation=False)
    x_input = interface.prepare_input(
        prompts, is_last_for_generation=False, add_eos_token=True
    ).to(interface.device)
    query_mask_part = (
        test_lang_x_input.split(cfg.task.split_token)[0] + cfg.task.split_token
    )
    mask_context = query_mask_part

    mask_length = interface.get_input_token_num(mask_context)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    info_score_list = []
    for batch in more_itertools.chunked(candidate_seq_data_list, cfg.batch_size):
        add_query_seq = [seq[:] + [test_data] for seq in batch]
        prompts = interface.transfer_prompts(
            add_query_seq, is_last_for_generation=False
        )

        add_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=False,
            add_eos_token=True,
        ).to(interface.device)
        icd_mask_prompt_list = [
            interface.concat_prompt(
                t[:-1],
                add_eos_token=False,
                add_image_token=True,
                is_last_for_generation=False,
            )
            for t in add_query_seq
        ]
        mask_context_list = [
            icd_mask_prompt + query_mask_part
            for icd_mask_prompt in icd_mask_prompt_list
        ]
        mask_length_list = [
            interface.get_input_token_num(mask_context)
            for mask_context in mask_context_list
        ]
        new_cond_prob = interface.get_cond_prob(
            add_icd_input, mask_length=mask_length_list
        )
        sub_info_score = new_cond_prob - cond_prob
        info_score_list.append(sub_info_score)
    scores = torch.cat(info_score_list)
    topk_scores, indices = scores.topk(cfg.topk)
    better_icd_seq = [candidate_seq_idx_list[i] + [test_data_id] for i in indices]
    better_score_list = topk_scores.cpu().tolist()

    return {test_data_id: {"id_list": better_icd_seq, "score_list": better_score_list}}


def gen_data(
    rank,
    cfg,
    sample_data,
    train_ds,
    candidate_set_idx,
    save_path,
):
    world_size = len(cfg.gpu_ids)
    process_device = f"cuda:{cfg.gpu_ids[rank]}"

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
    interface = init_interface(cfg, device=process_device)

    interface.tokenizer.padding_side = "right"

    final_res = {}
    sub_res_basename = (
        os.path.basename(save_path).split(".")[0]
        + f"_rank:{rank}_({subset_start}, {subset_end}).json"
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)
    if os.path.exists(save_path):
        final_res.update(json.load(open(save_path)))
        logger.info(
            f"Rank: {rank} reloading data from {save_path}, begin from {len(final_res)}"
        )
    if len(final_res) == subset_size:
        logger.info(f"Rank: {rank} task is Done.")
        return

    subset = subset.select(range(len(final_res), len(subset)))
    for i, test_data in enumerate(
        tqdm(
            subset,
            disable=(rank != world_size - 1),
            total=subset_size,
            initial=len(final_res),
            ncols=100,
        ),
    ):
        candidate_set = sub_cand_set_idx[i]
        res = generate_single_sample_icd(
            interface=interface,
            test_data=test_data,
            train_ds=train_ds,
            cfg=cfg,
            candidate_seq_idx_list=candidate_set,
        )
        final_res.update(res)
        with open(save_path, "w") as f:
            json.dump(final_res, f)
    return


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data_random.yaml"
)
def main(cfg: DictConfig):
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)
    cache_dir = cfg.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_dir = os.path.join(cfg.result_dir, "generated_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_proc_save_dir = os.path.join(save_dir, "sub_proc_data")
    if not os.path.exists(sub_proc_save_dir):
        os.makedirs(sub_proc_save_dir)

    save_file_name = (
        f"RandomSeq-{cfg.task.task_name}-{cfg.dataset.name}-"
        f"{cfg.infer_model.name}-scorer:{cfg.scorer}-"
        f"topk:{cfg.topk}-few_shot:{cfg.few_shot_num}-"
        f"candidate_num:{cfg.candidate_seq_num}-sample_num:{cfg.sample_num}.json"
    )

    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    # 加载数据集
    train_ds = load_ds(cfg, "train")

    # sample from train idx
    anchor_set_cache_filename = os.path.join(
        cache_dir, f"{cfg.dataset.name}-anchor_sample_num:{cfg.sample_num}.json"
    )
    if os.path.exists(anchor_set_cache_filename):
        logger.info("the anchor_set_cache_filename exists, loding...")
        anchor_idx_list = json.load(open(anchor_set_cache_filename, "r"))
    else:
        anchor_idx_list = random.sample(range(0, len(train_ds)), cfg.sample_num)
        with open(anchor_set_cache_filename, "w") as f:
            logger.info(f"save {anchor_set_cache_filename}...")
            json.dump(anchor_idx_list, f)
    anchor_data = train_ds.select(anchor_idx_list)

    candidate_set_idx = []
    for k in anchor_idx_list:
        k_cand_list = []
        for _ in range(cfg.candidate_seq_num):
            random_candidate_set = random.sample(
                range(0, len(train_ds)), cfg.few_shot_num
            )
            while k in random_candidate_set:
                random_candidate_set = random.sample(
                    list(range(len(train_ds))), cfg.few_shot_num
                )
            k_cand_list.append(random_candidate_set)
        candidate_set_idx.append(k_cand_list)

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
    # gen_data(
    #     0,
    #     cfg,
    #     anchor_data,
    #     train_ds,
    #     candidate_set_idx,
    #     sub_save_path,
    # )

    world_size = len(cfg.gpu_ids)
    subset_size = len(anchor_data) // world_size
    total_data = {}
    for rank in range(world_size):
        subset_start = rank * subset_size
        subset_end = (
            subset_start + subset_size if rank != world_size - 1 else len(anchor_data)
        )
        sub_res_basename = (
            os.path.basename(save_path).split(".")[0]
            + f"_rank:{rank}_({subset_start}, {subset_end}).json"
        )
        sub_save_path = sub_save_path.replace(
            os.path.basename(sub_save_path), sub_res_basename
        )
        with open(sub_save_path, "r") as f:
            data = json.load(f)
        logger.info(f"load the data from {sub_save_path}, the data length: {len(data)}")
        total_data.update(data)
    with open(save_path, "w") as f:
        json.dump(total_data, f)
    logger.info(f"save the final data to {save_path}")


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data_random.yaml"
)
def hydra_loguru_init(_) -> None:
    hydra_path = hydra.core.hydra_config.HydraConfig.get().run.dir
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name
    logger.remove()
    logger.add(sys.stderr, level=hydra.core.hydra_config.HydraConfig.get().verbose)
    logger.add(os.path.join(hydra_path, f"{job_name}.log"))


if __name__ == "__main__":
    load_dotenv()
    hydra_loguru_init()
    main()

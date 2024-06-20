import json
import os
import sys
from time import sleep
from typing import Dict

import hydra
import torch
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from torch.multiprocessing import spawn
from tqdm import tqdm

from lever_lm.utils import beam_filter, init_interface
from open_mmicl.interface import BaseInterface
from utils import get_cider_score, get_info_score, load_ds


@torch.inference_mode()
def generate_single_sample_icd(
    interface: BaseInterface,
    test_data: Dict,
    cfg: DictConfig,
    candidate_set: Dataset,
):
    test_data_id = test_data["idx"]
    # 构建candidate set
    candidateidx2data = {data["idx"]: data for data in candidate_set}
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
            choosed_icd_seq_list = [candidateidx2data[idx] for idx in icd_id_seq] + [
                test_data
            ]

            filtered_idx_list = sorted(list(filtered_candidateidx2data.keys()))
            if cfg.scorer == "infoscore":
                scores = get_info_score(
                    interface,
                    choosed_icd_seq_list=choosed_icd_seq_list,
                    candidate_set=filtered_candidateidx2data,
                    batch_size=cfg.batch_size,
                    split_token=cfg.task.split_token,
                    construct_order=cfg.construct_order,
                )
            elif cfg.scorer == "cider":
                assert (
                    "coco" in cfg.dataset.name
                ), f"Now CIDEr scorer only support mscoco task"
                scores = get_cider_score(
                    interface,
                    choosed_icd_seq_list,
                    candidate_set=filtered_candidateidx2data,
                    batch_size=cfg.batch_size,
                    train_ann_path=cfg.dataset.train_coco_annotation_file,
                    construct_order=cfg.construct_order,
                    gen_kwargs=cfg.task.gen_args,
                    model_name=cfg.infer_model.name,
                )

            # 选出最高的InfoScore
            topk_scores, indices = scores.topk(cfg.beam_size)
            indices = indices.tolist()
            indices = list(
                map(
                    lambda x: filtered_idx_list[x],
                    indices,
                )
            )
            topk_scores = topk_scores.tolist()

            for idx, score in zip(indices, topk_scores):
                new_test_data_id_list.append([idx, *test_data_id_seq])
                new_test_score_list.append(score)

        new_test_score_list, new_test_data_id_list = beam_filter(
            new_test_score_list, new_test_data_id_list, cfg.beam_size
        )
        test_data_id_list = new_test_data_id_list
    return {
        test_data_id: {"id_list": test_data_id_list, "score_list": new_test_score_list}
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
    if cfg.scorer == "infoscore":
        interface.tokenizer.padding_side = "right"
    elif cfg.scorer == "cider":
        interface.tokenizer.padding_side = "left"

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
        candidate_set = train_ds.select(sub_cand_set_idx[i])
        res = generate_single_sample_icd(
            interface=interface,
            test_data=test_data,
            cfg=cfg,
            candidate_set=candidate_set,
        )
        final_res.update(res)
        with open(save_path, "w") as f:
            json.dump(final_res, f)
    return


@hydra.main(
    version_base=None, config_path="./configs", config_name="generate_data.yaml"
)
def main(cfg: DictConfig):
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)
    cache_dir = cfg.sampler.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_dir = os.path.join(cfg.result_dir, "generated_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_proc_save_dir = os.path.join(save_dir, "sub_proc_data")
    if not os.path.exists(sub_proc_save_dir):
        os.makedirs(sub_proc_save_dir)

    save_file_name = (
        f"{cfg.task.task_name}-{cfg.dataset.name}-"
        f"{cfg.infer_model.name}-{cfg.sampler.sampler_name}-scorer:{cfg.scorer}-construct_order:{cfg.construct_order}-"
        f"beam_size:{cfg.beam_size}-few_shot:{cfg.few_shot_num}-"
        f"candidate_num:{cfg.sampler.candidate_num}-sample_num:{cfg.sample_num}.json"
    )

    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    # 加载数据集
    train_ds = load_ds(cfg, "train")

    # sample from train idx
    sampler = hydra.utils.instantiate(cfg.sampler)
    sampler_result = sampler(train_ds)

    anchor_data = train_ds.select(sampler_result["anchor_set"])
    candidate_set_idx = [
        sampler_result["candidate_set"][k] for k in sampler_result["anchor_set"]
    ]
    # spawn(
    #     gen_data,
    #     args=(
    #         cfg,
    #         anchor_data,
    #         train_ds,
    #         candidate_set_idx,
    #         sub_save_path,
    #     ),
    #     nprocs=len(cfg.gpu_ids),
    #     join=True,
    # )
    gen_data(
        0,
        cfg,
        anchor_data,
        train_ds,
        candidate_set_idx,
        sub_save_path,
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
    version_base=None, config_path="./configs", config_name="generate_data.yaml"
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

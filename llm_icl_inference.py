import datetime
import json
import os
import random
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoProcessor

from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import *
from icd_lm.utils import init_interface
from open_mmicl.vl_icl_inferencer import VLICLInferecer
from utils import caption_postprocess, load_ds, vqa_postprocess


def get_icd_lm_path(cfg):
    if cfg.icd_lm_path is None:
        logger.info(
            f"detect icd_lm_path is None, now try to find in {cfg.result_dir}/model_cpk/{cfg.ex_name}"
        )
        cpk_dir = os.path.join(
            cfg.result_dir, "model_cpk", cfg.task.task_name, cfg.ex_name
        )
        cpk_list = []
        for f in os.listdir(cpk_dir):
            cpk_list.append(os.path.join(cpk_dir, f))
        cpk_list = list(filter(lambda x: cfg.default_cpk_key in x, cpk_list))
        if cpk_list:
            logger.info(f"Detect {cpk_list[0]}, now begin to load cpk...")
            icd_lm_path = cpk_list[0]
        else:
            raise ValueError(
                f"The icd_lm_path is None and detect no checkpoint can use in {cpk_dir}"
            )
    else:
        icd_lm_path = cfg.icd_lm_path
    return icd_lm_path


def init_icd_lm(cfg, icd_lm_path):
    icd_lm = hydra.utils.instantiate(cfg.train.icd_lm)
    state_dict = torch.load(icd_lm_path)["state_dict"]
    state_dict = {k.replace("icd_lm.", ""): v for k, v in state_dict.items()}
    icd_lm.load_state_dict(state_dict)
    processor = AutoProcessor.from_pretrained(cfg.train.icd_lm.clip_name)
    return icd_lm, processor


def record(result_json_path: str, new_data: dict):
    recorded_data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            recorded_data = json.load(f)

    with open(result_json_path, "w") as f:
        recorded_data.update(new_data)
        json.dump(recorded_data, f, indent=4)


def evaluate_retriever(
    retriever_name,
    interface,
    retriever,
    ds,
    base_info,
    shot_num_list,
    result_json_path,
    cfg,
):
    retriever_res = {}
    info = base_info + retriever_name
    for shot_num in shot_num_list:
        logger.info(
            f"Now begin test {cfg.task.task_name}: {retriever_name} with {shot_num=}"
        )
        output_files = info + f"-bs:{cfg.inference_bs}-{shot_num=}"
        icd_idx_list = retriever.retrieve(shot_num)

        metric = inference_sst2(
            interface,
            icd_idx_list,
            ["negative", "postive"],
            ds["train"],
            ds["validation"],
        )
        retriever_res[f"{shot_num=}"] = metric
        logger.info(f"{output_files}: {metric=}")
        record(result_json_path, {info: retriever_res})


def inference_sst2(
    interface,
    icd_idx_list,
    labels,
    index_ds,
    test_ds,
):
    from tqdm import trange

    result = {}
    result_table = torch.zeros(size=(len(test_ds), len(labels)))
    for label_idx, label in enumerate(labels):
        for idx in trange(len(test_ds)):
            ice_idx = [index_ds[i] for i in icd_idx_list[idx]]
            query = test_ds[idx]
            temp_query = {"text": query["text"], "label_text": label}
            prompts = interface.transfer_prompts(
                ice_idx + [temp_query], is_last_for_generation=False
            )

            input_tensor_dict = interface.prepare_input(
                prompts, is_last_for_generation=False
            )
            data = {k: v.to(interface.device) for k, v in input_tensor_dict.items()}
            cond_prob = interface.get_cond_prob(data, [data["input_ids"].shape[-1] - 1])
            result[idx] = {}
            result[idx][label] = {"Prob": cond_prob, "prompts": prompts}
            result_table[idx][label_idx] = cond_prob

    pred = []
    pred = torch.argmax(result_table, dim=-1)
    gt = torch.tensor(test_ds["label"])

    return ((pred == gt).sum() / len(test_ds)).item()


def init_retriever(retriever_name, ds, cfg):
    if retriever_name == "ZeroShot":
        return ZeroRetriever(ds["train"], ds["validation"])
    elif retriever_name == "RandomRetriever":
        return RandRetriever(
            ds["train"],
            ds["validation"],
            seed=cfg.seed,
            fixed=cfg.random_retrieval_fixed,
        )
    elif retriever_name.startswith("MMTopKRetriever"):
        mode = retriever_name.split("-")[-1]
        index_field = (
            cfg.task.icd_text_feature_field
            if mode.endswith("t")
            else cfg.task.image_field
        )
        test_field = (
            cfg.task.image_field
            if mode.startswith("i")
            else cfg.task.icd_text_feature_field
        )

        cache_file = os.path.join(
            cfg.result_dir,
            "cache",
            f'{cfg.task.task_name}-{cfg.dataset.name}-{cfg.mmtopk_clip_name.split("/")[-1]}-{mode}-'
            f"index_field:{index_field}-test_data_num:{cfg.test_data_num}-"
            f"test_field:{test_field}-emb_cache.pth",
        )
        return MMTopkRetriever(
            ds["train"],
            ds["validation"],
            mode=mode,
            index_field=index_field,
            test_field=test_field,
            clip_model_name=cfg.mmtopk_clip_name,
            cache_file=cache_file,
            reversed_order=cfg.mmtopk_reversed_order,
            batch_size=32,
            num_workers=8,
        )
    elif retriever_name == "ICDLMRetriever":
        icd_lm_path = get_icd_lm_path(cfg)
        icd_lm, processor = init_icd_lm(cfg, icd_lm_path=icd_lm_path)
        return ICDLMRetriever(
            ds["train"],
            ds["validation"],
            icd_lm=icd_lm,
            processor=processor,
            query_image_field=cfg.train.icd_lm_ds.query_image_field,
            query_text_field=cfg.train.icd_lm_ds.query_text_field,
            icd_image_field=cfg.train.icd_lm_ds.icd_image_field,
            icd_text_field=cfg.train.icd_lm_ds.icd_text_field,
            device=cfg.device,
            infer_batch_size=cfg.icd_lm_bs,
            infer_num_workers=cfg.icd_lm_num_workers,
            reverse_seq=cfg.reverse_seq,
        )

    return None


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    logger.info(f"{cfg=}")
    result_dir = os.path.join(
        cfg.result_dir,
        "icl_inference",
        cfg.infer_model.name,
        cfg.task.task_name,
        cfg.ex_name,
    )
    result_json_path = os.path.join(result_dir, "metrics.json")

    test_data_num = cfg.test_data_num
    index_data_num = cfg.index_data_num

    ds = load_ds(cfg)

    if index_data_num != -1:
        ds["train"] = ds["train"].select(
            random.sample(range(len(ds["train"])), index_data_num)
        )
    if test_data_num != -1:
        ds["validation"] = ds["validation"].select(range(test_data_num))

    interface = init_interface(cfg, device=cfg.device)

    base_info = f"{str(datetime.datetime.now())}-{test_data_num=}-"

    retriever_list = [
        ("ZeroShot", [0] if cfg.test_zero_shot else []),
        ("RandomRetriever", cfg.shot_num_list if cfg.test_random else []),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2t',
            cfg.shot_num_list if cfg.test_i2t else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2i',
            cfg.shot_num_list if cfg.test_i2i else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-t2t',
            cfg.shot_num_list if cfg.test_t2t else [],
        ),
        (
            "ICDLMRetriever",
            cfg.shot_num_list if cfg.test_icd_lm else [],
        ),
    ]

    # Test for other
    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, ds, cfg)
            evaluate_retriever(
                retriever_name,
                interface,
                retriever_instance,
                ds,
                base_info,
                shot_nums,
                result_json_path,
                cfg,
            )


def shuffle_2d_list(matrix):
    new_matrix = [row.copy() for row in matrix]
    if len(new_matrix[0]) == 1:
        return new_matrix
    for i, row in enumerate(tqdm(new_matrix)):
        while row == matrix[i]:
            random.shuffle(row)
    return new_matrix


if __name__ == "__main__":
    load_dotenv()
    main()

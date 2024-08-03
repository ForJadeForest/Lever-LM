import os
from typing import Dict, List, Optional, Union

import hydra
import more_itertools
import torch
from loguru import logger
from transformers import AutoProcessor

from lever_lm.load_ds_utils import (
    load_coco_ds,
    load_hf_ds,
    load_vqav2_ds,
    load_vlicl_textocr,
)
from open_mmicl.interface import FlamingoInterface, IDEFICSInterface, LLMInterface
from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import postprocess_vqa_generation


def load_ds(cfg, split=None):
    if cfg.task.task_name == "caption":
        ds = load_coco_ds(
            name=cfg.dataset.name,
            train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
            train_coco_annotation_file=cfg.dataset.train_coco_annotation_file,
            val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
            val_coco_annotation_file=cfg.dataset.val_coco_annotation_file,
            karpathy_path=(
                cfg.dataset.karpathy_path
                if hasattr(cfg.dataset, "karpathy_path")
                else None
            ),
            split=split,
        )
    elif cfg.task.task_name == "vqa":
        if cfg.dataset.name == "vlicl_textocr":
            ds = load_vlicl_textocr(
                root_dir=cfg.dataset.root_dir,
                split=split,
            )
        elif cfg.dataset.name == "vqav2":
            ds = load_vqav2_ds(
                version=cfg.dataset.version,
                train_path=cfg.dataset.train_path,
                val_path=cfg.dataset.val_path,
                train_coco_dataset_root=cfg.dataset.train_coco_dataset_root,
                val_coco_dataset_root=cfg.dataset.val_coco_dataset_root,
                split=split,
            )

    else:
        try:
            ds = load_hf_ds(cfg.dataset.hf_ds_name, split=split)
        except Exception as e:
            raise ValueError(f"dataset load fail with error: {e}")
    return ds


@torch.inference_mode()
def get_info_score(
    interface: Union[FlamingoInterface, IDEFICSInterface, LLMInterface],
    choosed_icd_seq_list: List,
    candidate_set: Dict,
    batch_size: int,
    split_token: Optional[str] = None,
    construct_order="left",
):
    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    kwargs = dict(add_image_token=True)
    if isinstance(interface, LLMInterface):
        kwargs = dict()
    test_lang_x_input = interface.gen_text_with_label(
        choosed_icd_seq_list[-1], **kwargs
    )
    prompts = interface.transfer_prompts(
        choosed_icd_seq_list, is_last_for_generation=False
    )

    x_input = interface.prepare_input(
        prompts, is_last_for_generation=False, add_eos_token=True
    ).to(interface.device)

    icd_mask_prompt = interface.concat_prompt(
        choosed_icd_seq_list[:-1],
        add_eos_token=False,
        is_last_for_generation=False,
        **kwargs,
    )
    query_mask_part = test_lang_x_input.split(split_token)[0] + split_token

    mask_context = icd_mask_prompt + query_mask_part

    mask_length = interface.get_input_token_num(mask_context)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]

        # 2.1 拼接文本输入
        if construct_order == "left":
            add_new_icd_seq_list = [
                [new_icd] + choosed_icd_seq_list for new_icd in batch_data
            ]
        elif construct_order == "right":
            add_new_icd_seq_list = [
                choosed_icd_seq_list[:-1] + [new_icd] + [choosed_icd_seq_list[-1]]
                for new_icd in batch_data
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )

        prompts = interface.transfer_prompts(
            add_new_icd_seq_list, is_last_for_generation=False
        )

        add_new_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=False,
            add_eos_token=True,
        ).to(interface.device)
        icd_mask_prompt_list = [
            interface.concat_prompt(
                t[:-1],
                add_eos_token=False,
                is_last_for_generation=False,
                **kwargs,
            )
            for t in add_new_icd_seq_list
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
            add_new_icd_input, mask_length=mask_length_list
        )
        sub_info_score = new_cond_prob - cond_prob
        info_score_list.append(sub_info_score)
    return torch.cat(info_score_list)


@torch.inference_mode()
def get_cider_score(
    interface,
    choosed_icd_seq_list: List,
    candidate_set: Dict,
    batch_size: int,
    model_name: str,
    train_ann_path: str,
    construct_order="left",
    gen_kwargs: Dict = None,
):
    output_dict = {}

    prompts = interface.transfer_prompts(
        choosed_icd_seq_list, is_last_for_generation=True
    )

    x_input = interface.prepare_input(
        prompts, is_last_for_generation=True, add_eos_token=False
    ).to(interface.device)

    origin_outputs = interface.generate(
        **x_input,
        pad_token_id=interface.tokenizer.pad_token_id,
        eos_token_id=interface.tokenizer.eos_token_id,
        **gen_kwargs,
    )

    origin_outputs = origin_outputs.tolist()
    prompt_len = int(x_input["attention_mask"].shape[1])

    generated = interface.tokenizer.batch_decode(
        [output[prompt_len:] for output in origin_outputs],
        skip_special_tokens=True,
    )
    pred_coco = [
        {"image_id": choosed_icd_seq_list[-1]["image_id"], "caption": generated[0]}
    ]

    origin_cider_score = compute_cider(pred_coco, train_ann_path, reduce_cider=False)
    origin_cider_score = origin_cider_score[choosed_icd_seq_list[-1]["image_id"]][
        "CIDEr"
    ]

    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        if construct_order == "left":
            add_new_icd_seq_list = [
                [new_icd] + choosed_icd_seq_list for new_icd in batch_data
            ]
        elif construct_order == "right":
            add_new_icd_seq_list = [
                choosed_icd_seq_list[:-1] + [new_icd] + [choosed_icd_seq_list[-1]]
                for new_icd in batch_data
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )
        prompts = interface.transfer_prompts(
            add_new_icd_seq_list, is_last_for_generation=True
        )
        add_new_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=True,
            add_eos_token=False,
        ).to(interface.device)

        outputs = interface.generate(
            **add_new_icd_input,
            pad_token_id=interface.tokenizer.pad_token_id,
            eos_token_id=interface.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        outputs = outputs.tolist()
        prompt_len = int(add_new_icd_input["attention_mask"].shape[1])

        generated = interface.tokenizer.batch_decode(
            [output[prompt_len:] for output in outputs],
            skip_special_tokens=True,
        )
        for i, data in enumerate(batch_data):
            output_dict[data["idx"]] = {}
            output_dict[data["idx"]]["prediction"] = generated[i]
            output_dict[data["idx"]]["image_id"] = data["image_id"]

    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                "image_id": output_dict[idx]["image_id"],
                "caption": caption_postprocess(
                    output_dict[idx]["prediction"], model_name=model_name
                ),
            }
        )
    cider_score_info = compute_cider(pred_coco, train_ann_path, reduce_cider=False)
    cider_score = []
    for idx in cand_idx:
        img_id = candidate_set[idx]["image_id"]
        cider_score.append(cider_score_info[img_id]["CIDEr"])

    return torch.tensor(cider_score) - origin_cider_score


def caption_postprocess(text, model_name):
    if "flamingo" in model_name:
        return text.split("Output", 1)[0].replace('"', "")
    elif "idefics" in model_name:
        return text.split("Caption", 1)[0].replace('"', "").replace("\n", "")


def vqa_postprocess(text, model_name):
    if "flamingo" in model_name:
        return postprocess_vqa_generation(text)
    elif "idefics" in model_name:
        return postprocess_vqa_generation(text).replace("\n", "")


def get_lever_lm_path(cfg):
    if cfg.lever_lm_path is None:
        logger.info(
            f"detect lever_lm_path is None, now try to find in {cfg.result_dir}/model_cpk/{cfg.ex_name}"
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
            lever_lm_path = cpk_list[0]
        else:
            raise ValueError(
                f"The lever_lm_path is None and detect no checkpoint can use in {cpk_dir}"
            )
    else:
        lever_lm_path = cfg.lever_lm_path
    return lever_lm_path


def init_lever_lm(cfg, lever_lm_path):
    lever_lm = hydra.utils.instantiate(cfg.train.lever_lm)
    state_dict = torch.load(lever_lm_path)["state_dict"]
    state_dict = {k.replace("lever_lm.", ""): v for k, v in state_dict.items()}
    lever_lm.load_state_dict(state_dict)
    processor = AutoProcessor.from_pretrained(cfg.train.lever_lm.clip_name)
    return lever_lm, processor


def exact_match(results):
    import numpy as np

    acc = []
    for result in results:
        prediction = result["prediction"].strip()
        prediction = prediction.strip("\n")
        trunc_index = prediction.find("\n")
        if trunc_index <= 0:
            trunc_index = prediction.find(".")
        if trunc_index > 0:
            prediction = prediction[:trunc_index]

        if str(prediction).lower() == str(result["answer"]).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

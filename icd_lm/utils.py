import json
import os
from typing import List

import faiss
import more_itertools
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from open_mmicl.interface import FlamingoInterface, IDEFICSInterface


def init_lvlm(cfg, **kwargs):
    if "flamingo" in cfg.lvlm.name:
        return FlamingoInterface(
            lang_encoder_path=cfg.lvlm.lang_encoder_path,
            tokenizer_path=cfg.lvlm.tokenizer_path,
            flamingo_checkpoint_dir=cfg.lvlm.flamingo_checkpoint_dir,
            cross_attn_every_n_layers=cfg.lvlm.cross_attn_every_n_layers,
            hf_root=cfg.lvlm.hf_root,
            precision=cfg.precision,
            device=kwargs["device"],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            icd_join_char=cfg.lvlm.icd_join_char,
            load_from_local=cfg.lvlm.load_from_local,
            instruction=cfg.task.instruction,
            init_device=cfg.lvlm.init_device,
            image_field=cfg.task.image_field,
            label_field=cfg.task.output_column,
        )
    elif "idefics" in cfg.lvlm.name:
        return IDEFICSInterface(
            hf_root=cfg.lvlm.hf_root,
            load_from_local=cfg.lvlm.load_from_local,
            precision=cfg.precision,
            device=kwargs["device"],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            instruction=cfg.task.instruction,
            icd_join_char=cfg.lvlm.icd_join_char,
            image_field=cfg.task.image_field,
            label_field=cfg.task.output_column,
        )
    elif "Qwen" in cfg.lvlm.name:
        from open_mmicl.interface.llm_interface import LLMInterface
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(cfg.lvlm.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.lvlm.model_name)

        return LLMInterface(
            model,
            tokenizer,
            precision=cfg.precision,
            input_ids_field_name="input_ids",
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            instruction=cfg.task.instruction,
            icd_join_char=cfg.lvlm.icd_join_char,
            label_field=cfg.task.output_column,
            device=kwargs["device"],
        )

    else:
        raise ValueError("LVLM name error, now only support ['flamingo, idefics']")


def recall_sim_feature(test_vec, train_vec, top_k=200):
    logger.info(f"embedding shape: {train_vec.shape}")
    dim = train_vec.shape[-1]
    index_feat = faiss.IndexFlatIP(dim)
    index_feat.add(train_vec)
    dist, index = index_feat.search(test_vec, top_k)
    return dist, index


@torch.inference_mode()
def encode_text(
    train_ds,
    data_key,
    device,
    model_type="openai/clip-vit-large-patch14",
    batch_size=128,
):
    model = CLIPTextModelWithProjection.from_pretrained(model_type).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    final_text_feature = []

    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        text_list = [i[data_key] for i in batch]
        inputs = tokenizer(text_list, padding=True, return_tensors="pt").to(device)
        text_feature = model(**inputs).text_embeds
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        final_text_feature.append(text_feature)

    final_text_feature = torch.cat(final_text_feature, dim=0)
    return final_text_feature.detach().cpu().numpy()


@torch.inference_mode()
def encode_image(
    train_ds,
    data_key,
    device,
    model_type="openai/clip-vit-large-patch14",
    batch_size=128,
):
    model = CLIPVisionModelWithProjection.from_pretrained(model_type).to(device)
    processor = AutoProcessor.from_pretrained(model_type)
    model.eval()

    final_image_feature = []
    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        images = [i[data_key] for i in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_feature = model(**inputs).image_embeds
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        final_image_feature.append(image_feature)

    final_image_feature = torch.cat(final_image_feature, dim=0)
    return final_image_feature.detach().cpu().numpy()


def data_split(generated_data, train_ratio):
    # 获得有多少条test数据
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]["id_list"]
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少test数据用于训练 剩下部分用于监督val loss
    train_data_len = int(train_ratio * test_dataset_len)
    train_idx_set = set(sorted(list(test_dataset_id_set))[:train_data_len])
    val_idx_set = test_dataset_id_set - train_idx_set

    train_data_list = list()
    val_data_list = list()
    train_data_score = list()
    val_data_score = list()
    for d in generated_data:
        for i in range(len(generated_data[d]["id_list"])):
            query_idx = generated_data[d]["id_list"][i][-1]
            if int(query_idx) in train_idx_set:
                train_data_list.append(generated_data[d]["id_list"][i])
                train_data_score.append(generated_data[d]["score_list"][i])
            elif int(query_idx) in val_idx_set:
                val_data_list.append(generated_data[d]["id_list"][i])
                val_data_score.append(generated_data[d]["score_list"][i])
            else:
                raise ValueError()

    print(f"the train size {len(train_data_list)}, the test size {len(val_data_list)}")

    train_data = {
        "icd_seq": train_data_list,
        "icd_score": train_data_score,
    }
    val_data = {
        "icd_seq": val_data_list,
        "icd_score": val_data_score,
    }
    return train_data, val_data


def collate_fn(batch, processor: CLIPProcessor):
    bs = len(batch)
    collate_dict = {
        "icd_seq_idx": torch.stack([item["icd_seq_idx"] for item in batch]),
    }
    query_input = [d["query_input"] for d in batch]

    query_text_input = (
        [q["text"] for q in query_input] if "text" in query_input[0] else None
    )
    query_image_input = (
        [q["images"] for q in query_input] if "images" in query_input[0] else None
    )
    if query_text_input or query_image_input:
        query_input = processor(
            images=query_image_input,
            text=query_text_input,
            padding=True,
            return_tensors="pt",
        )
        collate_dict["query_input"] = query_input

    icd_input_list = [d["icd_input"] for d in batch]
    icd_image_input = icd_text_input = None
    if "text" in icd_input_list[0]:
        icd_num = len(icd_input_list[0]["text"])
        icd_text_input = [i["text"] for i in icd_input_list]
        icd_text_input = [i for icd_text in icd_text_input for i in icd_text]
    if "images" in icd_input_list[0]:
        icd_num = len(icd_input_list[0]["images"])
        icd_image_input = [i["images"] for i in icd_input_list]
        icd_image_input = [i for icd_image in icd_image_input for i in icd_image]

    if icd_image_input or icd_text_input:
        icd_input = processor(
            images=icd_image_input,
            text=icd_text_input,
            padding=True,
            return_tensors="pt",
        )
        if "input_ids" in icd_input:
            icd_input["input_ids"] = icd_input["input_ids"].view(bs, icd_num, -1)
            icd_input["attention_mask"] = icd_input["attention_mask"].view(
                bs, icd_num, -1
            )
        if "pixel_values" in icd_input:
            icd_input["pixel_values"] = icd_input["pixel_values"].view(
                bs, icd_num, *icd_input["pixel_values"].shape[1:]
            )
        collate_dict["icd_input"] = icd_input
    return collate_dict


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


class VLGenInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}
    origin_image_dict = {}

    def __init__(
        self,
        num: int,
    ) -> None:
        self.num = num
        self.output_dict = {}
        self.prediction_dict = {}
        self.results_dict = {}
        self.other_meta_info_dict = {}
        self.idx_list = []
        self.origin_prompt_dict = {}

    def creat_index(self, test_ds: Dataset):
        self.idx_list = [i for i in range(len(test_ds))]

    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        self.results_dict = {
            str(idx): {
                "output": self.output_dict[str(idx)][0],
                "pure_output": self.output_dict[str(idx)][1],
                "prediction": self.prediction_dict[str(idx)],
            }
            for idx in self.idx_list
        }
        for field in self.other_meta_info_dict:
            for idx in self.idx_list:
                if field in self.results_dict[str(idx)]:
                    logger.warning(
                        "the other meta info field name has duplicate! Please check for avoiding to losing info"
                    )
                    continue
                self.results_dict[str(idx)][field] = self.other_meta_info_dict[field][
                    str(idx)
                ]
        save_path = f"{output_json_filepath}/{output_json_filename}.json"
        if not os.path.exists(output_json_filepath):
            os.makedirs(output_json_filepath)
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
            json_file.close()

    def save_prediction_and_output(self, prediction, output, origin_prompt, idx):
        self.prediction_dict[str(idx)] = prediction
        self.output_dict[str(idx)] = output
        self.origin_prompt_dict[str(idx)] = origin_prompt

    def save_origin_info(self, meta_field: str, test_ds: Dataset):
        meta_dict = {}
        meta_list = test_ds[meta_field]
        for idx, m_d in enumerate(meta_list):
            meta_dict[str(idx)] = m_d
        self.other_meta_info_dict[meta_field] = meta_dict
